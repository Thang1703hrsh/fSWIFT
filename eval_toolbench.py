#!/usr/bin/env python3
"""
ToolBench evaluation: Act.EM, F1, HalluRate, Rouge-L
Used by scripts/eval_reasoning_agentic.sh

Expected data format — one JSON object per line (JSONL):
  {
    "id":                  str (optional),
    "query":               str,          # user instruction
    "tools": [                           # available tools for this query
      {
        "name":        str,
        "description": str,
        "parameters":  dict              # JSON Schema object
      }
    ],
    "reference_action":    str,          # ground-truth tool name to call
    "reference_arguments": dict,         # ground-truth call arguments
    "reference_response":  str           # ground-truth response text (for Rouge-L)
  }

Converting raw ToolLLM data
---------------------------
If you have the official ToolBench G123 data from
  https://huggingface.co/datasets/ToolBench/ToolBench
you can convert the test split with the snippet in TOOLBENCH_DATA_PREP.md
(the key fields map to: query→conversations[-2].content,
 tools→api_list, reference_action→target.tool_name, etc.)

Metrics
-------
  Act.EM       exact match of predicted tool name vs reference
  F1           token-level F1 between predicted and reference
               arguments (JSON-serialised, lowercased)
  HalluRate    fraction where the predicted tool is not in the
               available tool list (lower is better)
  Rouge-L      ROUGE-L between full predicted output and reference
               response text
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from rouge_score import rouge_scorer as rouge_lib

# ── Prompt template ──────────────────────────────────────────
_SYSTEM = (
    "You are a helpful AI assistant with access to external tools. "
    "When the user's request requires a tool call, output EXACTLY one JSON "
    "block wrapped in <tool_call> tags and nothing else before it:\n\n"
    "<tool_call>\n"
    '{"name": "<tool_name>", "arguments": {<key-value pairs>}}\n'
    "</tool_call>\n\n"
    "Available tools (JSON Schema):\n{tools}"
)


def build_prompt(query: str, tools: list) -> str:
    system = _SYSTEM.format(tools=json.dumps(tools, indent=2))
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{query}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


# ── Output parser ─────────────────────────────────────────────
def parse_tool_call(text: str) -> tuple:
    """Return (tool_name, arguments_dict) or (None, {}) if unparseable."""

    # Format 1: <tool_call>{...}</tool_call>  (primary)
    m = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(1))
            return data.get("name"), data.get("arguments") or {}
        except json.JSONDecodeError:
            pass

    # Format 2: fenced JSON block with "name" key
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(1))
            if "name" in data:
                return data.get("name"), data.get("arguments") or {}
        except json.JSONDecodeError:
            pass

    # Format 3: ReAct-style  Action: ...\nAction Input: {...}
    m_act = re.search(r"Action:\s*(\S+)", text)
    if m_act:
        args = {}
        m_inp = re.search(r"Action Input:\s*(\{.*?\})", text, re.DOTALL)
        if m_inp:
            try:
                args = json.loads(m_inp.group(1))
            except json.JSONDecodeError:
                pass
        return m_act.group(1).strip(), args

    return None, {}


# ── Metrics ───────────────────────────────────────────────────
def token_f1(pred: str, ref: str) -> float:
    p_tok = pred.lower().split()
    r_tok = ref.lower().split()
    if not p_tok or not r_tok:
        return 0.0
    common = sum((Counter(p_tok) & Counter(r_tok)).values())
    if common == 0:
        return 0.0
    prec = common / len(p_tok)
    rec  = common / len(r_tok)
    return 2 * prec * rec / (prec + rec)


def compute_metrics(samples: list) -> dict:
    scorer = rouge_lib.RougeScorer(["rougeL"], use_stemmer=False)

    act_em, f1s, hallur, rouges = [], [], [], []

    for s in samples:
        ref_action = s["reference_action"]
        ref_args   = s.get("reference_arguments") or {}
        ref_resp   = s.get("reference_response") or ""
        tool_names = {t["name"] for t in s.get("tools", [])}
        pred_text  = s["prediction"]

        pred_action, pred_args = parse_tool_call(pred_text)

        act_em.append(float(pred_action == ref_action))

        f1s.append(token_f1(
            json.dumps(pred_args,  sort_keys=True),
            json.dumps(ref_args,   sort_keys=True),
        ))

        # Hallucination: called a tool that doesn't exist in the list
        if pred_action is None:
            hallur.append(0.0)          # no tool called → not a hallucination
        else:
            hallur.append(0.0 if pred_action in tool_names else 1.0)

        rouges.append(
            scorer.score(ref_resp, pred_text)["rougeL"].fmeasure
        )

    n = len(samples)
    return {
        "n_samples":          n,
        "act_em":             sum(act_em)  / n,
        "f1":                 sum(f1s)     / n,
        "hallucination_rate": sum(hallur)  / n,
        "rouge_l":            sum(rouges)  / n,
    }


# ── Main ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="ToolBench evaluation (Act.EM, F1, HalluRate, Rouge-L)"
    )
    parser.add_argument("--model_path",      required=True,
                        help="HuggingFace model ID or local checkpoint directory")
    parser.add_argument("--data_path",       required=True,
                        help="JSONL test file (see module docstring for schema)")
    parser.add_argument("--output_path",     required=True,
                        help="Output JSON file for metrics + per-sample predictions")
    parser.add_argument("--max_samples",     type=int,   default=None,
                        help="Truncate to first N samples (default: all)")
    parser.add_argument("--max_new_tokens",  type=int,   default=512)
    parser.add_argument("--temperature",     type=float, default=0.0)
    parser.add_argument("--tensor_parallel", type=int,   default=1,
                        help="Tensor parallel degree for vLLM (ignored with --backend hf)")
    parser.add_argument("--backend",         choices=["vllm", "hf"], default="vllm",
                        help="Inference backend: 'vllm' (default, fast) or 'hf' (Kaggle-friendly)")
    args = parser.parse_args()

    # Load test data
    data_path = Path(args.data_path)
    samples = []
    with data_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    if args.max_samples:
        samples = samples[: args.max_samples]
    print(f"Loaded {len(samples)} samples from {data_path}")

    # Build prompts
    prompts = [build_prompt(s["query"], s.get("tools", [])) for s in samples]

    print(f"Running inference on {len(prompts)} samples … (backend={args.backend})")

    if args.backend == "vllm":
        from vllm import LLM, SamplingParams
        llm = LLM(
            model=args.model_path,
            tensor_parallel_size=args.tensor_parallel,
            dtype="float16",
            trust_remote_code=True,
            gpu_memory_utilization=0.9,
        )
        sampling = SamplingParams(
            temperature=args.temperature,
            max_tokens=args.max_new_tokens,
        )
        outputs = llm.generate(prompts, sampling)
        for s, out in zip(samples, outputs):
            s["prediction"] = out.outputs[0].text

    else:  # hf backend
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from tqdm import tqdm

        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()

        for s, prompt in tqdm(zip(samples, prompts), total=len(samples)):
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=(args.temperature > 0),
                    temperature=args.temperature if args.temperature > 0 else None,
                    pad_token_id=tokenizer.eos_token_id,
                )
            new_tokens = out[0][inputs["input_ids"].shape[1]:]
            s["prediction"] = tokenizer.decode(new_tokens, skip_special_tokens=True)

    # Compute metrics
    metrics = compute_metrics(samples)
    print("\n─── ToolBench Results ───────────────────────────")
    print(f"  Samples   : {metrics['n_samples']}")
    print(f"  Act.EM    : {metrics['act_em']*100:.2f}")
    print(f"  F1        : {metrics['f1']*100:.2f}")
    print(f"  HalluRate : {metrics['hallucination_rate']*100:.2f}")
    print(f"  Rouge-L   : {metrics['rouge_l']*100:.2f}")
    print("─────────────────────────────────────────────────")

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump({"metrics": metrics, "samples": samples}, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
