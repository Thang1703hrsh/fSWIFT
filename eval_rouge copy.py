"""
ROUGE-L evaluation for the knowledge distillation experiment (Table KD_RougeL).

For each dataset the student model generates a response to each test prompt,
then we compute ROUGE-L against the reference (chosen) response.

Usage:
    python eval_rouge.py \
        --model_path model_hub/gpt2-xl/distill \
        --data_dir   data/distillation \
        --datasets   dolly alpaca sni dialoguesum \
        --output     eval_results/distillation/results.json \
        --max_new_tokens 256 \
        --batch_size 8

Output JSON:
    {
      "dolly":       {"rouge_l": 25.94, "n": 500},
      "alpaca":      {"rouge_l": 30.69, "n": 500},
      "sni":         {"rouge_l": 26.43, "n": 500},
      "dialoguesum": {"rouge_l": 33.74, "n": 500},
      "avg":         29.20
    }
"""

import argparse
import json
import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from rouge_score import rouge_scorer
from tqdm import tqdm


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def generate_responses(model, tokenizer, prompts, max_new_tokens, batch_size, device):
    responses = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        batch_prompts = prompts[i: i + batch_size]
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=768,
        ).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        # Decode only the newly generated tokens (strip prompt)
        prompt_lengths = inputs["input_ids"].shape[1]
        for out in outputs:
            gen_tokens = out[prompt_lengths:]
            text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            responses.append(text)
    return responses


def compute_rouge_l(predictions, references):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = []
    for pred, ref in zip(predictions, references):
        score = scorer.score(ref, pred)["rougeL"].fmeasure
        scores.append(score)
    avg = 100.0 * sum(scores) / len(scores) if scores else 0.0
    return round(avg, 2)


def eval_dataset(model, tokenizer, data_path, max_new_tokens, batch_size, device):
    data = load_jsonl(data_path)
    prompts    = [ex["prompt"]  for ex in data]
    references = [ex["chosen"]  for ex in data]
    predictions = generate_responses(model, tokenizer, prompts, max_new_tokens, batch_size, device)
    rouge_l = compute_rouge_l(predictions, references)
    return {"rouge_l": rouge_l, "n": len(data)}


def main():
    parser = argparse.ArgumentParser(description="ROUGE-L evaluation for distillation.")
    parser.add_argument("--model_path", required=True, help="Path to the trained student model.")
    parser.add_argument("--data_dir",   required=True, help="Root dir from distill_data_prep.py.")
    parser.add_argument("--datasets", nargs="+",
                        default=["dolly", "alpaca", "sni", "dialoguesum"],
                        choices=["dolly", "alpaca", "sni", "dialoguesum"])
    parser.add_argument("--split", default="test", help="Which split to evaluate (default: test).")
    parser.add_argument("--output",   default="eval_results/distillation/results.json")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--batch_size",     type=int, default=8)
    parser.add_argument("--device",   default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {args.model_path} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to(device)
    model.eval()

    results = {}
    for ds_name in args.datasets:
        data_path = os.path.join(args.data_dir, ds_name, f"{args.split}.jsonl")
        if not os.path.exists(data_path):
            print(f"[SKIP] {ds_name}: file not found at {data_path}")
            results[ds_name] = {"rouge_l": None, "n": 0}
            continue
        print(f"\n===== Evaluating {ds_name} =====")
        results[ds_name] = eval_dataset(
            model, tokenizer, data_path,
            args.max_new_tokens, args.batch_size, device
        )
        print(f"  ROUGE-L: {results[ds_name]['rouge_l']:.2f}  (n={results[ds_name]['n']})")

    valid_scores = [v["rouge_l"] for v in results.values() if v["rouge_l"] is not None]
    results["avg"] = round(sum(valid_scores) / len(valid_scores), 2) if valid_scores else None

    print("\n" + "=" * 50)
    print("ROUGE-L Summary")
    print("=" * 50)
    for ds_name in args.datasets:
        r = results[ds_name]["rouge_l"]
        print(f"  {ds_name:<15} {r}")
    print(f"  {'avg':<15} {results['avg']}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
