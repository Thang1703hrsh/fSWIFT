"""
Prepare the four distillation benchmark datasets used in the paper (Table KD_RougeL):
  - Dolly        (databricks/databricks-dolly-15k)
  - Alpaca       (tatsu-lab/alpaca)
  - S-NI         (Super-Natural Instructions — allenai/natural_instructions v2)
  - DialogueSum  (knkarthick/dialogsum)

Each dataset is saved as JSONL with fields:
  {"prompt": "...", "chosen": "...", "rejected": "..."}

The "chosen" field is the ground-truth response.
The "rejected" field is a placeholder (same as chosen) — in the distillation
setting SWIFT uses only the teacher's token weights computed offline; the
preferred/rejected split is used by the training pipeline but the rejected
response will be overwritten by student-generated outputs during token weight
estimation (token_weight_estimation.py uses both chosen and rejected).

For the distillation experiment the pipeline is:
  1. Prepare data (this script)
  2. Compute token weights offline using teacher (Qwen2.5-7B-Instruct)
  3. Train student (GPT2-1.5B / gpt2-xl) with fswift loss
  4. Evaluate with ROUGE-L (eval_rouge.py)
"""

import os
import json
import argparse
import random
from datasets import load_dataset
from tqdm import tqdm

random.seed(42)


def save_jsonl(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  Saved {len(data)} examples → {path}")


def make_example(prompt, response):
    """Wrap into the format expected by preference_datasets.py.
    'rejected' is set to chosen as a placeholder — distillation.sh overwrites
    it with student-generated responses via generate_vllm.py before calling
    token_weight_estimation.py."""
    return {"prompt": prompt.strip(), "chosen": response.strip(), "rejected": response.strip()}


# ─────────────────────────────────────────────
# Dolly  (databricks/databricks-dolly-15k)
# ─────────────────────────────────────────────
def prep_dolly(out_dir, n_train=10000, n_test=500):
    print("Preparing Dolly...")
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    examples = []
    for row in ds:
        instruction = row.get("instruction", "").strip()
        context = row.get("context", "").strip()
        response = row.get("response", "").strip()
        if not instruction or not response:
            continue
        prompt = f"{instruction}\n{context}".strip() if context else instruction
        examples.append(make_example(prompt, response))
    random.shuffle(examples)
    save_jsonl(examples[:n_train], f"{out_dir}/dolly/train.jsonl")
    save_jsonl(examples[n_train:n_train + n_test], f"{out_dir}/dolly/test.jsonl")


# ─────────────────────────────────────────────
# Alpaca  (tatsu-lab/alpaca)
# ─────────────────────────────────────────────
def prep_alpaca(out_dir, n_train=10000, n_test=500):
    print("Preparing Alpaca...")
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    examples = []
    for row in ds:
        instruction = row.get("instruction", "").strip()
        inp = row.get("input", "").strip()
        output = row.get("output", "").strip()
        if not instruction or not output:
            continue
        prompt = f"{instruction}\n{inp}".strip() if inp else instruction
        examples.append(make_example(prompt, output))
    random.shuffle(examples)
    save_jsonl(examples[:n_train], f"{out_dir}/alpaca/train.jsonl")
    save_jsonl(examples[n_train:n_train + n_test], f"{out_dir}/alpaca/test.jsonl")


# ─────────────────────────────────────────────
# S-NI  (Super-Natural Instructions)
# Uses the community mirror: Muennighoff/natural-instructions
# ─────────────────────────────────────────────
def prep_sni(out_dir, n_train=10000, n_test=500):
    print("Preparing S-NI (Super-Natural Instructions)...")
    ds = load_dataset("Muennighoff/natural-instructions", split="train")
    examples = []
    for row in tqdm(ds, desc="S-NI"):
        definition = row.get("definition", "").strip()
        inputs = row.get("inputs", "").strip()
        targets = row.get("targets", "")
        if isinstance(targets, list):
            targets = targets[0] if targets else ""
        targets = targets.strip()
        if not inputs or not targets:
            continue
        prompt = f"{definition}\n\nInput: {inputs}" if definition else f"Input: {inputs}"
        examples.append(make_example(prompt, targets))
    random.shuffle(examples)
    save_jsonl(examples[:n_train], f"{out_dir}/sni/train.jsonl")
    save_jsonl(examples[n_train:n_train + n_test], f"{out_dir}/sni/test.jsonl")


# ─────────────────────────────────────────────
# DialogueSum  (knkarthick/dialogsum)
# ─────────────────────────────────────────────
def prep_dialoguesum(out_dir, n_test=500):
    print("Preparing DialogueSum...")
    train_ds = load_dataset("knkarthick/dialogsum", split="train")
    test_ds  = load_dataset("knkarthick/dialogsum", split="test")

    def process(ds):
        out = []
        for row in ds:
            dialogue = row.get("dialogue", "").strip()
            summary  = row.get("summary", "").strip()
            if not dialogue or not summary:
                continue
            prompt = f"Summarize the following dialogue:\n{dialogue}"
            out.append(make_example(prompt, summary))
        return out

    train_ex = process(train_ds)
    test_ex  = process(test_ds)
    random.shuffle(train_ex)
    save_jsonl(train_ex, f"{out_dir}/dialoguesum/train.jsonl")
    save_jsonl(test_ex[:n_test], f"{out_dir}/dialoguesum/test.jsonl")


def main():
    parser = argparse.ArgumentParser(description="Prepare distillation benchmark datasets.")
    parser.add_argument("--out_dir", type=str, default="data/distillation",
                        help="Root output directory for all datasets.")
    parser.add_argument("--n_train", type=int, default=10000,
                        help="Max training examples per dataset (Dolly/Alpaca/S-NI).")
    parser.add_argument("--n_test", type=int, default=500,
                        help="Test examples per dataset.")
    parser.add_argument("--datasets", nargs="+",
                        default=["dolly", "alpaca", "sni", "dialoguesum"],
                        choices=["dolly", "alpaca", "sni", "dialoguesum"],
                        help="Which datasets to prepare.")
    args = parser.parse_args()

    if "dolly" in args.datasets:
        prep_dolly(args.out_dir, args.n_train, args.n_test)
    if "alpaca" in args.datasets:
        prep_alpaca(args.out_dir, args.n_train, args.n_test)
    if "sni" in args.datasets:
        prep_sni(args.out_dir, args.n_train, args.n_test)
    if "dialoguesum" in args.datasets:
        prep_dialoguesum(args.out_dir, args.n_test)

    print("\nAll datasets prepared.")
    print(f"Data root: {args.out_dir}/")
    print("Each dataset has train.jsonl and test.jsonl with fields: prompt, chosen, rejected")


if __name__ == "__main__":
    main()
