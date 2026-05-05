"""
Prepare the four distillation benchmark datasets used in the paper (Table KD_RougeL):
  - Dolly        (databricks/databricks-dolly-15k)
  - Alpaca       (tatsu-lab/alpaca)
  - S-NI         (Super-Natural Instructions — allenai/natural_instructions v2)
  - DialogueSum  (knkarthick/dialogsum)

Each dataset is saved as JSONL with fields:
  {"prompt": "...", "chosen": "...", "rejected": "..."}

Split sizes (from paper):
  Dataset     Train   Valid   Test
  Dolly       11,435  1,000   500
  Alpaca      10,396  500     500
  S-NI        10,414  500     1,902
  DialogSum   12,460  500     1,500
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
    return {"prompt": prompt.strip(), "chosen": response.strip(), "rejected": response.strip()}


# ─────────────────────────────────────────────
# Dolly  — Train: 11435 | Valid: 1000 | Test: 500
# ─────────────────────────────────────────────
def prep_dolly(out_dir):
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
    save_jsonl(examples[:11435],              f"{out_dir}/dolly/train.jsonl")
    save_jsonl(examples[11435:12435],         f"{out_dir}/dolly/valid.jsonl")
    save_jsonl(examples[12435:12935],         f"{out_dir}/dolly/test.jsonl")


# ─────────────────────────────────────────────
# Alpaca  — Train: 10396 | Valid: 500 | Test: 500
# ─────────────────────────────────────────────
def prep_alpaca(out_dir):
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
    save_jsonl(examples[:10396],              f"{out_dir}/alpaca/train.jsonl")
    save_jsonl(examples[10396:10896],         f"{out_dir}/alpaca/valid.jsonl")
    save_jsonl(examples[10896:11396],         f"{out_dir}/alpaca/test.jsonl")


# ─────────────────────────────────────────────
# S-NI  — Train: 10414 | Valid: 500 | Test: 1902
# ─────────────────────────────────────────────
def prep_sni(out_dir):
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
    save_jsonl(examples[:10414],              f"{out_dir}/sni/train.jsonl")
    save_jsonl(examples[10414:10914],         f"{out_dir}/sni/valid.jsonl")
    save_jsonl(examples[10914:12816],         f"{out_dir}/sni/test.jsonl")


# ─────────────────────────────────────────────
# DialogueSum  — Train: 12460 | Valid: 500 | Test: 1500
# Uses official train/validation/test splits
# ─────────────────────────────────────────────
def prep_dialoguesum(out_dir):
    print("Preparing DialogueSum...")
    train_ds = load_dataset("knkarthick/dialogsum", split="train")
    valid_ds  = load_dataset("knkarthick/dialogsum", split="validation")
    test_ds   = load_dataset("knkarthick/dialogsum", split="test")

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
    valid_ex = process(valid_ds)
    test_ex  = process(test_ds)

    random.shuffle(train_ex)
    save_jsonl(train_ex[:12460],   f"{out_dir}/dialoguesum/train.jsonl")
    save_jsonl(valid_ex[:500],     f"{out_dir}/dialoguesum/valid.jsonl")
    save_jsonl(test_ex[:1500],     f"{out_dir}/dialoguesum/test.jsonl")


def main():
    parser = argparse.ArgumentParser(description="Prepare distillation benchmark datasets.")
    parser.add_argument("--out_dir", type=str, default="data/distillation")
    parser.add_argument("--datasets", nargs="+",
                        default=["dolly", "alpaca", "sni", "dialoguesum"],
                        choices=["dolly", "alpaca", "sni", "dialoguesum"])
    args = parser.parse_args()

    if "dolly" in args.datasets:
        prep_dolly(args.out_dir)
    if "alpaca" in args.datasets:
        prep_alpaca(args.out_dir)
    if "sni" in args.datasets:
        prep_sni(args.out_dir)
    if "dialoguesum" in args.datasets:
        prep_dialoguesum(args.out_dir)

    print("\nAll datasets prepared.")
    print(f"Data root: {args.out_dir}/")
    print("Each dataset has train.jsonl, valid.jsonl, test.jsonl")


if __name__ == "__main__":
    main()
