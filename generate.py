import json, argparse, time
from pathlib import Path
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model',        type=str,   required=True)
    p.add_argument('--input_dir',    type=str,   required=True)
    p.add_argument('--output_dir',   type=str,   required=True)
    p.add_argument('--output_dir2',   type=str,  default="tue",   required=False)
    p.add_argument('--data_frac',    type=int,   default=0)
    p.add_argument('--frac_len',     type=int,   default=0)
    p.add_argument('--batch_size',   type=int,   default=32)
    p.add_argument('--max_new_tokens', type=int, default=256)
    p.add_argument('--split',        type=str,   choices=['train','test'], default='train')
    return p.parse_args()

def read_jsonl(path):
    with open(path, 'r') as f:
        return [json.loads(l) for l in f]

def write_jsonl(path, items):
    with open(path, 'w') as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + '\n')

def main():
    args = parse_args()
    device = torch.device('cuda:0')

    # load model & tokenizer
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token

    # load and slice data
    data = read_jsonl(args.input_dir)
    if args.frac_len > 0:
        start = args.data_frac * args.frac_len
        end   = start + args.frac_len
        data = data[start:end]
    prompts = [d['prompt'] for d in data]
    truths  = [d['ground_truth'] for d in data]

    max_input_len = model.config.n_positions - args.max_new_tokens
    results = []
    t0 = time.time()
    for i in tqdm(range(0, len(prompts), args.batch_size)):
        batch = prompts[i:i+args.batch_size]
        enc = tokenizer(
            batch,
            return_tensors='pt',
            padding='longest',
            truncation=True,
            max_length=max_input_len,
            pad_to_multiple_of=8,
        ).to(device)


        out = model.generate(**enc,
                             max_new_tokens=args.max_new_tokens,
                             pad_token_id=tokenizer.eos_token_id)

        # remove prompt tokens
        gen = []
        for inp_ids, out_ids in zip(enc['input_ids'], out):
            gen_ids = out_ids[len(inp_ids):]
            txt = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            gen.append(txt)

        results.extend(gen)

    print(f"[generate_simple] elapsed {time.time()-t0:.1f}s on {len(prompts)} samples")

    # collect and dump
    items = []
    for p, ref, hyp in zip(prompts, truths, results):
        items.append({'prompt': p, 'chosen': ref, 'rejected': hyp})

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.frac_len > 60000:
        fname = f"{args.split}.jsonl"
    else:
        fname = f"{args.split}_{args.data_frac}.jsonl"
    write_jsonl(out_dir / fname, items)

    if args.output_dir2 != 'tue':
        out_dir2 = Path(args.output_dir2)
        out_dir2.mkdir(parents=True, exist_ok=True)
        if args.frac_len > 60000:
            fname = f"{args.split}.jsonl"
        else:
            fname = f"{args.split}_{args.data_frac}.jsonl"
        write_jsonl(out_dir2 / fname, items)

if __name__ == '__main__':
    main()
