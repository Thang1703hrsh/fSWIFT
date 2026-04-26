#! /bin/bash
set -euo pipefail

MODEL="model_hub/gpt2_120M_SPIN_ite0"
INPUT="data/Ultrachat200k/SFT/trainSFT.jsonl"
OUTDIR="data/Ultrachat200k/ite1/WPIN"
BATCH=8
MAX_NEW=256
FRAC_LEN=10000

# loop over data_frac = 0..70
for frac in {0..6}; do
  python generate.py \
    --model       "$MODEL" \
    --input_dir   "$INPUT" \
    --output_dir  "$OUTDIR" \
    --batch_size  $BATCH \
    --max_new_tokens $MAX_NEW \
    --data_frac   $frac \
    --frac_len    $FRAC_LEN \
    --split       train
done
