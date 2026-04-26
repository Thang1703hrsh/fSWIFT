#! /bin/bash
set -euo pipefail

MODEL="model_hub/Qwen1.5-1.8B/SPIN/ite0"
INPUT="data/Ultrachat200k/SFT/trainSFT.jsonl"
OUTDIR="data/Ultrachat200k/SPIN/ite1"
BATCH=8
MAX_NEW=512
FRAC_LEN=1000000

#### Gen sample for SPIN
# loop over data_frac = 0..70
for frac in {0..1}; do
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

python -u train.py \
  model=qwen \
  model.name_or_path=model_hub/Qwen1.5-1.8B/SPIN/ite0 \
  loss=dpo \
  base_data_dir=data \
  datasets='["Ultrachat200k/SPIN/ite0","Ultrachat200k/SPIN/ite1"]'
