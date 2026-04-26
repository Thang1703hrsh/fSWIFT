#! /bin/bash
set -euo pipefail

MODEL="model_hub/Qwen1.5-1.8B/SFT"
INPUT="data/Ultrachat200k/SFT/trainSFT.jsonl"
OUTDIR="data/Ultrachat200k/SPIN/ite0/train"
BATCH=8
MAX_NEW=512
FRAC_LEN=1000000
FRAC=0

#### Gen sample
# loop over data_frac = 0..70
python generate_vllm.py \
  --model       "$MODEL" \
  --input_dir   "$INPUT" \
  --output_dir  "$OUTDIR" \
  --max_new_tokens $MAX_NEW \
  --data_frac   $FRAC \
  --frac_len    $FRAC_LEN \
  --split       train

#### Train SPIN
python -u train.py \
  model=qwen \
  model.name_or_path=model_hub/Qwen1.5-1.8B/SFT/ \
  loss=dpo \
  base_data_dir=data \
  ckpt_dir=model_hub/Qwen1.5-1.8B/SPIN/ite0/ \
  datasets='["Ultrachat200k/SPIN/ite0"]'


MODEL="model_hub/Qwen1.5-1.8B/SPIN/ite0"
INPUT="data/Ultrachat200k/SFT/trainSFT.jsonl"
OUTDIR="data/Ultrachat200k/SPIN/ite1/train"
BATCH=8
MAX_NEW=512
FRAC_LEN=1000000
FRAC=0

#### Gen sample for SPIN
# loop over data_frac = 0..70
python generate_vllm.py \
  --model       "$MODEL" \
  --input_dir   "$INPUT" \
  --output_dir  "$OUTDIR" \
  --max_new_tokens $MAX_NEW \
  --data_frac   $FRAC \
  --frac_len    $FRAC_LEN \
  --split       train


python -u train.py \
  model=qwen \
  model.name_or_path=model_hub/Qwen1.5-1.8B/SPIN/ite0 \
  loss=dpo \
  base_data_dir=data \
  ckpt_dir=model_hub/Qwen1.5-1.8B/SPIN/ite1/ \
  datasets='["Ultrachat200k/SPIN/ite0","Ultrachat200k/SPIN/ite1"]'



MODEL="model_hub/Qwen1.5-1.8B/SPIN/ite1"
INPUT="data/Ultrachat200k/SFT/trainSFT.jsonl"
OUTDIR="data/Ultrachat200k/SPIN/ite2/train"
BATCH=8
MAX_NEW=512
FRAC_LEN=1000000
FRAC=0

# loop over data_frac = 0..70
python generate_vllm.py \
  --model       "$MODEL" \
  --input_dir   "$INPUT" \
  --output_dir  "$OUTDIR" \
  --max_new_tokens $MAX_NEW \
  --data_frac   $FRAC \
  --frac_len    $FRAC_LEN \
  --split       train



python -u train.py \
  model=qwen \
  model.name_or_path=model_hub/Qwen1.5-1.8B/SPIN/ite1 \
  loss=dpo \
  base_data_dir=data \
  ckpt_dir=model_hub/Qwen1.5-1.8B/SPIN/ite2/ \
  datasets='["Ultrachat200k/SPIN/ite1","Ultrachat200k/SPIN/ite2"]'




MODEL="model_hub/Qwen1.5-1.8B/SPIN/ite2"
INPUT="data/Ultrachat200k/SFT/trainSFT.jsonl"
OUTDIR="data/Ultrachat200k/SPIN/ite3/train"
BATCH=8
MAX_NEW=512
FRAC_LEN=1000000
FRAC=0

# loop over data_frac = 0..70
python generate_vllm.py \
  --model       "$MODEL" \
  --input_dir   "$INPUT" \
  --output_dir  "$OUTDIR" \
  --max_new_tokens $MAX_NEW \
  --data_frac   $FRAC \
  --frac_len    $FRAC_LEN \
  --split       train



python -u train.py \
  model=qwen \
  model.name_or_path=model_hub/Qwen1.5-1.8B/SPIN/ite2 \
  loss=dpo \
  base_data_dir=data \
  ckpt_dir=model_hub/Qwen1.5-1.8B/SPIN/ite3/ \
  datasets='["Ultrachat200k/SPIN/ite2","Ultrachat200k/SPIN/ite3"]'
