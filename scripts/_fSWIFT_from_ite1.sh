#!/bin/bash
set -euo pipefail

# ============================================================
# f-SWIFT: Resume from Iteration 1 (Iteration 0 already done)
# ============================================================

F_DIVERGENCE="${F_DIVERGENCE:-js}"  # default: Jensen-Shannon

MODEL_BASE="model_hub/Qwen1.5-1.8B"
TEACHER="model_hub/zephyr-7b-sft-full"
SFT_DATA="data/Ultrachat200k/SFT/trainSFT.jsonl"
DATA_BASE="data/Ultrachat200k/fSWIFT"

BATCH=2
MAX_NEW=512
FRAC_LEN=1000000
FRAC=0
WEIGHT_BATCH=8
NUM_GPUS=1

MAX_LENGTH=2048
MAX_PROMPT_LENGTH=1024

# ============================================================
# Iteration 1
# ============================================================
echo "===== f-SWIFT Iteration 1 (f*=${F_DIVERGENCE}) ====="

python generate_vllm.py \
  --model       "${MODEL_BASE}/fSWIFT/ite0" \
  --input_dir   "$SFT_DATA" \
  --output_dir  "${DATA_BASE}/ite1/train" \
  --max_new_tokens $MAX_NEW \
  --data_frac   $FRAC \
  --frac_len    $FRAC_LEN \
  --split       train

python token_weight_estimation.py \
  --model_name_1 "$TEACHER" \
  --model_name_2 "${MODEL_BASE}/fSWIFT/ite0" \
  --model1_template normal \
  --model2_template normal \
  --input_dir  "${DATA_BASE}/ite1" \
  --output_dir "${DATA_BASE}/ite1" \
  --max_length $MAX_LENGTH \
  --max_prompt_length $MAX_PROMPT_LENGTH \
  --batch_size $WEIGHT_BATCH \
  --num_gpus   $NUM_GPUS

python -u train.py \
  model=qwen \
  model.name_or_path="${MODEL_BASE}/fSWIFT/ite0" \
  loss=fswift \
  loss.f_divergence="${F_DIVERGENCE}" \
  base_data_dir=data \
  ckpt_dir="${MODEL_BASE}/fSWIFT/ite1/" \
  datasets='["Ultrachat200k/fSWIFT/ite0","Ultrachat200k/fSWIFT/ite1"]' \
  batch_size=$BATCH \
  lr=5e-7


# ============================================================
# Iteration 2
# ============================================================
echo "===== f-SWIFT Iteration 2 (f*=${F_DIVERGENCE}) ====="

python generate_vllm.py \
  --model       "${MODEL_BASE}/fSWIFT/ite1" \
  --input_dir   "$SFT_DATA" \
  --output_dir  "${DATA_BASE}/ite2/train" \
  --max_new_tokens $MAX_NEW \
  --data_frac   $FRAC \
  --frac_len    $FRAC_LEN \
  --split       train

python token_weight_estimation.py \
  --model_name_1 "$TEACHER" \
  --model_name_2 "${MODEL_BASE}/fSWIFT/ite1" \
  --model1_template normal \
  --model2_template normal \
  --input_dir  "${DATA_BASE}/ite2" \
  --output_dir "${DATA_BASE}/ite2" \
  --max_length $MAX_LENGTH \
  --max_prompt_length $MAX_PROMPT_LENGTH \
  --batch_size $WEIGHT_BATCH \
  --num_gpus   $NUM_GPUS

python -u train.py \
  model=qwen \
  model.name_or_path="${MODEL_BASE}/fSWIFT/ite1" \
  loss=fswift \
  loss.f_divergence="${F_DIVERGENCE}" \
  base_data_dir=data \
  ckpt_dir="${MODEL_BASE}/fSWIFT/ite2/" \
  datasets='["Ultrachat200k/fSWIFT/ite1","Ultrachat200k/fSWIFT/ite2"]' \
  batch_size=$BATCH \
  lr=1e-7


# ============================================================
# Iteration 3
# ============================================================
echo "===== f-SWIFT Iteration 3 (f*=${F_DIVERGENCE}) ====="

python generate_vllm.py \
  --model       "${MODEL_BASE}/fSWIFT/ite2" \
  --input_dir   "$SFT_DATA" \
  --output_dir  "${DATA_BASE}/ite3/train" \
  --max_new_tokens $MAX_NEW \
  --data_frac   $FRAC \
  --frac_len    $FRAC_LEN \
  --split       train

python token_weight_estimation.py \
  --model_name_1 "$TEACHER" \
  --model_name_2 "${MODEL_BASE}/fSWIFT/ite2" \
  --model1_template normal \
  --model2_template normal \
  --input_dir  "${DATA_BASE}/ite3" \
  --output_dir "${DATA_BASE}/ite3" \
  --max_length $MAX_LENGTH \
  --max_prompt_length $MAX_PROMPT_LENGTH \
  --batch_size $WEIGHT_BATCH \
  --num_gpus   $NUM_GPUS

python -u train.py \
  model=qwen \
  model.name_or_path="${MODEL_BASE}/fSWIFT/ite2" \
  loss=fswift \
  loss.f_divergence="${F_DIVERGENCE}" \
  base_data_dir=data \
  ckpt_dir="${MODEL_BASE}/fSWIFT/ite3/" \
  datasets='["Ultrachat200k/fSWIFT/ite2","Ultrachat200k/fSWIFT/ite3"]' \
  batch_size=$BATCH \
  lr=1e-7

echo "===== f-SWIFT training complete! ====="
