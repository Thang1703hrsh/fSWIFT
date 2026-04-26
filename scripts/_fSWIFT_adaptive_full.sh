#!/bin/bash
set -euo pipefail

# ============================================================
# f-SWIFT with Adaptive f-Scheduling
# Iterations 0-1: Jensen-Shannon (stable warm-up)
# Iterations 2-3: KL (aggressive refinement)
# ============================================================

MODEL_BASE="model_hub/Qwen1.5-1.8B"
TEACHER="model_hub/zephyr-7b-sft-full"
SFT_DATA="data/Ultrachat200k/SFT/trainSFT.jsonl"
DATA_BASE="data/Ultrachat200k/fSWIFT_adaptive"

BATCH=8
MAX_NEW=512
FRAC_LEN=1000000
FRAC=0
WEIGHT_BATCH=8
NUM_GPUS=1
MAX_LENGTH=1024
MAX_PROMPT_LENGTH=512

# ============================================================
# Iteration 0 — Jensen-Shannon (stable warm-up)
# ============================================================
echo "===== Iteration 0 (f*=js) ====="

python generate_vllm.py \
  --model       "${MODEL_BASE}/SFT" \
  --input_dir   "$SFT_DATA" \
  --output_dir  "${DATA_BASE}/ite0/train" \
  --max_new_tokens $MAX_NEW \
  --data_frac   $FRAC \
  --frac_len    $FRAC_LEN \
  --split       train

python token_weight_estimation.py \
  --model_name_1 "$TEACHER" \
  --model_name_2 "${MODEL_BASE}/SFT" \
  --model1_template normal \
  --model2_template normal \
  --input_dir  "${DATA_BASE}/ite0" \
  --output_dir "${DATA_BASE}/ite0" \
  --max_length $MAX_LENGTH \
  --max_prompt_length $MAX_PROMPT_LENGTH \
  --batch_size $WEIGHT_BATCH \
  --num_gpus   $NUM_GPUS

python -u train.py \
  model=qwen \
  model.name_or_path="${MODEL_BASE}/SFT/" \
  loss=fswift \
  loss.f_divergence=js \
  base_data_dir=data \
  ckpt_dir="${MODEL_BASE}/fSWIFT_adaptive/ite0/" \
  datasets='["Ultrachat200k/fSWIFT_adaptive/ite0"]'


# ============================================================
# Iteration 1 — Jensen-Shannon (stable warm-up)
# ============================================================
echo "===== Iteration 1 (f*=js) ====="

python generate_vllm.py \
  --model       "${MODEL_BASE}/fSWIFT_adaptive/ite0" \
  --input_dir   "$SFT_DATA" \
  --output_dir  "${DATA_BASE}/ite1/train" \
  --max_new_tokens $MAX_NEW \
  --data_frac   $FRAC \
  --frac_len    $FRAC_LEN \
  --split       train

python token_weight_estimation.py \
  --model_name_1 "$TEACHER" \
  --model_name_2 "${MODEL_BASE}/fSWIFT_adaptive/ite0" \
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
  model.name_or_path="${MODEL_BASE}/fSWIFT_adaptive/ite0" \
  loss=fswift \
  loss.f_divergence=js \
  base_data_dir=data \
  ckpt_dir="${MODEL_BASE}/fSWIFT_adaptive/ite1/" \
  datasets='["Ultrachat200k/fSWIFT_adaptive/ite0","Ultrachat200k/fSWIFT_adaptive/ite1"]'


# ============================================================
# Iteration 2 — KL (aggressive refinement)
# ============================================================
echo "===== Iteration 2 (f*=kl) ====="

python generate_vllm.py \
  --model       "${MODEL_BASE}/fSWIFT_adaptive/ite1" \
  --input_dir   "$SFT_DATA" \
  --output_dir  "${DATA_BASE}/ite2/train" \
  --max_new_tokens $MAX_NEW \
  --data_frac   $FRAC \
  --frac_len    $FRAC_LEN \
  --split       train

python token_weight_estimation.py \
  --model_name_1 "$TEACHER" \
  --model_name_2 "${MODEL_BASE}/fSWIFT_adaptive/ite1" \
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
  model.name_or_path="${MODEL_BASE}/fSWIFT_adaptive/ite1" \
  loss=fswift \
  loss.f_divergence=kl \
  base_data_dir=data \
  ckpt_dir="${MODEL_BASE}/fSWIFT_adaptive/ite2/" \
  datasets='["Ultrachat200k/fSWIFT_adaptive/ite1","Ultrachat200k/fSWIFT_adaptive/ite2"]'


# ============================================================
# Iteration 3 — KL (aggressive refinement)
# ============================================================
echo "===== Iteration 3 (f*=kl) ====="

python generate_vllm.py \
  --model       "${MODEL_BASE}/fSWIFT_adaptive/ite2" \
  --input_dir   "$SFT_DATA" \
  --output_dir  "${DATA_BASE}/ite3/train" \
  --max_new_tokens $MAX_NEW \
  --data_frac   $FRAC \
  --frac_len    $FRAC_LEN \
  --split       train

python token_weight_estimation.py \
  --model_name_1 "$TEACHER" \
  --model_name_2 "${MODEL_BASE}/fSWIFT_adaptive/ite2" \
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
  model.name_or_path="${MODEL_BASE}/fSWIFT_adaptive/ite2" \
  loss=fswift \
  loss.f_divergence=kl \
  base_data_dir=data \
  ckpt_dir="${MODEL_BASE}/fSWIFT_adaptive/ite3/" \
  datasets='["Ultrachat200k/fSWIFT_adaptive/ite2","Ultrachat200k/fSWIFT_adaptive/ite3"]'

echo "===== f-SWIFT adaptive training complete! ====="
