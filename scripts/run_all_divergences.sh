#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODEL_BASE="model_hub/Qwen1.5-1.8B"
TEACHER="model_hub/zephyr-7b-sft-full"
SFT_MODEL="${MODEL_BASE}/sft_v2"
SFT_DATA="data/Ultrachat200k/SFT/trainSFT.jsonl"


BATCH=64
GRAD_ACCUM=2
MAX_NEW=512
FRAC_LEN=1000000
FRAC=0
WEIGHT_BATCH=16        # token_weight_estimation uses all 4 GPUs via --num_gpus
NUM_GPUS=4
MAX_LENGTH=2048
MAX_PROMPT_LENGTH=1024
N_EPOCHS=2

SKIP_EXISTING="${SKIP_EXISTING:-0}"

ALL_DIVERGENCES=(js kl wasserstein)

# If a specific divergence is passed as argument, run only that one
if [ $# -ge 1 ]; then
    ALL_DIVERGENCES=("$1")
fi

run_divergence() {
    local DIV="$1"
    local CKPT_BASE="${MODEL_BASE}/fSWIFT_${DIV}"
    local DATA_BASE="data/Ultrachat200k/fSWIFT_${DIV}"

    echo ""
    echo "============================================================"
    echo " Starting f-SWIFT pipeline: f* = ${DIV}"
    echo " Checkpoints : ${CKPT_BASE}/iteX/"
    echo " Data        : ${DATA_BASE}/iteX/"
    echo " Started     : $(date)"
    echo "============================================================"

    # ----------------------------------------------------------
    # Iteration 0
    # ----------------------------------------------------------
    echo ""
    echo "===== [${DIV}] Iteration 0 ====="

    if [ "$SKIP_EXISTING" = "1" ] && [ -f "${CKPT_BASE}/ite0/model.safetensors" ]; then
        echo "[SKIP] ite0 checkpoint already exists."
    else
        python generate_vllm.py \
            --model        "$SFT_MODEL" \
            --input_dir    "$SFT_DATA" \
            --output_dir   "${DATA_BASE}/ite0/train" \
            --max_new_tokens $MAX_NEW \
            --data_frac    $FRAC \
            --frac_len     $FRAC_LEN \
            --split        train

        python token_weight_estimation.py \
            --model_name_1 "$TEACHER" \
            --model_name_2 "$SFT_MODEL" \
            --model1_template normal \
            --model2_template normal \
            --input_dir    "${DATA_BASE}/ite0" \
            --output_dir   "${DATA_BASE}/ite0" \
            --max_length   $MAX_LENGTH \
            --max_prompt_length $MAX_PROMPT_LENGTH \
            --batch_size   $WEIGHT_BATCH \
            --num_gpus     $NUM_GPUS

        python -u train.py \
            model=qwen \
            model.name_or_path="${SFT_MODEL}" \
            loss=fswift \
            loss.f_divergence="${DIV}" \
            trainer=FSDPTrainer \
            base_data_dir=data \
            ckpt_dir="${CKPT_BASE}/ite0/" \
            datasets="[\"Ultrachat200k/fSWIFT_${DIV}/ite0\"]" \
            batch_size=$BATCH \
            gradient_accumulation_steps=$GRAD_ACCUM \
            activation_checkpointing=true \
            n_epochs=$N_EPOCHS \
            lr=5e-7 \
            iteration=0
    fi

    # ----------------------------------------------------------
    # Iteration 1
    # ----------------------------------------------------------
    echo ""
    echo "===== [${DIV}] Iteration 1 ====="

    if [ "$SKIP_EXISTING" = "1" ] && [ -f "${CKPT_BASE}/ite1/model.safetensors" ]; then
        echo "[SKIP] ite1 checkpoint already exists."
    else
        python generate_vllm.py \
            --model        "${CKPT_BASE}/ite0" \
            --input_dir    "$SFT_DATA" \
            --output_dir   "${DATA_BASE}/ite1/train" \
            --max_new_tokens $MAX_NEW \
            --data_frac    $FRAC \
            --frac_len     $FRAC_LEN \
            --split        train

        python token_weight_estimation.py \
            --model_name_1 "$TEACHER" \
            --model_name_2 "${CKPT_BASE}/ite0" \
            --model1_template normal \
            --model2_template normal \
            --input_dir    "${DATA_BASE}/ite1" \
            --output_dir   "${DATA_BASE}/ite1" \
            --max_length   $MAX_LENGTH \
            --max_prompt_length $MAX_PROMPT_LENGTH \
            --batch_size   $WEIGHT_BATCH \
            --num_gpus     $NUM_GPUS

        python -u train.py \
            model=qwen \
            model.name_or_path="${CKPT_BASE}/ite0" \
            loss=fswift \
            loss.f_divergence="${DIV}" \
            trainer=FSDPTrainer \
            base_data_dir=data \
            ckpt_dir="${CKPT_BASE}/ite1/" \
            datasets="[\"Ultrachat200k/fSWIFT_${DIV}/ite0\",\"Ultrachat200k/fSWIFT_${DIV}/ite1\"]" \
            batch_size=$BATCH \
            gradient_accumulation_steps=$GRAD_ACCUM \
            activation_checkpointing=true \
            n_epochs=$N_EPOCHS \
            lr=5e-7 \
            iteration=1
    fi

    # ----------------------------------------------------------
    # Iteration 2
    # ----------------------------------------------------------
    echo ""
    echo "===== [${DIV}] Iteration 2 ====="

    if [ "$SKIP_EXISTING" = "1" ] && [ -f "${CKPT_BASE}/ite2/model.safetensors" ]; then
        echo "[SKIP] ite2 checkpoint already exists."
    else
        python generate_vllm.py \
            --model        "${CKPT_BASE}/ite1" \
            --input_dir    "$SFT_DATA" \
            --output_dir   "${DATA_BASE}/ite2/train" \
            --max_new_tokens $MAX_NEW \
            --data_frac    $FRAC \
            --frac_len     $FRAC_LEN \
            --split        train

        python token_weight_estimation.py \
            --model_name_1 "$TEACHER" \
            --model_name_2 "${CKPT_BASE}/ite1" \
            --model1_template normal \
            --model2_template normal \
            --input_dir    "${DATA_BASE}/ite2" \
            --output_dir   "${DATA_BASE}/ite2" \
            --max_length   $MAX_LENGTH \
            --max_prompt_length $MAX_PROMPT_LENGTH \
            --batch_size   $WEIGHT_BATCH \
            --num_gpus     $NUM_GPUS

        python -u train.py \
            model=qwen \
            model.name_or_path="${CKPT_BASE}/ite1" \
            loss=fswift \
            loss.f_divergence="${DIV}" \
            trainer=FSDPTrainer \
            base_data_dir=data \
            ckpt_dir="${CKPT_BASE}/ite2/" \
            datasets="[\"Ultrachat200k/fSWIFT_${DIV}/ite1\",\"Ultrachat200k/fSWIFT_${DIV}/ite2\"]" \
            batch_size=$BATCH \
            gradient_accumulation_steps=$GRAD_ACCUM \
            activation_checkpointing=true \
            n_epochs=$N_EPOCHS \
            lr=1e-7 \
            iteration=2
    fi

    # ----------------------------------------------------------
    # Iteration 3
    # ----------------------------------------------------------
    echo ""
    echo "===== [${DIV}] Iteration 3 ====="

    if [ "$SKIP_EXISTING" = "1" ] && [ -f "${CKPT_BASE}/ite3/model.safetensors" ]; then
        echo "[SKIP] ite3 checkpoint already exists."
    else
        python generate_vllm.py \
            --model        "${CKPT_BASE}/ite2" \
            --input_dir    "$SFT_DATA" \
            --output_dir   "${DATA_BASE}/ite3/train" \
            --max_new_tokens $MAX_NEW \
            --data_frac    $FRAC \
            --frac_len     $FRAC_LEN \
            --split        train

        python token_weight_estimation.py \
            --model_name_1 "$TEACHER" \
            --model_name_2 "${CKPT_BASE}/ite2" \
            --model1_template normal \
            --model2_template normal \
            --input_dir    "${DATA_BASE}/ite3" \
            --output_dir   "${DATA_BASE}/ite3" \
            --max_length   $MAX_LENGTH \
            --max_prompt_length $MAX_PROMPT_LENGTH \
            --batch_size   $WEIGHT_BATCH \
            --num_gpus     $NUM_GPUS

        python -u train.py \
            model=qwen \
            model.name_or_path="${CKPT_BASE}/ite2" \
            loss=fswift \
            loss.f_divergence="${DIV}" \
            trainer=FSDPTrainer \
            base_data_dir=data \
            ckpt_dir="${CKPT_BASE}/ite3/" \
            datasets="[\"Ultrachat200k/fSWIFT_${DIV}/ite2\",\"Ultrachat200k/fSWIFT_${DIV}/ite3\"]" \
            batch_size=$BATCH \
            gradient_accumulation_steps=$GRAD_ACCUM \
            activation_checkpointing=true \
            n_epochs=$N_EPOCHS \
            lr=1e-7 \
            iteration=3
    fi

    echo ""
    echo "===== [${DIV}] Pipeline complete! Finished: $(date) ====="
}

# ============================================================
# Main: run all divergences sequentially
# ============================================================
echo "============================================================"
echo " f-SWIFT: All-Divergence Training Run"
echo " Divergences : ${ALL_DIVERGENCES[*]}"
echo " Started     : $(date)"
echo " SKIP_EXISTING=${SKIP_EXISTING}"
echo "============================================================"

for DIV in "${ALL_DIVERGENCES[@]}"; do
    run_divergence "$DIV"
done

echo ""
echo "============================================================"
echo " ALL DIVERGENCES COMPLETE"
echo " Finished: $(date)"
echo "============================================================"
echo ""
echo "Next step — run evaluation:"
echo "  bash scripts/eval_all_divergences.sh"
