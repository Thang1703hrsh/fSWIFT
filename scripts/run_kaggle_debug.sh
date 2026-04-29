#!/bin/bash
# ── Kaggle-adapted version of run_all_divergences_debug.sh ───────────────────
# Kaggle provides: 2x T4 (16GB each) or 1x P100 (16GB)
# Adjust NUM_GPUS below to match your Kaggle accelerator setting.
#
# BEFORE RUNNING:
#   1. Upload project repo as a Kaggle Dataset mounted at /kaggle/input/f-swift
#   2. Set HF_TOKEN below (get from https://huggingface.co/settings/tokens)
#   3. pip install dependencies (see Cell 1 in notebook)
#   Models are downloaded automatically from HuggingFace.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Fix libcuda symlink for vLLM on Kaggle ────────────────────────────────────
# Kaggle only has libcuda.so.1, but the linker needs libcuda.so (no version suffix)
ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1 /usr/local/lib/libcuda.so 2>/dev/null || true
export LD_LIBRARY_PATH=/usr/local/lib:/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}

# ── HuggingFace token (required to download models) ──────────────────────────
HF_TOKEN="${HF_TOKEN:-}"   # set via: export HF_TOKEN=hf_...
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN is not set. Run: export HF_TOKEN=hf_..." >&2
    exit 1
fi
export HF_TOKEN

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT="/kaggle/working/f-SWIFT"
SFT_MODEL="${REPO_ROOT}/model_hub/Qwen1.5-1.8B/sft_v2"
TEACHER="${REPO_ROOT}/model_hub/zephyr-7b-sft-full"
SFT_DATA="${REPO_ROOT}/data/Ultrachat200k/SFT/trainSFT.jsonl"

# ── GPU config: Kaggle T4 x2 = 2 GPUs, P100 = 1 GPU ─────────────────────────
NUM_GPUS=2           # change to 1 if using P100/single GPU
GPU_IDS="0,1"        # change to "0" if single GPU

# ── Training hyperparams (scaled down for 2 GPUs) ────────────────────────────
# Rule: batch_size >= gradient_accumulation_steps * num_gpus
# With NUM_GPUS=2, GRAD_ACCUM=2: need batch_size >= 4
N_SAMPLES=200        # fewer samples to fit Kaggle 9-hour limit
N_EXAMPLES=200
BATCH=4              # 4 >= 2 (GRAD_ACCUM) * 2 (GPUs)
GRAD_ACCUM=2
MAX_NEW=128          # shorter responses → faster generation
FRAC=0
WEIGHT_BATCH=2
MAX_LENGTH=512
MAX_PROMPT_LENGTH=256
N_EPOCHS=1

SKIP_EXISTING="${SKIP_EXISTING:-0}"

ALL_DIVERGENCES=(js kl wasserstein)
if [ $# -ge 1 ]; then
    ALL_DIVERGENCES=("$1")
fi

# ── Step 0: Copy repo to writable working dir ────────────────────────────────
if [ ! -d "$REPO_ROOT" ]; then
    echo "Copying repo to working dir..."
    cp -r /kaggle/input/f-swift "$REPO_ROOT"
fi
cd "$REPO_ROOT"

# ── Step 0b: Download models from HuggingFace (skip if already present) ──────
download_model_if_missing() {
    local repo_id="$1"
    local local_dir="$2"
    if [ -d "$local_dir" ] && [ "$(ls -A "$local_dir" 2>/dev/null)" ]; then
        echo "[SKIP] Model already exists at ${local_dir}"
        return 0
    fi
    echo "Downloading ${repo_id} → ${local_dir} ..."
    mkdir -p "$local_dir"
    python - <<PYEOF
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="${repo_id}",
    repo_type="model",
    local_dir="${local_dir}",
    local_dir_use_symlinks=False,
    token="${HF_TOKEN}",
)
print("Done: ${local_dir}")
PYEOF
}

download_model_if_missing "ducthang1703/Qwen1.5-1.8B-sft-v2"      "$SFT_MODEL"
download_model_if_missing "alignment-handbook/zephyr-7b-sft-full"  "$TEACHER"

# ── Per-divergence pipeline ───────────────────────────────────────────────────
run_divergence() {
    local DIV="$1"
    local CKPT_BASE="${REPO_ROOT}/model_hub/fSWIFT_${DIV}_debug"
    local DATA_BASE="${REPO_ROOT}/data/Ultrachat200k/fSWIFT_${DIV}_debug"

    echo ""
    echo "============================================================"
    echo " [KAGGLE] f-SWIFT pipeline: f* = ${DIV}  (~${N_SAMPLES} samples)"
    echo " Checkpoints : ${CKPT_BASE}/iteX/"
    echo " Data        : ${DATA_BASE}/iteX/"
    echo " Started     : $(date)"
    echo "============================================================"

    for ITE in 0 1 2 3; do
        echo ""
        echo "===== [${DIV}] Iteration ${ITE} ====="

        if [ "$ITE" -eq 0 ]; then
            PREV_MODEL="$SFT_MODEL"
        else
            PREV_ITE=$((ITE - 1))
            PREV_MODEL="${CKPT_BASE}/ite${PREV_ITE}"
        fi

        DATASETS="[\"Ultrachat200k/fSWIFT_${DIV}_debug/ite0\""
        for D in $(seq 1 $ITE); do
            DATASETS="${DATASETS},\"Ultrachat200k/fSWIFT_${DIV}_debug/ite${D}\""
        done
        DATASETS="${DATASETS}]"

        if [ "$ITE" -le 1 ]; then
            LR=5e-7
        else
            LR=1e-7
        fi

        if [ "$SKIP_EXISTING" = "1" ] && [ -f "${CKPT_BASE}/ite${ITE}/model.safetensors" ]; then
            echo "[SKIP] ite${ITE} checkpoint already exists."
            continue
        fi

        mkdir -p "${DATA_BASE}/ite${ITE}"
        mkdir -p "${CKPT_BASE}/ite${ITE}"

        # Step 1: Generate responses with vLLM
        echo "[${DIV}] ite${ITE} — Generating ${N_SAMPLES} responses..."
        CUDA_VISIBLE_DEVICES=$GPU_IDS python generate_vllm.py \
            --model          "$PREV_MODEL" \
            --input_dir      "$SFT_DATA" \
            --output_dir     "${DATA_BASE}/ite${ITE}/train" \
            --max_new_tokens $MAX_NEW \
            --data_frac      $FRAC \
            --frac_len       $N_SAMPLES \
            --split          train

        # Step 2: Estimate token weights
        echo "[${DIV}] ite${ITE} — Estimating token weights..."
        CUDA_VISIBLE_DEVICES=$GPU_IDS python token_weight_estimation.py \
            --model_name_1      "$TEACHER" \
            --model_name_2      "$PREV_MODEL" \
            --model1_template   normal \
            --model2_template   normal \
            --input_dir         "${DATA_BASE}/ite${ITE}" \
            --output_dir        "${DATA_BASE}/ite${ITE}" \
            --max_length        $MAX_LENGTH \
            --max_prompt_length $MAX_PROMPT_LENGTH \
            --batch_size        $WEIGHT_BATCH \
            --num_gpus          $NUM_GPUS

        # Step 3: Train
        echo "[${DIV}] ite${ITE} — Training on ${N_EXAMPLES} examples..."
        CUDA_VISIBLE_DEVICES=$GPU_IDS python -u train.py \
            model=qwen \
            model.name_or_path="$PREV_MODEL" \
            loss=fswift \
            loss.f_divergence="${DIV}" \
            trainer=FSDPTrainer \
            base_data_dir="${REPO_ROOT}/data" \
            ckpt_dir="${CKPT_BASE}/ite${ITE}/" \
            datasets="${DATASETS}" \
            batch_size=$BATCH \
            gradient_accumulation_steps=$GRAD_ACCUM \
            activation_checkpointing=true \
            n_epochs=$N_EPOCHS \
            n_examples=$N_EXAMPLES \
            lr=$LR \
            iteration=$ITE

        echo "[${DIV}] ite${ITE} — Done: $(date)"
    done

    echo ""
    echo "===== [${DIV}] Pipeline complete! Finished: $(date) ====="
}

# ── Main ──────────────────────────────────────────────────────────────────────
echo "============================================================"
echo " [KAGGLE] f-SWIFT Debug Run"
echo " GPUs        : ${NUM_GPUS} (IDs: ${GPU_IDS})"
echo " Divergences : ${ALL_DIVERGENCES[*]}"
echo " Samples     : ~${N_SAMPLES} per iteration"
echo " Started     : $(date)"
echo "============================================================"

for DIV in "${ALL_DIVERGENCES[@]}"; do
    run_divergence "$DIV"
done

echo ""
echo "============================================================"
echo " ALL DIVERGENCES COMPLETE"
echo " Finished: $(date)"
echo "============================================================"
