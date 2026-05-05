#!/bin/bash
#SBATCH --account=le-lab
#SBATCH --gres=gpu:L40S:8
#SBATCH --mem=256GB
#SBATCH --time=336:00:00
#SBATCH --partition=general
#SBATCH --output=run_all_divergences-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tucnguye@iu.edu

# ── Environment setup ────────────────────────────────────────────────────────
# set -euo pipefail must come AFTER conda activate (conda may return non-zero)
nvidia-smi
eval "$(conda shell.bash hook)"
conda activate /data/project/le-lab/conda_env/WSPIN_v2
export LD_PRELOAD=/data/project/le-lab/conda_env/WSPIN/lib/libstdc++.so.6

set -euo pipefail

REPO_ROOT="/data/project/le-lab/fSWIFT"
cd "$REPO_ROOT"

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_BASE="model_hub/Qwen2.5-7B-Instruct"
TEACHER="${REPO_ROOT}/model_hub/Qwen3-32B/base"
SFT_MODEL="${REPO_ROOT}/${MODEL_BASE}/base"
SFT_DATA="data/Ultrachat200k/SFT/trainSFT.jsonl"

BATCH=64
GRAD_ACCUM=2
MAX_NEW=512
FRAC_LEN=1000000
FRAC=0
WEIGHT_BATCH=16
NUM_GPUS=8
MAX_LENGTH=2048
MAX_PROMPT_LENGTH=1024
N_EPOCHS=2

SKIP_EXISTING="${SKIP_EXISTING:-0}"
GPU_IDS="0,1,2,3,4,5,6,7"

ALL_DIVERGENCES=(js)

# If a specific divergence is passed as argument, run only that one
if [ $# -ge 1 ]; then
    ALL_DIVERGENCES=("$1")
fi

# ── Per-divergence pipeline ───────────────────────────────────────────────────
run_divergence() {
    local DIV="$1"
    local CKPT_BASE="${REPO_ROOT}/${MODEL_BASE}/fSWIFT_${DIV}"
    local DATA_BASE="data/Ultrachat200k/fSWIFT_qwen7b_${DIV}"

    echo ""
    echo "============================================================"
    echo " Starting f-SWIFT pipeline: f* = ${DIV}"
    echo " Checkpoints : ${CKPT_BASE}/iteX/"
    echo " Data        : ${DATA_BASE}/iteX/"
    echo " Started     : $(date)"
    echo "============================================================"

    for ITE in 0 1 2 3; do
        echo ""
        echo "===== [${DIV}] Iteration ${ITE} ====="

        # Pick model: ite0 starts from SFT, later iters from previous checkpoint
        if [ "$ITE" -eq 0 ]; then
            PREV_MODEL="$SFT_MODEL"
        else
            PREV_ITE=$((ITE - 1))
            PREV_MODEL="${CKPT_BASE}/ite${PREV_ITE}"
        fi
        if [ ! -d "$PREV_MODEL" ]; then
            echo "ERROR: PREV_MODEL not found: $PREV_MODEL" >&2
            exit 1
        fi

        # Build dataset list: SPIN sliding window — previous iter + current iter only
        # ite0: [ite0], ite1+: [ite(N-1), iteN]
        if [ "$ITE" -eq 0 ]; then
            DATASETS="[\"Ultrachat200k/fSWIFT_qwen7b_${DIV}/ite0\"]"
        else
            PREV_ITE=$((ITE - 1))
            DATASETS="[\"Ultrachat200k/fSWIFT_qwen7b_${DIV}/ite${PREV_ITE}\",\"Ultrachat200k/fSWIFT_qwen7b_${DIV}/ite${ITE}\"]"
        fi

        # Learning rate: higher for early iters, lower for later
        if [ "$ITE" -le 1 ]; then
            LR=5e-7
        else
            LR=1e-7
        fi

        if [ "$SKIP_EXISTING" = "1" ] && [ -f "${CKPT_BASE}/ite${ITE}/model.safetensors" ]; then
            echo "[SKIP] ite${ITE} checkpoint already exists."
            continue
        fi

        # Step 1: Generate self-play responses with vLLM
        echo "[${DIV}] ite${ITE} — Generating responses..."
        CUDA_VISIBLE_DEVICES=$GPU_IDS python generate_vllm.py \
            --model          "$PREV_MODEL" \
            --input_dir      "$SFT_DATA" \
            --output_dir     "${DATA_BASE}/ite${ITE}/train" \
            --max_new_tokens $MAX_NEW \
            --data_frac      $FRAC \
            --frac_len       $FRAC_LEN \
            --split          train

        # Step 2: Estimate token weights using teacher model
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

        # Step 3: Train with f-SWIFT loss
        # train.py uses mp.spawn internally — do NOT use torchrun (causes double process manager → SIGTERM)
        echo "[${DIV}] ite${ITE} — Training..."
        CUDA_VISIBLE_DEVICES=$GPU_IDS python -u train.py \
            model=qwen \
            model.name_or_path="$PREV_MODEL" \
            loss=fswift \
            loss.f_divergence="${DIV}" \
            trainer=FSDPTrainer \
            base_data_dir=data \
            ckpt_dir="${CKPT_BASE}/ite${ITE}/" \
            datasets="${DATASETS}" \
            batch_size=$BATCH \
            gradient_accumulation_steps=$GRAD_ACCUM \
            activation_checkpointing=true \
            n_epochs=$N_EPOCHS \
            lr=$LR \
            iteration=$ITE

        # FSDPTrainer saves to output/**/.../ite<N>_<timestamp>/ (nested path, not predictable)
        if [ ! -f "${CKPT_BASE}/ite${ITE}/model.safetensors" ] && \
           [ ! -f "${CKPT_BASE}/ite${ITE}/model-00001-of-00002.safetensors" ]; then
            LATEST_OUT=$(find "${REPO_ROOT}/output" -maxdepth 6 -type d -name "ite${ITE}_*" \
                         -newer "${REPO_ROOT}/scripts/run_divergences_qwen32b_7b.sh" \
                         2>/dev/null | xargs ls -dt 2>/dev/null | head -1)
            if [ -n "$LATEST_OUT" ] && [ -d "$LATEST_OUT" ]; then
                echo "[${DIV}] ite${ITE} — Copying checkpoint: ${LATEST_OUT} → ${CKPT_BASE}/ite${ITE}/"
                mkdir -p "${CKPT_BASE}/ite${ITE}"
                cp -r "${LATEST_OUT%/}"/. "${CKPT_BASE}/ite${ITE}/"
            else
                echo "ERROR: checkpoint not found in output/ after training ite${ITE}" >&2
                exit 1
            fi
        fi

        echo "[${DIV}] ite${ITE} — Done: $(date)"
    done

    echo ""
    echo "===== [${DIV}] Pipeline complete! Finished: $(date) ====="
}

# ── Main ─────────────────────────────────────────────────────────────────────
echo "============================================================"
echo " f-SWIFT: All-Divergence Training Run (SLURM)"
echo " Node        : $(hostname)"
echo " Divergences : ${ALL_DIVERGENCES[*]}"
echo " GPUs        : $NUM_GPUS x H100"
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