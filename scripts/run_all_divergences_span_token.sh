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
MODEL_BASE="model_hub/Qwen1.5-1.8B"
TEACHER="${REPO_ROOT}/model_hub/zephyr-7b-sft-full"
SFT_MODEL="${REPO_ROOT}/${MODEL_BASE}/sft_v2"
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

ALL_DIVERGENCES=(js kl hellinger)

# CO-CRAFT span-level divergences: prefixed with "cocraft_span_" to distinguish
# from plain f-SWIFT runs. The suffix after "cocraft_span_" is the f-divergence
# used for both the outer f* and the inner span f'-critic.
# Example: "cocraft_span_js", "cocraft_span_kl", "cocraft_span_hellinger"
ALL_COCRAFT_SPAN_DIVERGENCES=(cocraft_span_js cocraft_span_kl cocraft_span_hellinger)

# If a specific divergence is passed as argument, run only that one.
# Works for both plain fSWIFT divergences and cocraft_span_* variants.
if [ $# -ge 1 ]; then
    if [[ "$1" == cocraft_span_* ]]; then
        ALL_DIVERGENCES=()
        ALL_COCRAFT_SPAN_DIVERGENCES=("$1")
    else
        ALL_DIVERGENCES=("$1")
        ALL_COCRAFT_SPAN_DIVERGENCES=()
    fi
fi

# Optional: run CO-CRAFT span variants alongside plain fSWIFT.
# Set RUN_COCRAFT_SPAN=1 to also run CO-CRAFT span variants.
RUN_COCRAFT_SPAN="${RUN_COCRAFT_SPAN:-0}"

# ── CO-CRAFT Span hyper-parameters ──────────────────────────────────────────
# These are passed as loss.* overrides to train.py when loss=cocraft_span.
# Defaults match the paper's recommended setup (§9, §14 of CO_CRAFT_span_level.md).
SPAN_STRATEGY="${SPAN_STRATEGY:-fixed}"   # fixed | sentence | clause
SPAN_SIZE="${SPAN_SIZE:-8}"               # tokens per block (fixed strategy)
SPAN_MU="${SPAN_MU:-1.0}"                 # critic exponentiation temperature
SPAN_L="${SPAN_L:--2.0}"                  # critic lower clamp
SPAN_U="${SPAN_U:-2.0}"                   # critic upper clamp
SPAN_G_MIN="${SPAN_G_MIN:-0.25}"          # weight lower clip
SPAN_G_MAX="${SPAN_G_MAX:-4.0}"           # weight upper clip

# ── Per-divergence pipeline (f-SWIFT) ────────────────────────────────────────
run_divergence() {
    local DIV="$1"
    local CKPT_BASE="${REPO_ROOT}/${MODEL_BASE}/fSWIFT_${DIV}"
    local DATA_BASE="data/Ultrachat200k/fSWIFT_${DIV}"

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
            DATASETS="[\"Ultrachat200k/fSWIFT_${DIV}/ite0\"]"
        else
            PREV_ITE=$((ITE - 1))
            DATASETS="[\"Ultrachat200k/fSWIFT_${DIV}/ite${PREV_ITE}\",\"Ultrachat200k/fSWIFT_${DIV}/ite${ITE}\"]"
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
                         -newer "${REPO_ROOT}/scripts/run_all_divergences.sh" \
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

# ── CO-CRAFT Span-Level pipeline ─────────────────────────────────────────────
# DIV format: "cocraft_span_<f_divergence>", e.g. "cocraft_span_js"
run_cocraft_span_divergence() {
    local FULL_DIV="$1"
    # Extract the f-divergence suffix (js / kl / hellinger / …)
    local F_DIV="${FULL_DIV#cocraft_span_}"
    local CKPT_BASE="${REPO_ROOT}/${MODEL_BASE}/${FULL_DIV}"
    local DATA_BASE="data/Ultrachat200k/${FULL_DIV}"

    echo ""
    echo "============================================================"
    echo " Starting CO-CRAFT Span pipeline: f = ${F_DIV}"
    echo " Checkpoints : ${CKPT_BASE}/iteX/"
    echo " Data        : ${DATA_BASE}/iteX/"
    echo " Span config : strategy=${SPAN_STRATEGY} size=${SPAN_SIZE} mu=${SPAN_MU}"
    echo "               L=${SPAN_L} U=${SPAN_U} g_min=${SPAN_G_MIN} g_max=${SPAN_G_MAX}"
    echo " Started     : $(date)"
    echo "============================================================"

    for ITE in 0 1 2 3; do
        echo ""
        echo "===== [${FULL_DIV}] Iteration ${ITE} ====="

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

        # SPIN sliding-window dataset list (same logic as fSWIFT)
        if [ "$ITE" -eq 0 ]; then
            DATASETS="[\"Ultrachat200k/${FULL_DIV}/ite0\"]"
        else
            PREV_ITE=$((ITE - 1))
            DATASETS="[\"Ultrachat200k/${FULL_DIV}/ite${PREV_ITE}\",\"Ultrachat200k/${FULL_DIV}/ite${ITE}\"]"
        fi

        # Learning rate schedule
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
        # (Identical to fSWIFT — the generation step is method-agnostic.)
        echo "[${FULL_DIV}] ite${ITE} — Generating responses..."
        CUDA_VISIBLE_DEVICES=$GPU_IDS python generate_vllm.py \
            --model          "$PREV_MODEL" \
            --input_dir      "$SFT_DATA" \
            --output_dir     "${DATA_BASE}/ite${ITE}/train" \
            --max_new_tokens $MAX_NEW \
            --data_frac      $FRAC \
            --frac_len       $FRAC_LEN \
            --split          train

        # Step 2 (token_weight_estimation) is intentionally absent for cocraft_span.
        # Span importance weights are computed on-the-fly inside train.py from
        # Δ_t = log π_θ − log π_ref using the f'-critic. No teacher model is needed.

        # Step 2: Train with CO-CRAFT span-level loss
        # loss=cocraft_span activates:
        #   - outer f* objective (same as fswift)
        #   - inner span f'-critic for importance weighting
        # Span hyper-parameters are forwarded via loss.* overrides.
        echo "[${FULL_DIV}] ite${ITE} — Training (CO-CRAFT Span)..."
        CUDA_VISIBLE_DEVICES=$GPU_IDS python -u train.py \
            model=qwen \
            model.name_or_path="$PREV_MODEL" \
            loss=cocraft_span \
            loss.f_divergence="${F_DIV}" \
            loss.span_strategy="${SPAN_STRATEGY}" \
            loss.span_size="${SPAN_SIZE}" \
            loss.mu="${SPAN_MU}" \
            loss.span_L="${SPAN_L}" \
            loss.span_U="${SPAN_U}" \
            loss.span_g_min="${SPAN_G_MIN}" \
            loss.span_g_max="${SPAN_G_MAX}" \
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

        # Resolve checkpoint path (FSDPTrainer may save to a timestamped subdir)
        if [ ! -f "${CKPT_BASE}/ite${ITE}/model.safetensors" ] && \
           [ ! -f "${CKPT_BASE}/ite${ITE}/model-00001-of-00002.safetensors" ]; then
            LATEST_OUT=$(find "${REPO_ROOT}/output" -maxdepth 6 -type d -name "ite${ITE}_*" \
                         -newer "${REPO_ROOT}/scripts/run_all_divergences.sh" \
                         2>/dev/null | xargs ls -dt 2>/dev/null | head -1)
            if [ -n "$LATEST_OUT" ] && [ -d "$LATEST_OUT" ]; then
                echo "[${FULL_DIV}] ite${ITE} — Copying checkpoint: ${LATEST_OUT} → ${CKPT_BASE}/ite${ITE}/"
                mkdir -p "${CKPT_BASE}/ite${ITE}"
                cp -r "${LATEST_OUT%/}"/. "${CKPT_BASE}/ite${ITE}/"
            else
                echo "ERROR: checkpoint not found in output/ after training ite${ITE}" >&2
                exit 1
            fi
        fi

        echo "[${FULL_DIV}] ite${ITE} — Done: $(date)"
    done

    echo ""
    echo "===== [${FULL_DIV}] CO-CRAFT Span pipeline complete! Finished: $(date) ====="
}

# ── Main ─────────────────────────────────────────────────────────────────────
echo "============================================================"
echo " f-SWIFT: All-Divergence Training Run (SLURM)"
echo " Node        : $(hostname)"
echo " Divergences : ${ALL_DIVERGENCES[*]}"
if [ "$RUN_COCRAFT_SPAN" = "1" ] || [ ${#ALL_COCRAFT_SPAN_DIVERGENCES[@]} -gt 0 ]; then
echo " CO-CRAFT Span: ${ALL_COCRAFT_SPAN_DIVERGENCES[*]}"
fi
echo " GPUs        : $NUM_GPUS x H100"
echo " Started     : $(date)"
echo " SKIP_EXISTING=${SKIP_EXISTING}"
echo "============================================================"

for DIV in "${ALL_DIVERGENCES[@]}"; do
    run_divergence "$DIV"
done

# Run CO-CRAFT span variants if explicitly requested or passed as argument.
if [ "$RUN_COCRAFT_SPAN" = "1" ]; then
    for CDIV in "${ALL_COCRAFT_SPAN_DIVERGENCES[@]}"; do
        run_cocraft_span_divergence "$CDIV"
    done
elif [ ${#ALL_COCRAFT_SPAN_DIVERGENCES[@]} -gt 0 ] && \
     [ "${ALL_DIVERGENCES[*]}" = "" ]; then
    # Only cocraft_span args were passed (via $1 = cocraft_span_*).
    for CDIV in "${ALL_COCRAFT_SPAN_DIVERGENCES[@]}"; do
        run_cocraft_span_divergence "$CDIV"
    done
fi

echo ""
echo "============================================================"
echo " ALL DIVERGENCES COMPLETE"
echo " Finished: $(date)"
echo "============================================================"
echo ""
echo "Next step — run evaluation:"
echo "  bash scripts/eval_all_divergences.sh"