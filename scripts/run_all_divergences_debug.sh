#!/bin/bash
#SBATCH --account=le-lab
#SBATCH --gres=gpu:L40S:8
#SBATCH --mem=256GB
#SBATCH --time=12:00:00
#SBATCH --partition=general
#SBATCH --output=run_all_divergences_debug-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tucnguye@iu.edu

# ── Environment setup ────────────────────────────────────────────────────────
nvidia-smi
eval "$(conda shell.bash hook)"
conda activate /data/project/le-lab/conda_env/WSPIN_v2
export LD_PRELOAD=/data/project/le-lab/conda_env/WSPIN/lib/libstdc++.so.6

set -euo pipefail

REPO_ROOT="/data/project/le-lab/fSWIFT"
cd "$REPO_ROOT"

# ── Config (debug: ~500 samples) ─────────────────────────────────────────────
MODEL_BASE="model_hub/Qwen1.5-1.8B"
TEACHER="model_hub/zephyr-7b-sft-full"
SFT_MODEL="${MODEL_BASE}/sft_v2"
SFT_DATA="data/Ultrachat200k/SFT/trainSFT.jsonl"

N_SAMPLES=500          # number of samples to generate per iteration
N_EXAMPLES=500         # number of training examples used by train.py

BATCH=16
GRAD_ACCUM=2
MAX_NEW=256
FRAC=0
WEIGHT_BATCH=4
NUM_GPUS=8
MAX_LENGTH=1024
MAX_PROMPT_LENGTH=512
N_EPOCHS=1

SKIP_EXISTING="${SKIP_EXISTING:-0}"
GPU_IDS="0,1,2,3,4,5,6,7"

ALL_DIVERGENCES=(js kl hellinger)

if [ $# -ge 1 ]; then
    ALL_DIVERGENCES=("$1")
fi

# ── Per-divergence pipeline ───────────────────────────────────────────────────
run_divergence() {
    local DIV="$1"
    # Use separate debug dirs so full-scale runs are not overwritten
    local CKPT_BASE="${MODEL_BASE}/fSWIFT_${DIV}_debug"
    local DATA_BASE="data/Ultrachat200k/fSWIFT_${DIV}_debug"

    echo ""
    echo "============================================================"
    echo " [DEBUG] f-SWIFT pipeline: f* = ${DIV}  (~${N_SAMPLES} samples)"
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

        # Step 1: Generate ~500 responses with vLLM
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

        # Step 3: Train with f-SWIFT loss
        echo "[${DIV}] ite${ITE} — Training on ${N_EXAMPLES} examples..."
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
            n_examples=$N_EXAMPLES \
            lr=$LR \
            iteration=$ITE

        echo "[${DIV}] ite${ITE} — Done: $(date)"
    done

    echo ""
    echo "===== [${DIV}] Debug pipeline complete! Finished: $(date) ====="
}

# ── Main ─────────────────────────────────────────────────────────────────────
echo "============================================================"
echo " [DEBUG] f-SWIFT: All-Divergence Training Run"
echo " Node        : $(hostname)"
echo " Divergences : ${ALL_DIVERGENCES[*]}"
echo " GPUs        : $NUM_GPUS x L40S"
echo " Samples     : ~${N_SAMPLES} per iteration"
echo " Started     : $(date)"
echo " SKIP_EXISTING=${SKIP_EXISTING}"
echo "============================================================"

for DIV in "${ALL_DIVERGENCES[@]}"; do
    run_divergence "$DIV"
done

echo ""
echo "============================================================"
echo " ALL DIVERGENCES COMPLETE (debug run)"
echo " Finished: $(date)"
echo "============================================================"
