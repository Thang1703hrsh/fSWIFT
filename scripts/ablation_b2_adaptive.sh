#!/bin/bash
set -euo pipefail

# ============================================================
# Ablation B2: Adaptive f-Divergence Scheduling
#
# Compares fixed-divergence runs against adaptive schedules
# that switch f* mid-training (at iteration 2).
#
#   Schedule A: JS (ite0-1) → KL   (ite2-3)
#   Schedule B: JS (ite0-1) → chi2 (ite2-3)
#
# Fixed baselines (identity, js, kl) reuse B1 checkpoints if
# SKIP_EXISTING=1 is set; otherwise they are retrained.
#
# Usage:
#   bash scripts/ablation_b2_adaptive.sh
#   SKIP_EXISTING=1 bash scripts/ablation_b2_adaptive.sh
#
# Results:
#   eval_results/ablation_b2_adaptive/summary.tsv
# ============================================================

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODEL_BASE="model_hub/Qwen1.5-1.8B"
TEACHER="model_hub/zephyr-7b-sft-full"
SFT_MODEL="${MODEL_BASE}/sft_v2"
SFT_DATA="data/Ultrachat200k/SFT/trainSFT.jsonl"

BATCH=8
MAX_NEW=512
FRAC_LEN=1000000
FRAC=0
WEIGHT_BATCH=8
NUM_GPUS=1
MAX_LENGTH=2048
MAX_PROMPT_LENGTH=1024
N_EPOCHS=2

SKIP_EXISTING="${SKIP_EXISTING:-0}"
PYTHON=/media/volume/tuc_data/self_play_LLMs/miniconda3/envs/WSPIN/bin/python
RESULTS_DIR="$REPO_ROOT/eval_results/ablation_b2_adaptive"

# ============================================================
# Helper: generate + weight + train one iteration
# ============================================================
run_iter() {
    local NAME="$1"       # human label (e.g. "js_to_kl")
    local ITE="$2"        # 0 | 1 | 2 | 3
    local DIV="$3"        # f* for this iteration
    local INIT_CKPT="$4"  # starting model path
    local CKPT_BASE="${MODEL_BASE}/fSWIFT_${NAME}"
    local DATA_BASE="data/Ultrachat200k/fSWIFT_${NAME}"
    local DSET_BASE="Ultrachat200k/fSWIFT_${NAME}"

    echo ""
    echo "  --- [${NAME}] Iteration ${ITE} (f*=${DIV}) ---"

    if [ "$SKIP_EXISTING" = "1" ] && [ -f "${CKPT_BASE}/ite${ITE}/model.safetensors" ]; then
        echo "  [SKIP] ite${ITE} already done."
        return
    fi

    python generate_vllm.py \
        --model          "$INIT_CKPT" \
        --input_dir      "$SFT_DATA" \
        --output_dir     "${DATA_BASE}/ite${ITE}/train" \
        --max_new_tokens $MAX_NEW \
        --data_frac      $FRAC \
        --frac_len       $FRAC_LEN \
        --split          train

    python token_weight_estimation.py \
        --model_name_1      "$TEACHER" \
        --model_name_2      "$INIT_CKPT" \
        --model1_template   normal \
        --model2_template   normal \
        --input_dir         "${DATA_BASE}/ite${ITE}" \
        --output_dir        "${DATA_BASE}/ite${ITE}" \
        --max_length        $MAX_LENGTH \
        --max_prompt_length $MAX_PROMPT_LENGTH \
        --batch_size        $WEIGHT_BATCH \
        --num_gpus          $NUM_GPUS

    # Cumulative dataset: ite0 uses only ite0; ite1+ uses [prev, cur]
    local DATASETS
    if [ "$ITE" -eq 0 ]; then
        DATASETS="[\"${DSET_BASE}/ite0\"]"
    else
        local PREV=$((ITE - 1))
        DATASETS="[\"${DSET_BASE}/ite${PREV}\",\"${DSET_BASE}/ite${ITE}\"]"
    fi

    # LR schedule: 5e-7 for ite0/1, 1e-7 for ite2/3
    local LR
    if [ "$ITE" -le 1 ]; then
        LR=5e-7
    else
        LR=1e-7
    fi

    python -u train.py \
        model=qwen \
        model.name_or_path="${INIT_CKPT}" \
        loss=fswift \
        loss.f_divergence="${DIV}" \
        base_data_dir=data \
        ckpt_dir="${CKPT_BASE}/ite${ITE}/" \
        datasets="${DATASETS}" \
        batch_size=$BATCH \
        n_epochs=$N_EPOCHS \
        lr=$LR \
        iteration=$ITE
}

# ============================================================
# Run a full 4-iteration schedule
# Args: NAME  DIV_ITE0  DIV_ITE1  DIV_ITE2  DIV_ITE3
# ============================================================
run_schedule() {
    local NAME="$1"
    local DIVS=("$2" "$3" "$4" "$5")

    echo ""
    echo "============================================================"
    echo " [B2] Schedule: ${NAME}"
    echo "      ite0=${DIVS[0]}  ite1=${DIVS[1]}  ite2=${DIVS[2]}  ite3=${DIVS[3]}"
    echo " Started: $(date)"
    echo "============================================================"

    local CKPT_BASE="${MODEL_BASE}/fSWIFT_${NAME}"

    # ite0
    run_iter "$NAME" 0 "${DIVS[0]}" "$SFT_MODEL"
    # ite1
    run_iter "$NAME" 1 "${DIVS[1]}" "${CKPT_BASE}/ite0"
    # ite2
    run_iter "$NAME" 2 "${DIVS[2]}" "${CKPT_BASE}/ite1"
    # ite3
    run_iter "$NAME" 3 "${DIVS[3]}" "${CKPT_BASE}/ite2"

    echo ""
    echo "===== [${NAME}] Schedule complete! Finished: $(date) ====="
}

# ============================================================
# Evaluate a checkpoint
# ============================================================
eval_checkpoint() {
    local label="$1"
    local model_path="$2"
    local schedule="$3"
    local iter="$4"

    if [ ! -d "$model_path" ]; then
        echo "[SKIP] $label — not found."
        return
    fi

    echo ""
    echo "[EVAL] $label  ($model_path)"

    local out_dir="$RESULTS_DIR/$label"
    mkdir -p "$out_dir"

    cd "$REPO_ROOT/lm-evaluation-harness"
    declare -A TASK_FEWSHOT=(
        [arc_challenge]=25 [truthfulqa_mc2]=0 [winogrande]=5
        [gsm8k]=5 [mmlu]=5 [hellaswag]=10
    )
    for TASK in arc_challenge truthfulqa_mc2 winogrande gsm8k mmlu hellaswag; do
        $PYTHON -m lm_eval --model hf \
            --model_args pretrained="$model_path" \
            --tasks "$TASK" \
            --num_fewshot "${TASK_FEWSHOT[$TASK]}" \
            --device cuda:0 \
            --batch_size auto \
            --output_path "$out_dir/$TASK" 2>&1
    done
    cd "$REPO_ROOT"

    $PYTHON - "$out_dir" "$label" "$schedule" "$iter" "$RESULTS_DIR/summary.tsv" << 'PYEOF'
import sys, json, glob
out_dir, label, schedule, iter_, summary = sys.argv[1:]
results = {}
for task in ["arc_challenge","truthfulqa_mc2","winogrande","gsm8k","mmlu","hellaswag"]:
    hits = glob.glob(f"{out_dir}/{task}/**/results*.json", recursive=True)
    if hits:
        results[task] = json.load(open(hits[0])).get("results", {}).get(task, {})
arc   = results.get("arc_challenge",  {}).get("acc_norm,none", 0) * 100
truth = results.get("truthfulqa_mc2", {}).get("acc,none",      0) * 100
wino  = results.get("winogrande",     {}).get("acc,none",      0) * 100
gsm   = results.get("gsm8k",          {}).get("exact_match,strict-match", 0) * 100
mmlu  = results.get("mmlu",           {}).get("acc,none",      0) * 100
hella = results.get("hellaswag",      {}).get("acc_norm,none", 0) * 100
avg   = (arc + truth + wino + gsm + mmlu + hella) / 6
with open(summary, "a") as f:
    f.write(f"{label}\t{schedule}\t{iter_}\t{arc:.2f}\t{truth:.2f}\t{wino:.2f}\t{gsm:.2f}\t{mmlu:.2f}\t{hella:.2f}\t{avg:.2f}\n")
print(f"[RESULT] {label}: avg={avg:.2f}")
PYEOF
}

# ============================================================
# Main
# ============================================================
echo "============================================================"
echo " Ablation B2: Adaptive f-Divergence Scheduling"
echo " Started: $(date)"
echo "============================================================"

mkdir -p "$RESULTS_DIR"
echo -e "label\tschedule\titer\tarc\ttruthful\twino\tgsm8k\tmmlu\thellaswag\tavg" \
    > "$RESULTS_DIR/summary.tsv"

# --- Schedule A: JS → KL (switch at ite2) ---
run_schedule "js_to_kl"   js js kl kl

# --- Schedule B: JS → chi2 (switch at ite2) ---
run_schedule "js_to_chi2" js js chi2 chi2

# --- Fixed baselines (reuse B1 checkpoints if available) ---
# identity (SWIFT)
if [ "$SKIP_EXISTING" = "1" ] && [ -d "${MODEL_BASE}/fSWIFT_identity" ]; then
    echo "[INFO] Reusing B1 identity checkpoints."
else
    run_schedule "adaptive_identity" identity identity identity identity
fi

# fixed js
if [ "$SKIP_EXISTING" = "1" ] && [ -d "${MODEL_BASE}/fSWIFT_js" ]; then
    echo "[INFO] Reusing B1 js checkpoints."
else
    run_schedule "adaptive_js" js js js js
fi

# fixed kl
if [ "$SKIP_EXISTING" = "1" ] && [ -d "${MODEL_BASE}/fSWIFT_kl" ]; then
    echo "[INFO] Reusing B1 kl checkpoints."
else
    run_schedule "adaptive_kl" kl kl kl kl
fi

# --- Evaluate iter3 of every schedule ---
echo ""
echo "============================================================"
echo " Evaluating iter3 of all schedules"
echo "============================================================"

# adaptive schedules
eval_checkpoint "js_to_kl_ite3"   "${MODEL_BASE}/fSWIFT_js_to_kl/ite3"   "js→kl"   "ite3"
eval_checkpoint "js_to_chi2_ite3" "${MODEL_BASE}/fSWIFT_js_to_chi2/ite3" "js→chi2" "ite3"

# fixed baselines — point at B1 dirs if they exist, else adaptive dirs
for SCHED_NAME in "identity" "js" "kl"; do
    SRC_DIR="${MODEL_BASE}/fSWIFT_${SCHED_NAME}"
    if [ ! -d "$SRC_DIR" ]; then
        SRC_DIR="${MODEL_BASE}/fSWIFT_adaptive_${SCHED_NAME}"
    fi
    eval_checkpoint "fixed_${SCHED_NAME}_ite3" "${SRC_DIR}/ite3" "fixed_${SCHED_NAME}" "ite3"
done

echo ""
echo "============================================================"
echo " B2 COMPLETE"
echo " Summary: $RESULTS_DIR/summary.tsv"
echo " Finished: $(date)"
echo "============================================================"
echo ""
column -t -s $'\t' "$RESULTS_DIR/summary.tsv"
