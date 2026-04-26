#!/bin/bash
set -euo pipefail

# ============================================================
# Ablation B1: f-Divergence Selection for Self-Play
#
# Runs the full 4-iteration f-SWIFT pipeline for each of the
# 6 divergences and evaluates with the standard benchmark suite.
#
# Divergences: identity  kl  js  chi2  hellinger  wasserstein
#
# Usage:
#   bash scripts/ablation_b1_fdiv_selfplay.sh          # all 6
#   bash scripts/ablation_b1_fdiv_selfplay.sh js        # one
#   SKIP_EXISTING=1 bash scripts/ablation_b1_fdiv_selfplay.sh
#
# Results:
#   eval_results/divergence_comparison/summary.tsv
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

ALL_DIVERGENCES=(identity kl js chi2 hellinger wasserstein)

if [ $# -ge 1 ]; then
    ALL_DIVERGENCES=("$1")
fi

# ============================================================
# Core: full 4-iteration pipeline for one divergence
# ============================================================
run_divergence() {
    local DIV="$1"
    local CKPT_BASE="${MODEL_BASE}/fSWIFT_${DIV}"
    local DATA_BASE="data/Ultrachat200k/fSWIFT_${DIV}"
    local DSET_BASE="Ultrachat200k/fSWIFT_${DIV}"

    echo ""
    echo "============================================================"
    echo " [B1] f-SWIFT self-play: f* = ${DIV}"
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
            --model          "$SFT_MODEL" \
            --input_dir      "$SFT_DATA" \
            --output_dir     "${DATA_BASE}/ite0/train" \
            --max_new_tokens $MAX_NEW \
            --data_frac      $FRAC \
            --frac_len       $FRAC_LEN \
            --split          train

        python token_weight_estimation.py \
            --model_name_1      "$TEACHER" \
            --model_name_2      "$SFT_MODEL" \
            --model1_template   normal \
            --model2_template   normal \
            --input_dir         "${DATA_BASE}/ite0" \
            --output_dir        "${DATA_BASE}/ite0" \
            --max_length        $MAX_LENGTH \
            --max_prompt_length $MAX_PROMPT_LENGTH \
            --batch_size        $WEIGHT_BATCH \
            --num_gpus          $NUM_GPUS

        python -u train.py \
            model=qwen \
            model.name_or_path="${SFT_MODEL}" \
            loss=fswift \
            loss.f_divergence="${DIV}" \
            base_data_dir=data \
            ckpt_dir="${CKPT_BASE}/ite0/" \
            datasets="[\"${DSET_BASE}/ite0\"]" \
            batch_size=$BATCH \
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
            --model          "${CKPT_BASE}/ite0" \
            --input_dir      "$SFT_DATA" \
            --output_dir     "${DATA_BASE}/ite1/train" \
            --max_new_tokens $MAX_NEW \
            --data_frac      $FRAC \
            --frac_len       $FRAC_LEN \
            --split          train

        python token_weight_estimation.py \
            --model_name_1      "$TEACHER" \
            --model_name_2      "${CKPT_BASE}/ite0" \
            --model1_template   normal \
            --model2_template   normal \
            --input_dir         "${DATA_BASE}/ite1" \
            --output_dir        "${DATA_BASE}/ite1" \
            --max_length        $MAX_LENGTH \
            --max_prompt_length $MAX_PROMPT_LENGTH \
            --batch_size        $WEIGHT_BATCH \
            --num_gpus          $NUM_GPUS

        python -u train.py \
            model=qwen \
            model.name_or_path="${CKPT_BASE}/ite0" \
            loss=fswift \
            loss.f_divergence="${DIV}" \
            base_data_dir=data \
            ckpt_dir="${CKPT_BASE}/ite1/" \
            datasets="[\"${DSET_BASE}/ite0\",\"${DSET_BASE}/ite1\"]" \
            batch_size=$BATCH \
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
            --model          "${CKPT_BASE}/ite1" \
            --input_dir      "$SFT_DATA" \
            --output_dir     "${DATA_BASE}/ite2/train" \
            --max_new_tokens $MAX_NEW \
            --data_frac      $FRAC \
            --frac_len       $FRAC_LEN \
            --split          train

        python token_weight_estimation.py \
            --model_name_1      "$TEACHER" \
            --model_name_2      "${CKPT_BASE}/ite1" \
            --model1_template   normal \
            --model2_template   normal \
            --input_dir         "${DATA_BASE}/ite2" \
            --output_dir        "${DATA_BASE}/ite2" \
            --max_length        $MAX_LENGTH \
            --max_prompt_length $MAX_PROMPT_LENGTH \
            --batch_size        $WEIGHT_BATCH \
            --num_gpus          $NUM_GPUS

        python -u train.py \
            model=qwen \
            model.name_or_path="${CKPT_BASE}/ite1" \
            loss=fswift \
            loss.f_divergence="${DIV}" \
            base_data_dir=data \
            ckpt_dir="${CKPT_BASE}/ite2/" \
            datasets="[\"${DSET_BASE}/ite1\",\"${DSET_BASE}/ite2\"]" \
            batch_size=$BATCH \
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
            --model          "${CKPT_BASE}/ite2" \
            --input_dir      "$SFT_DATA" \
            --output_dir     "${DATA_BASE}/ite3/train" \
            --max_new_tokens $MAX_NEW \
            --data_frac      $FRAC \
            --frac_len       $FRAC_LEN \
            --split          train

        python token_weight_estimation.py \
            --model_name_1      "$TEACHER" \
            --model_name_2      "${CKPT_BASE}/ite2" \
            --model1_template   normal \
            --model2_template   normal \
            --input_dir         "${DATA_BASE}/ite3" \
            --output_dir        "${DATA_BASE}/ite3" \
            --max_length        $MAX_LENGTH \
            --max_prompt_length $MAX_PROMPT_LENGTH \
            --batch_size        $WEIGHT_BATCH \
            --num_gpus          $NUM_GPUS

        python -u train.py \
            model=qwen \
            model.name_or_path="${CKPT_BASE}/ite2" \
            loss=fswift \
            loss.f_divergence="${DIV}" \
            base_data_dir=data \
            ckpt_dir="${CKPT_BASE}/ite3/" \
            datasets="[\"${DSET_BASE}/ite2\",\"${DSET_BASE}/ite3\"]" \
            batch_size=$BATCH \
            n_epochs=$N_EPOCHS \
            lr=1e-7 \
            iteration=3
    fi

    echo ""
    echo "===== [${DIV}] Pipeline complete! Finished: $(date) ====="
}

# ============================================================
# Evaluate one checkpoint with full benchmark suite
# ============================================================
RESULTS_DIR="$REPO_ROOT/eval_results/divergence_comparison"

eval_checkpoint() {
    local label="$1"
    local div="$2"
    local iter="$3"
    local model_path="$4"

    if [ ! -d "$model_path" ]; then
        echo "[SKIP] $label — not found: $model_path"
        return
    fi

    echo ""
    echo "------------------------------------------------------------"
    echo "[EVAL] $label  ($model_path)"
    echo "------------------------------------------------------------"

    local out_dir="$RESULTS_DIR/$label"
    mkdir -p "$out_dir"

    cd "$REPO_ROOT/lm-evaluation-harness"

    declare -A TASK_FEWSHOT=(
        [arc_challenge]=25
        [truthfulqa_mc2]=0
        [winogrande]=5
        [gsm8k]=5
        [mmlu]=5
        [hellaswag]=10
    )
    for TASK in arc_challenge truthfulqa_mc2 winogrande gsm8k mmlu hellaswag; do
        $PYTHON -m lm_eval --model hf \
            --model_args pretrained="$model_path" \
            --tasks "$TASK" \
            --num_fewshot "${TASK_FEWSHOT[$TASK]}" \
            --device cuda:0 \
            --batch_size auto \
            --output_path "$out_dir/$TASK" \
            2>&1
    done

    cd "$REPO_ROOT"

    # Merge per-task JSON results and append to summary
    $PYTHON - "$out_dir" "$label" "$div" "$iter" "$RESULTS_DIR/summary.tsv" << 'PYEOF'
import sys, json, glob, os
out_dir, label, div, iter_, summary = sys.argv[1:]
results = {}
for task in ["arc_challenge","truthfulqa_mc2","winogrande","gsm8k","mmlu","hellaswag"]:
    hits = glob.glob(f"{out_dir}/{task}/**/results*.json", recursive=True)
    if hits:
        d = json.load(open(hits[0])).get("results", {})
        results[task] = d.get(task, {})
arc   = results.get("arc_challenge",  {}).get("acc_norm,none", 0) * 100
truth = results.get("truthfulqa_mc2", {}).get("acc,none",      0) * 100
wino  = results.get("winogrande",     {}).get("acc,none",      0) * 100
gsm   = results.get("gsm8k",          {}).get("exact_match,strict-match", 0) * 100
mmlu  = results.get("mmlu",           {}).get("acc,none",      0) * 100
hella = results.get("hellaswag",      {}).get("acc_norm,none", 0) * 100
avg   = (arc + truth + wino + gsm + mmlu + hella) / 6
with open(summary, "a") as f:
    f.write(f"{label}\t{div}\t{iter_}\t{arc:.2f}\t{truth:.2f}\t{wino:.2f}\t{gsm:.2f}\t{mmlu:.2f}\t{hella:.2f}\t{avg:.2f}\n")
print(f"[RESULT] {label}: arc={arc:.2f} truth={truth:.2f} wino={wino:.2f} gsm={gsm:.2f} mmlu={mmlu:.2f} hella={hella:.2f} avg={avg:.2f}")
PYEOF
}

# ============================================================
# Main
# ============================================================
echo "============================================================"
echo " Ablation B1: f-Divergence Selection"
echo " Divergences : ${ALL_DIVERGENCES[*]}"
echo " Started     : $(date)"
echo " SKIP_EXISTING=${SKIP_EXISTING}"
echo "============================================================"

# --- Training phase ---
for DIV in "${ALL_DIVERGENCES[@]}"; do
    run_divergence "$DIV"
done

# --- Evaluation phase ---
echo ""
echo "============================================================"
echo " Evaluating all checkpoints"
echo "============================================================"

mkdir -p "$RESULTS_DIR"
echo -e "label\tdivergence\titer\tarc\ttruthful\twino\tgsm8k\tmmlu\thellaswag\tavg" \
    > "$RESULTS_DIR/summary.tsv"

for DIV in "${ALL_DIVERGENCES[@]}"; do
    for ITE in 0 1 2 3; do
        eval_checkpoint "${DIV}_ite${ITE}" "$DIV" "ite${ITE}" \
            "${MODEL_BASE}/fSWIFT_${DIV}/ite${ITE}"
    done
done

echo ""
echo "============================================================"
echo " B1 COMPLETE"
echo " Summary: $RESULTS_DIR/summary.tsv"
echo " Finished: $(date)"
echo "============================================================"
echo ""
column -t -s $'\t' "$RESULTS_DIR/summary.tsv"
