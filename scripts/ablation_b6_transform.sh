#!/bin/bash
set -euo pipefail

# ============================================================
# Ablation B6: Weight Transform Strategy for f-SWIFT
#
# Tests whether the binary transform (SWIFT default) remains
# optimal when f* is nonlinear (JS here).  Iter0 only.
#
#   Transforms: binary  threshold  threshold_and_scale
#               rank_based  random
#   f* = js (fixed)
#
# Data generation and weight estimation are shared across all
# transform variants (token weights are computed once; the
# transform is applied inside train.py at training time).
#
# Usage:
#   bash scripts/ablation_b6_transform.sh
#   SKIP_EXISTING=1 bash scripts/ablation_b6_transform.sh
#
# Results:
#   eval_results/ablation_b6_transform/summary.tsv
# ============================================================

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODEL_BASE="model_hub/Qwen1.5-1.8B"
TEACHER="model_hub/zephyr-7b-sft-full"
SFT_MODEL="${MODEL_BASE}/sft_v2"
SFT_DATA="data/Ultrachat200k/SFT/trainSFT.jsonl"
DATA_BASE="data/Ultrachat200k/fSWIFT_transform_ablation"
DSET_BASE="Ultrachat200k/fSWIFT_transform_ablation"

BATCH=8
MAX_NEW=512
FRAC_LEN=1000000
FRAC=0
WEIGHT_BATCH=8
NUM_GPUS=1
MAX_LENGTH=2048
MAX_PROMPT_LENGTH=1024
N_EPOCHS=2
F_DIVERGENCE=js

SKIP_EXISTING="${SKIP_EXISTING:-0}"
PYTHON=/media/volume/tuc_data/self_play_LLMs/miniconda3/envs/WSPIN/bin/python
RESULTS_DIR="$REPO_ROOT/eval_results/ablation_b6_transform"

TRANSFORMS=(binary threshold threshold_and_scale rank_based random)

echo "============================================================"
echo " Ablation B6: Weight Transform Strategy  (f*=js, iter0)"
echo " Transforms: ${TRANSFORMS[*]}"
echo " Started   : $(date)"
echo "============================================================"

mkdir -p "$RESULTS_DIR"
echo -e "label\ttransform\tarc\ttruthful\twino\tgsm8k\tmmlu\thellaswag\tavg" \
    > "$RESULTS_DIR/summary.tsv"

# ────────────────────────────────────────────────────────────
# Step 1: Generate + weight data once (shared across transforms)
# ────────────────────────────────────────────────────────────
echo ""
echo "===== Step 1: Generating shared data (ite0) ====="

if [ "$SKIP_EXISTING" = "1" ] && python3 -c "
import json
with open('${DATA_BASE}/ite0/train.jsonl') as f:
    first = json.loads(f.readline())
exit(0 if 'chosen_weight' in first else 1)
" 2>/dev/null; then
    echo "[SKIP] Shared data with weights already at ${DATA_BASE}/ite0"
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
fi

# ────────────────────────────────────────────────────────────
# Step 2: Train one model per transform
# ────────────────────────────────────────────────────────────
for T in "${TRANSFORMS[@]}"; do
    CKPT="${MODEL_BASE}/ablation/transform_${T}/ite0"
    LABEL="transform_${T}"

    echo ""
    echo "============================================================"
    echo " [B6] transform=${T}  (f*=${F_DIVERGENCE})"
    echo "============================================================"

    if [ "$SKIP_EXISTING" = "1" ] && [ -f "${CKPT}/model.safetensors" ]; then
        echo "[SKIP] Checkpoint already at $CKPT"
    else
        python -u train.py \
            model=qwen \
            model.name_or_path="${SFT_MODEL}" \
            loss=fswift \
            loss.f_divergence="${F_DIVERGENCE}" \
            "transform.method=${T}" \
            base_data_dir=data \
            ckpt_dir="${CKPT}/" \
            datasets="[\"${DSET_BASE}/ite0\"]" \
            batch_size=$BATCH \
            n_epochs=$N_EPOCHS \
            lr=5e-7 \
            iteration=0
    fi
done

# ────────────────────────────────────────────────────────────
# Step 3: Evaluate all checkpoints
# ────────────────────────────────────────────────────────────
echo ""
echo "===== Step 3: Evaluating all transform checkpoints ====="

for T in "${TRANSFORMS[@]}"; do
    CKPT="${MODEL_BASE}/ablation/transform_${T}/ite0"
    LABEL="transform_${T}"

    if [ ! -d "$CKPT" ]; then
        echo "[SKIP EVAL] $CKPT not found."
        continue
    fi

    echo "[EVAL] transform=${T}"
    local_out="$RESULTS_DIR/$LABEL"
    mkdir -p "$local_out"

    cd "$REPO_ROOT/lm-evaluation-harness"
    declare -A TASK_FEWSHOT=(
        [arc_challenge]=25 [truthfulqa_mc2]=0 [winogrande]=5
        [gsm8k]=5 [mmlu]=5 [hellaswag]=10
    )
    for TASK in arc_challenge truthfulqa_mc2 winogrande gsm8k mmlu hellaswag; do
        $PYTHON -m lm_eval --model hf \
            --model_args pretrained="$CKPT" \
            --tasks "$TASK" \
            --num_fewshot "${TASK_FEWSHOT[$TASK]}" \
            --device cuda:0 \
            --batch_size auto \
            --output_path "$local_out/$TASK" 2>&1
    done
    cd "$REPO_ROOT"

    $PYTHON - "$local_out" "$LABEL" "$T" "$RESULTS_DIR/summary.tsv" << 'PYEOF'
import sys, json, glob
out_dir, label, transform, summary = sys.argv[1:]
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
    f.write(f"{label}\t{transform}\t{arc:.2f}\t{truth:.2f}\t{wino:.2f}\t{gsm:.2f}\t{mmlu:.2f}\t{hella:.2f}\t{avg:.2f}\n")
print(f"[RESULT] transform={transform}: arc={arc:.2f} truth={truth:.2f} wino={wino:.2f} gsm={gsm:.2f} mmlu={mmlu:.2f} hella={hella:.2f} avg={avg:.2f}")
PYEOF
done

echo ""
echo "============================================================"
echo " B6 COMPLETE"
echo " Summary: $RESULTS_DIR/summary.tsv"
echo " Finished: $(date)"
echo "============================================================"
echo ""
column -t -s $'\t' "$RESULTS_DIR/summary.tsv"
