#!/bin/bash
set -euo pipefail

# ============================================================
# Ablation B3: β Scaling for f-SWIFT (JS divergence)
#
# β is applied before f* so it shifts S_r into different
# nonlinear regimes.  We sweep 5 values using iter0 only to
# save compute.
#
#   β ∈ { 0.01, 0.05, 0.1, 0.2, 0.5 }
#   f* = js (fixed)
#
# The β=0.1 identity row reuses the SWIFT baseline if present.
#
# Usage:
#   bash scripts/ablation_b3_beta.sh
#   SKIP_EXISTING=1 bash scripts/ablation_b3_beta.sh
#
# Results:
#   eval_results/ablation_b3_beta/summary.tsv
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
F_DIVERGENCE=js

SKIP_EXISTING="${SKIP_EXISTING:-0}"
PYTHON=/media/volume/tuc_data/self_play_LLMs/miniconda3/envs/WSPIN/bin/python
RESULTS_DIR="$REPO_ROOT/eval_results/ablation_b3_beta"

BETAS=(0.01 0.05 0.1 0.2 0.5)

echo "============================================================"
echo " Ablation B3: β Scaling  (f*=js, iter0 only)"
echo " β values: ${BETAS[*]}"
echo " Started : $(date)"
echo "============================================================"

mkdir -p "$RESULTS_DIR"
echo -e "label\tbeta\tarc\ttruthful\twino\tgsm8k\tmmlu\thellaswag\tavg" \
    > "$RESULTS_DIR/summary.tsv"

for BETA in "${BETAS[@]}"; do
    LABEL="beta_${BETA}"
    CKPT="${MODEL_BASE}/ablation/beta_${BETA}/ite0"
    DATA_DIR="data/Ultrachat200k/fSWIFT_beta_${BETA}"
    DSET="Ultrachat200k/fSWIFT_beta_${BETA}"

    echo ""
    echo "============================================================"
    echo " [B3] β=${BETA}  (f*=${F_DIVERGENCE})"
    echo "============================================================"

    if [ "$SKIP_EXISTING" = "1" ] && [ -f "${CKPT}/model.safetensors" ]; then
        echo "[SKIP] β=${BETA} checkpoint already exists."
    else
        # Generate data only once per β (reuse if data dir exists)
        if [ "$SKIP_EXISTING" = "1" ] && [ -f "${DATA_DIR}/ite0/train.jsonl" ]; then
            echo "[SKIP] data already at ${DATA_DIR}/ite0"
        else
            python generate_vllm.py \
                --model          "$SFT_MODEL" \
                --input_dir      "$SFT_DATA" \
                --output_dir     "${DATA_DIR}/ite0/train" \
                --max_new_tokens $MAX_NEW \
                --data_frac      $FRAC \
                --frac_len       $FRAC_LEN \
                --split          train

            python token_weight_estimation.py \
                --model_name_1      "$TEACHER" \
                --model_name_2      "$SFT_MODEL" \
                --model1_template   normal \
                --model2_template   normal \
                --input_dir         "${DATA_DIR}/ite0" \
                --output_dir        "${DATA_DIR}/ite0" \
                --max_length        $MAX_LENGTH \
                --max_prompt_length $MAX_PROMPT_LENGTH \
                --batch_size        $WEIGHT_BATCH \
                --num_gpus          $NUM_GPUS
        fi

        python -u train.py \
            model=qwen \
            model.name_or_path="${SFT_MODEL}" \
            loss=fswift \
            loss.f_divergence="${F_DIVERGENCE}" \
            loss.beta="${BETA}" \
            base_data_dir=data \
            ckpt_dir="${CKPT}/" \
            datasets="[\"${DSET}/ite0\"]" \
            batch_size=$BATCH \
            n_epochs=$N_EPOCHS \
            lr=5e-7 \
            iteration=0
    fi

    # Evaluate
    if [ ! -d "$CKPT" ]; then
        echo "[SKIP EVAL] $CKPT not found."
        continue
    fi

    echo "[EVAL] β=${BETA}"
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

    $PYTHON - "$local_out" "$LABEL" "$BETA" "$RESULTS_DIR/summary.tsv" << 'PYEOF'
import sys, json, glob
out_dir, label, beta, summary = sys.argv[1:]
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
    f.write(f"{label}\t{beta}\t{arc:.2f}\t{truth:.2f}\t{wino:.2f}\t{gsm:.2f}\t{mmlu:.2f}\t{hella:.2f}\t{avg:.2f}\n")
print(f"[RESULT] beta={beta}: arc={arc:.2f} truth={truth:.2f} wino={wino:.2f} gsm={gsm:.2f} mmlu={mmlu:.2f} hella={hella:.2f} avg={avg:.2f}")
PYEOF
done

echo ""
echo "============================================================"
echo " B3 COMPLETE"
echo " Summary: $RESULTS_DIR/summary.tsv"
echo " Finished: $(date)"
echo "============================================================"
echo ""
column -t -s $'\t' "$RESULTS_DIR/summary.tsv"
