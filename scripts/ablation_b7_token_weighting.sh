#!/bin/bash
set -euo pipefail

# ============================================================
# Ablation B7: Token Weighting Strategy for f-SWIFT
#
# Compares five token weighting strategies at iter0:
#
#   random      — weights ~ Uniform[-0.5, 1.5] (no teacher signal)
#   equal       — all tokens weight = 1 (standard SFT-style)
#   contrastive — TIS-DPO style: separate pos/neg models infer
#                 token importance (loss=tisdpo)
#   reverse     — inverts SWIFT weights: w → 1 - w
#   swift       — our default: teacher-estimated token importance
#
# Data generation and weight estimation are shared by
# random / reverse / swift (teacher weights computed once).
# equal uses the same data but ignores weights (transform=binary top_percent=100).
# contrastive uses tisdpo loss directly (no transform needed).
#
# Usage:
#   bash scripts/ablation_b7_token_weighting.sh
#   SKIP_EXISTING=1 bash scripts/ablation_b7_token_weighting.sh
#
# Results:
#   eval_results/ablation_b7_token_weighting/summary.tsv
# ============================================================

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODEL_BASE="model_hub/Qwen1.5-1.8B"
TEACHER="model_hub/zephyr-7b-sft-full"
SFT_MODEL="${MODEL_BASE}/sft-v2"
SFT_DATA="data/Ultrachat200k/SFT/trainSFT.jsonl"
DATA_BASE="data/Ultrachat200k/fSWIFT_weighting_ablation"
DSET_BASE="Ultrachat200k/fSWIFT_weighting_ablation"

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
LR=5e-7

SKIP_EXISTING="${SKIP_EXISTING:-0}"
PYTHON=/media/volume/tuc_data/self_play_LLMs/miniconda3/envs/WSPIN/bin/python
RESULTS_DIR="$REPO_ROOT/eval_results/ablation_b7_token_weighting"

echo "============================================================"
echo " Ablation B7: Token Weighting Strategy  (f*=js, iter0)"
echo " Strategies: random  equal  contrastive  reverse  swift"
echo " Started   : $(date)"
echo "============================================================"

mkdir -p "$RESULTS_DIR"
echo -e "label\tweighting\tarc\ttruthful\twino\tgsm8k\tmmlu\thellaswag\tavg" \
    > "$RESULTS_DIR/summary.tsv"

# ────────────────────────────────────────────────────────────
# Step 1: Generate student responses (shared by all variants)
# ────────────────────────────────────────────────────────────
echo ""
echo "===== Step 1: Generating student responses (ite0) ====="

if [ "$SKIP_EXISTING" = "1" ] && [ -f "${DATA_BASE}/ite0/train.jsonl" ]; then
    echo "[SKIP] Student responses already at ${DATA_BASE}/ite0/train.jsonl"
else
    python generate_vllm.py \
        --model          "$SFT_MODEL" \
        --input_dir      "$SFT_DATA" \
        --output_dir     "${DATA_BASE}/ite0/train" \
        --max_new_tokens $MAX_NEW \
        --data_frac      $FRAC \
        --frac_len       $FRAC_LEN \
        --split          train
fi

# ────────────────────────────────────────────────────────────
# Step 2: Compute teacher token weights (shared by swift/random/reverse/equal)
# ────────────────────────────────────────────────────────────
echo ""
echo "===== Step 2: Computing teacher token importance weights ====="

if [ "$SKIP_EXISTING" = "1" ] && python3 -c "
import json
with open('${DATA_BASE}/ite0/train.jsonl') as f:
    first = json.loads(f.readline())
exit(0 if 'chosen_weight' in first else 1)
" 2>/dev/null; then
    echo "[SKIP] Weights already computed at ${DATA_BASE}/ite0/train.jsonl"
else
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
# Helper: evaluate a checkpoint
# ────────────────────────────────────────────────────────────
eval_checkpoint() {
    local label="$1"
    local model_path="$2"
    local weighting="$3"

    if [ ! -d "$model_path" ]; then
        echo "[SKIP EVAL] $label — checkpoint not found at $model_path"
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

    $PYTHON - "$out_dir" "$label" "$weighting" "$RESULTS_DIR/summary.tsv" << 'PYEOF'
import sys, json, glob
out_dir, label, weighting, summary = sys.argv[1:]
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
    f.write(f"{label}\t{weighting}\t{arc:.2f}\t{truth:.2f}\t{wino:.2f}\t{gsm:.2f}\t{mmlu:.2f}\t{hella:.2f}\t{avg:.2f}\n")
print(f"[RESULT] {label}: arc={arc:.2f} truth={truth:.2f} wino={wino:.2f} gsm={gsm:.2f} mmlu={mmlu:.2f} hella={hella:.2f} avg={avg:.2f}")
PYEOF
}

# ────────────────────────────────────────────────────────────
# Step 3a: RANDOM — weights ~ Uniform[-0.5, 1.5]
# ────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " [B7] Weighting: random  (Uniform[-0.5, 1.5])"
echo "============================================================"

CKPT_RANDOM="${MODEL_BASE}/ablation/weighting_random/ite0"
if [ "$SKIP_EXISTING" = "1" ] && [ -f "${CKPT_RANDOM}/model.safetensors" ]; then
    echo "[SKIP] random checkpoint already at $CKPT_RANDOM"
else
    python -u train.py \
        model=qwen \
        model.name_or_path="${SFT_MODEL}" \
        loss=fswift \
        loss.f_divergence="${F_DIVERGENCE}" \
        "transform.method=random" \
        "transform.random.min_val=-0.5" \
        "transform.random.max_val=1.5" \
        base_data_dir=data \
        ckpt_dir="${CKPT_RANDOM}/" \
        datasets="[\"${DSET_BASE}/ite0\"]" \
        batch_size=$BATCH \
        n_epochs=$N_EPOCHS \
        lr=$LR \
        iteration=0
fi

# ────────────────────────────────────────────────────────────
# Step 3b: EQUAL — all tokens weight = 1
# transform.method=origin passes raw weights unchanged;
# we override by setting all chosen_weight to 1 via a custom
# approach: use rank_based with min_scale=max_scale=1.0 so
# every token gets exactly 1 regardless of teacher signal.
# ────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " [B7] Weighting: equal  (all tokens = 1)"
echo "============================================================"

CKPT_EQUAL="${MODEL_BASE}/ablation/weighting_equal/ite0"
if [ "$SKIP_EXISTING" = "1" ] && [ -f "${CKPT_EQUAL}/model.safetensors" ]; then
    echo "[SKIP] equal checkpoint already at $CKPT_EQUAL"
else
    python -u train.py \
        model=qwen \
        model.name_or_path="${SFT_MODEL}" \
        loss=fswift \
        loss.f_divergence="${F_DIVERGENCE}" \
        "transform.method=rank_based" \
        "transform.rank_based.min_scale=1.0" \
        "transform.rank_based.max_scale=1.0" \
        base_data_dir=data \
        ckpt_dir="${CKPT_EQUAL}/" \
        datasets="[\"${DSET_BASE}/ite0\"]" \
        batch_size=$BATCH \
        n_epochs=$N_EPOCHS \
        lr=$LR \
        iteration=0
fi

# ────────────────────────────────────────────────────────────
# Step 3c: CONTRASTIVE — TIS-DPO style token weighting
# Trains with tisdpo loss; no external weight estimation needed
# ────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " [B7] Weighting: contrastive  (TIS-DPO)"
echo "============================================================"

CKPT_CONTRASTIVE="${MODEL_BASE}/ablation/weighting_contrastive/ite0"
if [ "$SKIP_EXISTING" = "1" ] && [ -f "${CKPT_CONTRASTIVE}/model.safetensors" ]; then
    echo "[SKIP] contrastive checkpoint already at $CKPT_CONTRASTIVE"
else
    python -u train.py \
        model=qwen \
        model.name_or_path="${SFT_MODEL}" \
        loss=tisdpo \
        loss.beta=0.1 \
        base_data_dir=data \
        ckpt_dir="${CKPT_CONTRASTIVE}/" \
        datasets="[\"${DSET_BASE}/ite0\"]" \
        batch_size=$BATCH \
        n_epochs=$N_EPOCHS \
        lr=$LR \
        iteration=0
fi

# ────────────────────────────────────────────────────────────
# Step 3d: REVERSE — invert SWIFT weights: w → 1 - w
# Uses the 'reverse' transform added to preference_datasets.py
# ────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " [B7] Weighting: reverse  (1 - w)"
echo "============================================================"

CKPT_REVERSE="${MODEL_BASE}/ablation/weighting_reverse/ite0"
if [ "$SKIP_EXISTING" = "1" ] && [ -f "${CKPT_REVERSE}/model.safetensors" ]; then
    echo "[SKIP] reverse checkpoint already at $CKPT_REVERSE"
else
    python -u train.py \
        model=qwen \
        model.name_or_path="${SFT_MODEL}" \
        loss=fswift \
        loss.f_divergence="${F_DIVERGENCE}" \
        "transform.method=reverse" \
        base_data_dir=data \
        ckpt_dir="${CKPT_REVERSE}/" \
        datasets="[\"${DSET_BASE}/ite0\"]" \
        batch_size=$BATCH \
        n_epochs=$N_EPOCHS \
        lr=$LR \
        iteration=0
fi

# ────────────────────────────────────────────────────────────
# Step 3e: SWIFT — default teacher-estimated token importance
# ────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " [B7] Weighting: swift  (teacher-estimated, default)"
echo "============================================================"

CKPT_SWIFT="${MODEL_BASE}/ablation/weighting_swift/ite0"
if [ "$SKIP_EXISTING" = "1" ] && [ -f "${CKPT_SWIFT}/model.safetensors" ]; then
    echo "[SKIP] swift checkpoint already at $CKPT_SWIFT"
else
    python -u train.py \
        model=qwen \
        model.name_or_path="${SFT_MODEL}" \
        loss=fswift \
        loss.f_divergence="${F_DIVERGENCE}" \
        "transform.method=binary" \
        base_data_dir=data \
        ckpt_dir="${CKPT_SWIFT}/" \
        datasets="[\"${DSET_BASE}/ite0\"]" \
        batch_size=$BATCH \
        n_epochs=$N_EPOCHS \
        lr=$LR \
        iteration=0
fi

# ────────────────────────────────────────────────────────────
# Step 4: Evaluate all checkpoints
# ────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " Step 4: Evaluating all weighting checkpoints"
echo "============================================================"

eval_checkpoint "weighting_random"      "$CKPT_RANDOM"      "random"
eval_checkpoint "weighting_equal"       "$CKPT_EQUAL"       "equal"
eval_checkpoint "weighting_contrastive" "$CKPT_CONTRASTIVE" "contrastive"
eval_checkpoint "weighting_reverse"     "$CKPT_REVERSE"     "reverse"
eval_checkpoint "weighting_swift"       "$CKPT_SWIFT"       "swift"

echo ""
echo "============================================================"
echo " B7 COMPLETE"
echo " Summary: $RESULTS_DIR/summary.tsv"
echo " Finished: $(date)"
echo "============================================================"
echo ""
column -t -s $'\t' "$RESULTS_DIR/summary.tsv"
