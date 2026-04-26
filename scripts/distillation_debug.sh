#!/bin/bash
set -euo pipefail

# Quick smoke-test: 200 examples, 1 epoch, 1 dataset (dolly only)
# Run: bash scripts/distillation_debug.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

TEACHER="${TEACHER:-model_hub/Qwen2.5-7B-Instruct}"
STUDENT_BASE="${STUDENT_BASE:-model_hub/gpt2-xl}"
STUDENT_OUT="model_hub/gpt2-xl/distill_debug"
DATA_ROOT="data/distillation_debug"
N_SAMPLES=200
BATCH=4
GRAD_ACCUM=2
MAX_LENGTH=512
MAX_PROMPT_LENGTH=384
F_DIVERGENCE="${F_DIVERGENCE:-js}"
RESULTS_DIR="eval_results/distillation_debug"

echo "============================================================"
echo " [DEBUG] Distillation smoke-test  (${N_SAMPLES} samples)"
echo " Teacher : $TEACHER"
echo " Student : $STUDENT_BASE → $STUDENT_OUT"
echo " F*      : $F_DIVERGENCE"
echo "============================================================"

# ── Step 1: Lấy 200 mẫu từ dolly ────────────────────────────
echo ""
echo "===== Step 1: Sampling ${N_SAMPLES} examples from dolly ====="
mkdir -p "${DATA_ROOT}/dolly"

if [ ! -f "${DATA_ROOT}/dolly/train.jsonl" ]; then
    head -${N_SAMPLES} data/distillation/dolly/train.jsonl > "${DATA_ROOT}/dolly/train.jsonl"
    echo "  Sampled ${N_SAMPLES} lines → ${DATA_ROOT}/dolly/train.jsonl"
else
    echo "[SKIP] Debug train data already exists"
fi

if [ ! -f "${DATA_ROOT}/dolly/test.jsonl" ]; then
    head -50 data/distillation/dolly/test.jsonl > "${DATA_ROOT}/dolly/test.jsonl"
    echo "  Sampled 50 test lines → ${DATA_ROOT}/dolly/test.jsonl"
else
    echo "[SKIP] Debug test data already exists"
fi

# ── Step 2: Generate student responses ───────────────────────
echo ""
echo "===== Step 2: Generating student responses ====="

TRAIN_FILE="${DATA_ROOT}/dolly/train.jsonl"

if python3 -c "
import json
with open('${TRAIN_FILE}') as f:
    first = json.loads(f.readline())
exit(0 if first.get('rejected','') != first.get('chosen','') else 1)
" 2>/dev/null; then
    echo "[SKIP] Student responses already generated"
else
    python generate_vllm.py \
        --model          "$STUDENT_BASE" \
        --input_dir      "$TRAIN_FILE" \
        --output_dir     "${DATA_ROOT}/dolly/train" \
        --max_new_tokens 128 \
        --data_frac      0 \
        --frac_len       1000000 \
        --split          train
fi

# ── Step 3: Compute token weights ────────────────────────────
echo ""
echo "===== Step 3: Computing token importance weights ====="

if python3 -c "
import json
with open('${TRAIN_FILE}') as f:
    first = json.loads(f.readline())
exit(0 if 'chosen_weight' in first else 1)
" 2>/dev/null; then
    echo "[SKIP] Weights already computed"
else
    python token_weight_estimation.py \
        --model_name_1      "$TEACHER" \
        --model_name_2      "$STUDENT_BASE" \
        --model1_template   normal \
        --model2_template   normal \
        --input_dir         "${DATA_ROOT}/dolly" \
        --output_dir        "${DATA_ROOT}/dolly" \
        --max_length        $MAX_LENGTH \
        --max_prompt_length $MAX_PROMPT_LENGTH \
        --batch_size        4 \
        --num_gpus          1
fi

# ── Step 4: Train ─────────────────────────────────────────────
echo ""
echo "===== Step 4: Training (f*=${F_DIVERGENCE}, n_examples=200) ====="

if [ -f "${STUDENT_OUT}/config.json" ]; then
    echo "[SKIP] Debug model already at $STUDENT_OUT — delete to retrain:"
    echo "       rm -rf $STUDENT_OUT"
else
    python -u train.py \
        model=gpt2-xl-distill \
        model.name_or_path="${STUDENT_BASE}" \
        loss=fswift \
        loss.f_divergence="${F_DIVERGENCE}" \
        base_data_dir=data \
        ckpt_dir="${STUDENT_OUT}" \
        datasets='["distillation_debug/dolly"]' \
        batch_size=$BATCH \
        gradient_accumulation_steps=$GRAD_ACCUM \
        activation_checkpointing=true \
        n_examples=200 \
        lr=5e-7 \
        max_length=$MAX_LENGTH \
        max_prompt_length=$MAX_PROMPT_LENGTH \
        iteration=0
fi

# ── Step 5: Eval ROUGE-L ──────────────────────────────────────
echo ""
echo "===== Step 5: ROUGE-L Evaluation ====="
mkdir -p "$RESULTS_DIR"

python eval_rouge.py \
    --model_path     "$STUDENT_OUT" \
    --data_dir       "$DATA_ROOT" \
    --datasets       dolly \
    --split          test \
    --output         "${RESULTS_DIR}/results.json" \
    --max_new_tokens 128 \
    --batch_size     8 \
    --device         cuda:0

echo ""
echo "============================================================"
echo " [DEBUG] Done! Results: ${RESULTS_DIR}/results.json"
python3 -c "
import json
r = json.load(open('${RESULTS_DIR}/results.json'))
print(f' ROUGE-L (dolly): {r[\"dolly\"][\"rougeL\"]:.4f}')
" 2>/dev/null || true
echo "============================================================"
