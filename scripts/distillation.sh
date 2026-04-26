#!/bin/bash
set -euo pipefail

# ============================================================
# SWIFT Knowledge Distillation Pipeline
# Teacher : Qwen2.5-7B-Instruct  (large, frozen)
# Student : GPT2-1.5B / gpt2-xl  (trained)
# Datasets: Dolly, Alpaca, S-NI, DialogueSum
# Metric  : ROUGE-L  (Table KD_RougeL in the paper)
#
# Pipeline:
#   1. Download student model (gpt2-xl)
#   2. Download teacher model (Qwen2.5-7B-Instruct)
#   3. Prepare the four benchmark datasets
#   4. Compute token importance weights (teacher offline pass)
#   5. Train student with f-SWIFT loss (single iteration)
#   6. Evaluate with ROUGE-L on all four benchmarks
#
# Usage:
#   bash scripts/distillation.sh
#
# Override defaults via env vars, e.g.:
#   TEACHER=model_hub/my-teacher bash scripts/distillation.sh
# ============================================================

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ── Models ───────────────────────────────────────────────────
TEACHER="${TEACHER:-model_hub/Qwen2.5-7B-Instruct}"
STUDENT_BASE="${STUDENT_BASE:-model_hub/gpt2-xl}"
STUDENT_OUT="${STUDENT_OUT:-model_hub/gpt2-xl/distill}"

# ── Data ─────────────────────────────────────────────────────
DATA_ROOT="data/distillation"
DATASETS=(dolly alpaca sni dialoguesum)

# ── Training hyperparams (matched to paper Appendix B) ───────
BATCH=4
GRAD_ACCUM=2                  # effective batch size = BATCH * GRAD_ACCUM = 8
WEIGHT_BATCH=4
NUM_GPUS=1
MAX_LENGTH=1024
MAX_PROMPT_LENGTH=768
LR=5e-7
F_DIVERGENCE="${F_DIVERGENCE:-js}"

# ── Results ──────────────────────────────────────────────────
RESULTS_DIR="eval_results/distillation"

echo "============================================================"
echo " SWIFT Distillation Pipeline"
echo " Teacher : $TEACHER"
echo " Student : $STUDENT_BASE  →  $STUDENT_OUT"
echo " F*      : $F_DIVERGENCE"
echo " Started : $(date)"
echo "============================================================"

# ────────────────────────────────────────────────────────────
# Step 1: Download student model (gpt2-xl = GPT2-1.5B)
# ────────────────────────────────────────────────────────────
if [ ! -d "$STUDENT_BASE" ]; then
    echo ""
    echo "===== Step 1: Downloading gpt2-xl ====="
    mkdir -p "$STUDENT_BASE"
    python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
print('Downloading gpt2-xl...')
tok = AutoTokenizer.from_pretrained('gpt2-xl')
tok.save_pretrained('$STUDENT_BASE')
model = AutoModelForCausalLM.from_pretrained('gpt2-xl')
model.save_pretrained('$STUDENT_BASE')
print('Saved to $STUDENT_BASE')
"
else
    echo "[SKIP] Student model already at $STUDENT_BASE"
fi

# ────────────────────────────────────────────────────────────
# Step 2: Download teacher model (Qwen2.5-7B-Instruct)
# ────────────────────────────────────────────────────────────
if [ ! -d "$TEACHER" ]; then
    echo ""
    echo "===== Step 2: Downloading Qwen2.5-7B-Instruct ====="
    mkdir -p "$TEACHER"
    python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
print('Downloading Qwen2.5-7B-Instruct...')
tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
tok.save_pretrained('$TEACHER')
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
model.save_pretrained('$TEACHER')
print('Saved to $TEACHER')
"
else
    echo "[SKIP] Teacher model already at $TEACHER"
fi

# ────────────────────────────────────────────────────────────
# Step 3: Prepare distillation datasets
# ────────────────────────────────────────────────────────────
echo ""
echo "===== Step 3: Preparing datasets ====="

DATASETS_PREPARED=1
for DS in "${DATASETS[@]}"; do
    if [ ! -f "${DATA_ROOT}/${DS}/train.jsonl" ]; then
        DATASETS_PREPARED=0
        break
    fi
done

if [ "$DATASETS_PREPARED" = "1" ]; then
    echo "[SKIP] All datasets already prepared."
else
    python distill_data_prep.py \
        --out_dir "$DATA_ROOT" \
        --n_train 10000 \
        --n_test  500
fi

# ────────────────────────────────────────────────────────────
# Step 4a: Generate student (rejected) responses for each dataset
#   Student generates responses to replace the placeholder rejected field
#   This matches the SWIFT/SPIN self-play setup where rejected = student output
# ────────────────────────────────────────────────────────────
echo ""
echo "===== Step 4a: Generating student responses (rejected) ====="

for DS in "${DATASETS[@]}"; do
    TRAIN_FILE="${DATA_ROOT}/${DS}/train.jsonl"
    GEN_OUT="${DATA_ROOT}/${DS}/train"

    if python3 -c "
import json
with open('${TRAIN_FILE}') as f:
    first = json.loads(f.readline())
# Check if rejected differs from chosen (i.e. already generated)
exit(0 if first.get('rejected','') != first.get('chosen','') else 1)
" 2>/dev/null; then
        echo "[SKIP] Student responses already generated for $DS"
    else
        echo "  Generating student responses for $DS..."
        python generate_vllm.py \
            --model          "$STUDENT_BASE" \
            --input_dir      "$TRAIN_FILE" \
            --output_dir     "$GEN_OUT" \
            --max_new_tokens 256 \
            --data_frac      0 \
            --frac_len       1000000 \
            --split          train
    fi
done

# ────────────────────────────────────────────────────────────
# Step 4b: Compute token importance weights for each dataset
#   Teacher = Qwen2.5-7B-Instruct (model_1)
#   Student = gpt2-xl base        (model_2)
# ────────────────────────────────────────────────────────────
echo ""
echo "===== Step 4b: Computing token importance weights ====="

for DS in "${DATASETS[@]}"; do
    WEIGHT_OUT="${DATA_ROOT}/${DS}"
    TRAIN_FILE="${WEIGHT_OUT}/train.jsonl"

    if python3 -c "
import json
with open('${TRAIN_FILE}') as f:
    first = json.loads(f.readline())
exit(0 if 'chosen_weight' in first else 1)
" 2>/dev/null; then
        echo "[SKIP] Weights already computed for $DS"
    else
        echo "  Computing weights for $DS..."
        python token_weight_estimation.py \
            --model_name_1      "$TEACHER" \
            --model_name_2      "$STUDENT_BASE" \
            --model1_template   normal \
            --model2_template   normal \
            --input_dir         "$WEIGHT_OUT" \
            --output_dir        "$WEIGHT_OUT" \
            --max_length        $MAX_LENGTH \
            --max_prompt_length $MAX_PROMPT_LENGTH \
            --batch_size        $WEIGHT_BATCH \
            --num_gpus          $NUM_GPUS
    fi
done

# ────────────────────────────────────────────────────────────
# Step 5: Train student on all four datasets (single iteration)
# ────────────────────────────────────────────────────────────
echo ""
echo "===== Step 5: Training student (f*=${F_DIVERGENCE}) ====="

if [ -f "${STUDENT_OUT}/config.json" ]; then
    echo "[SKIP] Trained student already at $STUDENT_OUT"
else
    python -u train.py \
        model=gpt2-xl-distill \
        model.name_or_path="${STUDENT_BASE}" \
        loss=fswift \
        loss.f_divergence="${F_DIVERGENCE}" \
        base_data_dir=data \
        ckpt_dir="${STUDENT_OUT}" \
        datasets='["distillation/dolly","distillation/alpaca","distillation/sni","distillation/dialoguesum"]' \
        batch_size=$BATCH \
        gradient_accumulation_steps=$GRAD_ACCUM \
        activation_checkpointing=true \
        n_epochs=2 \
        lr=$LR \
        max_length=$MAX_LENGTH \
        max_prompt_length=$MAX_PROMPT_LENGTH \
        iteration=0
fi

# ────────────────────────────────────────────────────────────
# Step 6: Evaluate with ROUGE-L
# ────────────────────────────────────────────────────────────
echo ""
echo "===== Step 6: ROUGE-L Evaluation ====="

mkdir -p "$RESULTS_DIR"

python eval_rouge.py \
    --model_path     "$STUDENT_OUT" \
    --data_dir       "$DATA_ROOT" \
    --datasets       dolly alpaca sni dialoguesum \
    --split          test \
    --output         "${RESULTS_DIR}/results.json" \
    --max_new_tokens 256 \
    --batch_size     $WEIGHT_BATCH \
    --device         cuda:0

echo ""
echo "============================================================"
echo " Distillation pipeline complete!"
echo " Results: ${RESULTS_DIR}/results.json"
echo " Finished: $(date)"
echo "============================================================"
