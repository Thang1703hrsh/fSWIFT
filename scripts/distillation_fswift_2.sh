#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ── Models ───────────────────────────────────────────────────
TEACHER="${TEACHER:-model_hub/Qwen1.5-1.8B/sft-distill}"
STUDENT_BASE="${STUDENT_BASE:-model_hub/gpt2-medium/sft}"
STUDENT_SFT="${STUDENT_SFT:-model_hub/gpt2-medium/sft}"
STUDENT_OUT="${STUDENT_OUT:-model_hub/gpt2-medium/sft_then_fswift}"

# ── Data ─────────────────────────────────────────────────────
# Separate DATA_ROOT to avoid conflicts with other distillation runs
# (gpt2-xl run uses data/distillation; this run uses data/distillation_gpt2medium)
DATA_ROOT="data/distillation_gpt2medium"
DATASETS=(dolly alpaca sni dialoguesum)

# ── Hyperparams ───────────────────────────────────────────────
BATCH=32
GRAD_ACCUM=16                 # effective batch size = 32, microbatch = 2/step
WEIGHT_BATCH=16
NUM_GPUS=1
MAX_LENGTH=1024
MAX_PROMPT_LENGTH=768
LR_SFT=1e-5                   # higher LR for SFT
LR_FSWIFT=5e-7                # lower LR for f-SWIFT (paper Appendix B)
N_EPOCHS_SFT=2
N_EPOCHS_FSWIFT=2
F_DIVERGENCE="${F_DIVERGENCE:-js}"

# ── Results ──────────────────────────────────────────────────
RESULTS_DIR="eval_results/distillation_qwen18b_gpt2medium"

echo "============================================================"
echo " SFT → f-SWIFT Distillation Pipeline"
echo " Teacher : $TEACHER  (already instruction-tuned)"
echo " Student : $STUDENT_BASE → SFT → $STUDENT_SFT → f-SWIFT → $STUDENT_OUT"
echo " F*      : $F_DIVERGENCE"
echo " Started : $(date)"
echo "============================================================"

# ────────────────────────────────────────────────────────────
# Step 1: Check student model (gpt2-medium/sft)
# ────────────────────────────────────────────────────────────
if [ ! -d "$STUDENT_BASE" ]; then
    echo "ERROR: Student model not found at $STUDENT_BASE"
    echo "Please run SFT for gpt2-medium first."
    exit 1
else
    echo "[SKIP] Student model already at $STUDENT_BASE"
fi

# ────────────────────────────────────────────────────────────
# Step 2: Check teacher model (Qwen1.5-1.8B/sft-distill)
# ────────────────────────────────────────────────────────────
if [ ! -d "$TEACHER" ]; then
    echo "ERROR: Teacher model not found at $TEACHER"
    echo "Please run SFT for Qwen1.5-1.8B first."
    exit 1
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
        --out_dir "$DATA_ROOT"
fi

# ────────────────────────────────────────────────────────────
# Step 4: Generate student (rejected) responses
# ────────────────────────────────────────────────────────────
echo ""
echo "===== Step 4: Generating student responses (rejected) ====="

for DS in "${DATASETS[@]}"; do
    TRAIN_FILE="${DATA_ROOT}/${DS}/train.jsonl"
    GEN_OUT="${DATA_ROOT}/${DS}/train"

    if python3 -c "
import json
with open('${TRAIN_FILE}') as f:
    first = json.loads(f.readline())
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
# Step 5: Compute token importance weights
#   Teacher = Qwen1.5-1.8B/sft-distill (model_1)
#   Student = gpt2-medium/sft           (model_2)
# ────────────────────────────────────────────────────────────
echo ""
echo "===== Step 5: Computing token importance weights ====="

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
# Step 6a: SFT student on ground-truth responses (chosen)
# ────────────────────────────────────────────────────────────
echo ""
echo "===== Step 6a: SFT student on ground-truth responses ====="

if [ -f "${STUDENT_SFT}/config.json" ]; then
    echo "[SKIP] SFT student already at $STUDENT_SFT"
else
    python -u train.py \
        model=gpt2-medium \
        model.name_or_path="${STUDENT_BASE}" \
        loss=sft \
        base_data_dir=data \
        ckpt_dir="${STUDENT_SFT}" \
        datasets='["distillation/dolly","distillation/alpaca","distillation/sni","distillation/dialoguesum"]' \
        batch_size=$BATCH \
        gradient_accumulation_steps=$GRAD_ACCUM \
        activation_checkpointing=true \
        n_epochs=$N_EPOCHS_SFT \
        lr=$LR_SFT \
        max_length=$MAX_LENGTH \
        max_prompt_length=$MAX_PROMPT_LENGTH \
        iteration=0
fi

# ────────────────────────────────────────────────────────────
# Step 6b: f-SWIFT from SFT checkpoint
#   Policy     = SFT student (π_θ₀ = gpt2-medium/sft)
#   Reference  = SFT student (same, frozen)
# ────────────────────────────────────────────────────────────
echo ""
echo "===== Step 6b: f-SWIFT from SFT checkpoint (f*=${F_DIVERGENCE}) ====="

if [ -f "${STUDENT_OUT}/config.json" ]; then
    echo "[SKIP] f-SWIFT student already at $STUDENT_OUT"
else
    python -u train.py \
        model=gpt2-medium \
        model.name_or_path="${STUDENT_SFT}" \
        loss=fswift \
        loss.f_divergence="${F_DIVERGENCE}" \
        base_data_dir=data \
        ckpt_dir="${STUDENT_OUT}" \
        datasets='["distillation/dolly","distillation/alpaca","distillation/sni","distillation/dialoguesum"]' \
        batch_size=$BATCH \
        gradient_accumulation_steps=$GRAD_ACCUM \
        activation_checkpointing=true \
        n_epochs=$N_EPOCHS_FSWIFT \
        lr=$LR_FSWIFT \
        max_length=$MAX_LENGTH \
        max_prompt_length=$MAX_PROMPT_LENGTH \
        iteration=0
fi

# ────────────────────────────────────────────────────────────
# Step 7: Evaluate both SFT and SFT→f-SWIFT with ROUGE-L
# ────────────────────────────────────────────────────────────
echo ""
echo "===== Step 7: ROUGE-L Evaluation ====="

mkdir -p "$RESULTS_DIR"

echo "  Evaluating SFT student..."
python eval_rouge.py \
    --model_path     "$STUDENT_SFT" \
    --data_dir       "$DATA_ROOT" \
    --datasets       dolly alpaca sni dialoguesum \
    --split          test \
    --output         "${RESULTS_DIR}/results_sft.json" \
    --max_new_tokens 256 \
    --batch_size     $WEIGHT_BATCH \
    --device         cuda:0

echo "  Evaluating SFT→f-SWIFT student..."
python eval_rouge.py \
    --model_path     "$STUDENT_OUT" \
    --data_dir       "$DATA_ROOT" \
    --datasets       dolly alpaca sni dialoguesum \
    --split          test \
    --output         "${RESULTS_DIR}/results_sft_then_fswift.json" \
    --max_new_tokens 256 \
    --batch_size     $WEIGHT_BATCH \
    --device         cuda:0

echo ""
echo "============================================================"
echo " SFT → f-SWIFT pipeline complete!"
echo " SFT results      : ${RESULTS_DIR}/results_sft.json"
echo " f-SWIFT results  : ${RESULTS_DIR}/results_sft_then_fswift.json"
echo " Finished: $(date)"
echo "============================================================"

# Print comparison
python3 -c "
import json
def avg(r): return sum(r[d]['rougeL'] for d in r) / len(r)
try:
    sft   = json.load(open('${RESULTS_DIR}/results_sft.json'))
    fswift = json.load(open('${RESULTS_DIR}/results_sft_then_fswift.json'))
    print('\n=== ROUGE-L Comparison ===')
    print(f'{'Dataset':<15} {'SFT':>8} {'SFT→f-SWIFT':>14}')
    print('-' * 40)
    for d in ['dolly','alpaca','sni','dialoguesum']:
        print(f'{d:<15} {sft[d][\"rougeL\"]:>8.4f} {fswift[d][\"rougeL\"]:>14.4f}')
    print('-' * 40)
    print(f'{'Average':<15} {avg(sft):>8.4f} {avg(fswift):>14.4f}')
except Exception as e:
    print(f'Could not print comparison: {e}')
"
