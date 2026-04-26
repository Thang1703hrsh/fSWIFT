#!/bin/bash
set -euo pipefail

# ============================================================
# Ablation B4: f-Divergence in Knowledge Distillation (ROUGE-L)
#
# Runs the full distillation pipeline once per divergence
# (single iteration, all 4 KD datasets) and compares ROUGE-L.
#
#   Teacher : Qwen2.5-7B-Instruct
#   Student : GPT2-XL (gpt2-xl / GPT2-1.5B)
#   Datasets: dolly, alpaca, sni, dialoguesum
#   Metric  : ROUGE-L
#
# Each divergence run reuses pre-built data (generate + weights)
# from the first run if SKIP_DATA=1.
#
# Usage:
#   bash scripts/ablation_b4_kd_fdiv.sh
#   SKIP_DATA=1 bash scripts/ablation_b4_kd_fdiv.sh  # skip data prep
#   SKIP_EXISTING=1 bash scripts/ablation_b4_kd_fdiv.sh
#
# Results:
#   eval_results/ablation_b4_kd_fdiv/summary.tsv
# ============================================================

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

TEACHER="${TEACHER:-model_hub/Qwen2.5-7B-Instruct}"
STUDENT_BASE="${STUDENT_BASE:-model_hub/gpt2-xl}"
DATA_ROOT="data/distillation"
DATASETS=(dolly alpaca sni dialoguesum)

BATCH=4
GRAD_ACCUM=2
WEIGHT_BATCH=4
NUM_GPUS=1
MAX_LENGTH=1024
MAX_PROMPT_LENGTH=768
LR=5e-7
N_EPOCHS=2

SKIP_EXISTING="${SKIP_EXISTING:-0}"
SKIP_DATA="${SKIP_DATA:-0}"
PYTHON=/media/volume/tuc_data/self_play_LLMs/miniconda3/envs/WSPIN/bin/python
RESULTS_DIR="$REPO_ROOT/eval_results/ablation_b4_kd_fdiv"

ALL_DIVERGENCES=(identity kl js chi2 hellinger wasserstein)

echo "============================================================"
echo " Ablation B4: f-Divergence in Knowledge Distillation"
echo " Teacher  : $TEACHER"
echo " Student  : $STUDENT_BASE"
echo " Divergences: ${ALL_DIVERGENCES[*]}"
echo " Started  : $(date)"
echo "============================================================"

mkdir -p "$RESULTS_DIR"

# ────────────────────────────────────────────────────────────
# Step 1: Download student model
# ────────────────────────────────────────────────────────────
if [ ! -d "$STUDENT_BASE" ]; then
    echo ""
    echo "===== Step 1: Downloading gpt2-xl ====="
    mkdir -p "$STUDENT_BASE"
    python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
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
# Step 2: Download teacher model
# ────────────────────────────────────────────────────────────
if [ ! -d "$TEACHER" ]; then
    echo ""
    echo "===== Step 2: Downloading Qwen2.5-7B-Instruct ====="
    mkdir -p "$TEACHER"
    python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
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
# Step 3: Prepare datasets (once)
# ────────────────────────────────────────────────────────────
if [ "$SKIP_DATA" = "1" ]; then
    echo "[SKIP] Data preparation skipped (SKIP_DATA=1)"
else
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

    # ────────────────────────────────────────────────────────
    # Step 4: Generate student (rejected) responses
    # ────────────────────────────────────────────────────────
    echo ""
    echo "===== Step 4: Generating student responses ====="

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

    # ────────────────────────────────────────────────────────
    # Step 5: Compute token importance weights
    # ────────────────────────────────────────────────────────
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
fi

# ────────────────────────────────────────────────────────────
# Step 6: Train + eval for each divergence
# ────────────────────────────────────────────────────────────
echo ""
echo -e "divergence\tdolly\talpaca\tsni\tdialogusum\tavg" \
    > "$RESULTS_DIR/summary.tsv"

for DIV in "${ALL_DIVERGENCES[@]}"; do
    STUDENT_OUT="${STUDENT_BASE}/distill_${DIV}"
    RESULT_FILE="$RESULTS_DIR/results_${DIV}.json"

    echo ""
    echo "============================================================"
    echo " [B4] Training  f*=${DIV}"
    echo "============================================================"

    if [ "$SKIP_EXISTING" = "1" ] && [ -f "${STUDENT_OUT}/config.json" ]; then
        echo "[SKIP] Model already trained: $STUDENT_OUT"
    else
        python -u train.py \
            model=gpt2-xl-distill \
            model.name_or_path="${STUDENT_BASE}" \
            loss=fswift \
            loss.f_divergence="${DIV}" \
            base_data_dir=data \
            ckpt_dir="${STUDENT_OUT}" \
            datasets='["distillation/dolly","distillation/alpaca","distillation/sni","distillation/dialoguesum"]' \
            batch_size=$BATCH \
            gradient_accumulation_steps=$GRAD_ACCUM \
            activation_checkpointing=true \
            n_epochs=$N_EPOCHS \
            lr=$LR \
            max_length=$MAX_LENGTH \
            max_prompt_length=$MAX_PROMPT_LENGTH \
            iteration=0
    fi

    echo ""
    echo "[EVAL] f*=${DIV}  →  ROUGE-L"

    if [ "$SKIP_EXISTING" = "1" ] && [ -f "$RESULT_FILE" ]; then
        echo "[SKIP] Results already at $RESULT_FILE"
    else
        python eval_rouge.py \
            --model_path     "$STUDENT_OUT" \
            --data_dir       "$DATA_ROOT" \
            --datasets       dolly alpaca sni dialoguesum \
            --split          test \
            --output         "$RESULT_FILE" \
            --max_new_tokens 256 \
            --batch_size     $WEIGHT_BATCH \
            --device         cuda:0
    fi

    # Append row to summary
    if [ -f "$RESULT_FILE" ]; then
        $PYTHON - "$RESULT_FILE" "$DIV" "$RESULTS_DIR/summary.tsv" << 'PYEOF'
import sys, json
path, div, out = sys.argv[1:]
r = json.load(open(path))
datasets = ["dolly", "alpaca", "sni", "dialoguesum"]
vals = [r.get(d, {}).get("rougeL", 0) for d in datasets]
avg = sum(vals) / len(vals)
row = "\t".join(f"{v:.4f}" for v in vals)
with open(out, "a") as f:
    f.write(f"{div}\t{row}\t{avg:.4f}\n")
print(f"[RESULT] f*={div}: dolly={vals[0]:.4f} alpaca={vals[1]:.4f} sni={vals[2]:.4f} dialoguesum={vals[3]:.4f} avg={avg:.4f}")
PYEOF
    fi
done

echo ""
echo "============================================================"
echo " B4 COMPLETE"
echo " Summary: $RESULTS_DIR/summary.tsv"
echo " Finished: $(date)"
echo "============================================================"
echo ""
column -t -s $'\t' "$RESULTS_DIR/summary.tsv"
