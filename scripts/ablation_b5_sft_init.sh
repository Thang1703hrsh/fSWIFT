#!/bin/bash
set -euo pipefail

# ============================================================
# Ablation B5: SFT Initialization for f-SWIFT Distillation
#
# Compares three initialisation strategies:
#   1. SFT baseline (student trained on ground-truth only)
#   2. Base → f-SWIFT JS (no SFT warmup)   ← distillation.sh
#   3. Base → SFT → f-SWIFT JS             ← distillation_fswift.sh
#
# Teacher : Qwen2.5-7B-Instruct
# Student : GPT2-XL
# Metric  : ROUGE-L on dolly / alpaca / sni / dialoguesum
#
# Usage:
#   bash scripts/ablation_b5_sft_init.sh
#   SKIP_EXISTING=1 bash scripts/ablation_b5_sft_init.sh
#
# Results:
#   eval_results/ablation_b5_sft_init/summary.tsv
# ============================================================

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SKIP_EXISTING="${SKIP_EXISTING:-0}"
PYTHON=/media/volume/tuc_data/self_play_LLMs/miniconda3/envs/WSPIN/bin/python
RESULTS_DIR="$REPO_ROOT/eval_results/ablation_b5_sft_init"

TEACHER="${TEACHER:-model_hub/Qwen2.5-7B-Instruct}"
STUDENT_BASE="${STUDENT_BASE:-model_hub/gpt2-xl}"
DATA_ROOT="data/distillation"

BATCH=4
GRAD_ACCUM=2
WEIGHT_BATCH=4
NUM_GPUS=1
MAX_LENGTH=1024
MAX_PROMPT_LENGTH=768
LR_SFT=1e-5
LR_FSWIFT=5e-7
N_EPOCHS=2
F_DIVERGENCE="${F_DIVERGENCE:-js}"

echo "============================================================"
echo " Ablation B5: SFT Initialization Comparison"
echo " f* = ${F_DIVERGENCE}"
echo " Started: $(date)"
echo "============================================================"

mkdir -p "$RESULTS_DIR"
echo -e "method\tdolly\talpaca\tsni\tdialoguesum\tavg" \
    > "$RESULTS_DIR/summary.tsv"

# ────────────────────────────────────────────────────────────
# Shared data preparation (models, datasets, weights)
# Delegate to distillation.sh env vars to avoid duplication.
# ────────────────────────────────────────────────────────────
echo ""
echo "===== Ensuring data is prepared (shared across all variants) ====="

# Lightweight check: if weights exist, assume data is ready
DATA_READY=1
for DS in dolly alpaca sni dialoguesum; do
    if ! python3 -c "
import json
with open('${DATA_ROOT}/${DS}/train.jsonl') as f:
    first = json.loads(f.readline())
exit(0 if 'chosen_weight' in first else 1)
" 2>/dev/null; then
        DATA_READY=0
        break
    fi
done

if [ "$DATA_READY" = "1" ]; then
    echo "[SKIP] All data + weights already prepared."
else
    echo "Running distillation.sh data-prep steps (Steps 1-4)..."
    # Run distillation.sh up through weight estimation, then stop.
    # We override STUDENT_OUT to a throwaway path so training is skipped
    # (distillation.sh will skip training if config.json already exists).
    SKIP_TRAIN=1 \
    STUDENT_OUT="${STUDENT_BASE}/_b5_data_prep_dummy" \
    F_DIVERGENCE="$F_DIVERGENCE" \
    bash scripts/distillation.sh || true
    echo "[INFO] If training ran in distillation.sh above, that's ok — we will reuse the data."
fi

# ────────────────────────────────────────────────────────────
# Variant 1: SFT baseline
# ────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " Variant 1: SFT baseline (student trained on chosen only)"
echo "============================================================"

STUDENT_SFT="${STUDENT_BASE}/sft"

if [ "$SKIP_EXISTING" = "1" ] && [ -f "${STUDENT_SFT}/config.json" ]; then
    echo "[SKIP] SFT model already at $STUDENT_SFT"
else
    python -u train.py \
        model=gpt2-xl-distill \
        model.name_or_path="${STUDENT_BASE}" \
        loss=sft \
        base_data_dir=data \
        ckpt_dir="${STUDENT_SFT}" \
        datasets='["distillation/dolly","distillation/alpaca","distillation/sni","distillation/dialoguesum"]' \
        batch_size=$BATCH \
        gradient_accumulation_steps=$GRAD_ACCUM \
        activation_checkpointing=true \
        n_epochs=$N_EPOCHS \
        lr=$LR_SFT \
        max_length=$MAX_LENGTH \
        max_prompt_length=$MAX_PROMPT_LENGTH \
        iteration=0
fi

SFT_RESULT="$RESULTS_DIR/results_sft.json"
if [ "$SKIP_EXISTING" = "1" ] && [ -f "$SFT_RESULT" ]; then
    echo "[SKIP] SFT eval results already at $SFT_RESULT"
else
    python eval_rouge.py \
        --model_path     "$STUDENT_SFT" \
        --data_dir       "$DATA_ROOT" \
        --datasets       dolly alpaca sni dialoguesum \
        --split          test \
        --output         "$SFT_RESULT" \
        --max_new_tokens 256 \
        --batch_size     $WEIGHT_BATCH \
        --device         cuda:0
fi

# ────────────────────────────────────────────────────────────
# Variant 2: Base → f-SWIFT JS (no SFT warmup)
# ────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " Variant 2: Base → f-SWIFT ${F_DIVERGENCE}  (no SFT warmup)"
echo "============================================================"

STUDENT_FSWIFT="${STUDENT_BASE}/distill_${F_DIVERGENCE}"

if [ "$SKIP_EXISTING" = "1" ] && [ -f "${STUDENT_FSWIFT}/config.json" ]; then
    echo "[SKIP] Base→f-SWIFT model already at $STUDENT_FSWIFT"
else
    python -u train.py \
        model=gpt2-xl-distill \
        model.name_or_path="${STUDENT_BASE}" \
        loss=fswift \
        loss.f_divergence="${F_DIVERGENCE}" \
        base_data_dir=data \
        ckpt_dir="${STUDENT_FSWIFT}" \
        datasets='["distillation/dolly","distillation/alpaca","distillation/sni","distillation/dialoguesum"]' \
        batch_size=$BATCH \
        gradient_accumulation_steps=$GRAD_ACCUM \
        activation_checkpointing=true \
        n_epochs=$N_EPOCHS \
        lr=$LR_FSWIFT \
        max_length=$MAX_LENGTH \
        max_prompt_length=$MAX_PROMPT_LENGTH \
        iteration=0
fi

BASE_FSWIFT_RESULT="$RESULTS_DIR/results_base_fswift.json"
if [ "$SKIP_EXISTING" = "1" ] && [ -f "$BASE_FSWIFT_RESULT" ]; then
    echo "[SKIP] Base→f-SWIFT eval results already exist."
else
    python eval_rouge.py \
        --model_path     "$STUDENT_FSWIFT" \
        --data_dir       "$DATA_ROOT" \
        --datasets       dolly alpaca sni dialoguesum \
        --split          test \
        --output         "$BASE_FSWIFT_RESULT" \
        --max_new_tokens 256 \
        --batch_size     $WEIGHT_BATCH \
        --device         cuda:0
fi

# ────────────────────────────────────────────────────────────
# Variant 3: Base → SFT → f-SWIFT JS
# ────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " Variant 3: Base → SFT → f-SWIFT ${F_DIVERGENCE}"
echo "============================================================"

STUDENT_SFT_FSWIFT="${STUDENT_BASE}/sft_then_fswift"

if [ "$SKIP_EXISTING" = "1" ] && [ -f "${STUDENT_SFT_FSWIFT}/config.json" ]; then
    echo "[SKIP] SFT→f-SWIFT model already at $STUDENT_SFT_FSWIFT"
else
    # SFT checkpoint is reused from Variant 1
    python -u train.py \
        model=gpt2-xl-distill \
        model.name_or_path="${STUDENT_SFT}" \
        loss=fswift \
        loss.f_divergence="${F_DIVERGENCE}" \
        base_data_dir=data \
        ckpt_dir="${STUDENT_SFT_FSWIFT}" \
        datasets='["distillation/dolly","distillation/alpaca","distillation/sni","distillation/dialoguesum"]' \
        batch_size=$BATCH \
        gradient_accumulation_steps=$GRAD_ACCUM \
        activation_checkpointing=true \
        n_epochs=$N_EPOCHS \
        lr=$LR_FSWIFT \
        max_length=$MAX_LENGTH \
        max_prompt_length=$MAX_PROMPT_LENGTH \
        iteration=0
fi

SFT_FSWIFT_RESULT="$RESULTS_DIR/results_sft_then_fswift.json"
if [ "$SKIP_EXISTING" = "1" ] && [ -f "$SFT_FSWIFT_RESULT" ]; then
    echo "[SKIP] SFT→f-SWIFT eval results already exist."
else
    python eval_rouge.py \
        --model_path     "$STUDENT_SFT_FSWIFT" \
        --data_dir       "$DATA_ROOT" \
        --datasets       dolly alpaca sni dialoguesum \
        --split          test \
        --output         "$SFT_FSWIFT_RESULT" \
        --max_new_tokens 256 \
        --batch_size     $WEIGHT_BATCH \
        --device         cuda:0
fi

# ────────────────────────────────────────────────────────────
# Collect results into summary table
# ────────────────────────────────────────────────────────────
echo ""
echo "===== Building summary table ====="

$PYTHON << PYEOF
import json, os

results_dir = "$RESULTS_DIR"
summary_path = f"{results_dir}/summary.tsv"

variants = [
    ("SFT_baseline",          f"{results_dir}/results_sft.json"),
    ("base_to_fswift_${F_DIVERGENCE}", f"{results_dir}/results_base_fswift.json"),
    ("sft_to_fswift_${F_DIVERGENCE}",  f"{results_dir}/results_sft_then_fswift.json"),
]

datasets = ["dolly", "alpaca", "sni", "dialoguesum"]
print(f"\n{'Method':<35} {'dolly':>7} {'alpaca':>7} {'sni':>7} {'dialoguesum':>12} {'avg':>7}")
print("-" * 80)

with open(summary_path, "w") as out:
    out.write("method\tdolly\talpaca\tsni\tdialoguesum\tavg\n")
    for name, path in variants:
        if not os.path.exists(path):
            print(f"  [MISSING] {path}")
            continue
        r = json.load(open(path))
        vals = [r.get(d, {}).get("rougeL", 0) for d in datasets]
        avg = sum(vals) / len(vals)
        row = "\t".join(f"{v:.4f}" for v in vals)
        out.write(f"{name}\t{row}\t{avg:.4f}\n")
        vals_str = "  ".join(f"{v:.4f}" for v in vals)
        print(f"  {name:<33} {vals[0]:>7.4f} {vals[1]:>7.4f} {vals[2]:>7.4f} {vals[3]:>12.4f} {avg:>7.4f}")
PYEOF

echo ""
echo "============================================================"
echo " B5 COMPLETE"
echo " Summary: $RESULTS_DIR/summary.tsv"
echo " Finished: $(date)"
echo "============================================================"
