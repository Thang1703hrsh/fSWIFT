#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ── Models ───────────────────────────────────────────────────
TEACHER="${TEACHER:-model_hub/Qwen2.5-7B-Instruct}"
STUDENT_BASE="${STUDENT_BASE:-model_hub/gpt2-xl}"
STUDENT_SFT="${STUDENT_SFT:-model_hub/gpt2-xl/sft}"

# ── Data ─────────────────────────────────────────────────────
DATA_ROOT="data/distillation"
DATASETS=(dolly alpaca sni dialoguesum)

BATCH=32
GRAD_ACCUM=2
WEIGHT_BATCH=16          # token_weight_estimation across 4 GPUs
NUM_GPUS=4
MAX_LENGTH=1024
MAX_PROMPT_LENGTH=768
LR_SFT=1e-5
LR_FSWIFT=5e-7
N_EPOCHS_SFT=2
N_EPOCHS_FSWIFT=2

# ── Divergences ───────────────────────────────────────────────
DIVERGENCES=(js kl wasserstein)

# ── Results ──────────────────────────────────────────────────
RESULTS_DIR="eval_results/distillation_sft_then_fswift"
SKIP_EXISTING="${SKIP_EXISTING:-0}"

echo "============================================================"
echo " SFT → f-SWIFT Distillation  (3 divergences × 4× H100)"
echo " Teacher    : $TEACHER"
echo " Student    : $STUDENT_BASE"
echo " Divergences: ${DIVERGENCES[*]}"
echo " Batch      : $BATCH  (per-GPU = $((BATCH / (GRAD_ACCUM * NUM_GPUS))))"
echo " Started    : $(date)"
echo "============================================================"

# ────────────────────────────────────────────────────────────
# Step 1: Download student model (gpt2-xl)
# ────────────────────────────────────────────────────────────
echo ""
echo "===== Step 1: Student model ====="
if [ ! -d "$STUDENT_BASE" ] || [ ! -f "${STUDENT_BASE}/config.json" ]; then
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
# Step 2: Download teacher model (Qwen2.5-7B-Instruct)
# ────────────────────────────────────────────────────────────
echo ""
echo "===== Step 2: Teacher model ====="
if [ ! -d "$TEACHER" ] || [ ! -f "${TEACHER}/config.json" ]; then
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
# Step 3: Prepare distillation datasets
# ────────────────────────────────────────────────────────────
echo ""
echo "===== Step 3: Preparing datasets ====="

DATASETS_PREPARED=1
for DS in "${DATASETS[@]}"; do
    if [ ! -f "${DATA_ROOT}/${DS}/train.jsonl" ]; then
        DATASETS_PREPARED=0; break
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
# Step 4: Generate student (rejected) responses
# ────────────────────────────────────────────────────────────
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
        echo "  Generating for $DS..."
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
# Step 5: Compute token importance weights (shared across divergences)
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
# Step 6a: SFT student — shared checkpoint for all divergences
# ────────────────────────────────────────────────────────────
echo ""
echo "===== Step 6a: SFT student (shared baseline) ====="

if [ "$SKIP_EXISTING" = "1" ] && [ -f "${STUDENT_SFT}/config.json" ]; then
    echo "[SKIP] SFT student already at $STUDENT_SFT"
else
    python -u train.py \
        model=gpt2-xl-distill \
        model.name_or_path="${STUDENT_BASE}" \
        loss=sft \
        trainer=FSDPTrainer \
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
# Step 6b: f-SWIFT from SFT checkpoint — one run per divergence
# ────────────────────────────────────────────────────────────
echo ""
echo "===== Step 6b: f-SWIFT training (3 divergences) ====="

for DIV in "${DIVERGENCES[@]}"; do
    STUDENT_OUT="model_hub/gpt2-xl/sft_then_fswift_${DIV}"

    echo ""
    echo "  --- f* = ${DIV} ---"

    if [ "$SKIP_EXISTING" = "1" ] && [ -f "${STUDENT_OUT}/config.json" ]; then
        echo "  [SKIP] Already trained: $STUDENT_OUT"
        continue
    fi

    python -u train.py \
        model=gpt2-xl-distill \
        model.name_or_path="${STUDENT_SFT}" \
        loss=fswift \
        loss.f_divergence="${DIV}" \
        trainer=FSDPTrainer \
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
done

# ────────────────────────────────────────────────────────────
# Step 7: ROUGE-L evaluation
# ────────────────────────────────────────────────────────────
echo ""
echo "===== Step 7: ROUGE-L Evaluation ====="

mkdir -p "$RESULTS_DIR"

# SFT baseline
SFT_RESULT="${RESULTS_DIR}/results_sft.json"
if [ "$SKIP_EXISTING" = "1" ] && [ -f "$SFT_RESULT" ]; then
    echo "[SKIP] SFT results already at $SFT_RESULT"
else
    echo "  Evaluating SFT baseline..."
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

# f-SWIFT per divergence
for DIV in "${DIVERGENCES[@]}"; do
    STUDENT_OUT="model_hub/gpt2-xl/sft_then_fswift_${DIV}"
    DIV_RESULT="${RESULTS_DIR}/results_${DIV}.json"

    if [ "$SKIP_EXISTING" = "1" ] && [ -f "$DIV_RESULT" ]; then
        echo "[SKIP] Results for f*=${DIV} already at $DIV_RESULT"
        continue
    fi

    echo "  Evaluating f*=${DIV}..."
    python eval_rouge.py \
        --model_path     "$STUDENT_OUT" \
        --data_dir       "$DATA_ROOT" \
        --datasets       dolly alpaca sni dialoguesum \
        --split          test \
        --output         "$DIV_RESULT" \
        --max_new_tokens 256 \
        --batch_size     $WEIGHT_BATCH \
        --device         cuda:0
done

# ────────────────────────────────────────────────────────────
# Summary table
# ────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " Pipeline complete! Finished: $(date)"
echo "============================================================"

python3 - << PYEOF
import json, os

results_dir = "$RESULTS_DIR"
divergences = ["js", "kl", "wasserstein"]
datasets    = ["dolly", "alpaca", "sni", "dialoguesum"]

def load(path):
    if not os.path.exists(path):
        return None
    return json.load(open(path))

def avg(r):
    return sum(r[d]["rougeL"] for d in datasets if d in r) / len(datasets)

# Build rows
rows = []
sft = load(f"{results_dir}/results_sft.json")
if sft:
    rows.append(("SFT_baseline", sft))
for div in divergences:
    r = load(f"{results_dir}/results_{div}.json")
    if r:
        rows.append((f"SFT→f-SWIFT({div})", r))

if not rows:
    print("No results found yet.")
else:
    # Write TSV
    tsv = f"{results_dir}/summary.tsv"
    with open(tsv, "w") as f:
        f.write("method\tdolly\talpaca\tsni\tdialoguesum\tavg\n")
        for name, r in rows:
            vals = [r.get(d, {}).get("rougeL", 0) for d in datasets]
            f.write(f"{name}\t" + "\t".join(f"{v:.4f}" for v in vals) + f"\t{avg(r):.4f}\n")

    # Print table
    col_w = 22
    header = f"{'Method':<{col_w}}" + "".join(f"{d:>10}" for d in datasets) + f"{'avg':>10}"
    print("\n=== ROUGE-L Comparison ===")
    print(header)
    print("-" * len(header))
    for name, r in rows:
        vals = [r.get(d, {}).get("rougeL", 0) for d in datasets]
        print(f"{name:<{col_w}}" + "".join(f"{v:>10.4f}" for v in vals) + f"{avg(r):>10.4f}")
    print(f"\nSummary TSV: {tsv}")
PYEOF
