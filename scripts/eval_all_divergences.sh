#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MODEL_BASE="$REPO_ROOT/model_hub/Qwen1.5-1.8B"
PYTHON=/media/volume/tuc_data/self_play_LLMs/miniconda3/envs/WSPIN/bin/python
export HF_TOKEN="${HF_TOKEN:-}"   # set via: export HF_TOKEN=hf_...

RESULTS_DIR="$REPO_ROOT/eval_results/divergence_comparison"
LOG_FILE="$RESULTS_DIR/eval_log.txt"
SUMMARY_FILE="$RESULTS_DIR/summary.tsv"

SKIP_EXISTING="${SKIP_EXISTING:-0}"

mkdir -p "$RESULTS_DIR"

# TSV header
echo -e "label\tdivergence\tgroup\titer\tarc\ttruthful\twino\tgsm8k\tmmlu\thellaswag\tavg" \
    > "$SUMMARY_FILE"

# Per-task few-shot counts
declare -A FEWSHOT=(
    [arc_challenge]=25
    [truthfulqa_mc2]=0
    [winogrande]=5
    [gsm8k]=5
    [mmlu]=5
    [hellaswag]=10
)

echo "============================================================"
echo " f-SWIFT Divergence Comparison — Full Benchmark Suite"
echo " Tasks   : arc  truthfulqa  winogrande  gsm8k  mmlu  hellaswag"
echo " Started : $(date)"
echo " Results : $RESULTS_DIR"
echo " SKIP_EXISTING=${SKIP_EXISTING}"
echo "============================================================"

# ============================================================
# Helper: evaluate one checkpoint on all 6 tasks
# Args: label  divergence  group  iter  model_path
# ============================================================
run_eval() {
    local label="$1"
    local div="$2"
    local group="$3"
    local iter="$4"
    local model_path="$5"

    if [ ! -d "$model_path" ]; then
        echo "[SKIP] $label — not found: $model_path"
        echo -e "$label\t$div\t$group\t$iter\tN/A\tN/A\tN/A\tN/A\tN/A\tN/A\tN/A" >> "$SUMMARY_FILE"
        return
    fi

    local out_dir="$RESULTS_DIR/$label"

    # Skip if all 6 task results already exist
    if [ "$SKIP_EXISTING" = "1" ]; then
        local all_done=1
        for TASK in arc_challenge truthfulqa_mc2 winogrande gsm8k mmlu hellaswag; do
            if [ -z "$(find "$out_dir/$TASK" -name "results*.json" 2>/dev/null | head -1)" ]; then
                all_done=0
                break
            fi
        done
        if [ "$all_done" = "1" ]; then
            echo "[SKIP] $label — all task results already exist."
            _append_summary "$label" "$div" "$group" "$iter" "$out_dir"
            return
        fi
    fi

    echo ""
    echo "------------------------------------------------------------"
    echo "[EVAL] $label"
    echo "       Path : $model_path"
    echo "       Time : $(date)"
    echo "------------------------------------------------------------"

    mkdir -p "$out_dir"
    cd "$REPO_ROOT/lm-evaluation-harness"

    for TASK in arc_challenge truthfulqa_mc2 winogrande gsm8k mmlu hellaswag; do
        echo "  -> $TASK (${FEWSHOT[$TASK]}-shot)"
        $PYTHON -m lm_eval --model hf \
            --model_args pretrained="$model_path" \
            --tasks "$TASK" \
            --num_fewshot "${FEWSHOT[$TASK]}" \
            --device cuda:0 \
            --batch_size auto \
            --output_path "$out_dir/$TASK" \
            2>&1 | tee -a "$LOG_FILE"
    done

    cd "$REPO_ROOT"
    _append_summary "$label" "$div" "$group" "$iter" "$out_dir"
}

# ============================================================
# Helper: read per-task JSONs and append one row to summary.tsv
# ============================================================
_append_summary() {
    local label="$1" div="$2" group="$3" iter="$4" out_dir="$5"

    $PYTHON - "$out_dir" "$label" "$div" "$group" "$iter" "$SUMMARY_FILE" << 'PYEOF'
import sys, json, glob
out_dir, label, div, group, iter_, summary = sys.argv[1:]

def get(task, key):
    hits = glob.glob(f"{out_dir}/{task}/**/results*.json", recursive=True)
    if not hits:
        return 0.0
    return json.load(open(hits[0])).get("results", {}).get(task, {}).get(key, 0.0)

# arc_challenge: acc_norm + stderr
arc   = (get("arc_challenge", "acc_norm,none") + get("arc_challenge", "acc_norm_stderr,none")) * 100
truth = get("truthfulqa_mc2", "acc,none")      * 100
wino  = get("winogrande",     "acc,none")      * 100
gsm   = get("gsm8k",          "exact_match,strict-match") * 100
mmlu  = get("mmlu",           "acc,none")      * 100
hella = get("hellaswag",      "acc_norm,none") * 100
avg   = (arc + truth + wino + gsm + mmlu + hella) / 6

with open(summary, "a") as f:
    f.write(f"{label}\t{div}\t{group}\t{iter_}\t"
            f"{arc:.2f}\t{truth:.2f}\t{wino:.2f}\t{gsm:.2f}\t{mmlu:.2f}\t{hella:.2f}\t{avg:.2f}\n")
print(f"[RESULT] {label}: arc={arc:.2f} truth={truth:.2f} wino={wino:.2f} "
      f"gsm={gsm:.2f} mmlu={mmlu:.2f} hella={hella:.2f} avg={avg:.2f}")
PYEOF
}

# # ============================================================
# # Baseline: sft_v2
# # ============================================================
# echo ""
# echo "===== BASELINE: sft_v2 ====="
# run_eval "sft_v2" "sft" "baseline" "-" \
#     "$MODEL_BASE/sft_v2"

# ============================================================
# js — fully sound, recommended default
# ============================================================
echo ""
echo "===== js — fully sound ====="
for ITE in 0 1 2 3; do
    run_eval "js_ite${ITE}" "js" "A" "ite${ITE}" \
        "$MODEL_BASE/fSWIFT_js/ite${ITE}"
done

# ============================================================
# kl — strongest gradient amplification
# ============================================================
echo ""
echo "===== kl — strongest gradient amplification ====="
for ITE in 0 1 2 3; do
    run_eval "kl_ite${ITE}" "kl" "B" "ite${ITE}" \
        "$MODEL_BASE/fSWIFT_kl/ite${ITE}"
done

# ============================================================
# wasserstein — WGAN-style clipped linear
# ============================================================
echo ""
echo "===== wasserstein — WGAN-style ====="
for ITE in 0 1 2 3; do
    run_eval "wasserstein_ite${ITE}" "wasserstein" "C" "ite${ITE}" \
        "$MODEL_BASE/fSWIFT_wasserstein/ite${ITE}"
done

# ============================================================
# Final summary table
# ============================================================
echo ""
echo "============================================================"
echo " FINAL SUMMARY"
echo "============================================================"
column -t -s $'\t' "$SUMMARY_FILE"
echo ""
echo " Summary TSV : $SUMMARY_FILE"
echo " Full log    : $LOG_FILE"
echo " Completed   : $(date)"
echo "============================================================"
