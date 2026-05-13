#!/bin/bash
set -euo pipefail
# Evaluate Base / DPO / SPIN / SWIFT / CRAFT on:
#   Reasoning : Big-Bench-Hard (BBH, 3-shot), DROP (3-shot)
#   Agentic   : ToolBench (Act.EM, F1, HalluRate, Rouge-L)
# Backbone    : Qwen2.5-7B-Instruct

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON=/media/volume/tuc_data/self_play_LLMs/miniconda3/envs/WSPIN/bin/python

RESULTS_DIR="$REPO_ROOT/eval_results/reasoning_agentic"
LOG_FILE="$RESULTS_DIR/eval_log.txt"
SKIP_EXISTING="${SKIP_EXISTING:-0}"

# ── Model paths ──────────────────────────────────────────────
# Set BASE to a HuggingFace ID or local path; others to your
# local checkpoints under model_hub/Qwen2.5-7B-Instruct/.
MODEL_ROOT="$REPO_ROOT/model_hub/Qwen2.5-7B-Instruct"

declare -A MODELS=(
    [base]="Qwen/Qwen2.5-7B-Instruct"
    [dpo]="$MODEL_ROOT/dpo"
    [spin]="$MODEL_ROOT/spin"
    [swift]="$MODEL_ROOT/swift"
    [craft]="$MODEL_ROOT/craft"
)

# ── ToolBench test data (JSONL) ───────────────────────────────
# See eval_toolbench.py for the expected data format.
# Download guidance is printed if the file is missing.
TOOLBENCH_DATA="${TOOLBENCH_DATA:-$REPO_ROOT/data/toolbench/test.jsonl}"

# ── Misc ──────────────────────────────────────────────────────
mkdir -p "$RESULTS_DIR"

REASONING_SUMMARY="$RESULTS_DIR/reasoning_summary.tsv"
TOOLBENCH_SUMMARY="$RESULTS_DIR/toolbench_summary.tsv"

echo -e "model\tbbh\tdrop\tavg"   > "$REASONING_SUMMARY"
echo -e "model\tact_em\tf1\thallur\trouge_l" > "$TOOLBENCH_SUMMARY"

echo "============================================================"
echo " Reasoning & Agentic Evaluation — Qwen2.5-7B-Instruct"
echo " Reasoning : BBH (3-shot), DROP (3-shot)"
echo " Agentic   : ToolBench (Act.EM, F1, HalluRate, Rouge-L)"
echo " Models    : base dpo spin swift craft"
echo " Started   : $(date)"
echo " Results   : $RESULTS_DIR"
echo "============================================================"

# ── Helper: check if model path is a local directory ─────────
is_local() { [[ "$1" != *"/"* ]] || [ -d "$1" ]; }

# ============================================================
# BBH + DROP via lm-eval
# ============================================================
run_reasoning_eval() {
    local name="$1"
    local model_path="$2"
    local out_dir="$RESULTS_DIR/$name"

    # Skip if local path doesn't exist (HuggingFace IDs are always valid)
    if [[ "$model_path" == *"/"* && "$model_path" != *":"* && ! -d "$model_path" ]]; then
        echo "[SKIP] $name — checkpoint not found: $model_path"
        echo -e "$name\tN/A\tN/A\tN/A" >> "$REASONING_SUMMARY"
        return
    fi

    if [ "$SKIP_EXISTING" = "1" ]; then
        bbh_done=$(find "$out_dir/bbh"  -name "results*.json" 2>/dev/null | head -1)
        drop_done=$(find "$out_dir/drop" -name "results*.json" 2>/dev/null | head -1)
        if [ -n "$bbh_done" ] && [ -n "$drop_done" ]; then
            echo "[SKIP] $name — reasoning results already exist"
            _append_reasoning_summary "$name" "$out_dir"
            return
        fi
    fi

    echo ""
    echo "------------------------------------------------------------"
    echo "[EVAL] $name — BBH + DROP"
    echo "       Model : $model_path"
    echo "       Time  : $(date)"
    echo "------------------------------------------------------------"
    mkdir -p "$out_dir"
    cd "$REPO_ROOT/lm-evaluation-harness"

    for TASK in bbh drop; do
        echo "  -> $TASK (3-shot)"
        $PYTHON -m lm_eval --model hf \
            --model_args pretrained="$model_path",dtype=bfloat16 \
            --tasks "$TASK" \
            --num_fewshot 3 \
            --device cuda:0 \
            --batch_size auto \
            --output_path "$out_dir/$TASK" \
            2>&1 | tee -a "$LOG_FILE"
    done

    cd "$REPO_ROOT"
    _append_reasoning_summary "$name" "$out_dir"
}

_append_reasoning_summary() {
    local name="$1"
    local out_dir="$2"

    $PYTHON - "$out_dir" "$name" "$REASONING_SUMMARY" << 'PYEOF'
import sys, json, glob
out_dir, name, summary_path = sys.argv[1:]

def extract_score(task_prefix, *candidate_keys):
    hits = glob.glob(f"{out_dir}/{task_prefix}/**/results*.json", recursive=True)
    if not hits:
        return None
    data = json.load(open(hits[0]))
    results = data.get("results", {})

    # Try top-level aggregate first (e.g. {"bbh": {"acc_norm,none": 0.65}})
    if task_prefix in results:
        for k in candidate_keys:
            if k in results[task_prefix]:
                return results[task_prefix][k]

    # Average across matching subtasks (e.g. {"bbh_boolean_understanding": {...}})
    vals = []
    for tname, tresults in results.items():
        if tname.startswith(task_prefix):
            for k in candidate_keys:
                if k in tresults:
                    vals.append(tresults[k])
                    break
    return (sum(vals) / len(vals)) if vals else None

# BBH: accuracy (normalized or plain)
bbh_raw = extract_score("bbh",  "acc_norm,none", "acc,none", "exact_match,none")
# DROP: F1 is the primary metric; fall back to EM
drop_raw = extract_score("drop", "f1,none", "em,none", "acc,none")

bbh  = (bbh_raw  * 100) if bbh_raw  is not None else float("nan")
drop = (drop_raw * 100) if drop_raw is not None else float("nan")

import math
valid = [x for x in [bbh, drop] if not math.isnan(x)]
avg  = sum(valid) / len(valid) if valid else float("nan")

def fmt(v): return f"{v:.2f}" if not math.isnan(v) else "N/A"

with open(summary_path, "a") as f:
    f.write(f"{name}\t{fmt(bbh)}\t{fmt(drop)}\t{fmt(avg)}\n")
print(f"[RESULT] {name}: bbh={fmt(bbh)}  drop={fmt(drop)}  avg={fmt(avg)}")
PYEOF
}

# ============================================================
# ToolBench evaluation
# ============================================================
run_toolbench_eval() {
    local name="$1"
    local model_path="$2"
    local out_file="$RESULTS_DIR/$name/toolbench.json"

    if [[ "$model_path" == *"/"* && "$model_path" != *":"* && ! -d "$model_path" ]]; then
        echo "[SKIP] $name — checkpoint not found: $model_path"
        echo -e "$name\tN/A\tN/A\tN/A\tN/A" >> "$TOOLBENCH_SUMMARY"
        return
    fi

    if [ ! -f "$TOOLBENCH_DATA" ]; then
        echo ""
        echo "[SKIP] ToolBench data not found: $TOOLBENCH_DATA"
        echo "       Provide a JSONL file at that path, or override:"
        echo "         TOOLBENCH_DATA=/your/path/test.jsonl bash $0"
        echo "       See eval_toolbench.py for the expected data schema."
        echo -e "$name\tN/A\tN/A\tN/A\tN/A" >> "$TOOLBENCH_SUMMARY"
        return
    fi

    if [ "$SKIP_EXISTING" = "1" ] && [ -f "$out_file" ]; then
        echo "[SKIP] $name — toolbench results already exist"
        _append_toolbench_summary "$name" "$out_file"
        return
    fi

    echo ""
    echo "------------------------------------------------------------"
    echo "[EVAL] $name — ToolBench"
    echo "       Model : $model_path"
    echo "       Data  : $TOOLBENCH_DATA"
    echo "       Time  : $(date)"
    echo "------------------------------------------------------------"
    mkdir -p "$(dirname "$out_file")"

    $PYTHON "$REPO_ROOT/eval_toolbench.py" \
        --model_path "$model_path" \
        --data_path  "$TOOLBENCH_DATA" \
        --output_path "$out_file" \
        2>&1 | tee -a "$LOG_FILE"

    _append_toolbench_summary "$name" "$out_file"
}

_append_toolbench_summary() {
    local name="$1"
    local out_file="$2"

    $PYTHON - "$out_file" "$name" "$TOOLBENCH_SUMMARY" << 'PYEOF'
import sys, json
out_file, name, summary_path = sys.argv[1:]

m = json.load(open(out_file)).get("metrics", {})
em     = m.get("act_em",             float("nan")) * 100
f1     = m.get("f1",                 float("nan")) * 100
hallur = m.get("hallucination_rate", float("nan")) * 100
rouge  = m.get("rouge_l",            float("nan")) * 100

def fmt(v):
    import math; return f"{v:.2f}" if not math.isnan(v) else "N/A"

with open(summary_path, "a") as f:
    f.write(f"{name}\t{fmt(em)}\t{fmt(f1)}\t{fmt(hallur)}\t{fmt(rouge)}\n")
print(f"[RESULT] {name}: act_em={fmt(em)}  f1={fmt(f1)}  hallur={fmt(hallur)}  rouge_l={fmt(rouge)}")
PYEOF
}

# ============================================================
# Main loop
# ============================================================
for NAME in base dpo spin swift craft; do
    MODEL="${MODELS[$NAME]}"
    run_reasoning_eval "$NAME" "$MODEL"
    run_toolbench_eval "$NAME" "$MODEL"
done

# ============================================================
# Final summary tables
# ============================================================
echo ""
echo "============================================================"
echo " REASONING SUMMARY (BBH / DROP — higher is better)"
echo "============================================================"
column -t -s $'\t' "$REASONING_SUMMARY"

echo ""
echo "============================================================"
echo " TOOLBENCH SUMMARY (Act.EM / F1 / Rouge-L ↑,  HalluRate ↓)"
echo "============================================================"
column -t -s $'\t' "$TOOLBENCH_SUMMARY"

echo ""
echo " Completed : $(date)"
echo " Logs      : $LOG_FILE"
echo " Results   : $RESULTS_DIR"
echo "============================================================"
