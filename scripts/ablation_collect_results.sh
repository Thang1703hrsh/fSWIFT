#!/bin/bash
set -euo pipefail

# ============================================================
# Ablation Results Collector
#
# Reads all per-scenario eval_results/ sub-directories and
# merges everything into one master TSV.
#
# Output:
#   eval_results/ablation_master.tsv
#
# Columns (self-play benchmarks):
#   scenario  variant  iter  arc  truthful  wino  gsm8k  mmlu  hellaswag  avg
#
# Columns (KD benchmarks):
#   scenario  variant  dolly  alpaca  sni  dialoguesum  avg
#
# Usage:
#   bash scripts/ablation_collect_results.sh
# ============================================================

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON=/media/volume/tuc_data/self_play_LLMs/miniconda3/envs/WSPIN/bin/python
MASTER="eval_results/ablation_master.tsv"
MASTER_KD="eval_results/ablation_master_kd.tsv"

echo "============================================================"
echo " Ablation Results Collection"
echo " Started: $(date)"
echo "============================================================"

mkdir -p eval_results

$PYTHON << 'PYEOF'
import json, os, glob, sys

def parse_lm_eval_dir(result_dir):
    # Each task is in a sub-dir: result_dir/<task>/**/results*.json
    results = {}
    for task in ["arc_challenge","truthfulqa_mc2","winogrande","gsm8k","mmlu","hellaswag"]:
        hits = glob.glob(f"{result_dir}/{task}/**/results*.json", recursive=True)
        if hits:
            results[task] = json.load(open(hits[0])).get("results", {}).get(task, {})
    return results if results else None

def parse_lm_eval(result_dir):
    d = parse_lm_eval_dir(result_dir)
    if d is None:
        return None
    return {
        "arc":   d.get("arc_challenge",  {}).get("acc_norm,none", 0) * 100,
        "truth": d.get("truthfulqa_mc2", {}).get("acc,none",      0) * 100,
        "wino":  d.get("winogrande",     {}).get("acc,none",      0) * 100,
        "gsm":   d.get("gsm8k",          {}).get("exact_match,strict-match", 0) * 100,
        "mmlu":  d.get("mmlu",           {}).get("acc,none",      0) * 100,
        "hella": d.get("hellaswag",      {}).get("acc_norm,none", 0) * 100,
    }

def avg6(r):
    return sum(r[k] for k in ("arc","truth","wino","gsm","mmlu","hella")) / 6

def parse_rouge(path):
    r = json.load(open(path))
    return {d: r.get(d, {}).get("rougeL", 0) for d in ["dolly","alpaca","sni","dialoguesum"]}

# ── Self-play benchmarks (B1, B2, B3, B6) ─────────────────────────────────
rows_sp = []
header_sp = "scenario\tvariant\titer\tarc\ttruthful\twino\tgsm8k\tmmlu\thellaswag\tavg"

# B1: divergence selection
for div in ["identity","kl","js","chi2","hellinger","wasserstein"]:
    for ite in range(4):
        label = f"{div}_ite{ite}"
        r = parse_lm_eval(f"eval_results/divergence_comparison/{label}")
        if r:
            rows_sp.append(f"B1_fdiv\t{div}\tite{ite}\t"
                           f"{r['arc']:.2f}\t{r['truth']:.2f}\t{r['wino']:.2f}\t"
                           f"{r['gsm']:.2f}\t{r['mmlu']:.2f}\t{r['hella']:.2f}\t{avg6(r):.2f}")

# B2: adaptive scheduling
for name in ["js_to_kl","js_to_chi2","fixed_identity","fixed_js","fixed_kl"]:
    label = f"{name}_ite3"
    r = parse_lm_eval(f"eval_results/ablation_b2_adaptive/{label}")
    if r:
        rows_sp.append(f"B2_adaptive\t{name}\tite3\t"
                       f"{r['arc']:.2f}\t{r['truth']:.2f}\t{r['wino']:.2f}\t"
                       f"{r['gsm']:.2f}\t{r['mmlu']:.2f}\t{r['hella']:.2f}\t{avg6(r):.2f}")

# B3: beta scaling
for beta in ["0.01","0.05","0.1","0.2","0.5"]:
    label = f"beta_{beta}"
    r = parse_lm_eval(f"eval_results/ablation_b3_beta/{label}")
    if r:
        rows_sp.append(f"B3_beta\tbeta={beta}\tite0\t"
                       f"{r['arc']:.2f}\t{r['truth']:.2f}\t{r['wino']:.2f}\t"
                       f"{r['gsm']:.2f}\t{r['mmlu']:.2f}\t{r['hella']:.2f}\t{avg6(r):.2f}")

# B6: weight transform
for t in ["binary","threshold","threshold_and_scale","rank_based","random"]:
    label = f"transform_{t}"
    r = parse_lm_eval(f"eval_results/ablation_b6_transform/{label}")
    if r:
        rows_sp.append(f"B6_transform\t{t}\tite0\t"
                       f"{r['arc']:.2f}\t{r['truth']:.2f}\t{r['wino']:.2f}\t"
                       f"{r['gsm']:.2f}\t{r['mmlu']:.2f}\t{r['hella']:.2f}\t{avg6(r):.2f}")

with open("eval_results/ablation_master.tsv", "w") as f:
    f.write(header_sp + "\n")
    for row in rows_sp:
        f.write(row + "\n")

print(f"Self-play table: {len(rows_sp)} rows → eval_results/ablation_master.tsv")

# ── KD benchmarks (B4, B5) ────────────────────────────────────────────────
rows_kd = []
header_kd = "scenario\tvariant\tdolly\talpaca\tsni\tdialoguesum\tavg"

# B4: divergence in KD
for div in ["identity","kl","js","chi2","hellinger","wasserstein"]:
    path = f"eval_results/ablation_b4_kd_fdiv/results_{div}.json"
    if os.path.exists(path):
        r = parse_rouge(path)
        vals = [r["dolly"],r["alpaca"],r["sni"],r["dialoguesum"]]
        avg = sum(vals)/len(vals)
        rows_kd.append(f"B4_kd\tf*={div}\t" + "\t".join(f"{v:.4f}" for v in vals) + f"\t{avg:.4f}")

# B5: SFT init
for name, fname in [
    ("SFT_baseline",      "results_sft.json"),
    ("base_to_fswift_js", "results_base_fswift.json"),
    ("sft_to_fswift_js",  "results_sft_then_fswift.json"),
]:
    path = f"eval_results/ablation_b5_sft_init/{fname}"
    if os.path.exists(path):
        r = parse_rouge(path)
        vals = [r["dolly"],r["alpaca"],r["sni"],r["dialoguesum"]]
        avg = sum(vals)/len(vals)
        rows_kd.append(f"B5_sft_init\t{name}\t" + "\t".join(f"{v:.4f}" for v in vals) + f"\t{avg:.4f}")

with open("eval_results/ablation_master_kd.tsv", "w") as f:
    f.write(header_kd + "\n")
    for row in rows_kd:
        f.write(row + "\n")

print(f"KD table       : {len(rows_kd)} rows → eval_results/ablation_master_kd.tsv")
PYEOF

echo ""
echo "============================================================"
echo " Self-Play Results (ablation_master.tsv)"
echo "============================================================"
if [ -f "$MASTER" ]; then
    column -t -s $'\t' "$MASTER"
else
    echo "  (no data yet)"
fi

echo ""
echo "============================================================"
echo " KD Results (ablation_master_kd.tsv)"
echo "============================================================"
if [ -f "$MASTER_KD" ]; then
    column -t -s $'\t' "$MASTER_KD"
else
    echo "  (no data yet)"
fi

echo ""
echo "============================================================"
echo " Collection complete: $(date)"
echo " Self-play: $MASTER"
echo " KD       : $MASTER_KD"
echo "============================================================"
