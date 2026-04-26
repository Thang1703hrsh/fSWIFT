<h1 align="center">f-SWIFT</h1>
<p align="center"><strong>f-Divergence Self-Play Weighted Fine-Tuning for Large Language Models</strong></p>

<p align="center">
  <a href="https://openreview.net/pdf?id=3VvdoCcVPU"><img src="https://img.shields.io/badge/Paper-OpenReview-b31b1b?style=flat-square" alt="Paper"></a>
  <img src="https://img.shields.io/badge/NeurIPS-2025%20Accepted-1f7a1f?style=flat-square" alt="NeurIPS 2025 Accepted">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-2f80ed?style=flat-square" alt="License"></a>
</p>

<p align="center">
f-SWIFT generalizes SWIFT by replacing the fixed KL-divergence objective with any f-divergence via the Fenchel conjugate f*(t), enabling richer optimization landscapes for self-play alignment.
</p>

## Overview

f-SWIFT extends [SWIFT (NeurIPS 2025)](https://openreview.net/pdf?id=3VvdoCcVPU) along two axes:

1. **f-Divergence generalization** — the self-play loss can use any f-divergence (JS, KL, χ², Hellinger, Wasserstein) by plugging in the corresponding conjugate function f*(t).
2. **Adaptive f-scheduling** — different divergences can be applied at different self-play iterations (e.g. JS for stable warm-up → KL for aggressive refinement).

Token-level importance weighting from a teacher model is preserved from SWIFT, providing better learning signals than uniform token treatment.

## Supported Divergences

| `f_divergence` | Description |
|---|---|
| `identity` | Recovers original SWIFT (linear f*) |
| `js` | Jensen-Shannon — bounded, stable gradients |
| `kl` | KL divergence — strongest gradient amplification |
| `chi2` | Chi-squared — heavy penalty on large ratios |
| `hellinger` | Hellinger distance — moderate, smooth |
| `wasserstein` | Wasserstein / WGAN-style clipped linear |

## Installation

```bash
# 1. Create conda environment
conda env create -f environment.yml
conda activate WSPIN

# 2. Install lm-evaluation-harness (for evaluation)
pip install -e lm-evaluation-harness/
```

> **Requirements:** CUDA 12.x, 4× A100/H100 recommended for full pipeline.

## Setup

### 1. Download models and data

```bash
export HF_TOKEN=hf_...   # your HuggingFace token

python scripts/download.py
```

This downloads:
- `Qwen/Qwen1.5-1.8B` → `model_hub/Qwen1.5-1.8B/base`
- `alignment-handbook/zephyr-7b-sft-full` → `model_hub/zephyr-7b-sft-full` (teacher)
- `ducthang1703/Qwen1.5-1.8B-sft-v2` → `model_hub/Qwen1.5-1.8B/sft_v2` (SFT init)
- `UCLA-AGI/SPIN_iter1` → `data_hub/SPIN_iter1`

### 2. Prepare self-play training data

```bash
python prepare_data.py
```

Samples 50k examples from `HuggingFaceH4/ultrachat_200k` → `data/Ultrachat200k/SFT/trainSFT.jsonl`.

## Training

### Self-Play (f-SWIFT)

Run all 3 divergences (JS, KL, Wasserstein) × 4 iterations on 4× H100:

```bash
bash scripts/run_all_divergences.sh
```

Checkpoints saved to `model_hub/Qwen1.5-1.8B/fSWIFT_{js,kl,wasserstein}/ite{0..3}/`.

To run a single divergence:

```bash
bash scripts/run_all_divergences.sh js
```

### Adaptive f-scheduling

```bash
bash scripts/_fSWIFT_adaptive_full.sh
```

Switches divergence mid-training (e.g. JS → KL at iteration 2).

### Distillation (f-SWIFT as KD)

Run knowledge distillation from `Qwen2.5-7B-Instruct` to `gpt2-xl` with all 3 divergences:

```bash
bash scripts/distillation_fswift_multi_div.sh
```

## Evaluation

After training, evaluate all checkpoints on 6 benchmarks (ARC, TruthfulQA, WinoGrande, GSM8K, MMLU, HellaSwag):

```bash
bash scripts/eval_all_divergences.sh
```

Results saved to `eval_results/divergence_comparison/summary.tsv`.

### Quick single-model eval

```bash
# ARC-Challenge (25-shot)
bash lm-evaluation-harness/eval_arc.sh model_hub/Qwen1.5-1.8B/fSWIFT_js/ite3

# HellaSwag (10-shot)
bash lm-evaluation-harness/eval_hella.sh model_hub/Qwen1.5-1.8B/fSWIFT_js/ite3

# Full 6-benchmark suite
bash lm-evaluation-harness/eval_llm.sh model_hub/Qwen1.5-1.8B/fSWIFT_js/ite3
```

## Ablation Studies

| Script | Description |
|---|---|
| `scripts/ablation_b2_adaptive.sh` | Adaptive f-scheduling strategies |
| `scripts/ablation_b3_beta.sh` | β scaling sweep (0.01 → 0.5) |
| `scripts/ablation_b5_sft_init.sh` | Effect of SFT initialization |
| `scripts/ablation_b6_transform.sh` | Token weight transform variants |
| `scripts/ablation_collect_results.sh` | Aggregate all ablation results |

## Project Structure

```text
f-SWIFT/
├── train.py                        # Main training entry point
├── trainers.py                     # FSDPTrainer + loss implementations
├── token_weight_estimation.py      # Teacher-guided token importance
├── generate_vllm.py                # vLLM-based response generation
├── prepare_data.py                 # UltraChat data preparation
├── eval_rouge.py                   # ROUGE-L evaluation (distillation)
├── config/
│   ├── config.yaml                 # Hydra base config
│   ├── loss/fswift.yaml            # f-SWIFT loss config
│   └── model/                      # Per-model configs (qwen, gpt2-xl, ...)
├── scripts/
│   ├── run_all_divergences.sh      # Main self-play pipeline
│   ├── eval_all_divergences.sh     # Full benchmark evaluation
│   ├── distillation_fswift_multi_div.sh  # KD with f-SWIFT
│   └── ablation_*.sh               # Ablation study scripts
└── lm-evaluation-harness/          # Evaluation framework
```

## Citation

```bibtex
@inproceedings{letoken,
  title={Token-Level Self-Play with Importance-Aware Guidance for Large Language Models},
  author={Le, Tue and Vuong, Hoang Tran and Tran, Quyen and Van, Linh Ngo and Harandi, Mehrtash and Le, Trung},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025},
}
```

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE).
