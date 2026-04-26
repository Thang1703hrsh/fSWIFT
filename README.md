<h1 align="center">SWIFT</h1>
<p align="center"><strong>Token-Level Self-Play with Importance-Aware Guidance for Large Language Models</strong></p>

<p align="center">
  <a href="https://openreview.net/pdf?id=3VvdoCcVPU"><img src="https://img.shields.io/badge/Paper-OpenReview-b31b1b?style=flat-square" alt="Paper"></a>
  <img src="https://img.shields.io/badge/NeurIPS-2025%20Accepted-1f7a1f?style=flat-square" alt="NeurIPS 2025 Accepted">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-2f80ed?style=flat-square" alt="License"></a>
</p>

<p align="center">
SWIFT (Self-Play Weighted Fine-Tuning) extends self-play alignment with teacher-guided token-level importance weighting.
</p>

## Overview

SWIFT builds on SPIN and improves token-level learning signals during self-play fine-tuning.  
Instead of treating every token equally, SWIFT uses token importance estimated from a stronger teacher model, enabling better alignment and stronger distillation behavior.

## Core Idea

- Token-level weighting: focus optimization on more informative tokens.
- Teacher-guided guidance: use a stronger model for token importance instead of direct logits matching.
- Practical tokenizer mapping: transfer token weights across teacher/student tokenizers.

## Installation

```bash
conda env create -f environment.yml
conda activate SWIFT
pip install -r requirements.txt
```

## Data Preparation

Download:
- SFT data: [Link SFT data](https://huggingface.co/datasets/UCLA-AGI/SPIN_iter1)
- Preference data: [Link DPO data](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized)

Organize under:

```text
data/
└── Ultrachat200k/
    ├── DPO/
    ├── SFT/
    ├── SPIN/
    └── SWIFT/
```

## Training Pipeline

### 1) Download Base Checkpoint

```bash
python scripts/download.py
```

### 2) Run Self-Play Training

```bash
bash scripts/_SPIN_full.sh
bash scripts/_SWIFT_full.sh
```

(`_SWIFT_full.sh` is the Self-Play Weighted Fine-Tuning pipeline used for SWIFT in this repo.)

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

