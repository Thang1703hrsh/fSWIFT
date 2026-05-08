#!/bin/bash
set -euo pipefail

# ============================================================
# CO-CRAFT Span-Level: Self-Play with f'-critic span weights
# f-divergence: JS  (outer f* and inner f'-critic)
# No teacher model — span weights computed on-the-fly from
#   Δ_t = log π_θ − log π_ref  via  f'_JS(exp(Δ_t)) − f'_JS(1)
#
# Server  : 4 × L40S 48GB (le-lab cluster)
# Trainer : FSDPTrainer (4-GPU — required to shard 7B optimizer states)
# Model   : Mistral-7B-v0.1
# ============================================================

# ── Environment (must come before set -euo pipefail catches conda errors) ──
eval "$(conda shell.bash hook)"
conda activate /data/project/le-lab/conda_env/WSPIN_v2
export LD_PRELOAD=/data/project/le-lab/conda_env/WSPIN/lib/libstdc++.so.6

REPO_ROOT="/data/project/le-lab/fSWIFT"
cd "$REPO_ROOT"

# ── Paths ────────────────────────────────────────────────────
SFT_MODEL="${REPO_ROOT}/model_hub/Mistral-7B-v0.1/base"
SFT_DATA="data/Ultrachat200k/SFT/trainSFT.jsonl"

CKPT_BASE="model_hub/Mistral-7B/cocraft_span_js"
DATA_BASE="data/Ultrachat200k/cocraft_span_js"
DSET_BASE="Ultrachat200k/cocraft_span_js"

# ── Hardware ─────────────────────────────────────────────────
GPU_IDS_TRAIN="0,1,2,3"   # all 4 L40S for FSDPTrainer
GPU_IDS_GEN="0"            # single GPU for vLLM (7B fits in 48GB)
NUM_GPUS=4

# ── Training hyper-parameters ────────────────────────────────
# BATCH=64, GRAD_ACCUM=4, NUM_GPUS=4:
#   per-GPU microbatch = 64 / (4 accum × 4 GPUs) = 4 samples
#   concat forward per GPU: 2B = 8  →  logits bf16 = 4.7 GiB  (fits in 48 GB)
#   gradient buffer fp32   = 9.4 GiB per GPU (fits in 48 GB)
BATCH=64
GRAD_ACCUM=4
N_EPOCHS=2
MAX_NEW=512
FRAC_LEN=1000000
FRAC=0
MAX_LENGTH=2048
MAX_PROMPT_LENGTH=1024

# ── CO-CRAFT span hyper-parameters ───────────────────────────
F_DIV="js"
SPAN_STRATEGY="fixed"
SPAN_SIZE=8
SPAN_MU=1.0
SPAN_L=-2.0
SPAN_U=2.0
SPAN_G_MIN=0.25
SPAN_G_MAX=4.0

SKIP_EXISTING="${SKIP_EXISTING:-0}"

# ── Logging ──────────────────────────────────────────────────
LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/cocraft_span_js_mistral7b.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"; }

# ── Helper: check if a checkpoint directory is complete ──────
# FSDPTrainer for 7B may save as model-00001-of-XXXXX.safetensors (sharded).
ckpt_exists() {
    local DST="$1"
    [ -f "${DST}/model.safetensors" ] || \
    ls "${DST}/model-"*".safetensors" 2>/dev/null | grep -q .
}

# ── Helper: resolve FSDP checkpoint ──────────────────────────
# FSDPTrainer saves to output/<exp_name>/iteN_<timestamp>/ rather than
# directly into ckpt_dir. This function copies it to the expected path.
resolve_ckpt() {
    local ITE="$1"
    local DST="${CKPT_BASE}/ite${ITE}"
    if ckpt_exists "$DST"; then
        return 0
    fi
    local LATEST_OUT
    LATEST_OUT=$(find "${REPO_ROOT}/output" -maxdepth 6 -type d -name "ite${ITE}_*" \
                 2>/dev/null | xargs ls -dt 2>/dev/null | head -1)
    if [ -n "$LATEST_OUT" ] && [ -d "$LATEST_OUT" ]; then
        log "Copying checkpoint: ${LATEST_OUT} → ${DST}/"
        mkdir -p "${DST}"
        cp -r "${LATEST_OUT%/}"/. "${DST}/"
    else
        echo "ERROR: checkpoint not found in output/ after training ite${ITE}" >&2
        exit 1
    fi
}

log "=================================================="
log " CO-CRAFT Span JS — Mistral-7B — 4×L40S"
log " SFT_MODEL : $SFT_MODEL"
log " CKPT_BASE : $REPO_ROOT/$CKPT_BASE"
log " DATA_BASE : $REPO_ROOT/$DATA_BASE"
log " GPU train : $GPU_IDS_TRAIN  (FSDPTrainer, 4×L40S 48GB)"
log " GPU gen   : $GPU_IDS_GEN   (vLLM single GPU)"
log " BATCH     : ${BATCH} global / GRAD_ACCUM ${GRAD_ACCUM} × ${NUM_GPUS} GPUs = $(( BATCH / GRAD_ACCUM / NUM_GPUS )) per GPU  (2B=$(( 2 * BATCH / GRAD_ACCUM / NUM_GPUS )) concat)"
log " LOG       : $LOG_FILE"
log "=================================================="

# ============================================================
# Iteration 0  —  SFT model -> cocraft_span_js/ite0
# ============================================================
log "===== Iteration 0 — generate ====="
if [ "$SKIP_EXISTING" = "1" ] && [ -f "${DATA_BASE}/ite0/train.jsonl" ]; then
  log "[SKIP] ite0 generate already exists."
else
  rm -f "${DATA_BASE}/ite0/train.jsonl"
  CUDA_VISIBLE_DEVICES=$GPU_IDS_GEN python generate_vllm.py \
    --model          "$SFT_MODEL" \
    --input_dir      "$SFT_DATA" \
    --output_dir     "${DATA_BASE}/ite0/train" \
    --max_new_tokens $MAX_NEW \
    --data_frac      $FRAC \
    --frac_len       $FRAC_LEN \
    --split          train \
    2>&1 | tee -a "$LOG_FILE"
fi

log "===== Iteration 0 — train ====="
if [ "$SKIP_EXISTING" = "1" ] && ckpt_exists "${CKPT_BASE}/ite0"; then
  log "[SKIP] ite0 checkpoint already exists."
else
  CUDA_VISIBLE_DEVICES=$GPU_IDS_TRAIN python -u train.py \
    model=mistra \
    model.name_or_path="$SFT_MODEL" \
    loss=cocraft_span \
    loss.f_divergence="${F_DIV}" \
    loss.span_strategy="${SPAN_STRATEGY}" \
    loss.span_size="${SPAN_SIZE}" \
    loss.mu="${SPAN_MU}" \
    loss.span_L="${SPAN_L}" \
    loss.span_U="${SPAN_U}" \
    loss.span_g_min="${SPAN_G_MIN}" \
    loss.span_g_max="${SPAN_G_MAX}" \
    trainer=FSDPTrainer \
    base_data_dir=data \
    ckpt_dir="${CKPT_BASE}/ite0/" \
    datasets="[\"${DSET_BASE}/ite0\"]" \
    batch_size=$BATCH \
    gradient_accumulation_steps=$GRAD_ACCUM \
    activation_checkpointing=true \
    max_length=$MAX_LENGTH \
    max_prompt_length=$MAX_PROMPT_LENGTH \
    n_epochs=$N_EPOCHS \
    lr=5e-7 \
    iteration=0 \
    2>&1 | tee -a "$LOG_FILE"
  resolve_ckpt 0
fi
log "===== Iteration 0 — done ====="

# ============================================================
# Iteration 1  —  ite0 -> cocraft_span_js/ite1
# ============================================================
log "===== Iteration 1 — generate ====="
if [ "$SKIP_EXISTING" = "1" ] && [ -f "${DATA_BASE}/ite1/train.jsonl" ]; then
  log "[SKIP] ite1 generate already exists."
else
  rm -f "${DATA_BASE}/ite1/train.jsonl"
  CUDA_VISIBLE_DEVICES=$GPU_IDS_GEN python generate_vllm.py \
    --model          "${CKPT_BASE}/ite0" \
    --input_dir      "$SFT_DATA" \
    --output_dir     "${DATA_BASE}/ite1/train" \
    --max_new_tokens $MAX_NEW \
    --data_frac      $FRAC \
    --frac_len       $FRAC_LEN \
    --split          train \
    2>&1 | tee -a "$LOG_FILE"
fi

log "===== Iteration 1 — train ====="
if [ "$SKIP_EXISTING" = "1" ] && ckpt_exists "${CKPT_BASE}/ite1"; then
  log "[SKIP] ite1 checkpoint already exists."
else
  CUDA_VISIBLE_DEVICES=$GPU_IDS_TRAIN python -u train.py \
    model=mistra \
    model.name_or_path="${CKPT_BASE}/ite0" \
    loss=cocraft_span \
    loss.f_divergence="${F_DIV}" \
    loss.span_strategy="${SPAN_STRATEGY}" \
    loss.span_size="${SPAN_SIZE}" \
    loss.mu="${SPAN_MU}" \
    loss.span_L="${SPAN_L}" \
    loss.span_U="${SPAN_U}" \
    loss.span_g_min="${SPAN_G_MIN}" \
    loss.span_g_max="${SPAN_G_MAX}" \
    trainer=FSDPTrainer \
    base_data_dir=data \
    ckpt_dir="${CKPT_BASE}/ite1/" \
    datasets="[\"${DSET_BASE}/ite0\",\"${DSET_BASE}/ite1\"]" \
    batch_size=$BATCH \
    gradient_accumulation_steps=$GRAD_ACCUM \
    activation_checkpointing=true \
    max_length=$MAX_LENGTH \
    max_prompt_length=$MAX_PROMPT_LENGTH \
    n_epochs=$N_EPOCHS \
    lr=5e-7 \
    iteration=1 \
    2>&1 | tee -a "$LOG_FILE"
  resolve_ckpt 1
fi
log "===== Iteration 1 — done ====="

# ============================================================
# Iteration 2  —  ite1 -> cocraft_span_js/ite2
# ============================================================
log "===== Iteration 2 — generate ====="
if [ "$SKIP_EXISTING" = "1" ] && [ -f "${DATA_BASE}/ite2/train.jsonl" ]; then
  log "[SKIP] ite2 generate already exists."
else
  rm -f "${DATA_BASE}/ite2/train.jsonl"
  CUDA_VISIBLE_DEVICES=$GPU_IDS_GEN python generate_vllm.py \
    --model          "${CKPT_BASE}/ite1" \
    --input_dir      "$SFT_DATA" \
    --output_dir     "${DATA_BASE}/ite2/train" \
    --max_new_tokens $MAX_NEW \
    --data_frac      $FRAC \
    --frac_len       $FRAC_LEN \
    --split          train \
    2>&1 | tee -a "$LOG_FILE"
fi

log "===== Iteration 2 — train ====="
if [ "$SKIP_EXISTING" = "1" ] && ckpt_exists "${CKPT_BASE}/ite2"; then
  log "[SKIP] ite2 checkpoint already exists."
else
  CUDA_VISIBLE_DEVICES=$GPU_IDS_TRAIN python -u train.py \
    model=mistra \
    model.name_or_path="${CKPT_BASE}/ite1" \
    loss=cocraft_span \
    loss.f_divergence="${F_DIV}" \
    loss.span_strategy="${SPAN_STRATEGY}" \
    loss.span_size="${SPAN_SIZE}" \
    loss.mu="${SPAN_MU}" \
    loss.span_L="${SPAN_L}" \
    loss.span_U="${SPAN_U}" \
    loss.span_g_min="${SPAN_G_MIN}" \
    loss.span_g_max="${SPAN_G_MAX}" \
    trainer=FSDPTrainer \
    base_data_dir=data \
    ckpt_dir="${CKPT_BASE}/ite2/" \
    datasets="[\"${DSET_BASE}/ite1\",\"${DSET_BASE}/ite2\"]" \
    batch_size=$BATCH \
    gradient_accumulation_steps=$GRAD_ACCUM \
    activation_checkpointing=true \
    max_length=$MAX_LENGTH \
    max_prompt_length=$MAX_PROMPT_LENGTH \
    n_epochs=$N_EPOCHS \
    lr=1e-7 \
    iteration=2 \
    2>&1 | tee -a "$LOG_FILE"
  resolve_ckpt 2
fi
log "===== Iteration 2 — done ====="

# ============================================================
# Iteration 3  —  ite2 -> cocraft_span_js/ite3
# ============================================================
log "===== Iteration 3 — generate ====="
if [ "$SKIP_EXISTING" = "1" ] && [ -f "${DATA_BASE}/ite3/train.jsonl" ]; then
  log "[SKIP] ite3 generate already exists."
else
  rm -f "${DATA_BASE}/ite3/train.jsonl"
  CUDA_VISIBLE_DEVICES=$GPU_IDS_GEN python generate_vllm.py \
    --model          "${CKPT_BASE}/ite2" \
    --input_dir      "$SFT_DATA" \
    --output_dir     "${DATA_BASE}/ite3/train" \
    --max_new_tokens $MAX_NEW \
    --data_frac      $FRAC \
    --frac_len       $FRAC_LEN \
    --split          train \
    2>&1 | tee -a "$LOG_FILE"
fi

log "===== Iteration 3 — train ====="
if [ "$SKIP_EXISTING" = "1" ] && ckpt_exists "${CKPT_BASE}/ite3"; then
  log "[SKIP] ite3 checkpoint already exists."
else
  CUDA_VISIBLE_DEVICES=$GPU_IDS_TRAIN python -u train.py \
    model=mistra \
    model.name_or_path="${CKPT_BASE}/ite2" \
    loss=cocraft_span \
    loss.f_divergence="${F_DIV}" \
    loss.span_strategy="${SPAN_STRATEGY}" \
    loss.span_size="${SPAN_SIZE}" \
    loss.mu="${SPAN_MU}" \
    loss.span_L="${SPAN_L}" \
    loss.span_U="${SPAN_U}" \
    loss.span_g_min="${SPAN_G_MIN}" \
    loss.span_g_max="${SPAN_G_MAX}" \
    trainer=FSDPTrainer \
    base_data_dir=data \
    ckpt_dir="${CKPT_BASE}/ite3/" \
    datasets="[\"${DSET_BASE}/ite2\",\"${DSET_BASE}/ite3\"]" \
    batch_size=$BATCH \
    gradient_accumulation_steps=$GRAD_ACCUM \
    activation_checkpointing=true \
    max_length=$MAX_LENGTH \
    max_prompt_length=$MAX_PROMPT_LENGTH \
    n_epochs=$N_EPOCHS \
    lr=1e-7 \
    iteration=3 \
    2>&1 | tee -a "$LOG_FILE"
  resolve_ckpt 3
fi
log "===== Iteration 3 — done ====="

log "=================================================="
log " CO-CRAFT Span JS — Mistral-7B — training complete"
log " Checkpoints : ${REPO_ROOT}/${CKPT_BASE}/ite{0..3}/"
log " Log file    : ${LOG_FILE}"
log "=================================================="
