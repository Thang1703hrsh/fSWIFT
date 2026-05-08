import math
import torch
torch.backends.cuda.matmul.allow_tf32 = True
import torch.nn.functional as F
import torch.nn as nn
import transformers
from omegaconf import DictConfig

import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    StateDictType,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload,
)
from torch.distributed.fsdp.api import FullStateDictConfig, FullOptimStateDictConfig
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import tensor_parallel as tp
import contextlib

from preference_datasets import get_batch_iterator
from utils import (
    slice_and_move_batch_for_device,
    formatted_dict,
    all_gather_if_needed,
    pad_to_length,
    get_block_class_from_model,
    rank0_print,
    get_local_dir,
)
import numpy as np
import wandb
import tqdm

import random
import os
from collections import defaultdict
import time
import json
import functools
from typing import Optional, Dict, List, Union, Tuple

import subprocess

def _tdpo_get_batch_logps(logits: torch.FloatTensor, reference_logits: torch.FloatTensor, labels: torch.LongTensor,
                          average_log_prob: bool = False):
    """Compute the kl divergence/log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        reference_logits: Logits of the reference model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

    Returns:
        Several tensors of shape (batch_size,) containing the average/sum kl divergence/log probabilities of the given labels under the given logits.
    """
    assert logits.shape[:-1] == labels.shape
    assert reference_logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    reference_logits = reference_logits[:, :-1, :]

    loss_mask = (labels != -100)

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == -100] = 0

    vocab_logps = logits.log_softmax(-1)

    reference_vocab_ps = reference_logits.softmax(-1)
    reference_vocab_logps = reference_vocab_ps.log()

    per_position_kl = (reference_vocab_ps * (reference_vocab_logps - vocab_logps)).sum(-1)
    per_token_logps = torch.gather(vocab_logps, dim=2, index=labels.unsqueeze(2)).squeeze(2)
    per_reference_token_logps = torch.gather(reference_vocab_logps, dim=2, index=labels.unsqueeze(2)).squeeze(2)

    logps_margin = per_token_logps - per_reference_token_logps

    if average_log_prob:
        return (logps_margin * loss_mask).sum(-1) / loss_mask.sum(-1), \
               (per_position_kl * loss_mask).sum(-1) / loss_mask.sum(-1), \
               (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (logps_margin * loss_mask).sum(-1), \
            (per_position_kl * loss_mask).sum(-1), \
            (per_token_logps * loss_mask).sum(-1)


def tdpo_loss(chosen_logps_margin: torch.FloatTensor,
              rejected_logps_margin: torch.FloatTensor,
              chosen_position_kl: torch.FloatTensor,
              rejected_position_kl: torch.FloatTensor,
              beta: float, alpha: float = 0.5, if_tdpo2: bool = True) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute the TDPO loss for a batch of policy and reference model log probabilities.

    Args:
        chosen_logps_margin: The difference of log probabilities between the policy model and the reference model for the chosen responses. Shape: (batch_size,)
        rejected_logps_margin: The difference of log probabilities between the policy model and the reference model for the rejected responses. Shape: (batch_size,)
        chosen_position_kl: The difference of sequential kl divergence between the policy model and the reference model for the chosen responses. Shape: (batch_size,)
        rejected_position_kl: The difference of sequential kl divergence between the policy model and the reference model for the rejected responses. Shape: (batch_size,)
        beta: Temperature parameter for the TDPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
        alpha: Temperature parameter for the TDPO loss, used to adjust the impact of sequential kl divergence.
        if_tdpo2: Determine whether to use method TDPO2, default is True; if False, then use method TDPO1.

    Returns:
        A tuple of two tensors: (losses, rewards).
        The losses tensor contains the TDPO loss for each example in the batch.
        The rewards tensors contain the rewards for response pair.
    """

    chosen_values = chosen_logps_margin + chosen_position_kl
    rejected_values = rejected_logps_margin + rejected_position_kl

    chosen_rejected_logps_margin = chosen_logps_margin - rejected_logps_margin

    if not if_tdpo2:
        logits = chosen_rejected_logps_margin - (rejected_position_kl - chosen_position_kl)    # tdpo1
    else:
        logits = chosen_rejected_logps_margin - alpha * (rejected_position_kl - chosen_position_kl.detach())  # tdpo2
    losses = -F.logsigmoid(beta * logits)

    chosen_rewards = beta * chosen_values.detach()
    rejected_rewards = beta * rejected_values.detach()

    return losses, chosen_rewards, rejected_rewards

def tisdpo_loss(chosen_logps_margin: torch.FloatTensor,
                rejected_logps_margin: torch.FloatTensor,
                chosen_position_kl: torch.FloatTensor,
                rejected_position_kl: torch.FloatTensor,
                beta: float, alpha: float = 0.5, token_level: bool = False) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    if token_level:
        chosen_values = chosen_logps_margin - chosen_position_kl
        rejected_values = rejected_logps_margin - rejected_position_kl
    else:
        chosen_values = chosen_logps_margin
        rejected_values = rejected_logps_margin

    chosen_rejected_logps_margin = chosen_logps_margin - rejected_logps_margin

    if token_level:
        logits = chosen_rejected_logps_margin - alpha * (chosen_position_kl - rejected_position_kl)  
    else:
        logits = chosen_rejected_logps_margin

    losses = -F.logsigmoid(beta * logits)

    chosen_rewards = beta * chosen_values.detach()
    rejected_rewards = beta * rejected_values.detach()

    return losses, chosen_rewards, rejected_rewards



def f_star_identity(t: torch.FloatTensor) -> torch.FloatTensor:
    """f*(t) = t. Recovers original SWIFT/IPM."""
    return t

def f_star_kl(t: torch.FloatTensor, clamp_max: float = 10.0) -> torch.FloatTensor:
    """Unshifted KL conjugate: f*(t) = e^(t-1). Satisfies f*(t) >= t (required by Prop. 2).
    NOTE: f*(0) = e^(-1) != 0, but f*(t) >= t holds everywhere — convergence guaranteed.
    The shifted variant f*(t) = e^(t-1) - e^(-1) satisfies f*(0)=0 but violates f*(t)>=t
    near t=0, breaking the convergence proof (paper Table 1, Remark 3)."""
    t_clamped = torch.clamp(t, -clamp_max, clamp_max)
    return torch.exp(t_clamped - 1.0)

def f_star_js(t: torch.FloatTensor, eps: float = 0.1) -> torch.FloatTensor:
    """Jensen-Shannon conjugate: f*(t) = -log(2 - e^t). Domain: t < log(2).
    Paper Section 3.6: recommended default, bounded and smooth."""
    t_clamped = torch.clamp(t, max=math.log(2.0) - eps)
    return -torch.log(2.0 - torch.exp(t_clamped))

def f_star_chi2(t: torch.FloatTensor, clamp_max: float = 10.0) -> torch.FloatTensor:
    """Pearson chi-squared conjugate: f*(t) = t^2/4 + t. Quadratic growth, clamp for stability."""
    t_clamped = torch.clamp(t, -clamp_max, clamp_max)
    return t_clamped * t_clamped / 4.0 + t_clamped

def f_star_hellinger(t: torch.FloatTensor, eps: float = 0.01) -> torch.FloatTensor:
    """Squared Hellinger conjugate: f*(t) = t / (1 - t). Domain: t < 1. Conservative."""
    t_clamped = torch.clamp(t, min=-10.0, max=1.0 - eps)
    return t_clamped / (1.0 - t_clamped)

def f_star_wasserstein(t: torch.FloatTensor, c: float = 1.0) -> torch.FloatTensor:
    """Wasserstein-1 (WGAN-style) conjugate: f*(t) = clamp(t, -c, c).
    Enforces c-Lipschitz constraint on the critic, matching the dual formulation
    of W1: sup_{||f||_Lip <= c} E[f(y)] - E[f(y')].
    - f*(0) = 0 satisfied (clamp(0) = 0). checkmark
    - f*(t) >= t NOT satisfied for t > c (clipped), so convergence proof
      (Prop. 2) does not apply. Gradient is 1 when |t| < c, 0 at boundary.
    - Best used in later iterations where S_r is large and JS/KL saturate.
    - Recommended c: 1.0 (standard WGAN) or tune in [0.5, 3.0]."""
    return torch.clamp(t, min=-c, max=c)

F_STAR_REGISTRY = {
    'identity': f_star_identity,
    'kl': f_star_kl,
    'js': f_star_js,
    'chi2': f_star_chi2,
    'hellinger': f_star_hellinger,
    'wasserstein': f_star_wasserstein,
}

def get_f_star(name: str):
    """Get f* function by name."""
    if name not in F_STAR_REGISTRY:
        raise ValueError(f"Unknown f-divergence type '{name}'. Available: {list(F_STAR_REGISTRY.keys())}")
    return F_STAR_REGISTRY[name]


# ── f' (derivative of f) for the CO-CRAFT centered span critic ────────────────
# The span critic is: a_f(Δ) = f'(exp(Δ)) − f'(1)
# Each f'(u) is the derivative of the generator f evaluated at u > 0.

def f_prime_identity(u: torch.FloatTensor) -> torch.FloatTensor:
    """f'(u) = 1 for f(u)=u (IPM / identity)."""
    return torch.ones_like(u)

def f_prime_kl(u: torch.FloatTensor, eps: float = 1e-8) -> torch.FloatTensor:
    """f'(u) = log(u) for KL divergence f(u) = u*log(u) - u + 1."""
    return torch.log(u.clamp(min=eps))

def f_prime_js(u: torch.FloatTensor, eps: float = 1e-8) -> torch.FloatTensor:
    """f'(u) = log(2u / (1+u)) for Jensen-Shannon f(u) = u*log(u) - (u+1)*log((u+1)/2)."""
    return torch.log((2.0 * u).clamp(min=eps) / (1.0 + u).clamp(min=eps))

def f_prime_chi2(u: torch.FloatTensor) -> torch.FloatTensor:
    """f'(u) = (u-1)/2 + 1 = (u+1)/2 for chi-squared f(u) = (u-1)^2 / 4.
    (The Pearson chi-squared generator is f(u)=(u-1)^2, so f'(u)=2(u-1), but
    we use the half-chi2 variant f(u)=(u-1)^2/4 matching f_star_chi2, giving f'(u)=(u-1)/2.)"""
    return (u - 1.0) / 2.0

def f_prime_hellinger(u: torch.FloatTensor, eps: float = 1e-8) -> torch.FloatTensor:
    """f'(u) = 1 - 1/sqrt(u) for squared Hellinger f(u) = (sqrt(u)-1)^2."""
    return 1.0 - 1.0 / (u.clamp(min=eps).sqrt())

def f_prime_wasserstein(u: torch.FloatTensor, c: float = 1.0) -> torch.FloatTensor:
    """f'(u) = clamp(u-1, -c, c) as a subgradient approximation for Wasserstein."""
    return torch.clamp(u - 1.0, min=-c, max=c)

F_PRIME_REGISTRY = {
    'identity':   f_prime_identity,
    'kl':         f_prime_kl,
    'js':         f_prime_js,
    'chi2':       f_prime_chi2,
    'hellinger':  f_prime_hellinger,
    'wasserstein': f_prime_wasserstein,
}

def get_f_prime(name: str):
    """Get the f' (f-generator derivative) function by divergence name."""
    if name not in F_PRIME_REGISTRY:
        raise ValueError(f"Unknown f-divergence type '{name}'. Available: {list(F_PRIME_REGISTRY.keys())}")
    return F_PRIME_REGISTRY[name]


# ── Span segmentation ─────────────────────────────────────────────────────────

def segment_into_spans(
    response_length: int,
    strategy: str = 'fixed',
    span_size: int = 8,
    token_ids: Optional[List[int]] = None,
    tokenizer=None,
) -> List[Tuple[int, int]]:
    """Segment response token indices [0, response_length) into non-overlapping spans.

    Args:
        response_length: number of response tokens (not including prompt).
        strategy: one of 'fixed', 'sentence', 'clause'.
            - 'fixed':    fixed-K token blocks (recommended default, fastest).
            - 'sentence': split at sentence-ending tokens (., !, ?).
            - 'clause':   split at sentence or clause boundaries (., !, ?, ,, ;, :).
        span_size: used only for 'fixed'; target tokens per block.
        token_ids: list of response token ids; required for 'sentence'/'clause'.
        tokenizer: tokenizer; used to decode single tokens for boundary detection.

    Returns:
        List of (l, r) half-open intervals [l, r) over [0, response_length).
        Each span contains at least one token.
    """
    if response_length <= 0:
        return []

    if strategy == 'fixed':
        spans = []
        for start in range(0, response_length, span_size):
            end = min(start + span_size, response_length)
            spans.append((start, end))
        return spans

    # Sentence and clause strategies: scan decoded single-token strings for punctuation.
    if token_ids is None or tokenizer is None:
        # Fall back to fixed if we don't have tokens.
        return segment_into_spans(response_length, 'fixed', span_size)

    sentence_enders = {'.', '!', '?'}
    clause_enders   = {'.', '!', '?', ',', ';', ':'}
    boundary_chars  = sentence_enders if strategy == 'sentence' else clause_enders

    spans = []
    span_start = 0
    for i, tid in enumerate(token_ids[:response_length]):
        decoded = tokenizer.decode([tid], skip_special_tokens=True)
        # Check if any boundary character appears in the decoded token.
        if any(ch in decoded for ch in boundary_chars):
            spans.append((span_start, i + 1))
            span_start = i + 1

    # Trailing tokens that didn't end on a boundary.
    if span_start < response_length:
        spans.append((span_start, response_length))

    # Safety: if no boundaries found at all, fall back to fixed.
    if not spans:
        return segment_into_spans(response_length, 'fixed', span_size)

    # Merge very short spans (< 2 tokens) with the previous span to avoid degenerate weights.
    merged = []
    for span in spans:
        if merged and (span[1] - span[0]) < 2:
            prev_l, prev_r = merged.pop()
            merged.append((prev_l, span[1]))
        else:
            merged.append(span)
    return merged


def length_normalized_alignment(m: int, M: int, N: int) -> int:
    """Map chosen span index m (0-indexed, out of M) to rejected span index (out of N).
    Implements floor(m * N / M) with clamp to [0, N-1]."""
    if N == 0:
        return 0
    return min(int(m * N / M), N - 1)


# ── CO-CRAFT Span-Level Weight Computation ────────────────────────────────────

def compute_span_weights(
    chosen_token_delta: torch.FloatTensor,
    rejected_token_delta: torch.FloatTensor,
    chosen_loss_mask: torch.BoolTensor,
    rejected_loss_mask: torch.BoolTensor,
    f_prime_fn,
    span_strategy: str = 'fixed',
    span_size: int = 8,
    mu: float = 1.0,
    L: float = -2.0,
    U: float = 2.0,
    g_min: float = 0.25,
    g_max: float = 4.0,
    eps: float = 1e-8,
    chosen_token_ids: Optional[List[List[int]]] = None,
    rejected_token_ids: Optional[List[List[int]]] = None,
    tokenizer=None,
) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    """Compute CO-CRAFT span-level importance weights and broadcast them to token positions.

    Implements the algorithm from CO_CRAFT_span_level.md §12:
      1. Segment responses into spans.
      2. Compute per-span delta = mean of token-level log-ratios inside the span.
      3. Apply the centered f-critic: a = f'(exp(delta)) - f'(1).
      4. Exponentiate, normalize (mean=1), and clip to [g_min, g_max].
      5. Broadcast span weight to all token positions inside the span.

    Args:
        chosen_token_delta:   (B, T_c) per-token log-ratio delta for chosen response.
                               Only response positions (where chosen_loss_mask is True) matter.
        rejected_token_delta: (B, T_r) per-token log-ratio delta for rejected response.
        chosen_loss_mask:     (B, T_c) bool mask; True where token belongs to the response.
        rejected_loss_mask:   (B, T_r) bool mask.
        f_prime_fn:           Callable f'(u), derivative of the f-generator.
        span_strategy:        'fixed', 'sentence', or 'clause'.
        span_size:            Block size for 'fixed' strategy.
        mu:                   Exponentiation temperature (0.5 or 1.0 recommended).
        L, U:                 Clamp range for critic scores before exponentiation.
        g_min, g_max:         Final weight clip range after normalization.
        eps:                  Small constant for numerical stability in normalization.
        chosen_token_ids:     Optional list[list[int]] of response token ids per batch element.
        rejected_token_ids:   Optional list[list[int]] of response token ids per batch element.
        tokenizer:            Tokenizer for sentence/clause decoding.

    Returns:
        chosen_weights:   (B, T_c) float tensor, detached. Prompt positions = 0.
        rejected_weights: (B, T_r) float tensor, detached. Prompt positions = 0.

    The caller should multiply these weights by the existing token-level weights
    (from token_weight_estimation) or use them as the sole weighting signal,
    depending on configuration.
    """
    B = chosen_token_delta.size(0)
    chosen_weights  = torch.zeros_like(chosen_token_delta)
    rejected_weights = torch.zeros_like(rejected_token_delta)

    # f'(1) is the baseline for the centered critic a_f(Δ) = f'(exp(Δ)) - f'(1).
    f_prime_at_one = f_prime_fn(torch.ones(1, device=chosen_token_delta.device)).squeeze()

    for b in range(B):
        # ── Chosen spans ───────────────────────────────────────────────────────
        c_mask  = chosen_loss_mask[b]                    # (T_c,) bool
        c_delta = chosen_token_delta[b]                  # (T_c,)
        c_resp_len = int(c_mask.sum().item())
        # Response-only deltas in order (mask is True for response positions)
        c_resp_delta = c_delta[c_mask]                   # (c_resp_len,)

        c_tids = (chosen_token_ids[b] if chosen_token_ids is not None else None)
        c_spans = segment_into_spans(
            c_resp_len, strategy=span_strategy, span_size=span_size,
            token_ids=c_tids, tokenizer=tokenizer,
        )

        # ── Rejected spans — compute first so chosen can align to them ──────────
        r_mask  = rejected_loss_mask[b]
        r_delta = rejected_token_delta[b]
        r_resp_len = int(r_mask.sum().item())
        r_resp_delta = r_delta[r_mask]

        r_tids = (rejected_token_ids[b] if rejected_token_ids is not None else None)
        r_spans = segment_into_spans(
            r_resp_len, strategy=span_strategy, span_size=span_size,
            token_ids=r_tids, tokenizer=tokenizer,
        )

        # Pre-compute mean delta per rejected span — used for cross-span alignment.
        N = len(r_spans)
        r_span_mean_delta: List[torch.Tensor] = []
        for (l, r_end) in r_spans:
            r_span_mean_delta.append(r_resp_delta[l:r_end].mean())

        if len(c_spans) == 0:
            # No response tokens: leave weights as 0, continue.
            pass
        else:
            M = len(c_spans)
            raw_c = []
            for m, (l, r) in enumerate(c_spans):
                c_span_delta = c_resp_delta[l:r].mean()              # scalar Δ_m (chosen)
                if N > 0:
                    # §5/§12: align chosen span m → rejected span A(m)
                    aligned_n = length_normalized_alignment(m, M, N)
                    r_span_delta = r_span_mean_delta[aligned_n]       # scalar Δ_{A(m)} (rejected)
                    # Cross-span critic: compare chosen span against its aligned rejected span
                    delta_diff = c_span_delta - r_span_delta
                    u = torch.exp(delta_diff)
                else:
                    u = torch.exp(c_span_delta)
                a = f_prime_fn(u.unsqueeze(0)).squeeze() - f_prime_at_one
                raw_weight = torch.exp(mu * torch.clamp(a, L, U))
                raw_c.append(raw_weight)

            raw_c_t = torch.stack(raw_c)                             # (M,)
            normed_c = raw_c_t / (raw_c_t.mean() + eps)
            normed_c = torch.clamp(normed_c, g_min, g_max)          # (M,)

            # Broadcast span weights to response token positions.
            resp_token_weights_c = torch.zeros(c_resp_len, device=chosen_token_delta.device)
            for m, (l, r) in enumerate(c_spans):
                resp_token_weights_c[l:r] = normed_c[m]

            # Scatter back to full sequence length at response positions.
            resp_idx = c_mask.nonzero(as_tuple=True)[0]
            chosen_weights[b, resp_idx] = resp_token_weights_c

        # ── Rejected span weights (symmetric: align rejected span n → chosen A^{-1}) ─
        if len(r_spans) == 0:
            pass
        else:
            # Pre-compute mean delta per chosen span for reverse alignment.
            M = len(c_spans)
            c_span_mean_delta: List[torch.Tensor] = []
            for (l, r) in c_spans:
                c_span_mean_delta.append(c_resp_delta[l:r].mean())

            raw_r = []
            for n, (l, r_end) in enumerate(r_spans):
                r_span_delta = r_resp_delta[l:r_end].mean()
                if M > 0:
                    # Symmetric alignment: rejected span n → chosen span B(n)
                    aligned_m = length_normalized_alignment(n, N, M)
                    c_span_delta = c_span_mean_delta[aligned_m]
                    # Cross-span critic: compare rejected span against its aligned chosen span
                    delta_diff = r_span_delta - c_span_delta
                    u = torch.exp(delta_diff)
                else:
                    u = torch.exp(r_span_delta)
                a = f_prime_fn(u.unsqueeze(0)).squeeze() - f_prime_at_one
                raw_weight = torch.exp(mu * torch.clamp(a, L, U))
                raw_r.append(raw_weight)

            raw_r_t = torch.stack(raw_r)
            normed_r = raw_r_t / (raw_r_t.mean() + eps)
            normed_r = torch.clamp(normed_r, g_min, g_max)

            resp_token_weights_r = torch.zeros(r_resp_len, device=rejected_token_delta.device)
            for n, (l, r_end) in enumerate(r_spans):
                resp_token_weights_r[l:r_end] = normed_r[n]

            resp_idx_r = r_mask.nonzero(as_tuple=True)[0]
            rejected_weights[b, resp_idx_r] = resp_token_weights_r

    return chosen_weights.detach(), rejected_weights.detach()


# ── CO-CRAFT Span-Level Loss ──────────────────────────────────────────────────

def cocraft_span_loss(
    policy_chosen_logps: torch.FloatTensor,
    policy_rejected_logps: torch.FloatTensor,
    reference_chosen_logps: torch.FloatTensor,
    reference_rejected_logps: torch.FloatTensor,
    beta: float,
    f_star_fn=f_star_js,
    label_smoothing: float = 0.0,
    reference_free: bool = False,
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """CO-CRAFT Span-Level loss.

    This function is called with *already span-weighted* sequence-level log-prob
    sums (policy_chosen_logps etc.), which have been produced by _get_batch_logps
    with the span-broadcast weights applied as token weights.

    The outer objective is identical to f-SWIFT:
        u = beta * S_c  −  f*(beta * S_r)
        l = log(1 + exp(-u))

    The difference from plain f-SWIFT is that S_c and S_r carry span-level
    CO-CRAFT importance weights baked in via _get_batch_logps(token_level=True).
    """
    S_c = policy_chosen_logps - reference_chosen_logps
    S_r = policy_rejected_logps - reference_rejected_logps

    if reference_free:
        S_c = policy_chosen_logps
        S_r = policy_rejected_logps

    logits = beta * S_c - f_star_fn(beta * S_r)
    losses = -F.logsigmoid(logits) * (1 - label_smoothing) - F.logsigmoid(-logits) * label_smoothing

    chosen_rewards  = beta * S_c.detach()
    rejected_rewards = beta * S_r.detach()
    return losses, chosen_rewards, rejected_rewards


def fswift_loss(policy_chosen_logps: torch.FloatTensor,
                policy_rejected_logps: torch.FloatTensor,
                reference_chosen_logps: torch.FloatTensor,
                reference_rejected_logps: torch.FloatTensor,
                beta: float,
                f_star_fn = f_star_js,
                label_smoothing: float = 0.0,
                reference_free: bool = False) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute the f-SWIFT loss for a batch of policy and reference model log probabilities.

    From the paper (Eq. u_fswift), the f-SWIFT objective is:
        u = beta * S_c - f*(beta * S_r)
    where S_c and S_r are the chosen and rejected weighted log-ratios.

    IMPORTANT: beta must be applied BEFORE f* to the rejected term, not after.
    This is because f* is nonlinear, so beta*f*(S_r) != f*(beta*S_r).
    The paper defines u with beta already inside the sum that f* wraps.
    """
    # S_c = sum_t w_t * log(pi_theta(y_t|x) / pi_theta_k(y_t|x))  (chosen)
    # S_r = sum_t w_t * log(pi_theta(y'_t|x) / pi_theta_k(y'_t|x))  (rejected)
    S_c = policy_chosen_logps - reference_chosen_logps
    S_r = policy_rejected_logps - reference_rejected_logps

    if reference_free:
        S_c = policy_chosen_logps
        S_r = policy_rejected_logps

    # f-SWIFT (Eq. u_fswift): u = beta*S_c - f*(beta*S_r)
    # beta scales the log-ratios BEFORE f* is applied to the rejected term.
    # The loss is l(u) = log(1 + exp(-u)), i.e. no additional beta multiplier.
    logits = beta * S_c - f_star_fn(beta * S_r)

    losses = -F.logsigmoid(logits) * (1 - label_smoothing) - F.logsigmoid(-logits) * label_smoothing

    chosen_rewards = beta * S_c.detach()
    rejected_rewards = beta * S_r.detach()

    return losses, chosen_rewards, rejected_rewards


def preference_loss(policy_chosen_logps: torch.FloatTensor,
                    policy_rejected_logps: torch.FloatTensor,
                    reference_chosen_logps: torch.FloatTensor,
                    reference_rejected_logps: torch.FloatTensor,
                    beta: float,
                    label_smoothing: float = 0.0,
                    ipo: bool = False,
                    reference_free: bool = False) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute the DPO loss for a batch of policy and reference model log probabilities.

    Args:
        policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
        reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
        reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
        beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
        label_smoothing: conservativeness for DPO loss, which assumes that preferences are noisy (flipped with probability label_smoothing)
        ipo: If True, use the IPO loss instead of the DPO loss.
        reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.

    Returns:
        A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
        The losses tensor contains the DPO loss for each example in the batch.
        The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    if reference_free:
        ref_logratios = 0

    logits = pi_logratios - ref_logratios  # also known as h_{\pi_\theta}^{y_w,y_l}

    if ipo:
        losses = (logits - 1/(2 * beta)) ** 2  # Eq. 17 of https://arxiv.org/pdf/2310.12036v2.pdf
    else:
        # Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
        losses = -F.logsigmoid(beta * logits) * (1 - label_smoothing) - F.logsigmoid(-beta * logits) * label_smoothing

    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

    return losses, chosen_rewards, rejected_rewards


def _get_batch_logps(logits: torch.FloatTensor, labels: torch.LongTensor, weights: torch.FloatTensor=None, average_log_prob: bool = False, token_level: bool = False) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    assert logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = (labels != -100)

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == -100] = 0

    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
    # import ipdb; ipdb.set_trace()
    if token_level:
        weights = weights[:, 1:].clone()
        batch_logps = (per_token_logps * loss_mask * weights).sum(-1)
    else:
        batch_logps = (per_token_logps * loss_mask).sum(-1)
    
    if average_log_prob:
        return batch_logps/loss_mask.sum(-1)
    else:
        return batch_logps


def _get_batch_logps_tisdpo(logits: torch.FloatTensor, reference_logits: torch.FloatTensor, labels: torch.LongTensor, weights: torch.FloatTensor=None, average_log_prob: bool = False) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        reference_logits: Logits of the reference model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
        weights: Weights for each token. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per 
        (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.
        token_level: If True, return the log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    assert logits.shape[:-1] == labels.shape

    

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    reference_logits = reference_logits[:, :-1, :]

    loss_mask = (labels != -100)

    labels[labels == -100] = 0

    vocab_ps = logits.softmax(-1)
    vocab_logps = vocab_ps.log()

    reference_vocab_ps = reference_logits.softmax(-1)
    reference_vocab_logps = reference_vocab_ps.log()

    per_position_kl = (vocab_ps * (vocab_logps - reference_vocab_logps)).sum(-1)

    per_token_logps = torch.gather(vocab_logps, dim=2, index=labels.unsqueeze(2)).squeeze(2)
    per_reference_token_logps = torch.gather(reference_vocab_logps, dim=2, index=labels.unsqueeze(2)).squeeze(2)

    logps_margin = per_token_logps - per_reference_token_logps
    weights = weights[:, 1:].clone()
    
    if average_log_prob:
        return (logps_margin * weights * loss_mask).sum(-1) / loss_mask.sum(-1), \
               (per_position_kl * weights * loss_mask).sum(-1) / loss_mask.sum(-1), \
               (per_token_logps * weights * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (logps_margin * weights * loss_mask).sum(-1), \
            (per_position_kl * weights * loss_mask).sum(-1), \
            (per_token_logps * weights * loss_mask).sum(-1)



def concatenated_inputs(batch: Dict[str, Union[List, torch.LongTensor]]) -> Dict[str, torch.LongTensor]:
    """Concatenate the chosen and rejected inputs into a single tensor.
    
    Args:
        batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).
        
    Returns:
        A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
    """
    max_length = max(batch['chosen_input_ids'].shape[1], batch['rejected_input_ids'].shape[1])
    concatenated_batch = {}
    for k in batch:
        if k.startswith('chosen') and isinstance(batch[k], torch.Tensor):
            pad_value = -100 if 'labels' in k else 0
            concatenated_key = k.replace('chosen', 'concatenated')
            concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
    for k in batch:
        if k.startswith('rejected') and isinstance(batch[k], torch.Tensor):
            pad_value = -100 if 'labels' in k else 0
            concatenated_key = k.replace('rejected', 'concatenated')
            concatenated_batch[concatenated_key] = torch.cat((
                concatenated_batch[concatenated_key],
                pad_to_length(batch[k], max_length, pad_value=pad_value),
            ), dim=0)
    return concatenated_batch


class BasicTrainer(object):
    def __init__(self, policy_tokenizer, policy: nn.Module, config: DictConfig, seed: int, run_dir: str, ckpt_dir: str, reference_model: Optional[nn.Module] = None, rank: int = 0, world_size: int = 1, transform_config = None):
        """A trainer for a language model, supporting either SFT or DPO training.
            
            If multiple GPUs are present, naively splits the model across them, effectively
            offering N times available memory, but without any parallel computation.
        """
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.config = config
        self.run_dir = run_dir
        self.ckpt_dir = ckpt_dir
        self.base_data_dir = config.base_data_dir
        self._current_iteration = config.get('iteration', 0)

        tokenizer_name_or_path = config.model.tokenizer_name_or_path or config.model.name_or_path
        rank0_print(f'Loading tokenizer {tokenizer_name_or_path}')
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        data_iterator_kwargs = dict(
            names=config.datasets,
            tokenizer=self.tokenizer,
            shuffle=True,
            max_length=config.max_length,
            max_prompt_length=config.max_prompt_length,
            sft_mode=config.loss.name == 'sft',
            seed=seed, 
            reverse_dataset=config.reverse_dataset, 
            base_data_dir=config.base_data_dir,
        )
        
        self.policy = policy
        self.policy_tokenizer = policy_tokenizer
        self.reference_model = reference_model
        
        # Use the passed transform_config if available
        self.transform_config = transform_config

        print(self.transform_config)
        
        self.train_iterator = get_batch_iterator(**data_iterator_kwargs, split='train', n_epochs=config.n_epochs, n_examples=config.n_examples, batch_size=config.batch_size, silent=rank != 0, transform_config=transform_config)
        rank0_print(f'Loaded train data iterator')
        # self.eval_iterator = get_batch_iterator(**data_iterator_kwargs, split='test', n_examples=config.n_eval_examples, batch_size=config.eval_batch_size, silent=rank != 0, transform_config=transform_config)
        # rank0_print('Step 1')
        # self.eval_batches = list(self.eval_iterator)
        # rank0_print('Step 2')
        # rank0_print(f'Loaded {len(self.eval_batches)} eval batches of size {config.eval_batch_size}')

    def get_batch_samples(self, batch: Dict[str, torch.LongTensor]) -> Tuple[str, str]:
        """Generate samples from the policy (and reference model, if doing DPO training) for the given batch of inputs."""

        # FSDP generation according to https://github.com/pytorch/pytorch/issues/100069
        ctx = lambda: (FSDP.summon_full_params(self.policy, writeback=False, recurse=False) if 'FSDP' in self.config.trainer else contextlib.nullcontext())
        with ctx():
            policy_output = self.policy.generate(
                batch['prompt_input_ids'], attention_mask=batch['prompt_attention_mask'], max_length=self.config.max_length, do_sample=True, pad_token_id=self.tokenizer.pad_token_id)

        if self.config.loss.name in {'dpo', 'ipo', 'tdpo', 'tisdpo', 'fswift', 'cocraft_span'}:
            ctx = lambda: (FSDP.summon_full_params(self.reference_model, writeback=False, recurse=False) if 'FSDP' in self.config.trainer else contextlib.nullcontext())
            with ctx():
                reference_output = self.reference_model.generate(
                    batch['prompt_input_ids'], attention_mask=batch['prompt_attention_mask'], max_length=self.config.max_length, do_sample=True, pad_token_id=self.tokenizer.pad_token_id)

        policy_output = pad_to_length(policy_output, self.config.max_length, self.tokenizer.pad_token_id)
        policy_output = all_gather_if_needed(policy_output, self.rank, self.world_size)
        policy_output_decoded = self.tokenizer.batch_decode(policy_output, skip_special_tokens=True)

        if self.config.loss.name in {'dpo', 'ipo', 'tdpo', 'tisdpo', 'fswift', 'cocraft_span'}:
            reference_output = pad_to_length(reference_output, self.config.max_length, self.tokenizer.pad_token_id)
            reference_output = all_gather_if_needed(reference_output, self.rank, self.world_size)
            reference_output_decoded = self.tokenizer.batch_decode(reference_output, skip_special_tokens=True)
        else:
            reference_output_decoded = []

        return policy_output_decoded, reference_output_decoded
    
    def concatenated_forward(self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

           We do this to avoid doing two forward passes, because it's faster for FSDP.
           Automatically moves batch to the model's device to support CPU-offloaded reference models.
        """
        concatenated_batch = concatenated_inputs(batch)
        model_device = next(model.parameters()).device
        input_ids = concatenated_batch['concatenated_input_ids'].to(model_device)
        attention_mask = concatenated_batch['concatenated_attention_mask'].to(model_device)
        labels = concatenated_batch['concatenated_labels'].to(model_device)
        weights = concatenated_batch['concatenated_weight'].to(model_device)

        if input_ids.shape[0] == 0:
            raise ValueError("concatenated_forward received an empty batch (0 examples). Check for empty responses in your dataset.")

        all_logits = model(input_ids, attention_mask=attention_mask).logits.to(torch.float32)
        all_logps = _get_batch_logps(all_logits, labels, weights, average_log_prob=False, token_level=self.config.loss.token_level)
        # move results back to the policy device (cuda) before returning
        all_logps = all_logps.to(self.policy.device if hasattr(self.policy, 'device') else 'cuda')
        chosen_logps = all_logps[:batch['chosen_input_ids'].shape[0]]
        rejected_logps = all_logps[batch['chosen_input_ids'].shape[0]:]
        return chosen_logps, rejected_logps
    
    def concatenated_forward_with_per_token_logps(
        self,
        model: nn.Module,
        batch: Dict[str, Union[List, torch.LongTensor]],
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor,
               torch.BoolTensor,  torch.BoolTensor]:
        """Run model on the concatenated batch and return raw per-token log-probs
        and response loss masks for CO-CRAFT span weight computation.

        Teacher-guided token weights (from token_weight_estimation.py) are NOT
        applied here. CO-CRAFT span uses only the span-level f'-critic for
        importance weighting; sequence-level scores are computed in get_batch_metrics
        after span weights are available.

        Returns:
            chosen_per_token_lp:   (B, T-1) raw per-token log-probs (chosen).
            rejected_per_token_lp: (B, T-1) raw per-token log-probs (rejected).
            chosen_loss_mask:      (B, T-1) bool mask, True at response positions.
            rejected_loss_mask:    (B, T-1) bool mask, True at response positions.
        """
        concatenated_batch = concatenated_inputs(batch)
        model_device = next(model.parameters()).device
        input_ids = concatenated_batch['concatenated_input_ids'].to(model_device)
        attn_mask = concatenated_batch['concatenated_attention_mask'].to(model_device)
        labels    = concatenated_batch['concatenated_labels'].to(model_device)

        if input_ids.shape[0] == 0:
            raise ValueError("concatenated_forward_with_per_token_logps received an empty batch (0 examples). Check for empty responses in your dataset.")

        # Keep logits in model dtype (bfloat16) to avoid materializing a 37 GiB float32 tensor.
        # Compute per-token log-probs one sample at a time: each (T-1, V) float32 chunk
        # is ~1.2 GiB and freed immediately, avoiding a 37 GiB intermediate.
        all_logits = model(input_ids, attention_mask=attn_mask).logits  # (2B, T, V) bf16

        labels_shifted = labels[:, 1:].clone()
        logits_shifted = all_logits[:, :-1, :]              # (2B, T-1, V) bf16 view
        loss_mask = (labels_shifted != -100)
        labels_shifted[labels_shifted == -100] = 0

        B2, T1, V = logits_shifted.shape
        per_token_logps_list = []
        for i in range(B2):
            per_token_logps_list.append(
                -torch.nn.functional.cross_entropy(
                    logits_shifted[i].float(),   # (T-1, V) float32: ~1.2 GiB, freed each step
                    labels_shifted[i],
                    reduction='none',
                )
            )
        per_token_logps = torch.stack(per_token_logps_list)  # (2B, T-1) float32

        B_half = batch['chosen_input_ids'].shape[0]
        chosen_per_token_lp   = per_token_logps[:B_half]
        rejected_per_token_lp = per_token_logps[B_half:]
        chosen_loss_mask      = loss_mask[:B_half]
        rejected_loss_mask    = loss_mask[B_half:]

        return (chosen_per_token_lp, rejected_per_token_lp,
                chosen_loss_mask, rejected_loss_mask)

    def tisdpo_concatenated_forward(self, model: nn.Module, reference_model: nn.Module,
                                  batch: Dict[str, Union[List, torch.LongTensor]]):
        """Run the policy model and the reference model on the given batch of inputs, concatenating the chosen and rejected inputs together.

           We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = concatenated_inputs(batch)
        all_logits = model(concatenated_batch['concatenated_input_ids'],
                           attention_mask=concatenated_batch['concatenated_attention_mask']).logits.to(torch.float32)
        
        with torch.no_grad():
            reference_all_logits = reference_model(concatenated_batch['concatenated_input_ids'],
                                                   attention_mask=concatenated_batch[
                                                       'concatenated_attention_mask']).logits.to(torch.float32)
        
        all_logps_margin, all_position_kl, all_logps = _get_batch_logps_tisdpo(all_logits, reference_all_logits, concatenated_batch['concatenated_labels'], concatenated_batch['concatenated_weight'], average_log_prob=False)

        chosen_logps_margin = all_logps_margin[:batch['chosen_input_ids'].shape[0]]
        rejected_logps_margin = all_logps_margin[batch['chosen_input_ids'].shape[0]:]
        chosen_position_kl = all_position_kl[:batch['chosen_input_ids'].shape[0]]
        rejected_position_kl = all_position_kl[batch['chosen_input_ids'].shape[0]:]

        chosen_logps = all_logps[:batch['chosen_input_ids'].shape[0]].detach()
        rejected_logps = all_logps[batch['chosen_input_ids'].shape[0]:].detach()

        return chosen_logps_margin, rejected_logps_margin, chosen_position_kl, rejected_position_kl, \
            chosen_logps, rejected_logps
    
    def tdpo_concatenated_forward(self, model: nn.Module, reference_model: nn.Module,
                                  batch: Dict[str, Union[List, torch.LongTensor]]):
        """Run the policy model and the reference model on the given batch of inputs, concatenating the chosen and rejected inputs together.

           We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = concatenated_inputs(batch)
        all_logits = model(concatenated_batch['concatenated_input_ids'],
                           attention_mask=concatenated_batch['concatenated_attention_mask']).logits.to(torch.float32)
        
        with torch.no_grad():
            reference_all_logits = reference_model(concatenated_batch['concatenated_input_ids'],
                                                   attention_mask=concatenated_batch[
                                                       'concatenated_attention_mask']).logits.to(torch.float32)
        all_logps_margin, all_position_kl, all_logps = _tdpo_get_batch_logps(all_logits, reference_all_logits, concatenated_batch['concatenated_labels'], average_log_prob=False)

        chosen_logps_margin = all_logps_margin[:batch['chosen_input_ids'].shape[0]]
        rejected_logps_margin = all_logps_margin[batch['chosen_input_ids'].shape[0]:]
        chosen_position_kl = all_position_kl[:batch['chosen_input_ids'].shape[0]]
        rejected_position_kl = all_position_kl[batch['chosen_input_ids'].shape[0]:]

        chosen_logps = all_logps[:batch['chosen_input_ids'].shape[0]].detach()
        rejected_logps = all_logps[batch['chosen_input_ids'].shape[0]:].detach()

        return chosen_logps_margin, rejected_logps_margin, chosen_position_kl, rejected_position_kl, \
            chosen_logps, rejected_logps



    def get_batch_metrics(self, batch: Dict[str, Union[List, torch.LongTensor]], loss_config: DictConfig, train=True):
        """Compute the SFT or DPO loss and other metrics for the given batch of inputs."""

        metrics = {}
        train_test = 'train' if train else 'eval'

        if loss_config.name in {'dpo', 'ipo'}:
            policy_chosen_logps, policy_rejected_logps = self.concatenated_forward(self.policy, batch)
            with torch.no_grad():
                reference_chosen_logps, reference_rejected_logps = self.concatenated_forward(self.reference_model, batch)

            if loss_config.name == 'dpo':
                loss_kwargs = {'beta': loss_config.beta, 'reference_free': loss_config.reference_free, 'label_smoothing': loss_config.label_smoothing, 'ipo': False}
            elif loss_config.name == 'ipo':
                loss_kwargs = {'beta': loss_config.beta, 'ipo': True}
            else:
                raise ValueError(f'unknown loss {loss_config.name}')

            losses, chosen_rewards, rejected_rewards = preference_loss(
                policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, **loss_kwargs)

            reward_accuracies = (chosen_rewards > rejected_rewards).float()

            chosen_rewards = all_gather_if_needed(chosen_rewards, self.rank, self.world_size)
            rejected_rewards = all_gather_if_needed(rejected_rewards, self.rank, self.world_size)
            reward_accuracies = all_gather_if_needed(reward_accuracies, self.rank, self.world_size)

            metrics[f'rewards_{train_test}/chosen'] = chosen_rewards.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/rejected'] = rejected_rewards.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/accuracies'] = reward_accuracies.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/margins'] = (chosen_rewards - rejected_rewards).cpu().numpy().tolist()

            policy_rejected_logps = all_gather_if_needed(policy_rejected_logps.detach(), self.rank, self.world_size)
            metrics[f'logps_{train_test}/rejected'] = policy_rejected_logps.cpu().numpy().tolist()
        elif loss_config.name == 'tdpo':
            chosen_logps_margin, rejected_logps_margin, chosen_position_kl, rejected_position_kl, policy_chosen_logps, policy_rejected_logps\
                = self.tdpo_concatenated_forward(self.policy, self.reference_model, batch)
            losses, chosen_rewards, rejected_rewards = tdpo_loss(chosen_logps_margin, rejected_logps_margin,
                                                                 chosen_position_kl, rejected_position_kl,
                                                                 beta=loss_config.beta, alpha=loss_config.alpha, if_tdpo2=loss_config.if_tdpo2)

            reward_accuracies = (chosen_rewards > rejected_rewards).float()

            chosen_rewards = all_gather_if_needed(chosen_rewards, self.rank, self.world_size)
            rejected_rewards = all_gather_if_needed(rejected_rewards, self.rank, self.world_size)
            reward_accuracies = all_gather_if_needed(reward_accuracies, self.rank, self.world_size)

            metrics[f'rewards_{train_test}/chosen'] = chosen_rewards.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/rejected'] = rejected_rewards.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/accuracies'] = reward_accuracies.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/margins'] = (chosen_rewards - rejected_rewards).cpu().numpy().tolist()

            all_device_chosen_position_kl = all_gather_if_needed(chosen_position_kl.detach(), self.rank, self.world_size)
            all_device_rejected_position_kl = all_gather_if_needed(rejected_position_kl.detach(), self.rank, self.world_size)

            metrics[f'kl_{train_test}/chosen'] = all_device_chosen_position_kl.cpu().numpy().tolist()
            metrics[f'kl_{train_test}/rejected'] = all_device_rejected_position_kl.cpu().numpy().tolist()
            metrics[f'kl_{train_test}/margin'] = (all_device_chosen_position_kl - all_device_rejected_position_kl).cpu().numpy().tolist()

            policy_rejected_logps = all_gather_if_needed(policy_rejected_logps.detach(), self.rank, self.world_size)
            metrics[f'logps_{train_test}/rejected'] = policy_rejected_logps.cpu().numpy().tolist()
        elif loss_config.name == 'tisdpo':
            chosen_logps_margin, rejected_logps_margin, chosen_position_kl, rejected_position_kl, policy_chosen_logps, policy_rejected_logps\
                = self.tisdpo_concatenated_forward(self.policy, self.reference_model, batch)
            losses, chosen_rewards, rejected_rewards = tisdpo_loss(chosen_logps_margin, rejected_logps_margin,
                                                                 chosen_position_kl, rejected_position_kl,
                                                                 beta=loss_config.beta, alpha=loss_config.alpha, token_level=loss_config.token_level)

            reward_accuracies = (chosen_rewards > rejected_rewards).float()

            chosen_rewards = all_gather_if_needed(chosen_rewards, self.rank, self.world_size)
            rejected_rewards = all_gather_if_needed(rejected_rewards, self.rank, self.world_size)
            reward_accuracies = all_gather_if_needed(reward_accuracies, self.rank, self.world_size)

            metrics[f'rewards_{train_test}/chosen'] = chosen_rewards.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/rejected'] = rejected_rewards.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/accuracies'] = reward_accuracies.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/margins'] = (chosen_rewards - rejected_rewards).cpu().numpy().tolist()

            all_device_chosen_position_kl = all_gather_if_needed(chosen_position_kl.detach(), self.rank, self.world_size)
            all_device_rejected_position_kl = all_gather_if_needed(rejected_position_kl.detach(), self.rank, self.world_size)

            metrics[f'kl_{train_test}/chosen'] = all_device_chosen_position_kl.cpu().numpy().tolist()
            metrics[f'kl_{train_test}/rejected'] = all_device_rejected_position_kl.cpu().numpy().tolist()
            metrics[f'kl_{train_test}/margin'] = (all_device_chosen_position_kl - all_device_rejected_position_kl).cpu().numpy().tolist()

            policy_rejected_logps = all_gather_if_needed(policy_rejected_logps.detach(), self.rank, self.world_size)
            metrics[f'logps_{train_test}/rejected'] = policy_rejected_logps.cpu().numpy().tolist()

        elif loss_config.name == 'fswift':
            policy_chosen_logps, policy_rejected_logps = self.concatenated_forward(self.policy, batch)
            with torch.no_grad():
                reference_chosen_logps, reference_rejected_logps = self.concatenated_forward(self.reference_model, batch)

            # Resolve f* function: support adaptive scheduling per iteration
            f_div_type = getattr(loss_config, 'f_divergence', 'js')
            if hasattr(loss_config, 'f_schedule') and loss_config.f_schedule:
                # Adaptive scheduling: use iteration index to pick f-divergence
                current_iter = getattr(self, '_current_iteration', 0)
                schedule = loss_config.f_schedule
                for entry in schedule:
                    if current_iter >= entry.get('from_iter', 0):
                        f_div_type = entry.get('f_divergence', f_div_type)
            f_star_fn = get_f_star(f_div_type)

            losses, chosen_rewards, rejected_rewards = fswift_loss(
                policy_chosen_logps, policy_rejected_logps,
                reference_chosen_logps, reference_rejected_logps,
                beta=loss_config.beta, f_star_fn=f_star_fn,
                label_smoothing=loss_config.label_smoothing,
                reference_free=loss_config.reference_free)

            reward_accuracies = (chosen_rewards > rejected_rewards).float()

            chosen_rewards = all_gather_if_needed(chosen_rewards, self.rank, self.world_size)
            rejected_rewards = all_gather_if_needed(rejected_rewards, self.rank, self.world_size)
            reward_accuracies = all_gather_if_needed(reward_accuracies, self.rank, self.world_size)

            metrics[f'rewards_{train_test}/chosen'] = chosen_rewards.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/rejected'] = rejected_rewards.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/accuracies'] = reward_accuracies.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/margins'] = (chosen_rewards - rejected_rewards).cpu().numpy().tolist()
            metrics[f'f_divergence_type'] = [f_div_type]

            policy_rejected_logps = all_gather_if_needed(policy_rejected_logps.detach(), self.rank, self.world_size)
            metrics[f'logps_{train_test}/rejected'] = policy_rejected_logps.cpu().numpy().tolist()

        elif loss_config.name == 'cocraft_span':
            # ── Step 1: Get reference per-token log-probs (frozen) ─────────────
            with torch.no_grad():
                (ref_chosen_per_tok, ref_rejected_per_tok,
                 chosen_loss_mask, rejected_loss_mask) = \
                    self.concatenated_forward_with_per_token_logps(self.reference_model, batch)

            # ── Step 2: Get policy per-token log-probs (trainable) ──────────────
            (pol_chosen_per_tok, pol_rejected_per_tok,
             _chosen_mask, _rejected_mask) = \
                self.concatenated_forward_with_per_token_logps(self.policy, batch)

            # ── Step 3: Compute per-token log-ratio deltas Δ = log π_θ - log π_ref ─
            # Shape: (B, T-1). Used by span critic only — no teacher weights involved.
            # Detach policy log-probs before subtracting so that span weights are
            # treated as fixed constants and gradients flow only through Step 6.
            # ref tensors come from CPU-offloaded reference model — move to policy device.
            model_device = next(self.policy.parameters()).device
            with torch.no_grad():
                chosen_token_delta  = (pol_chosen_per_tok.detach() - ref_chosen_per_tok.to(model_device)).detach()
                rejected_token_delta = (pol_rejected_per_tok.detach() - ref_rejected_per_tok.to(model_device)).detach()
                chosen_loss_mask   = chosen_loss_mask.to(model_device)
                rejected_loss_mask = rejected_loss_mask.to(model_device)

            # ── Step 4: Resolve f-divergence (supports adaptive scheduling) ────
            f_div_type = getattr(loss_config, 'f_divergence', 'js')
            if hasattr(loss_config, 'f_schedule') and loss_config.f_schedule:
                current_iter = getattr(self, '_current_iteration', 0)
                for entry in loss_config.f_schedule:
                    if current_iter >= entry.get('from_iter', 0):
                        f_div_type = entry.get('f_divergence', f_div_type)
            f_star_fn  = get_f_star(f_div_type)
            f_prime_fn = get_f_prime(f_div_type)

            # ── Step 5: Compute span weights and broadcast to token positions ───
            span_strategy = getattr(loss_config, 'span_strategy', 'fixed')
            span_size     = getattr(loss_config, 'span_size', 8)
            mu            = getattr(loss_config, 'mu', 1.0)
            span_L        = getattr(loss_config, 'span_L', -2.0)
            span_U        = getattr(loss_config, 'span_U', 2.0)
            span_g_min    = getattr(loss_config, 'span_g_min', 0.25)
            span_g_max    = getattr(loss_config, 'span_g_max', 4.0)

            # Optionally pass token ids for sentence/clause segmentation.
            # We only decode them if the strategy actually needs them.
            chosen_tids  = None
            rejected_tids = None
            if span_strategy in ('sentence', 'clause'):
                B_half = batch['chosen_input_ids'].shape[0]
                chosen_tids = []
                rejected_tids = []
                for b_idx in range(B_half):
                    c_lab = batch['chosen_labels'][b_idx]
                    r_lab = batch['rejected_labels'][b_idx]
                    # Response token ids are where label != -100.
                    chosen_tids.append(
                        [int(t) for t in batch['chosen_input_ids'][b_idx][c_lab != -100].tolist()]
                    )
                    rejected_tids.append(
                        [int(t) for t in batch['rejected_input_ids'][b_idx][r_lab != -100].tolist()]
                    )

            with torch.no_grad():
                span_chosen_weights, span_rejected_weights = compute_span_weights(
                    chosen_token_delta   = chosen_token_delta,
                    rejected_token_delta = rejected_token_delta,
                    chosen_loss_mask     = chosen_loss_mask,
                    rejected_loss_mask   = rejected_loss_mask,
                    f_prime_fn           = f_prime_fn,
                    span_strategy        = span_strategy,
                    span_size            = span_size,
                    mu                   = mu,
                    L                    = span_L,
                    U                    = span_U,
                    g_min                = span_g_min,
                    g_max                = span_g_max,
                    chosen_token_ids     = chosen_tids,
                    rejected_token_ids   = rejected_tids,
                    tokenizer            = self.tokenizer,
                )

            # ── Step 6: Apply span weights only (no teacher-guided token weights) ─
            # CO-CRAFT span uses span weights from the f'-critic as the sole
            # importance signal. Teacher token weights from token_weight_estimation.py
            # are intentionally ignored here — span-level weighting replaces them.

            # Recompute policy sequence-level scores using only span weights.
            policy_chosen_logps  = (pol_chosen_per_tok  * chosen_loss_mask  * span_chosen_weights).sum(-1)
            policy_rejected_logps = (pol_rejected_per_tok * rejected_loss_mask * span_rejected_weights).sum(-1)
            policy_chosen_logps   = policy_chosen_logps.to(model_device)
            policy_rejected_logps = policy_rejected_logps.to(model_device)

            # Recompute reference sequence-level scores using only span weights (no_grad).
            with torch.no_grad():
                ref_chosen_logps  = (ref_chosen_per_tok.to(model_device)  * chosen_loss_mask * span_chosen_weights).sum(-1)
                ref_rejected_logps = (ref_rejected_per_tok.to(model_device) * rejected_loss_mask * span_rejected_weights).sum(-1)

            # ── Step 7: Compute CO-CRAFT span loss ──────────────────────────────
            losses, chosen_rewards, rejected_rewards = cocraft_span_loss(
                policy_chosen_logps, policy_rejected_logps,
                ref_chosen_logps,    ref_rejected_logps,
                beta            = loss_config.beta,
                f_star_fn       = f_star_fn,
                label_smoothing = loss_config.label_smoothing,
                reference_free  = loss_config.reference_free,
            )

            reward_accuracies = (chosen_rewards > rejected_rewards).float()

            chosen_rewards   = all_gather_if_needed(chosen_rewards,   self.rank, self.world_size)
            rejected_rewards = all_gather_if_needed(rejected_rewards, self.rank, self.world_size)
            reward_accuracies = all_gather_if_needed(reward_accuracies, self.rank, self.world_size)

            metrics[f'rewards_{train_test}/chosen']    = chosen_rewards.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/rejected']  = rejected_rewards.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/accuracies'] = reward_accuracies.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/margins']   = (chosen_rewards - rejected_rewards).cpu().numpy().tolist()
            metrics[f'f_divergence_type']              = [f_div_type]
            metrics[f'span_strategy']                  = [span_strategy]

            policy_rejected_logps_gathered = all_gather_if_needed(policy_rejected_logps.detach(), self.rank, self.world_size)
            metrics[f'logps_{train_test}/rejected'] = policy_rejected_logps_gathered.cpu().numpy().tolist()

            # Re-assign for the shared logps/chosen logging below.
            policy_chosen_logps = policy_chosen_logps

        elif loss_config.name == 'sft':
            policy_chosen_logits = self.policy(batch['chosen_input_ids'], attention_mask=batch['chosen_attention_mask']).logits.to(torch.float32)
            policy_chosen_logps = _get_batch_logps(policy_chosen_logits, batch['chosen_labels'], average_log_prob=False, token_level=False)

            losses = -policy_chosen_logps

        policy_chosen_logps = all_gather_if_needed(policy_chosen_logps.detach(), self.rank, self.world_size)
        metrics[f'logps_{train_test}/chosen'] = policy_chosen_logps.cpu().numpy().tolist()

        all_devices_losses = all_gather_if_needed(losses.detach(), self.rank, self.world_size)
        metrics[f'loss/{train_test}'] = all_devices_losses.cpu().numpy().tolist()

        return losses.mean(), metrics

    def train(self):
        """Begin either SFT or DPO training, with periodic evaluation."""

        rank0_print(f'Using {self.config.optimizer} optimizer')
        self.optimizer = getattr(torch.optim, self.config.optimizer)(self.policy.parameters(), lr=self.config.lr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda step: min(1.0, (step + 1) / (self.config.warmup_steps + 1)))
    
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        if self.config.loss.name in {'dpo', 'ipo', 'tdpo', 'tisdpo', 'fswift', 'cocraft_span'}:
            self.reference_model.eval()

        self.example_counter = 0
        self.batch_counter = 0
        last_log = None

        # Per-epoch checkpoint: inferred from dataset size and batch_size
        examples_per_epoch = self.config.get('examples_per_epoch', None)
        last_saved_epoch = -1

        for batch in self.train_iterator:
            # #### BEGIN EVALUATION ####
            # if self.example_counter > 5: break  ### Early stop to obtain good checkpoint
            # if self.example_counter % 2000 == 0:
            if False:
                print('Saving model to LASTEST ...')
                self.policy.save_pretrained("model_hub/Qwen1.5-1.8B/LASTEST", safe_serialization=True)
                self.tokenizer.save_pretrained("model_hub/Qwen1.5-1.8B/LASTEST")
                print(f'Evaluating ...')
                lastest_path = os.path.abspath("model_hub/Qwen1.5-1.8B/LASTEST")
                bash_command = f"bash lm-evaluation-harness/eval_llm.sh {lastest_path}"
                try:
                    # subprocess.run(bash_command, shell=True, check=True)
                    result = subprocess.run(bash_command, shell=True, check=True, capture_output=True, text=True)
                    print(result.stdout)
                    wandb.log({"stdout": result.stdout})
                    print("Script executed successfully.")
                except subprocess.CalledProcessError as e:
                    print("An error occurred while executing the script.")
                    print("Error details:", e)
                    print("STDOUT:", e.stdout)
                    print("STDERR:", e.stderr)
            # #### END EVALUATION ####

            #### BEGIN TRAINING ####
            self.policy.train()

            start_time = time.time()
            batch_metrics = defaultdict(list)
            active_microbatches = 0
            for microbatch_idx in range(self.config.gradient_accumulation_steps):
                global_microbatch = slice_and_move_batch_for_device(batch, microbatch_idx, self.config.gradient_accumulation_steps, self.rank)
                local_microbatch = slice_and_move_batch_for_device(global_microbatch, self.rank, self.world_size, self.rank)
                # Skip empty microbatches that can appear at the end of a dataset epoch.
                if local_microbatch['chosen_input_ids'].shape[0] == 0:
                    continue
                active_microbatches += 1
                loss, metrics = self.get_batch_metrics(local_microbatch, self.config.loss, train=True)
                (loss / self.config.gradient_accumulation_steps).backward()

                for k, v in metrics.items():
                    batch_metrics[k].extend(v)

            grad_norm = self.clip_gradient()
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            step_time = time.time() - start_time
            examples_per_second = self.config.batch_size / step_time
            batch_metrics['examples_per_second'].append(examples_per_second)
            batch_metrics['grad_norm'].append(grad_norm)

            self.batch_counter += 1
            self.example_counter += self.config.batch_size

            # Save checkpoint at end of each epoch
            if examples_per_epoch is not None:
                current_epoch = (self.example_counter - 1) // examples_per_epoch
                if current_epoch > last_saved_epoch:
                    last_saved_epoch = current_epoch
                    epoch_ckpt_dir = os.path.join(self.ckpt_dir, f'epoch{current_epoch}')
                    rank0_print(f'Saving epoch {current_epoch} checkpoint to {epoch_ckpt_dir}')
                    os.makedirs(epoch_ckpt_dir, exist_ok=True)
                    self.policy.save_pretrained(epoch_ckpt_dir)
                    self.tokenizer.save_pretrained(epoch_ckpt_dir)

            if self.example_counter % 50 == 0:
                mean_train_metrics = {k: sum(v) / len(v) for k, v in batch_metrics.items() if isinstance(v[0], (int, float))}
                mean_train_metrics['counters/examples'] = self.example_counter
                mean_train_metrics['counters/updates'] = self.batch_counter
                rank0_print(f'train stats after {self.example_counter} examples: {formatted_dict(mean_train_metrics)}')

                if self.config.wandb.enabled and self.rank == 0:
                    wandb.log(mean_train_metrics, step=self.example_counter)

                last_log = time.time()
            # else:
                #rank0_print(f'skipping logging after {self.example_counter} examples to avoid logging too frequently')
            #### END TRAINING ####


    def clip_gradient(self):
        """Clip the gradient norm of the parameters of a non-FSDP policy."""
        return torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm).item()

    def write_state_dict(self, step: int, state: Dict[str, torch.Tensor], metrics: Dict, filename: str, dir_name: Optional[str] = None):
        """Write a checkpoint to disk."""
        if dir_name is None:
            dir_name = os.path.join(self.run_dir, f'LATEST')
        dir_name = self.ckpt_dir
        os.makedirs(dir_name, exist_ok=True)
        output_path = os.path.join(dir_name, filename)
        rank0_print(f'writing checkpoint to {output_path}...')
        torch.save({
            'step_idx': step,
            'state': state,
            'metrics': metrics if metrics is not None else {},
        }, output_path)
    
    def save(self, output_dir: Optional[str] = None, metrics: Optional[Dict] = None):
        """Save policy and tokenizer to disk."""
        if output_dir is None:
            model_save_dir = os.path.join(self.run_dir, f'LATEST')
        else:
            model_save_dir = output_dir
        model_save_dir = self.ckpt_dir
        os.makedirs(model_save_dir, exist_ok=True)
        
        # Save model using transformers save_pretrained
        self.policy.save_pretrained(model_save_dir)
        rank0_print(f"Model saved to {model_save_dir} using save_pretrained")
        
        # Save tokenizer alongside the model
        self.tokenizer.save_pretrained(model_save_dir)
        
        # Save metrics separately
        if metrics is not None:
            metrics_file = os.path.join(model_save_dir, "training_metrics.json")
            with open(metrics_file, "w") as f:
                json.dump({"step": self.example_counter, "metrics": metrics}, f)


class FSDPTrainer(BasicTrainer):
    def __init__(self, policy_tokenizer, policy: nn.Module, config: DictConfig, seed: int, run_dir: str, ckpt_dir: str, reference_model: Optional[nn.Module] = None, rank: int = 0, world_size: int = 1, transform_config = None):
        """A trainer subclass that uses PyTorch FSDP to shard the model across multiple GPUs.
        
           This trainer will shard both the policy and reference model across all available GPUs.
           Models are sharded at the block level, where the block class name is provided in the config.
        """

        super().__init__(policy_tokenizer, policy, config, seed, run_dir, ckpt_dir, reference_model, rank, world_size, transform_config=transform_config)
        assert config.model.block_name is not None, 'must specify model.block_name (e.g., GPT2Block or GPTNeoXLayer) for FSDP'

        wrap_class = get_block_class_from_model(policy, config.model.block_name)
        model_auto_wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={wrap_class},)

        shared_fsdp_kwargs = dict(
            auto_wrap_policy=model_auto_wrap_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            cpu_offload=CPUOffload(offload_params=False),
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            device_id=rank,
            ignored_modules=None,
            limit_all_gathers=False,
            use_orig_params=False,
            sync_module_states=False
        )

        rank0_print('Sharding policy...')
        mp_dtype = getattr(torch, config.model.fsdp_policy_mp) if config.model.fsdp_policy_mp is not None else None
        policy_mp_policy = MixedPrecision(param_dtype=mp_dtype, reduce_dtype=mp_dtype, buffer_dtype=mp_dtype)
        self.policy = FSDP(policy, **shared_fsdp_kwargs, mixed_precision=policy_mp_policy)

        if config.activation_checkpointing:
            rank0_print('Attempting to enable activation checkpointing...')
            try:
                # use activation checkpointing, according to:
                # https://pytorch.org/blog/scaling-multimodal-foundation-models-in-torchmultimodal-with-pytorch-distributed/
                #
                # first, verify we have FSDP activation support ready by importing:
                from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                    checkpoint_wrapper,
                    apply_activation_checkpointing,
                    CheckpointImpl,
                )
                non_reentrant_wrapper = functools.partial(
                    checkpoint_wrapper,
                    offload_to_cpu=False,
                    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                )
            except Exception as e:
                rank0_print('FSDP activation checkpointing not available:', e)
            else:
                check_fn = lambda submodule: isinstance(submodule, wrap_class)
                rank0_print('Applying activation checkpointing wrapper to policy...')
                apply_activation_checkpointing(self.policy, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn)
                rank0_print('FSDP activation checkpointing enabled!')

        if config.loss.name in {'dpo', 'ipo', 'tdpo', 'tisdpo', 'fswift', 'cocraft_span'}:
            rank0_print('Sharding reference model...')
            self.reference_model = FSDP(reference_model, **shared_fsdp_kwargs)
        
        print('Loaded model on rank', rank)
        dist.barrier()

    def clip_gradient(self):
        """Clip the gradient norm of the parameters of an FSDP policy, gathering the gradients across all GPUs."""
        return self.policy.clip_grad_norm_(self.config.max_grad_norm).item()
    
    def save(self, output_dir=None, metrics=None):
        """Save policy and tokenizer state to disk, gathering from all processes and saving only on the rank 0 process."""
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.policy, StateDictType.FULL_STATE_DICT, state_dict_config=save_policy):
            policy_state_dict = self.policy.state_dict()

        if self.rank == 0:
            # Save model using transformers save_pretrained
            if output_dir is None:
                model_save_dir = self.run_dir
            else:
                model_save_dir = output_dir
            
            os.makedirs(model_save_dir, exist_ok=True)
            
            # Get the original model class and instantiate it directly
            from transformers import AutoModelForCausalLM
            model_name = self.config.model.name_or_path
            unwrapped_model = AutoModelForCausalLM.from_pretrained(model_name)
            unwrapped_model.load_state_dict(policy_state_dict)
            
            # Save using transformers save_pretrained
            unwrapped_model.save_pretrained(model_save_dir)
            rank0_print(f"Model saved to {model_save_dir} using save_pretrained")
            del unwrapped_model
            
            # Save tokenizer alongside the model
            self.tokenizer.save_pretrained(model_save_dir)
            
            # Save metrics separately
            if metrics is not None:
                metrics_file = os.path.join(model_save_dir, "training_metrics.json")
                with open(metrics_file, "w") as f:
                    json.dump({"step": self.example_counter, "metrics": metrics}, f)
            
        del policy_state_dict
        dist.barrier()


class TensorParallelTrainer(BasicTrainer):
    def __init__(self, policy_tokenizer, policy, config, seed, run_dir, ckpt_dir: str, reference_model=None, rank=0, world_size=1, transform_config=None):
        """A trainer subclass that uses TensorParallel to shard the model across multiple GPUs.

           Based on https://github.com/BlackSamorez/tensor_parallel. Note sampling is extremely slow,
              see https://github.com/BlackSamorez/tensor_parallel/issues/66.
        """
        super().__init__(policy_tokenizer, policy, config, seed, run_dir, ckpt_dir, reference_model, rank, world_size, transform_config=transform_config)

        rank0_print('Sharding policy...')
        self.policy = tp.tensor_parallel(policy, sharded=True)
        if config.loss.name in {'dpo', 'ipo', 'tdpo', 'tisdpo', 'fswift', 'cocraft_span'}:
            rank0_print('Sharding reference model...')
            self.reference_model = tp.tensor_parallel(reference_model, sharded=False)

    def save(self, output_dir=None, metrics=None):
        """Save (unsharded) policy state to disk."""
        with tp.save_tensor_parallel(self.policy):
            policy_state_dict = self.policy.state_dict()
    
        # Save model using transformers save_pretrained
        if output_dir is None:
            model_save_dir = os.path.join(self.run_dir, f'LATEST')
        else:
            model_save_dir = output_dir
        model_save_dir = self.ckpt_dir
            
        os.makedirs(model_save_dir, exist_ok=True)
            
        # Get the original model class and instantiate it directly
        from transformers import AutoModelForCausalLM
        model_name = self.config.model.name_or_path
        unwrapped_model = AutoModelForCausalLM.from_pretrained(model_name)
        unwrapped_model.load_state_dict(policy_state_dict)
        
        # Save using transformers save_pretrained
        unwrapped_model.save_pretrained(model_save_dir)
        rank0_print(f"Model saved to {model_save_dir} using save_pretrained")
        del unwrapped_model
            
        # Save tokenizer alongside the model
        self.tokenizer.save_pretrained(model_save_dir)
            
        # Save metrics separately
        if metrics is not None:
            metrics_file = os.path.join(model_save_dir, "training_metrics.json")
            with open(metrics_file, "w") as f:
                json.dump({"step": self.example_counter, "metrics": metrics}, f)
        
        del policy_state_dict
        