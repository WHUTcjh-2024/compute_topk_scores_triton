# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Pure-Triton implementation of vLLM 0.29.0 ``compute_topk_scores``.

No PyTorch CUDA operator performs computation on this path. Top-K selection,
the log-softmax gather, the ``logits_mode`` gather, the token-id assembly and
the validity masking are all Triton kernels. PyTorch is only used for
allocations (``empty`` / ``new_empty``), stride metadata and the
``LogprobsTensors`` container.
"""

import numpy as np
import torch

from vllm.sampling_params import MAX_LOGPROB_TOKEN_IDS, SamplingParams
from vllm.triton_utils import tl, triton
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor, UvaBackedTensor

# Upper bound on the topk kernel's per-iteration gather width.
_MAX_TOPK_BLOCK = 1024

# Minimum scan width; grow to KP for large K so block selection stays in bounds.
_TOPK_BLOCK = 1024

# Ordering sentinel: one below the packed key of (``-inf``, index 0).
# Real keys are ``(monotonic_bits(logit) - 2**31) << 32 | index``.
_KEY_SENTINEL = tl.constexpr((-2139095041 << 32) - 1)


@triton.jit
def _order_key(logits, index):
    """Map fp32 logits to a monotonically ordered int64 key, tie-broken by index.

    The IEEE-754 bit pattern is folded so that unsigned comparison matches
    numeric comparison; the 32 index bits ride in the low half.
    """
    u = logits.to(tl.uint32, bitcast=True)
    # Signed zeros compare equal. Canonicalize NaNs so either sign sorts
    # above +inf, matching torch.topk's largest-value semantics.
    magnitude = u & 0x7FFFFFFF
    u = tl.where(magnitude == 0, 0, u)
    u = tl.where(magnitude > 0x7F800000, 0x7FFFFFFF, u)
    folded = u ^ tl.where(u >> 31 != 0, 0xFFFFFFFF, 0x80000000)
    return ((folded.to(tl.int64) - 0x80000000) << 32) + index.to(tl.int64)


@triton.jit
def _topk_ids_kernel(
    out_ids_ptr,
    out_ids_stride,
    logits_ptr,
    logits_stride,
    sampled_ptr,
    vocab_size,
    TOPK: tl.constexpr,
    WRITE_SAMPLED: tl.constexpr,
    BLOCK: tl.constexpr,
    KP: tl.constexpr,
):
    """Single-pass streaming Top-K over one vocab row per program.

    Keeps the running best ``KP`` entries as packed keys, so values never need
    a second pass. Blocks whose best key cannot enter the running set are
    skipped entirely, which drops most of the comparison work once the
    threshold has settled.
    """
    req_idx = tl.program_id(0).to(tl.int64)
    row_ptr = logits_ptr + req_idx * logits_stride
    out_ptr = out_ids_ptr + req_idx * out_ids_stride

    if WRITE_SAMPLED:
        tl.store(out_ptr, tl.load(sampled_ptr + req_idx))

    if TOPK > 0:
        k_off = tl.arange(0, KP)
        running = tl.full([KP], _KEY_SENTINEL, tl.int64)
        threshold = tl.full((), _KEY_SENTINEL, tl.int64)

        full = (vocab_size // BLOCK) * BLOCK
        for i in tl.range(0, full, BLOCK):
            block = i + tl.arange(0, BLOCK)
            keys = _order_key(tl.load(row_ptr + block).to(tl.float32), block)
            block_max = tl.max(keys)
            if block_max >= threshold:
                if KP == 1:
                    # Triton 3.7.1 topk(k=1) fails when reshaping a scalar.
                    running = tl.maximum(running, block_max)
                else:
                    block_top = tl.topk(keys, KP)
                    merged = tl.reshape(tl.join(running, block_top), [2 * KP])
                    running = tl.topk(merged, KP)
                threshold = tl.min(running)

        if full < vocab_size:
            block = full + tl.arange(0, BLOCK)
            mask = block < vocab_size
            keys = _order_key(
                tl.load(row_ptr + block, mask=mask, other=0.0).to(tl.float32),
                block,
            )
            keys = tl.where(mask, keys, _KEY_SENTINEL)
            block_max = tl.max(keys)
            if block_max >= threshold:
                if KP == 1:
                    running = tl.maximum(running, block_max)
                else:
                    block_top = tl.topk(keys, KP)
                    merged = tl.reshape(tl.join(running, block_top), [2 * KP])
                    running = tl.topk(merged, KP)

        ids = running & 0xFFFFFFFF
        tl.store(
            out_ptr + (1 if WRITE_SAMPLED else 0) + k_off, ids, mask=k_off < TOPK
        )


@triton.jit
def _topk_log_softmax_kernel(
    output_ptr,
    logits_ptr,
    logits_stride,
    topk_ids_ptr,
    topk,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    TOPK_BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0).to(tl.int64)
    row_ptr = logits_ptr + req_idx * logits_stride

    full = (vocab_size // BLOCK_SIZE) * BLOCK_SIZE
    max_val = float("-inf")
    se = 0.0
    for i in tl.range(
        0, full, BLOCK_SIZE, num_stages=2, loop_unroll_factor=2
    ):
        block = i + tl.arange(0, BLOCK_SIZE)
        logits = tl.load(row_ptr + block).to(tl.float32)
        block_max = tl.max(logits)
        new_max = tl.maximum(max_val, block_max)
        # Keep the accumulator at zero until a finite logit appears. An
        # all-masked prefix otherwise evaluates -inf - -inf and poisons se.
        normalizer = tl.where(new_max == float("-inf"), 0.0, new_max)
        se = se * tl.exp(max_val - normalizer) + tl.sum(
            tl.exp(logits - normalizer)
        )
        max_val = new_max

    if full < vocab_size:
        block = full + tl.arange(0, BLOCK_SIZE)
        logits = tl.load(
            row_ptr + block, mask=block < vocab_size, other=float("-inf")
        ).to(tl.float32)
        block_max = tl.max(logits)
        new_max = tl.maximum(max_val, block_max)
        normalizer = tl.where(new_max == float("-inf"), 0.0, new_max)
        se = se * tl.exp(max_val - normalizer) + tl.sum(
            tl.exp(logits - normalizer)
        )
        max_val = new_max

    lse = tl.log(se)

    for j in range(0, topk, TOPK_BLOCK_SIZE):
        k_offset = j + tl.arange(0, TOPK_BLOCK_SIZE)
        k_mask = k_offset < topk
        topk_ids = tl.load(
            topk_ids_ptr + req_idx * topk + k_offset, mask=k_mask, other=0
        ).to(tl.int64)
        logits = tl.load(row_ptr + topk_ids, mask=k_mask)
        logits = logits.to(tl.float32)
        o = logits - max_val - lse
        tl.store(output_ptr + req_idx * topk + k_offset, o, mask=k_mask)


@triton.jit
def _ranks_kernel(
    output_ptr,
    logits_ptr,
    logits_stride,
    token_ids_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0).to(tl.int64)
    row_ptr = logits_ptr + req_idx * logits_stride

    token_id = tl.load(token_ids_ptr + req_idx)
    x = tl.load(row_ptr + token_id)

    full = (vocab_size // BLOCK_SIZE) * BLOCK_SIZE
    n = 0
    for i in tl.range(0, full, BLOCK_SIZE, num_stages=4):
        block = i + tl.arange(0, BLOCK_SIZE)
        logits = tl.load(row_ptr + block)
        n += tl.sum((logits >= x).to(tl.int32))

    if full < vocab_size:
        block = full + tl.arange(0, BLOCK_SIZE)
        logits = tl.load(
            row_ptr + block, mask=block < vocab_size, other=float("-inf")
        )
        n += tl.sum((logits >= x).to(tl.int32))

    tl.store(output_ptr + req_idx, n)


@triton.jit
def _gather_logits_kernel(
    out_ptr,
    out_stride,
    logits_ptr,
    logits_stride,
    ids_ptr,
    ids_stride,
    num_cols,
    HAS_MASK: tl.constexpr,
    mask_ptr,
    mask_stride,
    BLOCK: tl.constexpr,
):
    """Gather one logits row by token id, emitting fp32, masked to ``-inf``."""
    req_idx = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    in_range = cols < num_cols

    ids = tl.load(
        ids_ptr + req_idx * ids_stride + cols, mask=in_range, other=0
    ).to(tl.int64)
    values = tl.load(
        logits_ptr + req_idx * logits_stride + ids, mask=in_range, other=0.0
    ).to(tl.float32)

    if HAS_MASK:
        valid = tl.load(
            mask_ptr + req_idx * mask_stride + cols, mask=in_range, other=0
        )
        values = tl.where(valid != 0, values, float("-inf"))

    tl.store(out_ptr + req_idx * out_stride + cols, values, mask=in_range)


@triton.jit
def _mask_scores_kernel(
    scores_ptr,
    scores_stride,
    mask_ptr,
    mask_stride,
    num_cols,
    BLOCK: tl.constexpr,
):
    """In-place ``masked_fill`` of scores where the validity mask is clear."""
    req_idx = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    in_range = cols < num_cols

    offset = req_idx * scores_stride + cols
    valid = tl.load(
        mask_ptr + req_idx * mask_stride + cols, mask=in_range, other=0
    )
    tl.store(
        scores_ptr + offset,
        float("-inf"),
        mask=in_range & (valid == 0),
    )


def compute_token_logprobs(
    logits: torch.Tensor, token_ids: torch.Tensor
) -> torch.Tensor:
    # NOTE(woosuk): To save GPU memory, we do not materialize the full
    # [batch_size, vocab_size] logprobs tensor. The kernel computes
    # max + logsumexp per row and only emits logprobs at `token_ids`.
    batch_size, vocab_size = logits.shape
    num_logprobs = token_ids.shape[1]
    logprobs = logits.new_empty((batch_size, num_logprobs), dtype=torch.float32)
    # Cap the kernel's per-iteration width so very large num_logprobs requests
    # stream the gather in bounded-size chunks, avoiding excessive mem use.
    topk_block_size = min(triton.next_power_of_2(num_logprobs), _MAX_TOPK_BLOCK)
    _topk_log_softmax_kernel[(batch_size,)](
        logprobs,
        logits,
        logits.stride(0),
        token_ids,
        num_logprobs,
        vocab_size,
        BLOCK_SIZE=1024,  # type: ignore
        TOPK_BLOCK_SIZE=topk_block_size,
    )
    return logprobs


def _gather_logits(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    valid_mask: torch.Tensor | None,
) -> torch.Tensor:
    batch_size = logits.shape[0]
    num_cols = token_ids.shape[1]
    scores = torch.empty(
        (batch_size, num_cols), dtype=torch.float32, device=logits.device
    )
    has_mask = valid_mask is not None
    _gather_logits_kernel[(batch_size,)](
        scores,
        scores.stride(0),
        logits,
        logits.stride(0),
        token_ids,
        token_ids.stride(0),
        num_cols,
        HAS_MASK=has_mask,
        mask_ptr=valid_mask if has_mask else scores,
        mask_stride=(
            valid_mask.stride(0) if has_mask else scores.stride(0)  # type: ignore[union-attr]
        ),
        BLOCK=triton.next_power_of_2(num_cols),
    )
    return scores


def _mask_invalid_scores(
    scores: torch.Tensor, valid_mask: torch.Tensor
) -> None:
    batch_size, num_cols = scores.shape
    _mask_scores_kernel[(batch_size,)](
        scores,
        scores.stride(0),
        valid_mask,
        valid_mask.stride(0),
        num_cols,
        BLOCK=triton.next_power_of_2(num_cols),
    )


def compute_topk_scores(
    logits: torch.Tensor,
    num_logprobs: int,
    sampled_token_ids: torch.Tensor,
    cu_num_logits: list[int] | torch.Tensor | None = None,
    logprob_token_ids_state: "LogprobTokenIdsState | None" = None,
    expanded_idx_mapping: torch.Tensor | None = None,
    max_per_req_token_ids: int = 0,
    logits_mode: bool = False,
) -> LogprobsTensors:
    assert num_logprobs >= 0
    batch_size, vocab_size = logits.shape
    if num_logprobs > vocab_size:
        raise ValueError(
            f"num_logprobs ({num_logprobs}) exceeds vocab_size ({vocab_size})"
        )

    if max_per_req_token_ids == 0:
        # Fast path: no request asked for custom logprob_token_ids. The Top-K
        # kernel writes the sampled token and the selected ids in one pass.
        if num_logprobs == 0:
            # Matches upstream exactly: a pure view, no kernel at all.
            logprob_token_ids = sampled_token_ids.unsqueeze(-1)
        else:
            logprob_token_ids = torch.empty(
                (batch_size, 1 + num_logprobs),
                dtype=torch.int64,
                device=logits.device,
            )
            _topk_ids_kernel[(batch_size,)](
                logprob_token_ids,
                logprob_token_ids.stride(0),
                logits,
                logits.stride(0),
                sampled_token_ids,
                vocab_size,
                TOPK=num_logprobs,
                WRITE_SAMPLED=True,
                BLOCK=max(_TOPK_BLOCK, triton.next_power_of_2(num_logprobs)),
                KP=triton.next_power_of_2(num_logprobs),
            )
        if logits_mode:
            scores = _gather_logits(logits, logprob_token_ids, None)
        else:
            scores = compute_token_logprobs(logits, logprob_token_ids)
    else:
        # Some requests specified logprob_token_ids. Build the [batch_size,
        # 1 + max_cols] token_ids matrix and validity mask on the GPU via a
        # single triton kernel, overriding the topk columns with per-request
        # tokens where applicable.
        assert logprob_token_ids_state is not None
        assert expanded_idx_mapping is not None

        if num_logprobs > 0:
            topk_token_ids = torch.empty(
                (batch_size, num_logprobs),
                # Match upstream and the custom-token buffer: the fill
                # kernel's dynamic source-pointer branches must agree.
                dtype=torch.int32,
                device=logits.device,
            )
            _topk_ids_kernel[(batch_size,)](
                topk_token_ids,
                topk_token_ids.stride(0),
                logits,
                logits.stride(0),
                sampled_token_ids,
                vocab_size,
                TOPK=num_logprobs,
                WRITE_SAMPLED=False,
                BLOCK=max(_TOPK_BLOCK, triton.next_power_of_2(num_logprobs)),
                KP=triton.next_power_of_2(num_logprobs),
            )
        else:
            # This tensor just used as an int32 pointer, data not accessed.
            topk_token_ids = logprob_token_ids_state.token_ids.gpu

        num_cols = max(num_logprobs, max_per_req_token_ids)
        logprob_token_ids = sampled_token_ids.new_empty(
            (batch_size, 1 + num_cols)
        )
        valid_mask = torch.empty(
            (batch_size, 1 + num_cols), dtype=torch.bool, device=logits.device
        )
        _fill_logprob_token_ids_kernel[(batch_size,)](
            logprob_token_ids,
            logprob_token_ids.stride(0),
            valid_mask,
            valid_mask.stride(0),
            sampled_token_ids,
            topk_token_ids,
            topk_token_ids.stride(0),
            expanded_idx_mapping,
            logprob_token_ids_state.num_token_ids.gpu,
            logprob_token_ids_state.token_ids.gpu,
            logprob_token_ids_state.token_ids.gpu.stride(0),
            NUM_TOPK=num_logprobs,
            NUM_COLS=num_cols,
            PADDED_COLS=triton.next_power_of_2(num_cols),
        )
        if logits_mode:
            scores = _gather_logits(logits, logprob_token_ids, valid_mask)
        else:
            scores = compute_token_logprobs(logits, logprob_token_ids)
            _mask_invalid_scores(scores, valid_mask)

    token_ranks = torch.empty(batch_size, dtype=torch.int64, device=logits.device)
    _ranks_kernel[(batch_size,)](
        token_ranks,
        logits,
        logits.stride(0),
        sampled_token_ids,
        vocab_size,
        BLOCK_SIZE=8192,  # type: ignore
    )
    is_tensor = isinstance(cu_num_logits, torch.Tensor)
    return LogprobsTensors(
        logprob_token_ids=logprob_token_ids,
        logprobs=scores,
        selected_token_ranks=token_ranks,
        cu_num_generated_tokens=None if is_tensor else cu_num_logits,
        cu_num_generated_tokens_tensor=cu_num_logits if is_tensor else None,
    )


@triton.jit
def _fill_logprob_token_ids_kernel(
    # [batch_size, 1 + num_cols]
    out_token_ids_ptr,
    out_token_ids_stride,
    # [batch_size, 1 + num_cols]
    out_valid_mask_ptr,
    out_valid_mask_stride,
    sampled_token_ids_ptr,  # [batch_size]
    topk_indices_ptr,  # [batch_size, NUM_TOPK] (unused when NUM_TOPK == 0)
    topk_indices_stride,
    expanded_idx_mapping_ptr,  # [batch_size] -> req_state_idx
    num_per_req_token_ids_ptr,  # [max_num_reqs]
    per_req_token_ids_ptr,  # [max_num_reqs, MAX_LOGPROB_TOKEN_IDS]
    per_req_token_ids_stride,
    NUM_TOPK: tl.constexpr,
    NUM_COLS,
    PADDED_COLS: tl.constexpr,
):
    batch_idx = tl.program_id(0)

    sampled = tl.load(sampled_token_ids_ptr + batch_idx)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + batch_idx)
    num_custom = tl.load(num_per_req_token_ids_ptr + req_state_idx)

    col = tl.max_contiguous(
        tl.multiple_of(tl.arange(0, PADDED_COLS), PADDED_COLS), PADDED_COLS
    )
    if num_custom > 0:
        src = per_req_token_ids_ptr + req_state_idx * per_req_token_ids_stride
        valid = col < num_custom
    else:
        src = topk_indices_ptr + batch_idx * topk_indices_stride
        valid = col < NUM_TOPK

    tokens = tl.load(src + col, mask=valid, other=0).to(tl.int64)

    tl.store(out_token_ids_ptr + batch_idx * out_token_ids_stride, sampled)
    tl.store(out_valid_mask_ptr + batch_idx * out_valid_mask_stride, 1)
    # Every remaining column is written, so callers may hand in uninitialised
    # buffers instead of pre-zeroed ones.
    in_range = col < NUM_COLS
    tl.store(
        out_token_ids_ptr + batch_idx * out_token_ids_stride + 1 + col,
        tokens,
        mask=in_range,
    )
    tl.store(
        out_valid_mask_ptr + batch_idx * out_valid_mask_stride + 1 + col,
        valid,
        mask=in_range,
    )


class LogprobTokenIdsState:
    """Per-request override of which token ids' logprobs to return.

    See `SamplingParams.logprob_token_ids`.
    """

    def __init__(self, max_num_reqs: int, device: torch.device):
        self.max_num_reqs = max_num_reqs
        self.num_token_ids = UvaBackedTensor(max_num_reqs, dtype=torch.int32)
        self.token_ids = StagedWriteTensor(
            (max_num_reqs, MAX_LOGPROB_TOKEN_IDS),
            dtype=torch.int32,
            device=device,
        )

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> None:
        token_ids = sampling_params.logprob_token_ids
        if not token_ids:
            self.num_token_ids.np[req_idx] = 0
            return
        n = len(token_ids)
        if n > MAX_LOGPROB_TOKEN_IDS:
            raise ValueError(
                f"Too many logprob_token_ids: {n}. The max is {MAX_LOGPROB_TOKEN_IDS}."
            )
        self.num_token_ids.np[req_idx] = n
        self.token_ids.stage_write(req_idx, 0, token_ids)

    def apply_staged_writes(self) -> None:
        self.num_token_ids.copy_to_uva()
        self.token_ids.apply_write()

    def max_num_token_ids(self, idx_mapping_np: np.ndarray) -> int:
        return int(self.num_token_ids.np[idx_mapping_np].max(initial=0))
