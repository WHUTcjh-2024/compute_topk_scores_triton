# compute_topk_scores — Pure Triton 实现

vLLM 0.29.0 `compute_topk_scores` 路径的 Pure Triton 实现：执行路径上不再有任何
PyTorch CUDA 算子参与计算。

- Reference：`vllm==0.29.0`，`vllm/v1/worker/gpu/sample/logprob.py`
- Candidate：`src/h20_topk/ops/compute_topk_scores.py`
- Target GPU：NVIDIA H20 / H100（SM90）

## 执行路径

```text
_topk_ids_kernel            Top-K 选择 + sampled token 列写入
  -> _topk_log_softmax_kernel   online log-sum-exp + 按 id gather logprob
  -> _ranks_kernel              sampled token 的 1-based rank
  -> _fill_logprob_token_ids_kernel / _gather_logits_kernel / _mask_scores_kernel
```

PyTorch 只用于分配（`empty` / `new_empty`）、读取 stride 和构造
`LogprobsTensors` 容器，不产生任何计算 kernel。

| 上游调用 | 现在 |
| --- | --- |
| `torch.topk(...).indices` | `_topk_ids_kernel` |
| `torch.cat` | 由 `_topk_ids_kernel` 直接写入第 0 列 |
| `logits.gather(-1, ids).to(float32)` | `_gather_logits_kernel` |
| `torch.zeros_like` / `new_zeros` | `_fill_logprob_token_ids_kernel` 全列写入 |
| `masked_fill(~valid_mask, -inf)` | `_gather_logits_kernel` / `_mask_scores_kernel` |
| `.to(torch.int32)` / `.to(torch.int64)` | kernel 内类型处理 |

## Kernel 说明

### `_topk_ids_kernel`

一行 vocab 一个 program，单遍流式选 Top-K：

- 把 fp32 logit 的 IEEE-754 位模式折叠成保序 uint32，再把 index 放进低 32 位，
  组成可排序的 int64 key，因此排序同时得到值与下标，不需要第二遍扫描求值。
- 每 `BLOCK` 个元素做一次块内排序，与 running top-K 做一次合并排序。
- 当整块的最大 key 都进不了 running 集合时，整块跳过排序。阈值稳定后大部分块
  不再进入排序。
- 支持把 sampled token 写成第 0 列（`WRITE_SAMPLED`），免掉一次 `torch.cat`。

### `_topk_log_softmax_kernel`

沿用已验证的 online log-sum-exp：单遍扫描同时维护 running max 与求和的
重标定，full-block / tail 拆分，`num_stages=2`、`loop_unroll_factor=2`。
`normalizer` 处理保证首段全 `-inf` 时不产生 NaN。

### `_ranks_kernel`

沿用已验证实现：`n = Σ(logits >= x)`，`x` 为 sampled token 的 logit，
full-block / tail 拆分，`num_stages=4`。

### `_gather_logits_kernel` / `_mask_scores_kernel` / `_fill_logprob_token_ids_kernel`

`logits_mode` 下按 id gather 并直接输出 fp32；有 per-request 自定义 token ids 时
由 `_fill_logprob_token_ids_kernel` 一次性写满整行（含 padding 的 0 / `False`），
因此调用方可以传入未初始化的缓冲区。

## 正确性

GPU 正确性测试保持原样，参考实现就是 vLLM 自己的 `compute_topk_scores`：

```bash
python -m pytest tests/test_compute_topk_scores.py -m compute_topk_scores -q
```

```bash
QUICK_MODE=1 python -m pytest tests/test_compute_topk_scores.py -m compute_topk_scores -q
```

```bash
python -m pytest tests/test_log_softmax_cpu.py -m compute_topk_scores -q
```

> 截至本次提交，Pure Triton 版本**尚未在 GPU 上运行**。改动只经过了 Triton
> CPU interpreter（`TRITON_INTERPRET=1`）与上游 torch 参考实现的逐项对比：
> 3 种 dtype × 5 种 vocab × 4 种 K、非连续行 stride、ties、首段全 `-inf`、
> `logits_mode`、custom token ids、int16 sampled ids、`cu_num_logits`、
> `V=8192 / K=50`。interpreter 对 0 维 bf16 标量 load 有已知缺陷，
> `_ranks_kernel` 的 bf16 分支无法在该环境下验证（该 kernel 未改动）。

## 性能

**尚未测量。** 下面这张表是上一版（仍使用 `torch.topk`）在 H20 上的数据，不能
代表当前 Pure Triton 版本，仅作历史参考。

| 指标 | 上一版结果 |
| --- | ---: |
| SpeedUp min | 0.982 |
| SpeedUp median | 1.163 |
| SpeedUp mean | 1.148 |
| SpeedUp max | 1.297 |

复测命令：

```bash
python scripts/bench_matrix.py --target vllm --output artifacts/vllm.json
```

```bash
python scripts/bench_matrix.py --target candidate --output artifacts/candidate.json
```

```bash
python scripts/compare_bench.py --reference artifacts/vllm.json --candidate artifacts/candidate.json
```

## 已知待验证项

- `tl.sort` 在本仓库固定的 `triton==3.7.1` 上对 int64 的支持，以及排序网络的
  实际吞吐；`_TOPK_BLOCK`（默认 1024）尚未在 SM90 上调优。
- `_topk_ids_kernel` 目前与 `_topk_log_softmax_kernel`、`_ranks_kernel` 各自
  完整读取一遍 logits，共三次。融合为单遍扫描（`scripts/ablate_stats.py` 有
  早期原型）是下一步的收益点。

## 复测环境

| 项目 | 值 |
| --- | --- |
| OS | Ubuntu 22.04.5 LTS |
| GPU | NVIDIA H20 |
| Compute Capability | 9.0 |
| NVIDIA Driver | 580.65.06 |
| CUDA Driver | 13.0 |
| CUDA Toolkit | 13.0 |
| Python | 3.12.15 |
| PyTorch | 2.13.0+cu130 |
| Triton | 3.7.1 |
| vLLM | 0.29.0 |
| pytest | 9.1.1 |
