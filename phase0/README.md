# Qwen Native Runtime — Phase 0

Baseline: antirez/ds4 upstream main at the worktree creation point.
Goal: validate a clean architecture boundary before implementing Qwen math.

## Decision

Do not convert the current MLX Qwen4-Exp weights to GGUF for the first PoC.
Use the existing 101-shard safetensors model directly:

    safetensors shard (MAP_SHARED, PROT_READ)
      -> page-aligned Metal newBufferWithBytesNoCopy
      -> lightweight tensor view (file offset + shape + dtype)
      -> Qwen layer implementation

This preserves the exact current 4-bit representation (U32 packed weights +
BF16 scales/biases), avoids a second quantization/serialization variable, and
lets mlx-serve remain the numerical oracle.

## DS4 code boundary found

Reusable runtime mechanisms:
- ds4.c model mmap path around the GGUF mapping code (MAP_SHARED for Metal).
- ds4_metal.m model mapped views via newBufferWithBytesNoCopy.
- Metal residency-set lifecycle and requestResidency.
- transient/model buffer cache cleanup and POSIX_MADV_DONTNEED paths.
- later: session/KV persistence and server/SSE keepalive.

DeepSeek/GLM-specific code that must not leak into the Qwen backend:
- GGUF metadata/model-profile validation.
- fixed tensor-name -> ds4 layer pointer binding.
- DeepSeek/GLM graph scheduling and expert-layout assumptions.

## Qwen model facts from current production weights

- model_type: qwen4_exp / qwen4_exp_text
- 48 layers, hidden size 2560, max context 262144.
- 24 attention heads, 2 KV heads, head_dim 256.
- 512 routed experts, top-10, routed intermediate 640, shared expert 640.
- Layer pattern: 3 linear-attention layers then 1 full-attention/QSA layer.
- 36 linear/GDN layers; 12 full/QSA layers (3,7,11,...,47).
- PLE tensors are present around the early layer path.
- 3167 tensors across 101 safetensors shards.
- Current quantized matrices are stored as U32 packed weights plus BF16
  scales/biases. Headers provide exact payload offsets without reading weights.

## Phase 1 target

First target should be one real full-attention layer (L11) because it exercises:
1. hyper-connection input/output path,
2. Q/K/V/O quantized projections,
3. QSA indexer tensors,
4. MoE/shared-expert tensors,
while avoiding PLE-specific early-layer behavior.

Before full L11 execution, implement and test only:
1. safetensors shard mmap;
2. page-aligned no-copy MTLBuffer creation;
3. tensor subviews using manifest offsets;
4. one quantized projection + BF16 scale/bias correctness against MLX;
5. release residency + madvise(DONTNEED), then measure physical footprint.

Pass criteria for the first memory PoC:
- no tensor payload copy during load;
- output matches MLX oracle within quantization tolerance;
- pages become resident on execution;
- idle physical footprint drops materially without closing the logical model;
- rerun faults pages back and produces identical output.

## Phase 1A result (2026-08-31)

The first real Qwen weight path is validated without materializing the model in
MLX. `layers.11.self_attn.q_proj` is an 8-bit affine non-expert projection
(group size 64), despite the model-wide expert default being 4-bit. Its
safetensors payload is mapped with `MAP_SHARED`, exposed to Metal with
`newBufferWithBytesNoCopy`, and consumed directly by a correctness-first
Metal matrix-vector kernel.

Reference: MLX `quantized_matmul(... bits=8, group_size=64, mode="affine")`.
For a deterministic BF16 input vector, the native mmap path reached cosine
0.99999859 with FP32 accumulation. This validates the storage offsets, packed
U32 byte order, BF16 scales/biases, and direct file-backed Metal access.

Phase 1B should separate the mmap/file-backing primitives from the probe and
implement an optimized Q8 affine matmul path, then measure page residency and
reclaim behavior under repeated map/run/release cycles.

## Phase 1C checkpoint

The mmap-backed runtime primitive now owns shard mappings and strong Metal no-copy
views. L11 q/k/v/o affine-8 projections all execute directly from the existing
safetensors payload without MLX array materialization.

Correctness against MLX 0.32.1 `quantized_matmul` on fixed inputs:
- q/k/v/o cosine: 1.000000000000
- max absolute error: 5.96e-8 to 1.49e-7

Correctness-first native steady wall latency (one command buffer per projection):
- q: ~0.56 ms
- k: ~0.44 ms
- v: ~0.44 ms
- o: ~0.49 ms

MLX median reference on the same tensors/inputs:
- q: ~0.42 ms
- k: ~0.25 ms
- v: ~0.26 ms
- o: ~0.28 ms

The remaining gap is small enough to proceed; next test is scheduling/fusion before
replacing the scalar-per-row correctness kernel with a DS4-style tiled matmul.
