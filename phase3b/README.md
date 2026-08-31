# Phase 3B — Qwen4Exp GDN native runtime

## Status

- Gated DeltaNet decode recurrence implemented in Metal.
- Q/K L2 normalization + 1/sqrt(128) scale matches FLA recurrent reference.
- Depthwise causal conv width=4 state update implemented.
- Real q8 projections, A_log/dt/beta gates, RMSNormGated, out projection, and attention Hyper-Connection validated.
- Manifest-backed per-tensor shard resolver supports GDN tensors split across shard boundaries (notably layer 20).
- All 36 linear-attention layers pass attention-half E2E with conv/recurrent cache updates.
- All 36 linear-attention layers also pass complete decoder-layer E2E: GDN + both Hyper-Connections + routed/shared MoE.

## Architecture

- 16 key heads x 128, repeated 3x to 48 value heads.
- 48 value heads x 128.
- conv channels: 10240, width 4.
- recurrent state: 48 x 128 x 128 FP32 (~3 MiB/layer).
- conv state: 10240 x 4 FP32 (~160 KiB/layer).
- 36-layer recurrent state budget ~108 MiB FP32 plus ~5.6 MiB conv state.

## Validation

Sweep: layers 0-47 excluding full-attention layers 3,7,...,47.
All 36 layers: output cosine=1.0, conv-state cosine=1.0, recurrent-state cosine=1.0.
Worst observed attention-half max abs error: ~1.97e-6 (L46).

Special topology cases:
- L1 includes PLE tensors; GDN attention was validated on the post-PLE residual-state boundary.
- L20 GDN tensors span two shards; per-tensor resolver correctly maps them.

Complete GDN decoder-layer sweep: router top-10 mismatch=0 for all 36 layers; final cosine=1.0 for all. Worst final max abs error ~1.97e-6 (L46).

Next: handle L1 PLE separately, then assemble the 48-layer trunk and model-level embedding/final mixer/lm-head path.

## L1 PLE special case

- `ple_layer_ids=[2]` maps to zero-based decoder layer 1 only.
- 16 hashed n-gram heads are resolved directly from the mmap-backed `ngram_table.bin`; lookup is bit-exact against the MLX oracle.
- Decode PLE state is two prior token ids plus a 10240 x 9 dilated-conv state.
- key/value projection, grouped RMSNorm, signed-sqrt gate, normalized dilated conv, and residual PLE output are validated.
- PLE block output max abs error observed: ~4.47e-8 on the L1 composition input.
- With this special case validated, all 48 decoder-layer structures are covered by native math.
