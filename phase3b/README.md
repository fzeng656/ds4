# Phase 3B — Qwen4Exp GDN native runtime

## Status

- Gated DeltaNet decode recurrence implemented in Metal.
- Q/K L2 normalization + 1/sqrt(128) scale matches FLA recurrent reference.
- Depthwise causal conv width=4 state update implemented.
- Real q8 projections, A_log/dt/beta gates, RMSNormGated, out projection, and attention Hyper-Connection validated.
- Manifest-backed per-tensor shard resolver supports GDN tensors split across shard boundaries (notably layer 20).
- All 36 linear-attention layers pass attention-half E2E with conv/recurrent cache updates.

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

Next: combine GDN attention-half with MLP Hyper-Connection + MoE for complete linear-attention decoder layers, then handle L1 PLE separately.
