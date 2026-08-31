# Phase 4A — model-level I/O

Validated native model entry/exit for the Qwen4Exp text model:

- `embed_tokens`: q4/group64 mmap row lookup, 248320 x 2560, exact against MLX-pack dequantization.
- final `hyper_connection_mixer`: four 2560 residual streams -> one 2560 hidden vector, using the same low-rank Hyper-Connection math validated at layer level.
- `lm_head`: q8/group64 mmap-backed Metal matvec, 2560 -> 248320 logits.
- Full logits cosine = 1.0 and argmax token matches the oracle.

MTP is intentionally excluded from the minimum viable main-token decode path and can be attached after base-model correctness and scheduling are stable.
