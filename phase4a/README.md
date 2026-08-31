# Phase 4A — model-level I/O

Validated native model entry/exit for the Qwen4Exp text model:

- `embed_tokens`: q4/group64 mmap row lookup, 248320 x 2560, exact against MLX-pack dequantization.
- final `hyper_connection_mixer`: four 2560 residual streams -> one 2560 hidden vector, using the same low-rank Hyper-Connection math validated at layer level.
- `lm_head`: q8/group64 mmap-backed Metal matvec, 2560 -> 248320 logits.
- Full logits cosine = 1.0 and argmax token matches the oracle.

MTP is intentionally excluded from the minimum viable main-token decode path and can be attached after base-model correctness and scheduling are stable.

## Phase 4A-2 — full decode trunk

The model owner now runs a complete greedy decode step:

`token -> q4 embedding -> 48 decoder layers -> final Hyper-Connection mixer -> q8 lm_head -> argmax`

Runtime state is owned at model scope:

- 36 GDN conv + recurrent states (~114 MiB fixed state including PLE state).
- 12 QSA index/K/V caches, allocated lazily per layer and grown geometrically.
- L1 PLE two-token lexical history plus dilated-conv state.
- QSA positions 0-2 use the official causal-tail behavior before the first complete 4-token index block exists.

### MLX-pack routing compatibility

The current `mlx-serve` Qwen4Exp pack renormalizes the selected top-10 MoE probabilities to sum to 1 before expert reduction. This differs from a literal reading of the current Transformers `norm_topk_prob=None` path, but it is required to reproduce the deployed MLX-pack model. The native QSA and GDN MoE paths therefore apply the same selected-top-k renormalization.

Strict raw-completion reference (`prompt_tokens=1`, input token 12345 / text `" Van"`, greedy):

`12345 -> 10655 -> 4963 -> 318 -> 16 -> 24`

The native model-level decode path reproduces all five generated token IDs exactly.
