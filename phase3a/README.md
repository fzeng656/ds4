# Phase 3A — Manifest-backed QSA layer runtime

The native Qwen4 full-attention runtime no longer depends on Layer 11 file offsets or shard numbers. `qn_qwen4_layer_open()` can resolve a QSA layer from `model_dir + manifest_path + layer_index`, mmap only the three role shards used by that layer, and build all 57 tensor views from manifest metadata.

All 12 QSA layers were validated end-to-end at decode position 2560 against an MLX oracle, including Hyper-Connection, QSA routing with causal tail, sparse GQA, MoE top-10 routing, routed 4-bit experts, shared expert, and the second Hyper-Connection.

| Layer | Role shards | Router mismatch | Max abs error | Cosine |
|---:|---|---:|---:|---:|
| 3 | 49/50/51 | 0 | 2.68e-7 | 1.0 |
| 7 | 93/94/95 | 0 | 1.27e-7 | 1.0 |
| 11 | 8/9/10 | 0 | 5.96e-7 | 1.0 |
| 15 | 16/17/18 | 0 | 3.13e-7 | 1.0 |
| 19 | 24/25/26 | 0 | 2.09e-7 | 1.0 |
| 23 | 35/36/37 | 0 | 4.47e-7 | 1.0 |
| 27 | 43/44/45 | 0 | 7.75e-7 | 1.0 |
| 31 | 53/54/55 | 0 | 1.85e-7 | 1.0 |
| 35 | 61/62/63 | 0 | 1.49e-7 | 1.0 |
| 39 | 69/70/71 | 0 | 8.94e-8 | 1.0 |
| 43 | 79/80/81 | 0 | 2.98e-8 | 1.0 |
| 47 | 87/88/89 | 0 | 7.30e-7 | 1.0 |

Each QSA layer uses the same three-shard role pattern: A = attention Hyper-Connection, B = routed expert gate/up banks, C = attention + router + expert down + shared expert + MLP Hyper-Connection.
