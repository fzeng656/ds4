# L1 PLE native decode

Qwen4Exp has one Per-Layer Embedding (PLE) injection at decoder layer index 1 (`ple_layer_ids=[2]`).

Native decode path validated:

1. bigram/trigram hash from the latest three token IDs; 8 heads each, 16 total,
2. mmap random lookup from `ngram_table.bin`,
3. q4/group32 row dequantization (16 rows x 160 dims -> 2560),
4. q8 key/value projections,
5. four-stream group RMSNorm and signed-sqrt gate,
6. dilated depthwise conv (kernel=4, dilation=3, state length=9),
7. add PLE output before layer-1 attention Hyper-Connection.

`ngram_table.bin` is ~30 GB and is not expanded: 320,001,536 rows, q4 group32, 160 dims/row. Only 16 rows are touched per decoded token.

Validation:
- mmap q4 lookup: exact (0 error),
- complete PLE output max abs error ~5.96e-8,
- PLE conv cache max abs error ~1.91e-6,
- integrated L1 PLE + GDN + Hyper + MoE: router mismatch 0, final cosine 1.0, max abs error ~8.64e-7.
