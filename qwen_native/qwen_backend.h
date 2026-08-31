#ifndef QWEN_NATIVE_BACKEND_H
#define QWEN_NATIVE_BACKEND_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* Phase-0 contract only. No model execution is wired yet. */
typedef enum {
    QN_DTYPE_BF16 = 1,
    QN_DTYPE_U32_PACKED = 2,
} qn_dtype;

typedef struct {
    const char *name;
    const char *shard_path;
    qn_dtype dtype;
    uint64_t file_offset;
    uint64_t bytes;
    uint32_t ndim;
    uint64_t shape[4];
} qn_tensor_desc;

typedef enum {
    QN_LAYER_LINEAR_ATTN = 1,
    QN_LAYER_FULL_ATTN = 2,
} qn_layer_type;

typedef struct {
    uint32_t hidden_size;
    uint32_t num_layers;
    uint32_t num_attention_heads;
    uint32_t num_kv_heads;
    uint32_t head_dim;
    uint32_t num_experts;
    uint32_t experts_per_token;
    uint32_t moe_intermediate_size;
    uint32_t max_context;
} qn_model_shape;

/* Runtime boundary intended for Phase 1:
 *   safetensors MAP_SHARED -> page-aligned Metal no-copy views -> tensor views.
 * Model math must depend on this interface, not on GGUF internals. */
typedef struct qn_weight_store qn_weight_store;

int qn_weight_store_open(qn_weight_store **out, const char *model_dir,
                         char *err, size_t errlen);
void qn_weight_store_close(qn_weight_store *store);
int qn_weight_store_find(qn_weight_store *store, const char *name,
                         qn_tensor_desc *out);
int qn_weight_store_advise_layer(qn_weight_store *store, uint32_t layer,
                                 bool will_need);

#endif
