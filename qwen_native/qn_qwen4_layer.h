#ifndef QN_QWEN4_LAYER_H
#define QN_QWEN4_LAYER_H
#include <stddef.h>
#include <stdint.h>

typedef struct qn_qwen4_layer qn_qwen4_layer;

typedef struct {
    const char *shard8_path;
    const char *shard9_path;
    const char *shard10_path;
    uint32_t layer_index;
    const char *model_dir;      /* Phase3A: preferred manifest-backed path */
    const char *manifest_path;  /* phase0 qwen38fn_manifest.json */
    const char *tensor_prefix;  /* optional exact prefix, e.g. language_model.mtp.layers.0 */
} qn_qwen4_layer_config;

typedef struct {
    const float *hyper_state;          /* 4 x 2560 */
    const float *indexer_raw_history; /* (position) x 128 */
    const float *key_history;         /* (position) x 2 x 256 */
    const float *value_history;       /* (position) x 2 x 256 */
    uint32_t position;                /* zero based cache/history position */
    uint32_t position_base;           /* absolute RoPE base; target QSA uses 0, MTP uses prompt/generation base */
} qn_qwen4_decode_input;

typedef struct {
    float *hyper_state;               /* 4 x 2560 output */
    float *indexer_raw_key;           /* optional 128-float cache append */
    float *key_cache;                 /* optional 2 x 256 cache append */
    float *value_cache;               /* optional 2 x 256 cache append */
    uint32_t selected_expert_ids[10];
    float selected_expert_weights[10];
    double attention_ms;
    double moe_ms;
    double total_ms;
} qn_qwen4_decode_output;

int qn_qwen4_layer_open(qn_qwen4_layer **out,
                        const qn_qwen4_layer_config *config,
                        char *err, size_t errlen);
int qn_qwen4_layer_forward_attention(qn_qwen4_layer *layer,
                                     const qn_qwen4_decode_input *input,
                                     float *hyper_output,
                                     double *elapsed_ms,
                                     char *err, size_t errlen);

int qn_qwen4_layer_forward_moe(qn_qwen4_layer *layer,
                               const float *hidden, float *output,
                               uint32_t selected_ids[10],
                               float selected_weights[10],
                               double *elapsed_ms,
                               char *err, size_t errlen);

int qn_qwen4_layer_forward_prefill_batch(qn_qwen4_layer *layer,
                                           const float *hyper_states,
                                           uint32_t base_position, uint32_t tokens,
                                           float *index_history, float *key_history, float *value_history,
                                           float *hyper_outputs, double *elapsed_ms,
                                           char *err, size_t errlen);

int qn_qwen4_layer_forward_decode(qn_qwen4_layer *layer,
                                  const qn_qwen4_decode_input *input,
                                  qn_qwen4_decode_output *output,
                                  char *err, size_t errlen);
#ifdef __OBJC__
#import <Metal/Metal.h>
void qn_qwen4_layer_set_command_queue(qn_qwen4_layer *layer,id<MTLCommandQueue> queue);
int qn_qwen4_layer_forward_decode_buffer(qn_qwen4_layer *layer,id<MTLBuffer> hyper_buffer,const qn_qwen4_decode_input *input,qn_qwen4_decode_output *output,char *err,size_t errlen);
int qn_qwen4_layer_forward_decode_buffer_submit(qn_qwen4_layer *layer,id<MTLBuffer> hyper_buffer,const qn_qwen4_decode_input *input,id<MTLBuffer> *gpu_out,id<MTLCommandBuffer> *submitted,char *err,size_t errlen);
int qn_qwen4_layer_copy_current_cache(qn_qwen4_layer *layer,float *index_raw,float *key,float *value);
#endif
int qn_qwen4_layer_warm_mps64(qn_qwen4_layer *l,char *err,size_t errlen);
int qn_qwen4_layer_prefill_mps_enabled(qn_qwen4_layer *l);
int qn_qwen4_layer_moe_bm32_enabled(qn_qwen4_layer *l);
void qn_qwen4_layer_close(qn_qwen4_layer *layer);

#endif
