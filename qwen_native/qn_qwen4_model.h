#ifndef QN_QWEN4_MODEL_H
#define QN_QWEN4_MODEL_H
#include <stddef.h>
#include <stdint.h>
typedef struct qn_qwen4_model qn_qwen4_model;
typedef struct {
    const char *model_dir;
    const char *manifest_path;
    const char *ngram_table_path;
    uint32_t qsa_cache_capacity; /* initial tokens/layer; 0 => 16, grows on demand */
} qn_qwen4_model_config;
typedef struct {
    uint32_t next_token;
    double trunk_ms;
    double logits_ms;
    double total_ms;
} qn_qwen4_step_output;
typedef struct {
    uint32_t num_layers;
    uint32_t qsa_layers;
    uint32_t gdn_layers;
    uint64_t gdn_conv_state_bytes;
    uint64_t gdn_recurrent_state_bytes;
    uint64_t ple_state_bytes;
    uint32_t opened_qsa_layers;
    uint32_t opened_gdn_layers;
    int ple_opened;
    uint32_t qsa_cache_capacity;
    uint64_t qsa_cache_bytes;
    uint32_t position;
} qn_qwen4_model_stats;
int qn_qwen4_model_open(qn_qwen4_model **out,const qn_qwen4_model_config *cfg,char *err,size_t errlen);
int qn_qwen4_model_ensure_layer(qn_qwen4_model *m,uint32_t layer_index,char *err,size_t errlen);
int qn_qwen4_model_ensure_ple(qn_qwen4_model *m,char *err,size_t errlen);
int qn_qwen4_model_seed_qsa_cache(qn_qwen4_model *m,uint32_t layer_index,uint32_t tokens,const float *index_raw,const float *keys,const float *values,char *err,size_t errlen);
int qn_qwen4_model_forward_token(qn_qwen4_model *m,uint32_t token_id,float hyper_state[10240],char *err,size_t errlen);
int qn_qwen4_model_step(qn_qwen4_model *m,uint32_t token_id,float *logits,qn_qwen4_step_output *out,char *err,size_t errlen);
void qn_qwen4_model_set_position(qn_qwen4_model *m,uint32_t position);
void qn_qwen4_model_get_stats(qn_qwen4_model *m,qn_qwen4_model_stats *stats);
void qn_qwen4_model_close(qn_qwen4_model *m);
#endif
