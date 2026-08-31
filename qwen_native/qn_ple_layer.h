#ifndef QN_PLE_LAYER_H
#define QN_PLE_LAYER_H
#include <stddef.h>
#include <stdint.h>
typedef struct qn_ple_layer qn_ple_layer;
typedef struct { const char *model_dir; const char *manifest_path; const char *ngram_table_path; } qn_ple_config;
typedef struct { const float *hyper_state; int64_t token_history[3]; float *conv_state; } qn_ple_input;
typedef struct { float *ple_output; int64_t ngram_ids[16]; double elapsed_ms; } qn_ple_output;
int qn_ple_layer_open(qn_ple_layer **out,const qn_ple_config *cfg,char *err,size_t errlen);
int qn_ple_layer_forward(qn_ple_layer *l,const qn_ple_input *in,qn_ple_output *out,char *err,size_t errlen);
void qn_ple_layer_close(qn_ple_layer *l);
#endif
