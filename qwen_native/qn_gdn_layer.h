#ifndef QN_GDN_LAYER_H
#define QN_GDN_LAYER_H
#include <stddef.h>
#include <stdint.h>
typedef struct qn_gdn_layer qn_gdn_layer;
typedef struct { const char *model_dir; const char *manifest_path; uint32_t layer_index; } qn_gdn_layer_config;
typedef struct { const float *hyper_state; float *conv_state; float *recurrent_state; } qn_gdn_decode_input;
typedef struct { float *hyper_state; double elapsed_ms; } qn_gdn_decode_output;
int qn_gdn_layer_open(qn_gdn_layer **out,const qn_gdn_layer_config *cfg,char *err,size_t errlen);
int qn_gdn_layer_forward_attention(qn_gdn_layer *l,const qn_gdn_decode_input *in,qn_gdn_decode_output *out,char *err,size_t errlen);
void qn_gdn_layer_close(qn_gdn_layer *l);
#endif
