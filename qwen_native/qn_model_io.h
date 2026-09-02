#ifndef QN_MODEL_IO_H
#define QN_MODEL_IO_H
#include <stddef.h>
#include <stdint.h>
typedef struct qn_model_io qn_model_io;
typedef struct { const char *model_dir; const char *manifest_path; } qn_model_io_config;
int qn_model_io_open(qn_model_io **out,const qn_model_io_config *cfg,char *err,size_t errlen);
int qn_model_io_embed(qn_model_io *m,uint32_t token_id,float out[2560],char *err,size_t errlen);
int qn_model_io_logits(qn_model_io *m,const float hyper_state[10240],float *logits,uint32_t *argmax_token,double *elapsed_ms,char *err,size_t errlen);
int qn_model_io_lm_head(qn_model_io *m,const float hidden[2560],float *logits,uint32_t *argmax_token,double *elapsed_ms,char *err,size_t errlen);
void qn_model_io_close(qn_model_io *m);
#endif
