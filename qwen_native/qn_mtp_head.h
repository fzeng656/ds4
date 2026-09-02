#ifndef QN_MTP_HEAD_H
#define QN_MTP_HEAD_H
#include <stddef.h>
#include <stdint.h>
#include "qn_model_io.h"

typedef struct qn_mtp_head qn_mtp_head;
typedef struct {
    const char *model_dir;
    const char *manifest_path;
    qn_model_io *io;
} qn_mtp_head_config;

typedef struct {
    uint32_t draft_token;
    double preprocess_ms;
    double layer_ms;
    double mixer_lmhead_ms;
    double total_ms;
} qn_mtp_head_output;

int qn_mtp_head_open(qn_mtp_head **out,const qn_mtp_head_config *cfg,char *err,size_t errlen);
int qn_mtp_head_reset(qn_mtp_head *h);
int qn_mtp_head_truncate(qn_mtp_head *h,uint32_t seq_offset);
int qn_mtp_head_append_history(qn_mtp_head *h,const float stream_prev[10240],uint32_t token_id,uint32_t abs_position,double *elapsed_ms,char *err,size_t errlen);
int qn_mtp_head_forward(qn_mtp_head *h,const float stream_prev[10240],uint32_t token_id,uint32_t abs_position,float stream_out[10240],qn_mtp_head_output *out,char *err,size_t errlen);
uint32_t qn_mtp_head_seq_offset(qn_mtp_head *h);
void qn_mtp_head_close(qn_mtp_head *h);
#endif
