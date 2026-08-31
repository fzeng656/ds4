#include "qn_qwen4_model.h"
#include "qn_qwen4_layer.h"
#include "qn_gdn_layer.h"
#include "qn_ple_layer.h"
#include "qn_model_io.h"
#include <mach/mach_time.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

struct qn_qwen4_model {
    char model_dir[1024], manifest_path[1024], ngram_path[1024];
    qn_qwen4_layer *qsa[48];
    qn_gdn_layer *gdn[48];
    qn_ple_layer *ple;
    qn_model_io *io;
    float *qsa_index[48], *qsa_key[48], *qsa_value[48];
    size_t qsa_cap[48];
    float *ple_conv;
    int64_t ple_token_history[2];
    uint32_t opened_qsa, opened_gdn, position, qsa_initial_capacity;
    uint64_t qsa_cache_bytes;
};

static void E(char *e,size_t n,const char *s){if(e&&n)snprintf(e,n,"%s",s?s:"model runtime error");}
static double msnow(void){static mach_timebase_info_data_t tb;if(!tb.denom)mach_timebase_info(&tb);return (double)mach_absolute_time()*tb.numer/tb.denom/1e6;}
static int is_qsa(uint32_t i){return i<48 && (i&3u)==3u;}
static uint64_t qsa_bytes_for(size_t cap){return (uint64_t)cap*(128u+512u+512u)*sizeof(float);}

static int ensure_qsa_cache(qn_qwen4_model *m,uint32_t i,size_t need,char *e,size_t n){
    if(!is_qsa(i)){E(e,n,"QSA cache requested for non-QSA layer");return -1;}
    if(m->qsa_cap[i]>=need)return 0;
    size_t old=m->qsa_cap[i], cap=old?old:m->qsa_initial_capacity;
    while(cap<need){if(cap>SIZE_MAX/2){E(e,n,"QSA cache capacity overflow");return -1;}cap*=2;}
    float *ni=calloc(cap*128,sizeof(float)), *nk=calloc(cap*512,sizeof(float)), *nv=calloc(cap*512,sizeof(float));
    if(!ni||!nk||!nv){free(ni);free(nk);free(nv);E(e,n,"QSA cache allocation failed");return -1;}
    if(old){memcpy(ni,m->qsa_index[i],old*128*sizeof(float));memcpy(nk,m->qsa_key[i],old*512*sizeof(float));memcpy(nv,m->qsa_value[i],old*512*sizeof(float));}
    free(m->qsa_index[i]);free(m->qsa_key[i]);free(m->qsa_value[i]);
    m->qsa_index[i]=ni;m->qsa_key[i]=nk;m->qsa_value[i]=nv;m->qsa_cap[i]=cap;
    m->qsa_cache_bytes-=qsa_bytes_for(old);m->qsa_cache_bytes+=qsa_bytes_for(cap);
    return 0;
}

int qn_qwen4_model_open(qn_qwen4_model **out,const qn_qwen4_model_config *c,char *e,size_t n){
    if(!out||!c||!c->model_dir||!c->manifest_path||!c->ngram_table_path){E(e,n,"invalid model config");return -1;}
    qn_qwen4_model *m=calloc(1,sizeof(*m));if(!m){E(e,n,"calloc model failed");return -1;}
    snprintf(m->model_dir,sizeof(m->model_dir),"%s",c->model_dir);snprintf(m->manifest_path,sizeof(m->manifest_path),"%s",c->manifest_path);snprintf(m->ngram_path,sizeof(m->ngram_path),"%s",c->ngram_table_path);m->qsa_initial_capacity=c->qsa_cache_capacity?c->qsa_cache_capacity:16;
    m->ple_conv=calloc((size_t)10240*9,sizeof(float));if(!m->ple_conv){E(e,n,"PLE state allocation failed");qn_qwen4_model_close(m);return -1;}
    m->ple_token_history[0]=m->ple_token_history[1]=248044; /* model EOS */
    qn_model_io_config io={m->model_dir,m->manifest_path};if(qn_model_io_open(&m->io,&io,e,n)){qn_qwen4_model_close(m);return -1;}
    *out=m;return 0;
}

int qn_qwen4_model_ensure_layer(qn_qwen4_model *m,uint32_t i,char *e,size_t n){
    if(!m||i>=48){E(e,n,"invalid layer index");return -1;}
    if(is_qsa(i)){if(m->qsa[i])return 0;qn_qwen4_layer_config c={0};c.layer_index=i;c.model_dir=m->model_dir;c.manifest_path=m->manifest_path;if(qn_qwen4_layer_open(&m->qsa[i],&c,e,n))return -1;m->opened_qsa++;return 0;}
    if(m->gdn[i])return 0;qn_gdn_layer_config c={m->model_dir,m->manifest_path,i};if(qn_gdn_layer_open(&m->gdn[i],&c,e,n))return -1;m->opened_gdn++;return 0;
}
int qn_qwen4_model_ensure_ple(qn_qwen4_model *m,char *e,size_t n){if(!m){E(e,n,"invalid model");return -1;}if(m->ple)return 0;qn_ple_config c={m->model_dir,m->manifest_path,m->ngram_path};return qn_ple_layer_open(&m->ple,&c,e,n);}
void qn_qwen4_model_set_position(qn_qwen4_model *m,uint32_t p){if(m)m->position=p;}

int qn_qwen4_model_seed_qsa_cache(qn_qwen4_model *m,uint32_t i,uint32_t tokens,const float *idx,const float *k,const float *v,char *e,size_t n){
    if(!m||!is_qsa(i)||(!idx&&tokens)||(!k&&tokens)||(!v&&tokens)){E(e,n,"invalid QSA cache seed");return -1;}if(ensure_qsa_cache(m,i,tokens?tokens:1,e,n))return -1;if(tokens){memcpy(m->qsa_index[i],idx,(size_t)tokens*128*4);memcpy(m->qsa_key[i],k,(size_t)tokens*512*4);memcpy(m->qsa_value[i],v,(size_t)tokens*512*4);}return 0;
}

static int run_trunk(qn_qwen4_model *m,uint32_t token,float h[10240],char *e,size_t n){
    float next[10240];
    int profile=getenv("QN_MODEL_PROFILE")!=NULL; double gdn_ms=0,qsa_ms=0,ple_ms=0;
    for(uint32_t i=0;i<48;i++){
        if(i==1){if(qn_qwen4_model_ensure_ple(m,e,n))return -1;float po[10240];qn_ple_input pi={0};pi.hyper_state=h;pi.token_history[0]=m->ple_token_history[0];pi.token_history[1]=m->ple_token_history[1];pi.token_history[2]=(int64_t)token;pi.conv_state=m->ple_conv;qn_ple_output py={0};py.ple_output=po;double pt=msnow();if(qn_ple_layer_forward(m->ple,&pi,&py,e,n))return -1;ple_ms+=msnow()-pt;for(uint32_t j=0;j<10240;j++)h[j]+=po[j];}
        if(qn_qwen4_model_ensure_layer(m,i,e,n))return -1;
        if(is_qsa(i)){
            if(ensure_qsa_cache(m,i,(size_t)m->position+1,e,n))return -1;
            float ci[128],ck[512],cv[512];qn_qwen4_decode_input in={h,m->qsa_index[i],m->qsa_key[i],m->qsa_value[i],m->position};qn_qwen4_decode_output o={0};o.hyper_state=next;o.indexer_raw_key=ci;o.key_cache=ck;o.value_cache=cv;
            double lt=msnow();if(qn_qwen4_layer_forward_decode(m->qsa[i],&in,&o,e,n))return -1;double lms=msnow()-lt;qsa_ms+=lms;if(profile)fprintf(stderr,"[model-prof] L%02u QSA %.4f ms (att %.4f moe %.4f)\n",i,lms,o.attention_ms,o.moe_ms);
            memcpy(m->qsa_index[i]+(size_t)m->position*128,ci,128*4);memcpy(m->qsa_key[i]+(size_t)m->position*512,ck,512*4);memcpy(m->qsa_value[i]+(size_t)m->position*512,cv,512*4);
        }else{qn_gdn_decode_input in={h,NULL,NULL};qn_gdn_decode_output o={0};o.hyper_state=next;double lt=msnow();if(qn_gdn_layer_forward_full(m->gdn[i],&in,&o,e,n))return -1;double lms=msnow()-lt;gdn_ms+=lms;if(profile)fprintf(stderr,"[model-prof] L%02u GDN %.4f ms\n",i,lms);}
        memcpy(h,next,10240*4);
    }
    if(profile)fprintf(stderr,"[model-prof] totals GDN=%.4f QSA=%.4f PLE=%.4f layers=%.4f ms\n",gdn_ms,qsa_ms,ple_ms,gdn_ms+qsa_ms+ple_ms);
    return 0;
}

int qn_qwen4_model_forward_token(qn_qwen4_model *m,uint32_t token,float h[10240],char *e,size_t n){
    if(!m||!h||token>=248320){E(e,n,"invalid forward token args");return -1;}if(run_trunk(m,token,h,e,n))return -1;m->ple_token_history[0]=m->ple_token_history[1];m->ple_token_history[1]=(int64_t)token;m->position++;return 0;
}

int qn_qwen4_model_step(qn_qwen4_model *m,uint32_t token,float *logits,qn_qwen4_step_output *out,char *e,size_t n){
    if(!m||!out||token>=248320){E(e,n,"invalid model step");return -1;}double t0=msnow();float emb[2560],h[10240];if(qn_model_io_embed(m->io,token,emb,e,n))return -1;for(int g=0;g<4;g++)memcpy(h+g*2560,emb,2560*4);if(run_trunk(m,token,h,e,n))return -1;double t1=msnow();uint32_t nt=0;double lm=0;if(qn_model_io_logits(m->io,h,logits,&nt,&lm,e,n))return -1;m->ple_token_history[0]=m->ple_token_history[1];m->ple_token_history[1]=(int64_t)token;m->position++;out->next_token=nt;out->trunk_ms=t1-t0;out->logits_ms=lm;out->total_ms=msnow()-t0;return 0;
}

void qn_qwen4_model_get_stats(qn_qwen4_model *m,qn_qwen4_model_stats *s){if(!m||!s)return;memset(s,0,sizeof(*s));s->num_layers=48;s->qsa_layers=12;s->gdn_layers=36;s->gdn_conv_state_bytes=(uint64_t)36*10240*4*4;s->gdn_recurrent_state_bytes=(uint64_t)36*48*128*128*4;s->ple_state_bytes=(uint64_t)10240*9*4+2*8;s->opened_qsa_layers=m->opened_qsa;s->opened_gdn_layers=m->opened_gdn;s->ple_opened=m->ple!=0;s->qsa_cache_capacity=m->qsa_initial_capacity;s->qsa_cache_bytes=m->qsa_cache_bytes;s->position=m->position;}
void qn_qwen4_model_close(qn_qwen4_model *m){if(!m)return;for(uint32_t i=0;i<48;i++){if(m->qsa[i])qn_qwen4_layer_close(m->qsa[i]);if(m->gdn[i])qn_gdn_layer_close(m->gdn[i]);free(m->qsa_index[i]);free(m->qsa_key[i]);free(m->qsa_value[i]);}if(m->ple)qn_ple_layer_close(m->ple);if(m->io)qn_model_io_close(m->io);free(m->ple_conv);free(m);}
