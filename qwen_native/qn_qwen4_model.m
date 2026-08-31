#include "qn_qwen4_model.h"
#include "qn_qwen4_layer.h"
#include "qn_gdn_layer.h"
#include "qn_ple_layer.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
struct qn_qwen4_model {
 char model_dir[1024], manifest_path[1024], ngram_path[1024];
 qn_qwen4_layer *qsa[48]; qn_gdn_layer *gdn[48]; qn_ple_layer *ple;
 float *gdn_conv[48]; float *gdn_state[48];
 float *ple_conv; int64_t ple_token_history[2];
 uint32_t opened_qsa,opened_gdn;
};
static void E(char*e,size_t n,const char*s){if(e&&n)snprintf(e,n,"%s",s?s:"model runtime error");}
static int is_qsa(uint32_t i){return i<48 && (i&3u)==3u;}
int qn_qwen4_model_open(qn_qwen4_model **out,const qn_qwen4_model_config*c,char*e,size_t n){
 if(!out||!c||!c->model_dir||!c->manifest_path||!c->ngram_table_path){E(e,n,"invalid model config");return -1;}qn_qwen4_model*m=calloc(1,sizeof(*m));if(!m){E(e,n,"calloc model failed");return -1;}snprintf(m->model_dir,sizeof(m->model_dir),"%s",c->model_dir);snprintf(m->manifest_path,sizeof(m->manifest_path),"%s",c->manifest_path);snprintf(m->ngram_path,sizeof(m->ngram_path),"%s",c->ngram_table_path);
 for(uint32_t i=0;i<48;i++)if(!is_qsa(i)){m->gdn_conv[i]=calloc((size_t)10240*4,sizeof(float));m->gdn_state[i]=calloc((size_t)48*128*128,sizeof(float));if(!m->gdn_conv[i]||!m->gdn_state[i]){E(e,n,"GDN state allocation failed");qn_qwen4_model_close(m);return -1;}}
 m->ple_conv=calloc((size_t)10240*9,sizeof(float));if(!m->ple_conv){E(e,n,"PLE state allocation failed");qn_qwen4_model_close(m);return -1;}m->ple_token_history[0]=m->ple_token_history[1]=151646; /* EOS initial context */
 *out=m;return 0;
}
int qn_qwen4_model_ensure_layer(qn_qwen4_model*m,uint32_t i,char*e,size_t n){if(!m||i>=48){E(e,n,"invalid layer index");return -1;}if(is_qsa(i)){if(m->qsa[i])return 0;qn_qwen4_layer_config c={0};c.layer_index=i;c.model_dir=m->model_dir;c.manifest_path=m->manifest_path;if(qn_qwen4_layer_open(&m->qsa[i],&c,e,n))return -1;m->opened_qsa++;return 0;}if(m->gdn[i])return 0;qn_gdn_layer_config c={m->model_dir,m->manifest_path,i};if(qn_gdn_layer_open(&m->gdn[i],&c,e,n))return -1;m->opened_gdn++;return 0;}
int qn_qwen4_model_ensure_ple(qn_qwen4_model*m,char*e,size_t n){if(!m){E(e,n,"invalid model");return -1;}if(m->ple)return 0;qn_ple_config c={m->model_dir,m->manifest_path,m->ngram_path};if(qn_ple_layer_open(&m->ple,&c,e,n))return -1;return 0;}
void qn_qwen4_model_get_stats(qn_qwen4_model*m,qn_qwen4_model_stats*s){if(!m||!s)return;memset(s,0,sizeof(*s));s->num_layers=48;s->qsa_layers=12;s->gdn_layers=36;s->gdn_conv_state_bytes=(uint64_t)36*10240*4*4;s->gdn_recurrent_state_bytes=(uint64_t)36*48*128*128*4;s->ple_state_bytes=(uint64_t)10240*9*4+2*8;s->opened_qsa_layers=m->opened_qsa;s->opened_gdn_layers=m->opened_gdn;s->ple_opened=m->ple!=0;}
void qn_qwen4_model_close(qn_qwen4_model*m){if(!m)return;for(uint32_t i=0;i<48;i++){if(m->qsa[i])qn_qwen4_layer_close(m->qsa[i]);if(m->gdn[i])qn_gdn_layer_close(m->gdn[i]);free(m->gdn_conv[i]);free(m->gdn_state[i]);}if(m->ple)qn_ple_layer_close(m->ple);free(m->ple_conv);free(m);}
