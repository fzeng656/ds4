#include "qn_qwen4_model.h"
#include "qn_qwen4_layer.h"
#include "qn_gdn_layer.h"
#include "qn_ple_layer.h"
#include "qn_model_io.h"
#include "qn_runtime.h"
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
    int stable_mode, production_prepared;
    float last_stream[10240]; int last_stream_valid;
    uint32_t *verify_next_ids; uint32_t verify_capture_count;
    float *verify_streams;
    __strong id<MTLDevice> shadow_device; __strong id<MTLCommandQueue> shadow_queue;
    __strong id<MTLCommandBuffer> shadow_copy_cb; int shadow_copy_pending;
    float *shadow_ple_conv; float shadow_last_stream[10240]; int64_t shadow_ple_hist[2]; uint32_t shadow_position; int shadow_last_valid, shadow_active;
    char moe_metallib_path[1024];
};

static void E(char *e,size_t n,const char *s){if(e&&n)snprintf(e,n,"%s",s?s:"model runtime error");}
static double msnow(void){static mach_timebase_info_data_t tb;if(!tb.denom)mach_timebase_info(&tb);return (double)mach_absolute_time()*tb.numer/tb.denom/1e6;}
static int is_qsa(uint32_t i){return i<48 && (i&3u)==3u;}
static int gpu_block4_enabled(void){const char*v=getenv("QN_GPU_BLOCK4");return !v||v[0]!='0';}
static int gpu_trunk_chain_enabled(void){const char*v=getenv("QN_GPU_TRUNK_CHAIN"),*r=getenv("QN_QSA_RESIDENT_CACHE");return (!v||v[0]!='0')&&(!r||r[0]!='0');}
static int gpu_full_trunk_enabled(void){const char*v=getenv("QN_GPU_FULL_TRUNK"),*r=getenv("QN_QSA_RESIDENT_CACHE");return (!v||v[0]!='0')&&(!r||r[0]!='0');}
#define QN_QWEN4_PRODUCTION_CONTEXT 8192u
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
    qn_runtime_config_status rs={0};if(qn_runtime_configure_from_env(&rs,e,n))return -1;
    qn_qwen4_model *m=calloc(1,sizeof(*m));if(!m){E(e,n,"calloc model failed");return -1;}
    m->stable_mode=rs.stable_mode;if(rs.moe_metallib_configured)snprintf(m->moe_metallib_path,sizeof(m->moe_metallib_path),"%s",rs.moe_metallib_path);
    snprintf(m->model_dir,sizeof(m->model_dir),"%s",c->model_dir);snprintf(m->manifest_path,sizeof(m->manifest_path),"%s",c->manifest_path);snprintf(m->ngram_path,sizeof(m->ngram_path),"%s",c->ngram_table_path);m->qsa_initial_capacity=c->qsa_cache_capacity?c->qsa_cache_capacity:16;
    m->ple_conv=calloc((size_t)10240*9,sizeof(float));m->shadow_ple_conv=calloc((size_t)10240*9,sizeof(float));if(!m->ple_conv||!m->shadow_ple_conv){E(e,n,"PLE state allocation failed");qn_qwen4_model_close(m);return -1;}@autoreleasepool{m->shadow_device=MTLCreateSystemDefaultDevice();m->shadow_queue=[m->shadow_device newCommandQueue];if(!m->shadow_queue){E(e,n,"spec shadow Metal queue unavailable");qn_qwen4_model_close(m);return -1;}}
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
int qn_qwen4_model_prepare_prefill64(qn_qwen4_model *m,char *e,size_t n){
    if(!m){E(e,n,"invalid model");return -1;}
    const uint32_t T=64;size_t H=(size_t)T*10240,I=(size_t)T*128,K=(size_t)T*512;
    float *hin=calloc(H,sizeof(float)),*hout=calloc(H,sizeof(float)),*idx=calloc(I,sizeof(float)),*key=calloc(K,sizeof(float)),*val=calloc(K,sizeof(float));
    if(!hin||!hout||!idx||!key||!val){free(hin);free(hout);free(idx);free(key);free(val);E(e,n,"prefill warm allocation failed");return -1;}
    for(uint32_t i=0;i<48;i++){
        if(qn_qwen4_model_ensure_layer(m,i,e,n)){free(hin);free(hout);free(idx);free(key);free(val);return -1;}
        if(is_qsa(i)){
            /* QSA has no retained recurrent state: a complete dummy T64 run safely
               warms FA256 + MPS GEMM dispatch shapes using caller-owned scratch caches. */
            memset(idx,0,I*sizeof(float));memset(key,0,K*sizeof(float));memset(val,0,K*sizeof(float));double warm_ms=0;
            if(qn_qwen4_layer_forward_prefill_batch(m->qsa[i],hin,0,T,idx,key,val,hout,&warm_ms,e,n)){free(hin);free(hout);free(idx);free(key);free(val);return -1;}
        }else{
            if(qn_gdn_layer_warm_mps64(m->gdn[i],e,n)){free(hin);free(hout);free(idx);free(key);free(val);return -1;}
            /* Full dummy prefill faults expert banks and batch kernels resident.
               GDN owns recurrent state, so restore it to the clean session state. */
            if(qn_gdn_layer_forward_full_batch(m->gdn[i],hin,T,hout,e,n)){free(hin);free(hout);free(idx);free(key);free(val);return -1;}
            if(qn_gdn_layer_reset_state(m->gdn[i])){free(hin);free(hout);free(idx);free(key);free(val);E(e,n,"GDN warm reset failed");return -1;}
        }
    }
    free(hin);free(hout);free(idx);free(key);free(val);return qn_qwen4_model_ensure_ple(m,e,n);
}
void qn_qwen4_model_get_runtime_status(qn_qwen4_model*m,qn_qwen4_runtime_status*s){
    if(!s)return;memset(s,0,sizeof(*s));if(!m)return;s->stable_mode=m->stable_mode;s->production_prepared=m->production_prepared;
    snprintf(s->moe_metallib_path,sizeof(s->moe_metallib_path),"%s",m->moe_metallib_path);
    for(uint32_t i=0;i<48;i++){
        if(is_qsa(i)){if(m->qsa[i]){s->qsa_mps_layers+=qn_qwen4_layer_prefill_mps_enabled(m->qsa[i])?1:0;s->qsa_bm32_layers+=qn_qwen4_layer_moe_bm32_enabled(m->qsa[i])?1:0;}}
        else if(m->gdn[i]){s->gdn_mps_layers+=qn_gdn_layer_prefill_mps_enabled(m->gdn[i])?1:0;s->gdn_bm32_layers+=qn_gdn_layer_moe_bm32_enabled(m->gdn[i])?1:0;}
    }
}
int qn_qwen4_model_prepare_production(qn_qwen4_model*m,char*e,size_t n){
    if(!m){E(e,n,"invalid model");return -1;}
    if(!m->stable_mode){E(e,n,"production prepare requires QN_RUNTIME_MODE=stable before model open");return -1;}
    if(qn_qwen4_model_prepare_prefill64(m,e,n))return -1;
    qn_qwen4_runtime_status s;qn_qwen4_model_get_runtime_status(m,&s);
    if(s.gdn_mps_layers!=36||s.qsa_mps_layers!=12){E(e,n,"stable production requires MPS prefill on all 48 layers");return -1;}
    if(s.gdn_bm32_layers!=36||s.qsa_bm32_layers!=12){E(e,n,"stable production requires BM32 sorted-MoE on all 48 layers");return -1;}
    m->production_prepared=1;return 0;
}
int qn_qwen4_model_reset_session(qn_qwen4_model*m,char*e,size_t n){
    if(!m){E(e,n,"invalid model");return -1;}
    m->position=0;m->last_stream_valid=0;m->ple_token_history[0]=m->ple_token_history[1]=248044;
    if(m->ple_conv)memset(m->ple_conv,0,(size_t)10240*9*sizeof(float));
    for(uint32_t i=0;i<48;i++)if(m->gdn[i]&&qn_gdn_layer_reset_state(m->gdn[i])){E(e,n,"GDN session reset failed");return -1;}
    return 0;
}
void qn_qwen4_model_set_position(qn_qwen4_model *m,uint32_t p){if(m)m->position=p;}

int qn_qwen4_model_seed_qsa_cache(qn_qwen4_model *m,uint32_t i,uint32_t tokens,const float *idx,const float *k,const float *v,char *e,size_t n){
    if(!m||!is_qsa(i)||(!idx&&tokens)||(!k&&tokens)||(!v&&tokens)){E(e,n,"invalid QSA cache seed");return -1;}if(ensure_qsa_cache(m,i,tokens?tokens:1,e,n))return -1;if(tokens){memcpy(m->qsa_index[i],idx,(size_t)tokens*128*4);memcpy(m->qsa_key[i],k,(size_t)tokens*512*4);memcpy(m->qsa_value[i],v,(size_t)tokens*512*4);}return 0;
}

static int run_trunk_gpu_full(qn_qwen4_model*m,uint32_t token,float h[10240],char*e,size_t n){
 @autoreleasepool{
  __strong id<MTLCommandBuffer> cbs[40]={nil};uint32_t nc=0;id<MTLBuffer>buf=nil;id<MTLCommandBuffer>cb=nil;id<MTLCommandQueue>queue=nil;
  if(qn_qwen4_model_ensure_layer(m,0,e,n)||qn_qwen4_model_ensure_ple(m,e,n)||qn_qwen4_model_ensure_layer(m,1,e,n)||qn_qwen4_model_ensure_layer(m,2,e,n)||qn_qwen4_model_ensure_layer(m,3,e,n))return -1;
  if(ensure_qsa_cache(m,3,(size_t)m->position+1,e,n))return -1;
  if(qn_gdn_layer_submit_host(m->gdn[0],h,nil,&buf,&cb,e,n))return -1;queue=cb.commandQueue;cbs[nc++]=cb;
  int64_t ph[3]={m->ple_token_history[0],m->ple_token_history[1],(int64_t)token};if(qn_ple_layer_submit_buffer(m->ple,buf,ph,m->ple_conv,queue,&buf,&cb,e,n))return -1;cbs[nc++]=cb;
  if(qn_gdn_layer_submit_buffer(m->gdn[1],buf,queue,&buf,&cb,e,n))return -1;cbs[nc++]=cb;
  if(qn_gdn_layer_submit_buffer(m->gdn[2],buf,queue,&buf,&cb,e,n))return -1;cbs[nc++]=cb;
  qn_qwen4_layer_set_command_queue(m->qsa[3],queue);qn_qwen4_decode_input in3={NULL,m->qsa_index[3],m->qsa_key[3],m->qsa_value[3],m->position,0};if(qn_qwen4_layer_forward_decode_buffer_submit(m->qsa[3],buf,&in3,&buf,&cb,e,n))return -1;cbs[nc++]=cb;
  for(uint32_t i=4;i<48;i+=4){uint32_t qi=i+3;if(qn_qwen4_model_ensure_layer(m,i,e,n)||qn_qwen4_model_ensure_layer(m,i+1,e,n)||qn_qwen4_model_ensure_layer(m,i+2,e,n)||qn_qwen4_model_ensure_layer(m,qi,e,n))return -1;if(ensure_qsa_cache(m,qi,(size_t)m->position+1,e,n))return -1;
   if(qn_gdn_group_forward3_submit_buffer(m->gdn[i],m->gdn[i+1],m->gdn[i+2],buf,queue,&buf,&cb,e,n))return -1;cbs[nc++]=cb;qn_qwen4_layer_set_command_queue(m->qsa[qi],queue);qn_qwen4_decode_input in={NULL,m->qsa_index[qi],m->qsa_key[qi],m->qsa_value[qi],m->position,0};if(qn_qwen4_layer_forward_decode_buffer_submit(m->qsa[qi],buf,&in,&buf,&cb,e,n))return -1;cbs[nc++]=cb;
  }
  [cbs[nc-1] waitUntilCompleted];for(uint32_t j=0;j<nc;j++)if(cbs[j].status==MTLCommandBufferStatusError){E(e,n,cbs[j].error.description.UTF8String);return -1;}memcpy(h,buf.contents,10240*4);if(qn_ple_layer_copy_state(m->ple,m->ple_conv)){E(e,n,"PLE state mirror failed");return -1;}
  for(uint32_t qi=3;qi<48;qi+=4){float*ci=m->qsa_index[qi]+(size_t)m->position*128,*ck=m->qsa_key[qi]+(size_t)m->position*512,*cv=m->qsa_value[qi]+(size_t)m->position*512;if(qn_qwen4_layer_copy_current_cache(m->qsa[qi],ci,ck,cv)){E(e,n,"QSA cache mirror failed");return -1;}}
 }
 return 0;
}

static int run_trunk_gpu_chain_from4(qn_qwen4_model*m,float h[10240],char*e,size_t n){
 @autoreleasepool{
  __strong id<MTLCommandBuffer> cbs[32]={nil};uint32_t nc=0;id<MTLBuffer>buf=nil;id<MTLCommandBuffer>cb=nil;id<MTLCommandQueue>queue=nil;
  for(uint32_t i=4;i<48;i+=4){uint32_t qi=i+3;
   if(qn_qwen4_model_ensure_layer(m,i,e,n)||qn_qwen4_model_ensure_layer(m,i+1,e,n)||qn_qwen4_model_ensure_layer(m,i+2,e,n)||qn_qwen4_model_ensure_layer(m,qi,e,n))return -1;
   if(ensure_qsa_cache(m,qi,(size_t)m->position+1,e,n))return -1;
   if(i==4){if(qn_gdn_group_forward3_submit_host(m->gdn[i],m->gdn[i+1],m->gdn[i+2],h,&buf,&cb,e,n))return -1;queue=cb.commandQueue;}
   else if(qn_gdn_group_forward3_submit_buffer(m->gdn[i],m->gdn[i+1],m->gdn[i+2],buf,queue,&buf,&cb,e,n))return -1;
   cbs[nc++]=cb;qn_qwen4_layer_set_command_queue(m->qsa[qi],queue);
   qn_qwen4_decode_input in={NULL,m->qsa_index[qi],m->qsa_key[qi],m->qsa_value[qi],m->position,0};
   if(qn_qwen4_layer_forward_decode_buffer_submit(m->qsa[qi],buf,&in,&buf,&cb,e,n))return -1;cbs[nc++]=cb;
  }
  if(!nc||!buf){E(e,n,"empty GPU trunk chain");return -1;}[cbs[nc-1] waitUntilCompleted];
  for(uint32_t j=0;j<nc;j++)if(cbs[j].status==MTLCommandBufferStatusError){E(e,n,cbs[j].error.description.UTF8String);return -1;}
  memcpy(h,buf.contents,10240*4);
  for(uint32_t qi=7;qi<48;qi+=4){float*ci=m->qsa_index[qi]+(size_t)m->position*128,*ck=m->qsa_key[qi]+(size_t)m->position*512,*cv=m->qsa_value[qi]+(size_t)m->position*512;if(qn_qwen4_layer_copy_current_cache(m->qsa[qi],ci,ck,cv)){E(e,n,"QSA cache mirror failed");return -1;}}
 }
 return 0;
}

static int run_trunk(qn_qwen4_model *m,uint32_t token,float h[10240],char *e,size_t n){
    if(gpu_full_trunk_enabled())return run_trunk_gpu_full(m,token,h,e,n);
    float next[10240];
    for(uint32_t i=0;i<48;i++){
        if(i==4 && gpu_trunk_chain_enabled())return run_trunk_gpu_chain_from4(m,h,e,n);
        if(i==1){if(qn_qwen4_model_ensure_ple(m,e,n))return -1;float po[10240];qn_ple_input pi={0};pi.hyper_state=h;pi.token_history[0]=m->ple_token_history[0];pi.token_history[1]=m->ple_token_history[1];pi.token_history[2]=(int64_t)token;pi.conv_state=m->ple_conv;qn_ple_output py={0};py.ple_output=po;if(qn_ple_layer_forward(m->ple,&pi,&py,e,n))return -1;for(uint32_t j=0;j<10240;j++)h[j]+=po[j];}
        if(i>=4 && (i%4)==0){
            if(qn_qwen4_model_ensure_layer(m,i,e,n)||qn_qwen4_model_ensure_layer(m,i+1,e,n)||qn_qwen4_model_ensure_layer(m,i+2,e,n))return -1;
            if(gpu_block4_enabled()){
                uint32_t qi=i+3;if(qn_qwen4_model_ensure_layer(m,qi,e,n))return -1;if(ensure_qsa_cache(m,qi,(size_t)m->position+1,e,n))return -1;
                id<MTLBuffer>gbuf=nil;id<MTLCommandBuffer>gcb=nil;if(qn_gdn_group_forward3_submit_host(m->gdn[i],m->gdn[i+1],m->gdn[i+2],h,&gbuf,&gcb,e,n))return -1;
                qn_qwen4_layer_set_command_queue(m->qsa[qi],gcb.commandQueue);
                float ci[128],ck[512],cv[512];qn_qwen4_decode_input in={h,m->qsa_index[qi],m->qsa_key[qi],m->qsa_value[qi],m->position,0};qn_qwen4_decode_output o={0};o.hyper_state=next;o.indexer_raw_key=ci;o.key_cache=ck;o.value_cache=cv;
                if(qn_qwen4_layer_forward_decode_buffer(m->qsa[qi],gbuf,&in,&o,e,n))return -1;
                if(gcb.status==MTLCommandBufferStatusError){E(e,n,gcb.error.description.UTF8String);return -1;}
                memcpy(m->qsa_index[qi]+(size_t)m->position*128,ci,128*4);memcpy(m->qsa_key[qi]+(size_t)m->position*512,ck,512*4);memcpy(m->qsa_value[qi]+(size_t)m->position*512,cv,512*4);
                memcpy(h,next,10240*4);i+=3;continue;
            }
            if(qn_gdn_group_forward3(m->gdn[i],m->gdn[i+1],m->gdn[i+2],h,next,e,n))return -1;
            memcpy(h,next,10240*4); i+=2; continue;
        }
        if(qn_qwen4_model_ensure_layer(m,i,e,n))return -1;
        if(is_qsa(i)){
            if(ensure_qsa_cache(m,i,(size_t)m->position+1,e,n))return -1;
            float ci[128],ck[512],cv[512];qn_qwen4_decode_input in={h,m->qsa_index[i],m->qsa_key[i],m->qsa_value[i],m->position,0};qn_qwen4_decode_output o={0};o.hyper_state=next;o.indexer_raw_key=ci;o.key_cache=ck;o.value_cache=cv;
            if(qn_qwen4_layer_forward_decode(m->qsa[i],&in,&o,e,n))return -1;
            memcpy(m->qsa_index[i]+(size_t)m->position*128,ci,128*4);memcpy(m->qsa_key[i]+(size_t)m->position*512,ck,512*4);memcpy(m->qsa_value[i]+(size_t)m->position*512,cv,512*4);
        }else{qn_gdn_decode_input in={h,NULL,NULL};qn_gdn_decode_output o={0};o.hyper_state=next;if(qn_gdn_layer_forward_full(m->gdn[i],&in,&o,e,n))return -1;}
        memcpy(h,next,10240*4);
    }
    return 0;
}

int qn_qwen4_model_forward_token(qn_qwen4_model *m,uint32_t token,float h[10240],char *e,size_t n){
    if(m&&m->stable_mode&&!m->production_prepared){E(e,n,"stable runtime requires qn_qwen4_model_prepare_production before inference");return -1;}
    if(!m||!h||token>=248320){E(e,n,"invalid forward token args");return -1;}if(m->position>=QN_QWEN4_PRODUCTION_CONTEXT){E(e,n,"production context limit reached (8192 tokens)");return -1;}if(run_trunk(m,token,h,e,n))return -1;memcpy(m->last_stream,h,sizeof(m->last_stream));m->last_stream_valid=1;m->ple_token_history[0]=m->ple_token_history[1];m->ple_token_history[1]=(int64_t)token;m->position++;return 0;
}

int qn_qwen4_model_step(qn_qwen4_model *m,uint32_t token,float *logits,qn_qwen4_step_output *out,char *e,size_t n){
    if(m&&m->stable_mode&&!m->production_prepared){E(e,n,"stable runtime requires qn_qwen4_model_prepare_production before inference");return -1;}
    if(!m||!out||token>=248320){E(e,n,"invalid model step");return -1;}if(m->position>=QN_QWEN4_PRODUCTION_CONTEXT){E(e,n,"production context limit reached (8192 tokens)");return -1;}double t0=msnow();float emb[2560],h[10240];if(qn_model_io_embed(m->io,token,emb,e,n))return -1;for(int g=0;g<4;g++)memcpy(h+g*2560,emb,2560*4);if(run_trunk(m,token,h,e,n))return -1;memcpy(m->last_stream,h,sizeof(m->last_stream));m->last_stream_valid=1;double t1=msnow();uint32_t nt=0;double lm=0;if(qn_model_io_logits(m->io,h,logits,&nt,&lm,e,n))return -1;m->ple_token_history[0]=m->ple_token_history[1];m->ple_token_history[1]=(int64_t)token;m->position++;out->next_token=nt;out->trunk_ms=t1-t0;out->logits_ms=lm;out->total_ms=msnow()-t0;return 0;
}

int qn_qwen4_model_prefill_tokens(qn_qwen4_model *m,const uint32_t *tokens,size_t count,float *logits,qn_qwen4_prefill_output *out,char *e,size_t n){
    if(m&&m->stable_mode&&!m->production_prepared){E(e,n,"stable runtime requires qn_qwen4_model_prepare_production before inference");return -1;}
    if(!m||!tokens||!count||!out){E(e,n,"invalid prefill arguments");return -1;}if(count>QN_QWEN4_PRODUCTION_CONTEXT || (uint64_t)m->position+count>QN_QWEN4_PRODUCTION_CONTEXT){E(e,n,"prefill exceeds production context limit (8192 tokens)");return -1;}
    /* Production/model-owner prefill is deliberately chunked at 64 tokens.
       This keeps every layer on the warmed MPS/BM32 shape, bounds retained
       scratch independently of prompt length, and preserves causal state via
       the model's GDN/PLE/QSA caches between chunks. Layer-level APIs remain
       available for larger research batches. */
    if(count>64){
        size_t off=0;double ps=0,ls=0,ts=0;qn_qwen4_prefill_output part={0};
        while(off<count){size_t nleft=count-off,chunk=nleft>64?64:nleft;float *lp=(off+chunk==count)?logits:NULL;memset(&part,0,sizeof(part));if(qn_qwen4_model_prefill_tokens(m,tokens+off,chunk,lp,&part,e,n))return -1;ps+=part.prefill_ms;ls+=part.logits_ms;ts+=part.total_ms;off+=chunk;}
        out->next_token=part.next_token;out->prompt_tokens=(uint32_t)count;out->prefill_ms=ps;out->logits_ms=ls;out->total_ms=ts;return 0;
    }
    if(count==1){
        double t0=msnow();float emb[2560],h[10240];uint32_t token=tokens[0];if(token>=248320){E(e,n,"prefill token out of range");return -1;}if(qn_model_io_embed(m->io,token,emb,e,n))return -1;for(int g=0;g<4;g++)memcpy(h+g*2560,emb,2560*sizeof(float));if(run_trunk(m,token,h,e,n))return -1;memcpy(m->last_stream,h,sizeof(m->last_stream));m->last_stream_valid=1;double t1=msnow();uint32_t nt=0;double lm=0;if(qn_model_io_logits(m->io,h,logits,&nt,&lm,e,n))return -1;m->ple_token_history[0]=m->ple_token_history[1];m->ple_token_history[1]=(int64_t)token;m->position++;out->next_token=nt;out->prompt_tokens=1;out->prefill_ms=t1-t0;out->logits_ms=lm;out->total_ms=msnow()-t0;return 0;
    }
    if(count>UINT32_MAX){E(e,n,"prefill token count too large");return -1;}
    uint32_t T=(uint32_t)count,base=m->position;
    for(size_t t=0;t<count;t++)if(tokens[t]>=248320){E(e,n,"prefill token out of range");return -1;}
    size_t elems=count*10240;float *a=malloc(elems*sizeof(float)),*b=malloc(elems*sizeof(float));if(!a||!b){free(a);free(b);E(e,n,"prefill stream allocation failed");return -1;}
    double t0=msnow();
    for(uint32_t t=0;t<T;t++){float emb[2560];if(qn_model_io_embed(m->io,tokens[t],emb,e,n)){free(a);free(b);return -1;}for(int g=0;g<4;g++)memcpy(a+(size_t)t*10240+g*2560,emb,2560*sizeof(float));}
    int64_t ple_hist[2]={m->ple_token_history[0],m->ple_token_history[1]};
    for(uint32_t i=0;i<48;i++){
        if(i==1){
            if(qn_qwen4_model_ensure_ple(m,e,n)){free(a);free(b);return -1;}
            float *po=malloc(elems*sizeof(float));int64_t *ids64=malloc((size_t)T*sizeof(int64_t));if(!po||!ids64){free(po);free(ids64);free(a);free(b);E(e,n,"PLE prefill allocation failed");return -1;}for(uint32_t t=0;t<T;t++)ids64[t]=(int64_t)tokens[t];
            if(qn_ple_layer_forward_batch(m->ple,a,ids64,T,ple_hist,m->ple_conv,po,e,n)){free(po);free(ids64);free(a);free(b);return -1;}
            for(size_t z=0;z<elems;z++)a[z]+=po[z];free(po);free(ids64);
        }
        if(qn_qwen4_model_ensure_layer(m,i,e,n)){free(a);free(b);return -1;}
        if(i>=4&&!is_qsa(i)&&i+2<48&&!is_qsa(i+1)&&!is_qsa(i+2)){
            if(qn_qwen4_model_ensure_layer(m,i+1,e,n)||qn_qwen4_model_ensure_layer(m,i+2,e,n)){free(a);free(b);return -1;}
            if(qn_gdn_group_forward3_batch(m->gdn[i],m->gdn[i+1],m->gdn[i+2],a,T,b,e,n)){free(a);free(b);return -1;}
            float *tmp=a;a=b;b=tmp;i+=2;continue;
        }
        if(is_qsa(i)){
            if(ensure_qsa_cache(m,i,(size_t)base+T,e,n)){free(a);free(b);return -1;}
            if((uint64_t)base+T<=8192){
                double qms=0;if(qn_qwen4_layer_forward_prefill_batch(m->qsa[i],a,base,T,m->qsa_index[i],m->qsa_key[i],m->qsa_value[i],b,&qms,e,n)){free(a);free(b);return -1;}
            }else{
                for(uint32_t t=0;t<T;t++){
                    uint32_t pos=base+t;float ci[128],ck[512],cv[512];qn_qwen4_decode_input in={a+(size_t)t*10240,m->qsa_index[i],m->qsa_key[i],m->qsa_value[i],pos,0};qn_qwen4_decode_output o={0};o.hyper_state=b+(size_t)t*10240;o.indexer_raw_key=ci;o.key_cache=ck;o.value_cache=cv;
                    if(qn_qwen4_layer_forward_decode(m->qsa[i],&in,&o,e,n)){free(a);free(b);return -1;}
                    memcpy(m->qsa_index[i]+(size_t)pos*128,ci,128*sizeof(float));memcpy(m->qsa_key[i]+(size_t)pos*512,ck,512*sizeof(float));memcpy(m->qsa_value[i]+(size_t)pos*512,cv,512*sizeof(float));
                }
            }
        }else{
            if(qn_gdn_layer_forward_full_batch(m->gdn[i],a,T,b,e,n)){free(a);free(b);return -1;}
        }
        float *tmp=a;a=b;b=tmp;
    }
    if(m->verify_streams&&m->verify_capture_count==T)memcpy(m->verify_streams,a,(size_t)T*10240*sizeof(float));
    double t1=msnow();memcpy(m->last_stream,a+(size_t)(T-1)*10240,sizeof(m->last_stream));m->last_stream_valid=1;uint32_t nt=0;double lm=0;if(m->verify_next_ids&&m->verify_capture_count==T){if(qn_model_io_logits_batch(m->io,a,T,m->verify_next_ids,&lm,e,n)){free(a);free(b);return -1;}nt=m->verify_next_ids[T-1];if(logits){double one=0;if(qn_model_io_logits(m->io,a+(size_t)(T-1)*10240,logits,&nt,&one,e,n)){free(a);free(b);return -1;}lm+=one;}}else if(qn_model_io_logits(m->io,a+(size_t)(T-1)*10240,logits,&nt,&lm,e,n)){free(a);free(b);return -1;}
    m->ple_token_history[0]=ple_hist[0];m->ple_token_history[1]=ple_hist[1];m->position=base+T;
    out->next_token=nt;out->prompt_tokens=T;out->prefill_ms=t1-t0;out->logits_ms=lm;out->total_ms=msnow()-t0;free(a);free(b);return 0;
}

int qn_qwen4_model_spec_shadow_dispatch(qn_qwen4_model*m,char*e,size_t n){
 if(!m||m->shadow_active||m->shadow_copy_pending){E(e,n,"invalid spec shadow dispatch");return -1;}for(uint32_t i=0;i<48;i++)if(!is_qsa(i)&&qn_qwen4_model_ensure_layer(m,i,e,n))return -1;
 m->shadow_position=m->position;m->shadow_ple_hist[0]=m->ple_token_history[0];m->shadow_ple_hist[1]=m->ple_token_history[1];memcpy(m->shadow_ple_conv,m->ple_conv,(size_t)10240*9*sizeof(float));m->shadow_last_valid=m->last_stream_valid;if(m->last_stream_valid)memcpy(m->shadow_last_stream,m->last_stream,sizeof(m->last_stream));
 @autoreleasepool{id<MTLCommandBuffer>cb=[m->shadow_queue commandBuffer];id<MTLBlitCommandEncoder>b=[cb blitCommandEncoder];for(uint32_t i=0;i<48;i++)if(!is_qsa(i)&&qn_gdn_layer_encode_state_copy_to_shadow(m->gdn[i],b)){[b endEncoding];E(e,n,"GDN shadow encode failed");return -1;}[b endEncoding];[cb commit];m->shadow_copy_cb=cb;m->shadow_copy_pending=1;}return 0;
}
int qn_qwen4_model_spec_shadow_activate(qn_qwen4_model*m,double*wait_ms,char*e,size_t n){if(!m||!m->shadow_copy_pending||m->shadow_active){E(e,n,"invalid spec shadow activate");return -1;}double t0=msnow();@autoreleasepool{[m->shadow_copy_cb waitUntilCompleted];if(m->shadow_copy_cb.status==MTLCommandBufferStatusError){E(e,n,m->shadow_copy_cb.error.description.UTF8String);m->shadow_copy_cb=nil;m->shadow_copy_pending=0;return -1;}m->shadow_copy_cb=nil;}m->shadow_copy_pending=0;for(uint32_t i=0;i<48;i++)if(!is_qsa(i))qn_gdn_layer_swap_state_buffers(m->gdn[i]);m->shadow_active=1;if(wait_ms)*wait_ms=msnow()-t0;return 0;}
int qn_qwen4_model_spec_shadow_begin(qn_qwen4_model*m,double*elapsed,char*e,size_t n){double t0=msnow();if(qn_qwen4_model_spec_shadow_dispatch(m,e,n))return -1;if(qn_qwen4_model_spec_shadow_activate(m,NULL,e,n))return -1;if(elapsed)*elapsed=msnow()-t0;return 0;}
int qn_qwen4_model_spec_shadow_commit(qn_qwen4_model*m){if(!m||!m->shadow_active)return -1;m->shadow_active=0;return 0;}
int qn_qwen4_model_spec_shadow_rollback(qn_qwen4_model*m,char*e,size_t n){if(!m||!m->shadow_active){E(e,n,"invalid spec shadow rollback");return -1;}for(uint32_t i=0;i<48;i++)if(!is_qsa(i)&&m->gdn[i])qn_gdn_layer_swap_state_buffers(m->gdn[i]);m->position=m->shadow_position;m->ple_token_history[0]=m->shadow_ple_hist[0];m->ple_token_history[1]=m->shadow_ple_hist[1];memcpy(m->ple_conv,m->shadow_ple_conv,(size_t)10240*9*sizeof(float));m->last_stream_valid=m->shadow_last_valid;if(m->shadow_last_valid)memcpy(m->last_stream,m->shadow_last_stream,sizeof(m->last_stream));for(uint32_t i=0;i<48;i++)if(is_qsa(i)&&m->qsa[i])qn_qwen4_layer_set_resident_cache_tokens(m->qsa[i],m->shadow_position);m->shadow_active=0;return 0;}

int qn_qwen4_model_copy_last_stream(qn_qwen4_model*m,float out[10240]){if(!m||!out||!m->last_stream_valid)return -1;memcpy(out,m->last_stream,sizeof(m->last_stream));return 0;}
int qn_qwen4_model_verify_tokens_capture(qn_qwen4_model*m,const uint32_t*tokens,size_t count,uint32_t*next_ids,float*streams,qn_qwen4_prefill_output*out,char*e,size_t n){if(!m||!tokens||count<2||count>64||!next_ids||!out){E(e,n,"invalid verify args");return -1;}m->verify_next_ids=next_ids;m->verify_streams=streams;m->verify_capture_count=(uint32_t)count;int rc=qn_qwen4_model_prefill_tokens(m,tokens,count,NULL,out,e,n);m->verify_next_ids=NULL;m->verify_streams=NULL;m->verify_capture_count=0;return rc;}
int qn_qwen4_model_verify_tokens(qn_qwen4_model*m,const uint32_t*tokens,size_t count,uint32_t*next_ids,qn_qwen4_prefill_output*out,char*e,size_t n){return qn_qwen4_model_verify_tokens_capture(m,tokens,count,next_ids,NULL,out,e,n);}

void qn_qwen4_model_get_stats(qn_qwen4_model *m,qn_qwen4_model_stats *s){if(!m||!s)return;memset(s,0,sizeof(*s));s->num_layers=48;s->qsa_layers=12;s->gdn_layers=36;s->gdn_conv_state_bytes=(uint64_t)36*10240*4*4;s->gdn_recurrent_state_bytes=(uint64_t)36*48*128*128*4;s->ple_state_bytes=(uint64_t)10240*9*4+2*8;s->opened_qsa_layers=m->opened_qsa;s->opened_gdn_layers=m->opened_gdn;s->ple_opened=m->ple!=0;s->qsa_cache_capacity=m->qsa_initial_capacity;s->qsa_cache_bytes=m->qsa_cache_bytes;s->position=m->position;}
void qn_qwen4_model_close(qn_qwen4_model *m){if(!m)return;@autoreleasepool{m->shadow_copy_cb=nil;m->shadow_queue=nil;m->shadow_device=nil;}for(uint32_t i=0;i<48;i++){if(m->qsa[i])qn_qwen4_layer_close(m->qsa[i]);if(m->gdn[i])qn_gdn_layer_close(m->gdn[i]);free(m->qsa_index[i]);free(m->qsa_key[i]);free(m->qsa_value[i]);}if(m->ple)qn_ple_layer_close(m->ple);if(m->io)qn_model_io_close(m->io);free(m->ple_conv);free(m->shadow_ple_conv);free(m);}
