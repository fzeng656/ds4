#import <Foundation/Foundation.h>
#include "../qn_gdn_layer.h"
#include <mach/mach_time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
static double ms(){static mach_timebase_info_data_t tb;if(!tb.denom)mach_timebase_info(&tb);return(double)mach_absolute_time()*tb.numer/tb.denom/1e6;}
static int run(uint32_t T){const char*D=getenv("QWEN4_TEST_MODEL");const char*M=getenv("QWEN4_TEST_MANIFEST");if(!D||!M)return 77;size_t N=(size_t)T*10240;float*in=malloc(N*4),*seq=malloc(N*4),*bat=malloc(N*4);for(size_t i=0;i<N;i++)in[i]=sinf((float)i*.00137f)*.03f+cosf((float)i*.00031f)*.01f;char e[512]={0};qn_gdn_layer*a=0,*b=0;qn_gdn_layer_config c={D,M,0};if(qn_gdn_layer_open(&a,&c,e,sizeof(e))||qn_gdn_layer_open(&b,&c,e,sizeof(e))){fprintf(stderr,"open %s\n",e);return 1;}double t0=ms();for(uint32_t t=0;t<T;t++){qn_gdn_decode_input di={in+(size_t)t*10240,NULL,NULL};qn_gdn_decode_output o={0};o.hyper_state=seq+(size_t)t*10240;if(qn_gdn_layer_forward_full(a,&di,&o,e,sizeof(e))){fprintf(stderr,"seq %s\n",e);return 2;}}double st=ms()-t0;t0=ms();if(qn_gdn_layer_forward_full_batch(b,in,T,bat,e,sizeof(e))){fprintf(stderr,"batch %s\n",e);return 3;}double bt=ms()-t0;double mx=0,se=0,dot=0,aa=0,bb=0;for(size_t i=0;i<N;i++){double z=(double)seq[i]-bat[i];mx=fmax(mx,fabs(z));se+=z*z;dot+=(double)seq[i]*bat[i];aa+=(double)seq[i]*seq[i];bb+=(double)bat[i]*bat[i];}printf("T=%u seq=%.3fms batch=%.3fms speedup=%.3fx max=%.3g rmse=%.3g cos=%.12f\n",T,st,bt,st/bt,mx,sqrt(se/N),dot/sqrt(aa*bb));qn_gdn_layer_close(a);qn_gdn_layer_close(b);free(in);free(seq);free(bat);return mx>1e-4?4:0;}
int main(){@autoreleasepool{int r=run(5);if(r)return r;return run(16);}}
