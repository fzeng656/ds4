#include "../ds4_gpu.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#define CK(x) do{if(!(x)){fprintf(stderr,"FAIL %d draw %u\n",__LINE__,draw);return 1;}}while(0)
static uint32_t seed=19871;static uint32_t rnd(void){seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;return seed;}
int main(int argc,char **argv){unsigned draw=0,draws=argc>1?atoi(argv[1]):64;CK(ds4_gpu_init()); ds4_gpu_set_ssd_streaming(true);
 const unsigned rows=64,heads=32,width=512,comp=8192,raw=256;const size_t n=rows*heads*width;
 float *x=malloc((n>comp*width?n:comp*width)*4),*ref=malloc((n+128)*4),*got=malloc((n+128)*4);int *ids=malloc(rows*512*4);void *model=NULL;CK(!posix_memalign(&model,getpagesize(),4096));memset(model,0,4096);CK(x&&ref&&got&&ids&&ds4_gpu_set_model_map(model,4096));
 ds4_gpu_tensor *q=ds4_gpu_tensor_alloc(n*4),*r=ds4_gpu_tensor_alloc(raw*width*4),*c=ds4_gpu_tensor_alloc(comp*width*4),*ix=ds4_gpu_tensor_alloc(rows*512*4),*root=ds4_gpu_tensor_alloc((n+128)*4);CK(q&&r&&c&&ix&&root);ds4_gpu_tensor *o=ds4_gpu_tensor_view(root,256,n*4);CK(o);
 for(draw=0;draw<draws+12;draw++){
  if(draw<draws){for(unsigned i=0;i<comp*width;i++)x[i]=(int32_t)rnd()*0x1p-32f;CK(ds4_gpu_tensor_write(c,0,x,comp*width*4));for(unsigned i=0;i<raw*width;i++)x[i]=(int32_t)rnd()*0x1p-32f;CK(ds4_gpu_tensor_write(r,0,x,raw*width*4));for(size_t i=0;i<n;i++)x[i]=(int32_t)rnd()*0x1p-32f;CK(ds4_gpu_tensor_write(q,0,x,n*4));for(unsigned t=0;t<rows;t++)for(unsigned j=0;j<512;j++)ids[t*512+j]=((j*127+draw)%comp);if(draw%2)for(unsigned j=0;j<rows*512;j+=17)ids[j]=-1;CK(ds4_gpu_tensor_write(ix,0,ids,rows*512*4));}
  for(unsigned arm=draw<draws?0:2;arm<3;arm++){
   if(!arm)setenv("DS4_METAL_DISABLE_V41_PREFILL_RB16","1",1);else unsetenv("DS4_METAL_DISABLE_V41_PREFILL_RB16");
   if(!arm)unsetenv("DS4_V41_EXPERIMENT_STAGE_COMPKV_F16");else setenv("DS4_V41_EXPERIMENT_STAGE_COMPKV_F16","1",1);
   if(arm==1)setenv("DS4_METAL_DISABLE_V41_PREFILL_LEAN","1",1);else unsetenv("DS4_METAL_DISABLE_V41_PREFILL_LEAN");
   for(size_t i=0;i<n+128;i++)((uint32_t*)got)[i]=0x7fc12345;CK(ds4_gpu_tensor_write(root,0,got,(n+128)*4));CK(ds4_gpu_begin_commands());
   CK(ds4_gpu_attention_indexed_mixed_batch_heads_tensor(o,model,4096,0,q,r,c,0,ix,rows,20000,raw,raw,128,comp,512,128,2,heads,width));
   CK(ds4_gpu_end_commands()&&ds4_gpu_tensor_read(root,0,arm?got:ref,(n+128)*4));
   if(arm){if(memcmp(ref,got,(n+128)*4)){for(size_t j=0;j<n+128;j++)if(((uint32_t*)ref)[j]!=((uint32_t*)got)[j]){fprintf(stderr,"MISMATCH arm%u word%zu %08x %08x\n",arm,j,((uint32_t*)ref)[j],((uint32_t*)got)[j]);break;}CK(0);}for(size_t j=64;j<n+64;j++)CK((((uint32_t*)got)[j]&0x7f800000u)!=0x7f800000u);}
  }
 }

 fprintf(stderr,"PASS dual-rb4 correctness rows=%u draws=%u fixed-mem guard parity\n",rows,draws);
 for(int pass=0;pass<8;pass++){
    const int arm=pass%2;
    if(arm)setenv("DS4_V41_EXPERIMENT_STAGE_COMPKV_F16","1",1);
    else unsetenv("DS4_V41_EXPERIMENT_STAGE_COMPKV_F16");
    // Exclude pipeline compilation by creating each once before timed samples.
    CK(ds4_gpu_begin_commands());
    CK(ds4_gpu_attention_indexed_mixed_batch_heads_tensor(o,model,4096,0,q,r,c,0,ix,
       rows,20000,raw,raw,128,comp,512,128,2,heads,width));
    CK(ds4_gpu_end_commands());
    struct timespec begin,end;
    clock_gettime(CLOCK_MONOTONIC,&begin);
    for(int i=0;i<16;i++){
      CK(ds4_gpu_begin_commands());
      CK(ds4_gpu_attention_indexed_mixed_batch_heads_tensor(o,model,4096,0,q,r,c,0,ix,
        rows,20000,raw,raw,128,comp,512,128,2,heads,width));
      CK(ds4_gpu_end_commands());
    }
    clock_gettime(CLOCK_MONOTONIC,&end);
    const double ms=1000.0*(end.tv_sec-begin.tv_sec)+0.000001*(end.tv_nsec-begin.tv_nsec);
    printf("BENCH rows=%u head=%u run=%d arm=%s avg_ms=%.4f\n",rows,heads,pass,arm?"stage_f16":"metal4_dual",ms/16.0);
    fflush(stdout);
 }
 return 0;
}
