#include "ds4_gpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
static double now_ms(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return 1e3*t.tv_sec+1e-6*t.tv_nsec;}
#define CK(x) do{if(!(x)){fprintf(stderr,"FAIL %d\n",__LINE__);return 2;}}while(0)
int main(void){
 CK(ds4_gpu_init());
 const unsigned widths[]={4096,16384,65536,131072}, row_counts[]={16,32,64};
 for(unsigned w=0;w<4;w++)for(unsigned r=0;r<3;r++){
  unsigned width=widths[w],rows=row_counts[r],ratio=2,start=(width-rows-20)*ratio-1u;
  size_t n=(size_t)width*rows; float *v=malloc(n*4);CK(v);
  for(size_t i=0;i<n;i++){unsigned u=(unsigned)((i*7919ull+i/width*1009ull)%1000003ull);v[i]=(float)u*0.000025f-15.0f;}
  ds4_gpu_tensor *scores=ds4_gpu_tensor_alloc(n*4),*out=ds4_gpu_tensor_alloc((size_t)rows*512*4);
  CK(scores&&out&&ds4_gpu_tensor_write(scores,0,v,n*4));
  for(int arm=0;arm<2;arm++){
   if(arm)setenv("DS4_V41_EXPERIMENT_CAUSAL_RADIX","1",1);else unsetenv("DS4_V41_EXPERIMENT_CAUSAL_RADIX");
   for(int z=0;z<3;z++)CK(ds4_gpu_dsv41_indexer_topk_batch(out,scores,width,rows,start,ratio));
  }
  for(int pass=0;pass<8;pass++){
   int arm=pass%2;
   if(arm)setenv("DS4_V41_EXPERIMENT_CAUSAL_RADIX","1",1);else unsetenv("DS4_V41_EXPERIMENT_CAUSAL_RADIX");
   double t=now_ms();
   for(int j=0;j<16;j++)CK(ds4_gpu_dsv41_indexer_topk_batch(out,scores,width,rows,start,ratio));
   printf("TOPK_AB width=%u rows=%u pass=%d variant=%s ms=%.4f\n",width,rows,pass,arm?"radix":"ladder",(now_ms()-t)/16);fflush(stdout);
  }
  ds4_gpu_tensor_free(scores);ds4_gpu_tensor_free(out);free(v);
 }
 return 0;
}
