#include "ds4_gpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <string.h>
static double now_ms(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1000.0+t.tv_nsec*1e-6;}
#define CK(x) do{if(!(x)){fprintf(stderr,"FAIL line %d\n",__LINE__);return 3;}}while(0)
int main(int argc,char**argv){
 int trials=argc>1?atoi(argv[1]):12; CK(ds4_gpu_init());
 const unsigned widths[]={4096,16384,65536,131072};
 const unsigned counts[]={1,16,64};
 for(unsigned wi=0;wi<sizeof(widths)/sizeof(widths[0]);wi++){
   unsigned width=widths[wi];
   for(unsigned ci=0;ci<sizeof(counts)/sizeof(counts[0]);ci++){
     unsigned rows=counts[ci],ratio=2,start=(width-rows-20)*ratio-1u;
     size_t n=(size_t)width*rows;
     float *v=malloc(n*sizeof(float)); CK(v);
     for(size_t i=0;i<n;i++){unsigned u=(unsigned)((i*7919ull+i/width*1009ull)%1000003ull);v[i]=(float)u*0.000025f-15.0f;}
     ds4_gpu_tensor *scores=ds4_gpu_tensor_alloc(n*4),*out=ds4_gpu_tensor_alloc((size_t)rows*512*4);
     CK(scores&&out);CK(ds4_gpu_tensor_write(scores,0,v,n*4));
     for(int i=0;i<3;i++){CK(ds4_gpu_dsv41_indexer_topk_batch(out,scores,width,rows,start,ratio));}
     double t0=now_ms();
     for(int i=0;i<trials;i++){CK(ds4_gpu_dsv41_indexer_topk_batch(out,scores,width,rows,start,ratio));}
     printf("TOPKBENCH width=%u rows=%u ratio=%u trials=%d per_call_ms=%.4f\n",width,rows,ratio,trials,(now_ms()-t0)/trials);fflush(stdout);
     ds4_gpu_tensor_free(scores);ds4_gpu_tensor_free(out);free(v);
   }
 }
 return 0;
}
