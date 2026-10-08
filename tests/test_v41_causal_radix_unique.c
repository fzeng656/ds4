#include "ds4_gpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <string.h>
static int comp_id(const void *a,const void *b){int x=*(const int*)a,y=*(const int*)b;return (x>y)-(x<y);}
#define CK(x) do{if(!(x)){fprintf(stderr,"FAIL line=%d width=%u rows=%u mode=%d\n",__LINE__,width,rows,pattern);return 3;}}while(0)
int main(void){
 if(!ds4_gpu_init()) return 2;
 const unsigned widths[]={4096,8192,16384,65536,131072};
 const unsigned nrows[]={1,8,32};
 for(unsigned wi=0;wi<5;wi++)for(unsigned ri=0;ri<3;ri++)for(unsigned ratio=1;ratio<=2;ratio++){
   unsigned width=widths[wi],rows=nrows[ri],start=(width-rows-8)*ratio-1u;
   size_t n=(size_t)width*rows;
   float *v=malloc(n*4);int32_t *orig=malloc((size_t)rows*512*4),*got=malloc((size_t)rows*512*4);
   if(!v||!orig||!got)return 4;
   ds4_gpu_tensor *scores=ds4_gpu_tensor_alloc(n*4),*a=ds4_gpu_tensor_alloc((size_t)rows*512*4),*b=ds4_gpu_tensor_alloc((size_t)rows*512*4);
   if(!scores||!a||!b)return 4;
   for(int pattern=0;pattern<1;pattern++){
     for(unsigned t=0;t<rows;t++){
       unsigned visible=(start+t+1u)/ratio;
       for(unsigned i=0;i<width;i++){
         float f=pattern==0?(float)((i*7919ull+t*1009ull)%104729ull)-50000.0f:
                 pattern==1?(float)(i%19):((i%7)?-INFINITY:(float)(i%13));
         if(i>=visible)f=123456.0f;
         v[(size_t)t*width+i]=f;
       }
     }
     CK(ds4_gpu_tensor_write(scores,0,v,n*4));
     unsetenv("DS4_V41_EXPERIMENT_CAUSAL_RADIX");
     CK(ds4_gpu_dsv41_indexer_topk_batch(a,scores,width,rows,start,ratio));
     CK(ds4_gpu_tensor_read(a,0,orig,(size_t)rows*512*4));
     setenv("DS4_V41_EXPERIMENT_CAUSAL_RADIX","1",1);
     CK(ds4_gpu_dsv41_indexer_topk_batch(b,scores,width,rows,start,ratio));
     CK(ds4_gpu_tensor_read(b,0,got,(size_t)rows*512*4));
     unsigned mismatch=0;
     for(size_t j=0;j<(size_t)rows*512;j++)if(orig[j]!=got[j]){
       if(mismatch<1)fprintf(stderr,"MISMATCH w=%u rows=%u mode=%d idx=%zu baseline=%d candidate=%d bval=%.2f cval=%.2f\n",
          width,rows,pattern,j,orig[j],got[j],v[(j/512)*width+orig[j]],v[(j/512)*width+got[j]]);
       mismatch++;
     }
     unsigned set_mismatch=0,value_mismatch=0;
     for(unsigned t=0;t<rows;t++){
       int32_t *os=malloc(512*4),*cs=malloc(512*4);
       memcpy(os,orig+t*512,512*4);memcpy(cs,got+t*512,512*4);
       qsort(os,512,4,comp_id);qsort(cs,512,4,comp_id);
       for(unsigned j=0;j<512;j++){
         set_mismatch+=(os[j]!=cs[j]);
         value_mismatch+=(v[(size_t)t*width+orig[t*512+j]]!=v[(size_t)t*width+got[t*512+j]]);
       }
       free(os);free(cs);
     }
     if (mismatch || set_mismatch || value_mismatch) return 5;
     printf("TOPK_UNIQUE_PASS width=%u rows=%u ratio=%u exact=1\n",width,rows,ratio);fflush(stdout);
   }
   ds4_gpu_tensor_free(scores);ds4_gpu_tensor_free(a);ds4_gpu_tensor_free(b);free(v);free(orig);free(got);
 }
 return 0;
}
