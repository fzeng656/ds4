#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

typedef struct { void *base; size_t bytes; int fd; } MMap;
static MMap map_file(const char *p) {
    MMap m={0}; m.fd=open(p,O_RDONLY); if(m.fd<0){perror("open"); exit(1);} struct stat st; if(fstat(m.fd,&st)){perror("fstat");exit(1);} m.bytes=(size_t)st.st_size; m.base=mmap(NULL,m.bytes,PROT_READ,MAP_SHARED,m.fd,0); if(m.base==MAP_FAILED){perror("mmap");exit(1);} return m;
}
static void unmap_file(MMap *m){ if(m->base && m->base!=MAP_FAILED){ posix_madvise(m->base,m->bytes,POSIX_MADV_DONTNEED); munmap(m->base,m->bytes);} if(m->fd>=0) close(m->fd); }
static void *read_all(const char *p,size_t expect){ FILE *f=fopen(p,"rb"); if(!f){perror("fopen");exit(1);} void *x=malloc(expect); if(fread(x,1,expect,f)!=expect){fprintf(stderr,"short read %s\n",p);exit(1);} fclose(f); return x; }

int main(int argc,char **argv){
 if(argc!=7){fprintf(stderr,"usage: %s shard w_off s_off b_off x.bin ref.bin\n",argv[0]);return 2;}
 const uint64_t woff=strtoull(argv[2],0,10), soff=strtoull(argv[3],0,10), boff=strtoull(argv[4],0,10);
 const uint32_t OUT=12288, IN=2560, GROUP=64, WCOLS=640, GROUPS=40;
 const size_t wbytes=(size_t)OUT*WCOLS*4, sbytes=(size_t)OUT*GROUPS*2;
 MMap m=map_file(argv[1]); float *x=read_all(argv[5],IN*4); float *ref=read_all(argv[6],OUT*4);
 if(woff+wbytes>m.bytes||soff+sbytes>m.bytes||boff+sbytes>m.bytes){fprintf(stderr,"range OOB\n");return 1;}
 @autoreleasepool {
  id<MTLDevice> dev=MTLCreateSystemDefaultDevice(); id<MTLCommandQueue> q=[dev newCommandQueue]; NSError *err=nil;
  NSString *src=@"#include <metal_stdlib>\nusing namespace metal;\ninline float bf16_to_f32(ushort v){ return as_type<float>((uint)v << 16); }\nkernel void q8_affine_mv(device const uint *w [[buffer(0)]], device const ushort *sc [[buffer(1)]], device const ushort *bi [[buffer(2)]], device const float *x [[buffer(3)]], device float *y [[buffer(4)]], uint row [[thread_position_in_grid]]) { const uint OUT=12288, IN=2560, GROUP=64, WCOLS=640, GROUPS=40; if(row>=OUT) return; float acc=0.0f; device const uint *wr=w + row*WCOLS; device const ushort *sr=sc + row*GROUPS; device const ushort *br=bi + row*GROUPS; for(uint g=0; g<GROUPS; ++g){ float s=bf16_to_f32(sr[g]); float b=bf16_to_f32(br[g]); uint xbase=g*GROUP; uint wbase=g*16; for(uint j=0;j<16;++j){ uint p=wr[wbase+j]; uint k=xbase+j*4; float4 qv=float4((p)&255u,(p>>8)&255u,(p>>16)&255u,(p>>24)&255u); float4 xv=float4(x[k],x[k+1],x[k+2],x[k+3]); acc += dot(xv, qv*s + b); } } y[row]=acc; }";
  id<MTLLibrary> lib=[dev newLibraryWithSource:src options:nil error:&err]; if(!lib){fprintf(stderr,"compile %s\n",err.description.UTF8String);return 1;}
  id<MTLFunction> fn=[lib newFunctionWithName:@"q8_affine_mv"]; id<MTLComputePipelineState> ps=[dev newComputePipelineStateWithFunction:fn error:&err]; if(!ps){fprintf(stderr,"pipeline %s\n",err.description.UTF8String);return 1;}
  id<MTLBuffer> wb=[dev newBufferWithBytesNoCopy:(uint8_t*)m.base+woff length:wbytes options:MTLResourceStorageModeShared deallocator:nil];
  id<MTLBuffer> sb=[dev newBufferWithBytesNoCopy:(uint8_t*)m.base+soff length:sbytes options:MTLResourceStorageModeShared deallocator:nil];
  id<MTLBuffer> bb=[dev newBufferWithBytesNoCopy:(uint8_t*)m.base+boff length:sbytes options:MTLResourceStorageModeShared deallocator:nil];
  id<MTLBuffer> xb=[dev newBufferWithBytes:x length:IN*4 options:MTLResourceStorageModeShared]; id<MTLBuffer> yb=[dev newBufferWithLength:OUT*4 options:MTLResourceStorageModeShared];
  if(!wb||!sb||!bb){fprintf(stderr,"no-copy buffer failed wb=%p sb=%p bb=%p\n",wb,sb,bb);return 1;}
  id<MTLCommandBuffer> cb=[q commandBuffer]; id<MTLComputeCommandEncoder> ce=[cb computeCommandEncoder]; [ce setComputePipelineState:ps]; [ce setBuffer:wb offset:0 atIndex:0]; [ce setBuffer:sb offset:0 atIndex:1]; [ce setBuffer:bb offset:0 atIndex:2]; [ce setBuffer:xb offset:0 atIndex:3]; [ce setBuffer:yb offset:0 atIndex:4]; NSUInteger tg=MIN((NSUInteger)ps.maxTotalThreadsPerThreadgroup,256); [ce dispatchThreads:MTLSizeMake(OUT,1,1) threadsPerThreadgroup:MTLSizeMake(tg,1,1)]; [ce endEncoding]; [cb commit]; [cb waitUntilCompleted]; if(cb.status==MTLCommandBufferStatusError){fprintf(stderr,"command %s\n",cb.error.description.UTF8String);return 1;}
  float *y=(float*)yb.contents; double maxabs=0,mae=0,mse=0; uint32_t maxi=0; for(uint32_t i=0;i<OUT;i++){ double e=(double)y[i]-ref[i]; double a=fabs(e); if(a>maxabs){maxabs=a;maxi=i;} mae+=a; mse+=e*e;} mae/=OUT; mse=sqrt(mse/OUT); double dot=0,na=0,nb=0; for(uint32_t i=0;i<OUT;i++){dot+=(double)y[i]*ref[i];na+=(double)y[i]*y[i];nb+=(double)ref[i]*ref[i];} double cos=dot/sqrt(na*nb);
  printf("first8 native:"); for(int i=0;i<8;i++) printf(" %.8g",y[i]); printf("\n"); printf("first8 ref:   "); for(int i=0;i<8;i++) printf(" %.8g",ref[i]); printf("\n"); printf("max_abs=%.9g at %u native=%.9g ref=%.9g mae=%.9g rmse=%.9g cosine=%.12f\n",maxabs,maxi,y[maxi],ref[maxi],mae,mse,cos);
  double rmax=0,rmae=0,rmse2=0; uint32_t rexact=0; for(uint32_t i=0;i<OUT;i++){ union { float f; uint32_t u; } v={.f=y[i]}; uint32_t lsb=(v.u>>16)&1u; v.u += 0x7fffu + lsb; v.u &= 0xffff0000u; double e=(double)v.f-ref[i], a=fabs(e); if(a>rmax) rmax=a; rmae+=a; rmse2+=e*e; if(v.f==ref[i]) rexact++; } printf("native->bf16: exact=%u/%u (%.2f%%) max_abs=%.9g mae=%.9g rmse=%.9g\n",rexact,OUT,100.0*rexact/OUT,rmax,rmae/OUT,sqrt(rmse2/OUT));
  wb=nil;sb=nil;bb=nil;xb=nil;yb=nil;ps=nil;fn=nil;lib=nil;q=nil;
 }
 free(x);free(ref);unmap_file(&m);return 0;
}
