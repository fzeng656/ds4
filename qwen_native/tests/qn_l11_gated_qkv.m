#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import "qn_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mach/mach_time.h>

static void *read_all(const char *p,size_t n){FILE*f=fopen(p,"rb");if(!f){perror("fopen");exit(1);}void*x=malloc(n);if(fread(x,1,n,f)!=n){fprintf(stderr,"short %s\n",p);exit(1);}fclose(f);return x;}
static double now_ms(void){static mach_timebase_info_data_t tb; if(!tb.denom) mach_timebase_info(&tb); return (double)mach_absolute_time()*tb.numer/tb.denom/1e6;}

static NSString *kernels(void){return @R"METAL(
#include <metal_stdlib>
using namespace metal;
inline float bf16f(ushort v){return as_type<float>((uint)v<<16);}

kernel void q8_mv_simd(device const uint*w [[buffer(0)]],device const ushort*sc [[buffer(1)]],device const ushort*bi [[buffer(2)]],device const float*x [[buffer(3)]],device float*y [[buffer(4)]],constant uint&outdim [[buffer(5)]],constant uint&indim [[buffer(6)]],uint tid [[thread_position_in_grid]],uint lane [[thread_index_in_simdgroup]]){
 uint row=tid/32; if(row>=outdim)return; uint groups=indim/64,wcols=indim/4; float acc=0;
 device const uint*wr=w+row*wcols; device const ushort*sr=sc+row*groups; device const ushort*br=bi+row*groups;
 for(uint g=0;g<groups;g++){float s=bf16f(sr[g]),b=bf16f(br[g]); uint xb=g*64,wb=g*16;
   for(uint j=lane;j<16;j+=32){uint p=wr[wb+j],k=xb+j*4; float4 q=float4(p&255u,(p>>8)&255u,(p>>16)&255u,(p>>24)&255u); float4 xv=float4(x[k],x[k+1],x[k+2],x[k+3]); acc+=dot(xv,q*s+b);}}
 acc=simd_sum(acc); if(lane==0)y[row]=acc;
}

kernel void gated_qkv_post(device const float*qraw [[buffer(0)]],device const float*kraw [[buffer(1)]],device const float*vraw [[buffer(2)]],device const ushort*qw [[buffer(3)]],device const ushort*kw [[buffer(4)]],device float*qout [[buffer(5)]],device float*gate [[buffer(6)]],device float*kout [[buffer(7)]],device float*vout [[buffer(8)]],uint tid [[thread_position_in_grid]],uint lane [[thread_index_in_simdgroup]]){
 // One SIMD group per 256-wide head. 24 Q heads, 2 KV heads.
 uint head=tid/32; if(head<24){
   float ss=0; for(uint j=lane;j<256;j+=32){float z=qraw[head*512+j]; ss+=z*z;} ss=simd_sum(ss); float inv=rsqrt(ss/256.0f+1e-6f);
   for(uint j=lane;j<256;j+=32){qout[head*256+j]=qraw[head*512+j]*inv*bf16f(qw[j]); gate[head*256+j]=qraw[head*512+256+j];}
 } else if(head<26){uint kh=head-24; float ss=0; for(uint j=lane;j<256;j+=32){float z=kraw[kh*256+j];ss+=z*z;} ss=simd_sum(ss); float inv=rsqrt(ss/256.0f+1e-6f);
   for(uint j=lane;j<256;j+=32){kout[kh*256+j]=kraw[kh*256+j]*inv*bf16f(kw[j]);vout[kh*256+j]=vraw[kh*256+j];}
 }
}
)METAL";}

static void encode_mv(id<MTLComputeCommandEncoder>ce,id<MTLComputePipelineState>ps,qn_affine_metal_view*v,id<MTLBuffer>x,id<MTLBuffer>y,uint32_t out,uint32_t in){
 [ce setComputePipelineState:ps];[ce setBuffer:v->weight offset:0 atIndex:0];[ce setBuffer:v->scales offset:0 atIndex:1];[ce setBuffer:v->biases offset:0 atIndex:2];[ce setBuffer:x offset:0 atIndex:3];[ce setBuffer:y offset:0 atIndex:4];[ce setBytes:&out length:4 atIndex:5];[ce setBytes:&in length:4 atIndex:6];[ce dispatchThreads:MTLSizeMake((NSUInteger)out*32,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
}
static void errstat(const char*n,const float*a,const float*b,size_t N){double mx=0,se=0,ae=0;for(size_t i=0;i<N;i++){double e=(double)a[i]-b[i],z=fabs(e);if(z>mx)mx=z;ae+=z;se+=e*e;}printf("%s max=%.4g mae=%.4g rmse=%.4g\n",n,mx,ae/N,sqrt(se/N));}

int main(int ac,char**av){if(ac!=7){fprintf(stderr,"usage: %s shard x.bin qref gateref kref vref\n",av[0]);return 2;} char er[256];qn_file_map fm;if(qn_file_map_open(&fm,av[1],er,sizeof(er))){fprintf(stderr,"%s\n",er);return 1;}
 float*x=read_all(av[2],2560*4),*qr=read_all(av[3],6144*4),*gr=read_all(av[4],6144*4),*kr=read_all(av[5],512*4),*vr=read_all(av[6],512*4);
 @autoreleasepool{id<MTLDevice>d=MTLCreateSystemDefaultDevice();id<MTLCommandQueue>cq=[d newCommandQueue];NSError*e=nil;id<MTLLibrary>lib=[d newLibraryWithSource:kernels() options:nil error:&e];if(!lib){fprintf(stderr,"%s\n",e.description.UTF8String);return 1;}id<MTLComputePipelineState>mv=[d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"q8_mv_simd"] error:&e],post=[d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"gated_qkv_post"] error:&e];
 qn_affine_desc qd={505402312,536859592,537842632,12288,2560,64,8},kd={487297480,488608200,488649160,512,2560,64,8},vd={538825672,540136392,540177352,512,2560,64,8}; qn_affine_metal_view qv={0},kv={0},vv={0};qn_affine_make_view(d,&fm,&qd,&qv,er,sizeof(er));qn_affine_make_view(d,&fm,&kd,&kv,er,sizeof(er));qn_affine_make_view(d,&fm,&vd,&vv,er,sizeof(er));
 qn_bf16_metal_view qnw={0},knw={0}; qn_bf16_make_view(d,&fm,505401800,256,&qnw,er,sizeof(er));qn_bf16_make_view(d,&fm,487296968,256,&knw,er,sizeof(er));
 id<MTLBuffer>xb=[d newBufferWithBytes:x length:2560*4 options:MTLResourceStorageModeShared],qraw=[d newBufferWithLength:12288*4 options:MTLResourceStorageModeShared],kraw=[d newBufferWithLength:512*4 options:MTLResourceStorageModeShared],vraw=[d newBufferWithLength:512*4 options:MTLResourceStorageModeShared],qo=[d newBufferWithLength:6144*4 options:MTLResourceStorageModeShared],go=[d newBufferWithLength:6144*4 options:MTLResourceStorageModeShared],ko=[d newBufferWithLength:512*4 options:MTLResourceStorageModeShared],vo=[d newBufferWithLength:512*4 options:MTLResourceStorageModeShared];
 double t0=now_ms();id<MTLCommandBuffer>cb=[cq commandBuffer];id<MTLComputeCommandEncoder>ce=[cb computeCommandEncoder];encode_mv(ce,mv,&qv,xb,qraw,12288,2560);encode_mv(ce,mv,&kv,xb,kraw,512,2560);encode_mv(ce,mv,&vv,xb,vraw,512,2560);[ce setComputePipelineState:post];[ce setBuffer:qraw offset:0 atIndex:0];[ce setBuffer:kraw offset:0 atIndex:1];[ce setBuffer:vraw offset:0 atIndex:2];[ce setBuffer:qnw.buffer offset:0 atIndex:3];[ce setBuffer:knw.buffer offset:0 atIndex:4];[ce setBuffer:qo offset:0 atIndex:5];[ce setBuffer:go offset:0 atIndex:6];[ce setBuffer:ko offset:0 atIndex:7];[ce setBuffer:vo offset:0 atIndex:8];[ce dispatchThreads:MTLSizeMake(26*32,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[ce endEncoding];[cb commit];[cb waitUntilCompleted];double t1=now_ms();
 printf("gated_qkv total_ms=%.4f\n",t1-t0);errstat("qnorm",qo.contents,qr,6144);errstat("gate",go.contents,gr,6144);errstat("knorm",ko.contents,kr,512);errstat("value",vo.contents,vr,512);
 }
 free(x);free(qr);free(gr);free(kr);free(vr);qn_file_map_close(&fm);return 0;}
