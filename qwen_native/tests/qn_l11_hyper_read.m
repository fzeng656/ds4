#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import "qn_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static void *rd(const char*p,size_t n){FILE*f=fopen(p,"rb");if(!f){perror("fopen");exit(1);}void*x=malloc(n);if(fread(x,1,n,f)!=n){fprintf(stderr,"short %s\n",p);exit(1);}fclose(f);return x;}
static void estat(const char*n,const float*a,const float*b,size_t N){double mx=0,ma=0,se=0;for(size_t i=0;i<N;i++){double e=(double)a[i]-b[i],z=fabs(e);if(z>mx)mx=z;ma+=z;se+=e*e;}printf("%s max=%.6g mae=%.6g rmse=%.6g\n",n,mx,ma/N,sqrt(se/N));}
static NSString*src(void){return @R"METAL(
#include <metal_stdlib>
using namespace metal;
inline float bf16f(ushort v){return as_type<float>((uint)v<<16);}
inline float sigm(float x){return 1.0f/(1.0f+exp(-x));}

kernel void group_rms4(device const float*x[[buffer(0)]],device const ushort*w[[buffer(1)]],device float*y[[buffer(2)]],uint tid[[thread_position_in_grid]],uint lane[[thread_index_in_simdgroup]]){
 uint g=tid/32;if(g>=4)return;float ss=0;for(uint j=lane;j<2560;j+=32){float z=x[g*2560+j];ss+=z*z;}ss=simd_sum(ss);float inv=rsqrt(ss/2560.0f+1e-6f);for(uint j=lane;j<2560;j+=32)y[g*2560+j]=x[g*2560+j]*inv*bf16f(w[g*2560+j]);
}
kernel void q8_mv(device const uint*w[[buffer(0)]],device const ushort*sc[[buffer(1)]],device const ushort*bi[[buffer(2)]],device const float*x[[buffer(3)]],device float*y[[buffer(4)]],constant uint&outdim[[buffer(5)]],constant uint&indim[[buffer(6)]],uint tid[[thread_position_in_grid]],uint lane[[thread_index_in_simdgroup]]){
 uint row=tid/32;if(row>=outdim)return;uint groups=indim/64,wcols=indim/4;float acc=0;device const uint*wr=w+row*wcols;device const ushort*sr=sc+row*groups;device const ushort*br=bi+row*groups;for(uint g=0;g<groups;g++){float s=bf16f(sr[g]),b=bf16f(br[g]);uint xb=g*64,wb=g*16;for(uint j=lane;j<16;j+=32){uint p=wr[wb+j],k=xb+j*4;float4 q=float4(p&255u,(p>>8)&255u,(p>>16)&255u,(p>>24)&255u);float4 xv=float4(x[k],x[k+1],x[k+2],x[k+3]);acc+=dot(xv,q*s+b);}}acc=simd_sum(acc);if(lane==0)y[row]=acc;
}
kernel void silu_div4(device float*x[[buffer(0)]],constant uint&n[[buffer(1)]],uint i[[thread_position_in_grid]]){if(i<n){float z=x[i]*0.25f;x[i]=z*sigm(z);}}
kernel void sigmoid_inplace(device float*x[[buffer(0)]],constant uint&n[[buffer(1)]],uint i[[thread_position_in_grid]]){if(i<n)x[i]=sigm(x[i]);}
kernel void mix4(device const float*xnorm[[buffer(0)]],device const float*mixw[[buffer(1)]],device float*out[[buffer(2)]],uint i[[thread_position_in_grid]]){if(i<2560){float z=0;for(uint g=0;g<4;g++)z+=xnorm[g*2560+i]*mixw[g*2560+i];out[i]=z*0.25f;}}
kernel void bf16_mv4(device const ushort*w[[buffer(0)]],device const float*x[[buffer(1)]],device float*y[[buffer(2)]],uint tid[[thread_position_in_grid]],uint lane[[thread_index_in_simdgroup]]){uint row=tid/32;if(row>=4)return;float a=0;for(uint j=lane;j<10240;j+=32)a+=bf16f(w[row*10240+j])*x[j];a=simd_sum(a);if(lane==0)y[row]=2.0f*sigm(a*0.25f);}
)METAL";}
static void encq(id<MTLComputeCommandEncoder>ce,id<MTLComputePipelineState>ps,qn_affine_metal_view*v,id<MTLBuffer>x,id<MTLBuffer>y,uint32_t O,uint32_t I){[ce setComputePipelineState:ps];[ce setBuffer:v->weight offset:0 atIndex:0];[ce setBuffer:v->scales offset:0 atIndex:1];[ce setBuffer:v->biases offset:0 atIndex:2];[ce setBuffer:x offset:0 atIndex:3];[ce setBuffer:y offset:0 atIndex:4];[ce setBytes:&O length:4 atIndex:5];[ce setBytes:&I length:4 atIndex:6];[ce dispatchThreads:MTLSizeMake((NSUInteger)O*32,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];}
int main(int ac,char**av){if(ac!=6){fprintf(stderr,"usage: %s shard hyper.bin mixed_ref.bin inj_ref.bin xnorm_ref.bin\n",av[0]);return 2;}float*x=rd(av[2],10240*4),*mr=rd(av[3],2560*4),*ir=rd(av[4],4*4),*nr=rd(av[5],10240*4);qn_file_map fm;char er[256];if(qn_file_map_open(&fm,av[1],er,sizeof(er))){fprintf(stderr,"%s\n",er);return 1;}
 @autoreleasepool{id<MTLDevice>d=MTLCreateSystemDefaultDevice();id<MTLCommandQueue>cq=[d newCommandQueue];NSError*e=nil;id<MTLLibrary>lib=[d newLibraryWithSource:src() options:nil error:&e];if(!lib){fprintf(stderr,"%s\n",e.description.UTF8String);return 1;}
 id<MTLComputePipelineState> norm=[d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"group_rms4"] error:&e];
 id<MTLComputePipelineState> q8=[d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"q8_mv"] error:&e];
 id<MTLComputePipelineState> silu=[d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"silu_div4"] error:&e];
 id<MTLComputePipelineState> sig=[d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"sigmoid_inplace"] error:&e];
 id<MTLComputePipelineState> mix=[d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mix4"] error:&e];
 id<MTLComputePipelineState> inj=[d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"bf16_mv4"] error:&e];
 qn_bf16_metal_view nw={0},iw={0};qn_bf16_make_view(d,&fm,485631776,10240,&nw,er,sizeof(er));qn_bf16_make_view(d,&fm,485549856,4*10240,&iw,er,sizeof(er));qn_affine_desc dd={485652256,488929056,489031456,320,10240,64,8},ud={489133856,492410656,492513056,10240,320,64,8};qn_affine_metal_view dv={0},uv={0};qn_affine_make_view(d,&fm,&dd,&dv,er,sizeof(er));qn_affine_make_view(d,&fm,&ud,&uv,er,sizeof(er));
 id<MTLBuffer>xb=[d newBufferWithBytes:x length:10240*4 options:MTLResourceStorageModeShared],xn=[d newBufferWithLength:10240*4 options:MTLResourceStorageModeShared],lo=[d newBufferWithLength:320*4 options:MTLResourceStorageModeShared],mw=[d newBufferWithLength:10240*4 options:MTLResourceStorageModeShared],mo=[d newBufferWithLength:2560*4 options:MTLResourceStorageModeShared],io=[d newBufferWithLength:4*4 options:MTLResourceStorageModeShared];uint32_t n320=320,n10240=10240;
 id<MTLCommandBuffer>cb=[cq commandBuffer];id<MTLComputeCommandEncoder>ce=[cb computeCommandEncoder];[ce setComputePipelineState:norm];[ce setBuffer:xb offset:0 atIndex:0];[ce setBuffer:nw.buffer offset:0 atIndex:1];[ce setBuffer:xn offset:0 atIndex:2];[ce dispatchThreads:MTLSizeMake(4*32,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];encq(ce,q8,&dv,xn,lo,320,10240);[ce setComputePipelineState:silu];[ce setBuffer:lo offset:0 atIndex:0];[ce setBytes:&n320 length:4 atIndex:1];[ce dispatchThreads:MTLSizeMake(320,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];encq(ce,q8,&uv,lo,mw,10240,320);[ce setComputePipelineState:sig];[ce setBuffer:mw offset:0 atIndex:0];[ce setBytes:&n10240 length:4 atIndex:1];[ce dispatchThreads:MTLSizeMake(10240,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[ce setComputePipelineState:mix];[ce setBuffer:xn offset:0 atIndex:0];[ce setBuffer:mw offset:0 atIndex:1];[ce setBuffer:mo offset:0 atIndex:2];[ce dispatchThreads:MTLSizeMake(2560,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[ce setComputePipelineState:inj];[ce setBuffer:iw.buffer offset:0 atIndex:0];[ce setBuffer:xn offset:0 atIndex:1];[ce setBuffer:io offset:0 atIndex:2];[ce dispatchThreads:MTLSizeMake(4*32,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[ce endEncoding];[cb commit];[cb waitUntilCompleted];if(cb.status==MTLCommandBufferStatusError){fprintf(stderr,"%s\n",cb.error.description.UTF8String);return 1;}estat("xnorm",xn.contents,nr,10240);estat("mixed",mo.contents,mr,2560);estat("inject",io.contents,ir,4);
 }
 free(x);free(mr);free(ir);free(nr);qn_file_map_close(&fm);return 0;}
