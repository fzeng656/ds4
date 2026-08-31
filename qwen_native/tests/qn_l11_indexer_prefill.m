#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import "qn_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static void *rd(const char*p,size_t n){FILE*f=fopen(p,"rb");if(!f){perror("fopen");exit(1);}void*x=malloc(n);if(fread(x,1,n,f)!=n){fprintf(stderr,"short %s\n",p);exit(1);}fclose(f);return x;}
static void staterr(const char*n,const float*a,const float*b,size_t N){double mx=0,ma=0,se=0;for(size_t i=0;i<N;i++){double e=(double)a[i]-b[i],z=fabs(e);if(z>mx)mx=z;ma+=z;se+=e*e;}printf("%s max=%.5g mae=%.5g rmse=%.5g\n",n,mx,ma/N,sqrt(se/N));}
static NSString*src(void){return @R"METAL(
#include <metal_stdlib>
using namespace metal;
inline float bf16f(ushort v){return as_type<float>((uint)v<<16);}

kernel void q8_mm_rows(device const uint*w[[buffer(0)]],device const ushort*sc[[buffer(1)]],device const ushort*bi[[buffer(2)]],device const float*x[[buffer(3)]],device float*y[[buffer(4)]],constant uint&outdim[[buffer(5)]],constant uint&indim[[buffer(6)]],constant uint&tokens[[buffer(7)]],uint tid[[thread_position_in_grid]],uint lane[[thread_index_in_simdgroup]]){
 uint gidx=tid/32; uint tok=gidx/outdim,row=gidx%outdim; if(tok>=tokens)return; uint groups=indim/64,wcols=indim/4; float acc=0; device const uint*wr=w+row*wcols;device const ushort*sr=sc+row*groups;device const ushort*br=bi+row*groups;device const float*xr=x+tok*indim;
 for(uint g=0;g<groups;g++){float s=bf16f(sr[g]),b=bf16f(br[g]);uint xb=g*64,wb=g*16;for(uint j=lane;j<16;j+=32){uint p=wr[wb+j],k=xb+j*4;float4 q=float4(p&255u,(p>>8)&255u,(p>>16)&255u,(p>>24)&255u);float4 xv=float4(xr[k],xr[k+1],xr[k+2],xr[k+3]);acc+=dot(xv,q*s+b);}}
 acc=simd_sum(acc);if(lane==0)y[tok*outdim+row]=acc;
}

kernel void index_q_norm(device const float*raw[[buffer(0)]],device const ushort*w[[buffer(1)]],device float*out[[buffer(2)]],constant uint&tokens[[buffer(3)]],uint tid[[thread_position_in_grid]],uint lane[[thread_index_in_simdgroup]]){
 uint gh=tid/32;uint tok=gh/4,h=gh%4;if(tok>=tokens)return;device const float*q=raw+tok*640+h*128;float ss=0;for(uint j=lane;j<128;j+=32){float z=q[j];ss+=z*z;}ss=simd_sum(ss);float inv=rsqrt(ss/128.0f+1e-6f);for(uint j=lane;j<128;j+=32)out[tok*512+h*128+j]=q[j]*inv*bf16f(w[j]);
}

kernel void index_k_compress_norm(device const float*raw[[buffer(0)]],device const ushort*w[[buffer(1)]],device float*out[[buffer(2)]],constant uint&blocks[[buffer(3)]],uint tid[[thread_position_in_grid]],uint lane[[thread_index_in_simdgroup]]){
 uint block=tid/32;if(block>=blocks)return;float vals[4];float ss=0;for(uint j=lane;j<128;j+=32){float z=0;for(uint t=0;t<4;t++)z+=raw[(block*4+t)*640+512+j];z*=0.25f;vals[j/32]=z;ss+=z*z;}ss=simd_sum(ss);float inv=rsqrt(ss/128.0f+1e-6f);for(uint j=lane;j<128;j+=32){float z=0;for(uint t=0;t<4;t++)z+=raw[(block*4+t)*640+512+j];z*=0.25f;out[block*128+j]=z*inv*bf16f(w[j]);}
}
)METAL";}

int main(int ac,char**av){if(ac!=6){fprintf(stderr,"usage: %s shard hidden.bin qref.bin kref.bin tokens\n",av[0]);return 2;}uint32_t T=(uint32_t)atoi(av[5]);if(!T||T%4){fprintf(stderr,"tokens must be multiple of 4\n");return 2;}uint32_t B=T/4;float*x=rd(av[2],(size_t)T*2560*4),*qr=rd(av[3],(size_t)T*512*4),*kr=rd(av[4],(size_t)B*128*4);qn_file_map fm;char er[256];if(qn_file_map_open(&fm,av[1],er,sizeof(er))){fprintf(stderr,"%s\n",er);return 1;}
 @autoreleasepool{id<MTLDevice>d=MTLCreateSystemDefaultDevice();id<MTLCommandQueue>cq=[d newCommandQueue];NSError*e=nil;id<MTLLibrary>lib=[d newLibraryWithSource:src() options:nil error:&e];if(!lib){fprintf(stderr,"%s\n",e.description.UTF8String);return 1;}id<MTLComputePipelineState>mm=[d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"q8_mm_rows"] error:&e],qp=[d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"index_q_norm"] error:&e],kp=[d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"index_k_compress_norm"] error:&e];
 qn_affine_desc idesc={485555656,487194056,487245256,640,2560,64,8};qn_affine_metal_view iv={0};if(qn_affine_make_view(d,&fm,&idesc,&iv,er,sizeof(er))){fprintf(stderr,"%s\n",er);return 1;}qn_bf16_metal_view qw={0},kw={0};qn_bf16_make_view(d,&fm,487296712,128,&qw,er,sizeof(er));qn_bf16_make_view(d,&fm,487296456,128,&kw,er,sizeof(er));
 id<MTLBuffer>xb=[d newBufferWithBytes:x length:(size_t)T*2560*4 options:MTLResourceStorageModeShared],raw=[d newBufferWithLength:(size_t)T*640*4 options:MTLResourceStorageModeShared],qo=[d newBufferWithLength:(size_t)T*512*4 options:MTLResourceStorageModeShared],ko=[d newBufferWithLength:(size_t)B*128*4 options:MTLResourceStorageModeShared];uint32_t O=640,I=2560;
 id<MTLCommandBuffer>cb=[cq commandBuffer];id<MTLComputeCommandEncoder>ce=[cb computeCommandEncoder];[ce setComputePipelineState:mm];[ce setBuffer:iv.weight offset:0 atIndex:0];[ce setBuffer:iv.scales offset:0 atIndex:1];[ce setBuffer:iv.biases offset:0 atIndex:2];[ce setBuffer:xb offset:0 atIndex:3];[ce setBuffer:raw offset:0 atIndex:4];[ce setBytes:&O length:4 atIndex:5];[ce setBytes:&I length:4 atIndex:6];[ce setBytes:&T length:4 atIndex:7];[ce dispatchThreads:MTLSizeMake((NSUInteger)T*O*32,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
 [ce setComputePipelineState:qp];[ce setBuffer:raw offset:0 atIndex:0];[ce setBuffer:qw.buffer offset:0 atIndex:1];[ce setBuffer:qo offset:0 atIndex:2];[ce setBytes:&T length:4 atIndex:3];[ce dispatchThreads:MTLSizeMake((NSUInteger)T*4*32,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
 [ce setComputePipelineState:kp];[ce setBuffer:raw offset:0 atIndex:0];[ce setBuffer:kw.buffer offset:0 atIndex:1];[ce setBuffer:ko offset:0 atIndex:2];[ce setBytes:&B length:4 atIndex:3];[ce dispatchThreads:MTLSizeMake((NSUInteger)B*32,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[ce endEncoding];[cb commit];[cb waitUntilCompleted];if(cb.status==MTLCommandBufferStatusError){fprintf(stderr,"%s\n",cb.error.description.UTF8String);return 1;}
 staterr("index_q",qo.contents,qr,(size_t)T*512);staterr("index_k4",ko.contents,kr,(size_t)B*128);
 }
 free(x);free(qr);free(kr);qn_file_map_close(&fm);return 0;}
