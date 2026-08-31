#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import "qn_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

static void *rd(const char*p,size_t n){FILE*f=fopen(p,"rb");if(!f){perror("fopen");exit(1);}void*x=malloc(n);if(fread(x,1,n,f)!=n){fprintf(stderr,"short %s\n",p);exit(1);}fclose(f);return x;}
static NSString*src(void){return @R"METAL(
#include <metal_stdlib>
using namespace metal;
inline float bf16f(ushort v){return as_type<float>((uint)v<<16);}
inline float rope_inv(uint j){return pow(10000000.0f,-(2.0f*(float)j)/64.0f);}

kernel void q8_mm_rows(device const uint*w[[buffer(0)]],device const ushort*sc[[buffer(1)]],device const ushort*bi[[buffer(2)]],device const float*x[[buffer(3)]],device float*y[[buffer(4)]],constant uint&outdim[[buffer(5)]],constant uint&indim[[buffer(6)]],constant uint&tokens[[buffer(7)]],uint tid[[thread_position_in_grid]],uint lane[[thread_index_in_simdgroup]]){
 uint gi=tid/32,tok=gi/outdim,row=gi%outdim;if(tok>=tokens)return;uint groups=indim/64,wcols=indim/4;float acc=0;device const uint*wr=w+row*wcols;device const ushort*sr=sc+row*groups;device const ushort*br=bi+row*groups;device const float*xr=x+tok*indim;
 for(uint g=0;g<groups;g++){float s=bf16f(sr[g]),b=bf16f(br[g]);uint xb=g*64,wb=g*16;for(uint j=lane;j<16;j+=32){uint p=wr[wb+j],k=xb+j*4;float4 q=float4(p&255u,(p>>8)&255u,(p>>16)&255u,(p>>24)&255u);float4 xv=float4(xr[k],xr[k+1],xr[k+2],xr[k+3]);acc+=dot(xv,q*s+b);}}
 acc=simd_sum(acc);if(lane==0)y[tok*outdim+row]=acc;
}

kernel void q_last_norm_rope(device const float*raw[[buffer(0)]],device const ushort*w[[buffer(1)]],device float*out[[buffer(2)]],constant uint&tokens[[buffer(3)]],uint tid[[thread_position_in_grid]],uint lane[[thread_index_in_simdgroup]]){
 uint h=tid/32;if(h>=4)return;uint tok=tokens-1;device const float*q=raw+tok*640+h*128;threadgroup float normv;float ss=0;for(uint j=lane;j<128;j+=32){float z=q[j];ss+=z*z;}ss=simd_sum(ss);float inv=rsqrt(ss/128.0f+1e-6f);
 // store normalized first, then rotate first 64 dims as two 32-d halves
 for(uint j=lane;j<128;j+=32)out[h*128+j]=q[j]*inv*bf16f(w[j]);
 threadgroup_barrier(mem_flags::mem_threadgroup); // output is device mem; command ordering within thread is enough for own lanes below
 for(uint j=lane;j<32;j+=32){ /* lane-specific vector loop below */ }
 for(uint j=lane;j<32;j+=32){float a=out[h*128+j],b=out[h*128+32+j],ang=(float)tok*rope_inv(j),c=cos(ang),s=sin(ang);out[h*128+j]=a*c-b*s;out[h*128+32+j]=b*c+a*s;}
}

kernel void k_compress_norm_rope(device const float*raw[[buffer(0)]],device const ushort*w[[buffer(1)]],device float*out[[buffer(2)]],constant uint&blocks[[buffer(3)]],uint tid[[thread_position_in_grid]],uint lane[[thread_index_in_simdgroup]]){
 uint block=tid/32;if(block>=blocks)return;threadgroup float tmp[128];float ss=0;for(uint j=lane;j<128;j+=32){float z=0;for(uint t=0;t<4;t++)z+=raw[(block*4+t)*640+512+j];z*=0.25f;tmp[j]=z;ss+=z*z;}ss=simd_sum(ss);float inv=rsqrt(ss/128.0f+1e-6f);for(uint j=lane;j<128;j+=32)tmp[j]=tmp[j]*inv*bf16f(w[j]);threadgroup_barrier(mem_flags::mem_threadgroup);
 uint pos=block*4;for(uint j=lane;j<32;j+=32){float a=tmp[j],b=tmp[32+j],ang=(float)pos*rope_inv(j),c=cos(ang),s=sin(ang);tmp[j]=a*c-b*s;tmp[32+j]=b*c+a*s;}threadgroup_barrier(mem_flags::mem_threadgroup);for(uint j=lane;j<128;j+=32)out[block*128+j]=tmp[j];
}

kernel void score_blocks(device const float*q[[buffer(0)]],device const float*k[[buffer(1)]],device float*scores[[buffer(2)]],constant uint&blocks[[buffer(3)]],uint tid[[thread_position_in_grid]],uint lane[[thread_index_in_simdgroup]]){
 uint b=tid/32;if(b>=blocks)return;float total=0;for(uint h=0;h<4;h++){float dotv=0;for(uint j=lane;j<128;j+=32)dotv+=q[h*128+j]*k[b*128+j];dotv=simd_sum(dotv);if(lane==0)total+=max(dotv,0.0f);}
 if(lane==0)scores[b]=total*0.08838834764831845f; // 1/sqrt(128)
}
)METAL";}

typedef struct{float s;uint32_t i;} Pair;static int cmp(const void*a,const void*b){float x=((const Pair*)a)->s,y=((const Pair*)b)->s;return x<y?1:x>y?-1:0;}
int main(int ac,char**av){if(ac!=7){fprintf(stderr,"usage: %s shard hidden scores_ref selected_ref tokens blocks\n",av[0]);return 2;}uint32_t T=atoi(av[5]),B=atoi(av[6]);float*x=rd(av[2],(size_t)T*2560*4),*sr=rd(av[3],(size_t)B*4);unsigned char*mr=rd(av[4],B);qn_file_map fm;char er[256];if(qn_file_map_open(&fm,av[1],er,sizeof(er))){fprintf(stderr,"%s\n",er);return 1;}
 @autoreleasepool{id<MTLDevice>d=MTLCreateSystemDefaultDevice();id<MTLCommandQueue>cq=[d newCommandQueue];NSError*e=nil;id<MTLLibrary>lib=[d newLibraryWithSource:src() options:nil error:&e];if(!lib){fprintf(stderr,"compile %s\n",e.description.UTF8String);return 1;}id<MTLComputePipelineState>mm=[d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"q8_mm_rows"] error:&e],qp=[d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"q_last_norm_rope"] error:&e],kp=[d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"k_compress_norm_rope"] error:&e],sp=[d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"score_blocks"] error:&e];
 qn_affine_desc idesc={485555656,487194056,487245256,640,2560,64,8};qn_affine_metal_view iv={0};qn_affine_make_view(d,&fm,&idesc,&iv,er,sizeof(er));qn_bf16_metal_view qw={0},kw={0};qn_bf16_make_view(d,&fm,487296712,128,&qw,er,sizeof(er));qn_bf16_make_view(d,&fm,487296456,128,&kw,er,sizeof(er));
 id<MTLBuffer>xb=[d newBufferWithBytes:x length:(size_t)T*2560*4 options:MTLResourceStorageModeShared],raw=[d newBufferWithLength:(size_t)T*640*4 options:MTLResourceStorageModeShared],q=[d newBufferWithLength:512*4 options:MTLResourceStorageModeShared],k=[d newBufferWithLength:(size_t)B*128*4 options:MTLResourceStorageModeShared],scores=[d newBufferWithLength:(size_t)B*4 options:MTLResourceStorageModeShared];uint32_t O=640,I=2560;
 id<MTLCommandBuffer>cb=[cq commandBuffer];id<MTLComputeCommandEncoder>ce=[cb computeCommandEncoder];[ce setComputePipelineState:mm];[ce setBuffer:iv.weight offset:0 atIndex:0];[ce setBuffer:iv.scales offset:0 atIndex:1];[ce setBuffer:iv.biases offset:0 atIndex:2];[ce setBuffer:xb offset:0 atIndex:3];[ce setBuffer:raw offset:0 atIndex:4];[ce setBytes:&O length:4 atIndex:5];[ce setBytes:&I length:4 atIndex:6];[ce setBytes:&T length:4 atIndex:7];[ce dispatchThreads:MTLSizeMake((NSUInteger)T*O*32,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
 [ce setComputePipelineState:qp];[ce setBuffer:raw offset:0 atIndex:0];[ce setBuffer:qw.buffer offset:0 atIndex:1];[ce setBuffer:q offset:0 atIndex:2];[ce setBytes:&T length:4 atIndex:3];[ce dispatchThreads:MTLSizeMake(4*32,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
 [ce setComputePipelineState:kp];[ce setBuffer:raw offset:0 atIndex:0];[ce setBuffer:kw.buffer offset:0 atIndex:1];[ce setBuffer:k offset:0 atIndex:2];[ce setBytes:&B length:4 atIndex:3];[ce dispatchThreads:MTLSizeMake((NSUInteger)B*32,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
 [ce setComputePipelineState:sp];[ce setBuffer:q offset:0 atIndex:0];[ce setBuffer:k offset:0 atIndex:1];[ce setBuffer:scores offset:0 atIndex:2];[ce setBytes:&B length:4 atIndex:3];[ce dispatchThreads:MTLSizeMake((NSUInteger)B*32,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[ce endEncoding];[cb commit];[cb waitUntilCompleted];if(cb.status==MTLCommandBufferStatusError){fprintf(stderr,"cmd %s\n",cb.error.description.UTF8String);return 1;}
 float*ss=(float*)scores.contents;double mx=0,ma=0,se=0;for(uint32_t i=0;i<B;i++){double z=fabs((double)ss[i]-sr[i]);if(z>mx)mx=z;ma+=z;se+=z*z;}printf("scores max=%.6g mae=%.6g rmse=%.6g\n",mx,ma/B,sqrt(se/B));Pair*p=malloc((size_t)B*sizeof(*p));for(uint32_t i=0;i<B;i++)p[i]=(Pair){ss[i],i};qsort(p,B,sizeof(*p),cmp);unsigned char*mask=calloc(B,1);uint32_t top=B<512?B:512;for(uint32_t i=0;i<top;i++)mask[p[i].i]=1;uint32_t mismatch=0;for(uint32_t i=0;i<B;i++)if(mask[i]!=mr[i])mismatch++;printf("selected blocks=%u/%u mismatch=%u",top,B,mismatch);if(B>512){printf(" excluded_native=");for(uint32_t i=0;i<B;i++)if(!mask[i])printf("%u ",i);printf(" excluded_ref=");for(uint32_t i=0;i<B;i++)if(!mr[i])printf("%u ",i);}printf("\n");free(p);free(mask);
 }
 free(x);free(sr);free(mr);qn_file_map_close(&fm);return 0;}
