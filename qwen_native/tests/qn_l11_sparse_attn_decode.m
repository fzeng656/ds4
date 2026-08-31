#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import "qn_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static void *rd(const char*p,size_t n){FILE*f=fopen(p,"rb");if(!f){perror("fopen");exit(1);}void*x=malloc(n);if(fread(x,1,n,f)!=n){fprintf(stderr,"short %s\n",p);exit(1);}fclose(f);return x;}
static void estat(const char*n,const float*a,const float*b,size_t N){double mx=0,ma=0,se=0,dot=0,aa=0,bb=0;for(size_t i=0;i<N;i++){double e=(double)a[i]-b[i],z=fabs(e);if(z>mx)mx=z;ma+=z;se+=e*e;dot+=(double)a[i]*b[i];aa+=(double)a[i]*a[i];bb+=(double)b[i]*b[i];}printf("%s max=%.6g mae=%.6g rmse=%.6g cos=%.12f\n",n,mx,ma/N,sqrt(se/N),dot/sqrt(aa*bb));}
static NSString*src(void){return @R"METAL(
#include <metal_stdlib>
using namespace metal;
// correctness-first: one 32-lane SIMD group per Q head. 24 Q heads, 2 KV heads => 12 Q/KV groups.
kernel void sparse_gqa_decode(device const float*q[[buffer(0)]],device const float*k[[buffer(1)]],device const float*v[[buffer(2)]],device const int*ids[[buffer(3)]],device float*out[[buffer(4)]],constant uint&nsel[[buffer(5)]],uint tid[[thread_position_in_grid]],uint lane[[thread_index_in_simdgroup]]){
 uint h=tid/32;if(h>=24)return;uint kvh=h/12;float maxs=-INFINITY;
 // pass1 max score
 for(uint t=0;t<nsel;t++){int id=ids[t];float z=0;for(uint j=lane;j<256;j+=32)z+=q[h*256+j]*k[((uint)id*2+kvh)*256+j];z=simd_sum(z)*(1.0f/16.0f);if(lane==0)maxs=max(maxs,z);}
 maxs=simd_broadcast_first(maxs);
 // pass2 denominator
 float den=0;for(uint t=0;t<nsel;t++){int id=ids[t];float z=0;for(uint j=lane;j<256;j+=32)z+=q[h*256+j]*k[((uint)id*2+kvh)*256+j];z=simd_sum(z)*(1.0f/16.0f);if(lane==0)den+=exp(z-maxs);}den=simd_broadcast_first(den);
 // pass3 weighted value, each lane owns 8 output dims
 for(uint j=lane;j<256;j+=32){float acc=0;for(uint t=0;t<nsel;t++){int id=ids[t];float z=0;for(uint d=0;d<256;d++)z+=q[h*256+d]*k[((uint)id*2+kvh)*256+d];float p=exp(z*(1.0f/16.0f)-maxs)/den;acc+=p*v[((uint)id*2+kvh)*256+j];}out[h*256+j]=acc;}
}
kernel void apply_gate(device float*x[[buffer(0)]],device const float*g[[buffer(1)]],uint i[[thread_position_in_grid]]){if(i<6144)x[i]*=1.0f/(1.0f+exp(-g[i]));}
kernel void q8_mv(device const uint*w[[buffer(0)]],device const ushort*sc[[buffer(1)]],device const ushort*bi[[buffer(2)]],device const float*x[[buffer(3)]],device float*y[[buffer(4)]],constant uint&outdim[[buffer(5)]],constant uint&indim[[buffer(6)]],uint tid[[thread_position_in_grid]],uint lane[[thread_index_in_simdgroup]]){
 uint row=tid/32;if(row>=outdim)return;uint groups=indim/64,wcols=indim/4;float acc=0;device const uint*wr=w+row*wcols;device const ushort*sr=sc+row*groups;device const ushort*br=bi+row*groups;for(uint g=0;g<groups;g++){float s=as_type<float>((uint)sr[g]<<16),b=as_type<float>((uint)br[g]<<16);uint xb=g*64,wb=g*16;for(uint j=lane;j<16;j+=32){uint p=wr[wb+j],kk=xb+j*4;float4 qq=float4(p&255u,(p>>8)&255u,(p>>16)&255u,(p>>24)&255u);float4 xv=float4(x[kk],x[kk+1],x[kk+2],x[kk+3]);acc+=dot(xv,qq*s+b);}}acc=simd_sum(acc);if(lane==0)y[row]=acc;
}
)METAL";}
int main(int ac,char**av){if(ac!=10){fprintf(stderr,"usage: %s shard q k v gate ids attn_ref out_ref nsel\n",av[0]);return 2;}uint32_t N=atoi(av[9]);float*q=rd(av[2],6144*4),*k=rd(av[3],(size_t)N*2*256*4),*v=rd(av[4],(size_t)N*2*256*4),*g=rd(av[5],6144*4);int*ids=rd(av[6],N*4);float*ar=rd(av[7],6144*4),*or=rd(av[8],2560*4);qn_file_map fm;char er[256];if(qn_file_map_open(&fm,av[1],er,sizeof(er))){fprintf(stderr,"%s\n",er);return 1;}
 @autoreleasepool{id<MTLDevice>d=MTLCreateSystemDefaultDevice();id<MTLCommandQueue>cq=[d newCommandQueue];NSError*e=nil;id<MTLLibrary>lib=[d newLibraryWithSource:src() options:nil error:&e];if(!lib){fprintf(stderr,"compile %s\n",e.description.UTF8String);return 1;}id<MTLComputePipelineState>att=[d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"sparse_gqa_decode"] error:&e],gatep=[d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"apply_gate"] error:&e],mm=[d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"q8_mv"] error:&e];
 qn_affine_desc od={488690120,504418760,504910280,2560,6144,64,8};qn_affine_metal_view ov={0};if(qn_affine_make_view(d,&fm,&od,&ov,er,sizeof(er))){fprintf(stderr,"%s\n",er);return 1;}id<MTLBuffer>qb=[d newBufferWithBytes:q length:6144*4 options:MTLResourceStorageModeShared],kb=[d newBufferWithBytes:k length:(size_t)N*2*256*4 options:MTLResourceStorageModeShared],vb=[d newBufferWithBytes:v length:(size_t)N*2*256*4 options:MTLResourceStorageModeShared],gb=[d newBufferWithBytes:g length:6144*4 options:MTLResourceStorageModeShared],ib=[d newBufferWithBytes:ids length:N*4 options:MTLResourceStorageModeShared],ao=[d newBufferWithLength:6144*4 options:MTLResourceStorageModeShared],oo=[d newBufferWithLength:2560*4 options:MTLResourceStorageModeShared];
 // IDs supplied to this test are compact 0..N-1 into K/V buffers, so normalize identity.
 int*ii=(int*)ib.contents;for(uint32_t z=0;z<N;z++)ii[z]=(int)z;
 id<MTLCommandBuffer>cb=[cq commandBuffer];id<MTLComputeCommandEncoder>ce=[cb computeCommandEncoder];[ce setComputePipelineState:att];[ce setBuffer:qb offset:0 atIndex:0];[ce setBuffer:kb offset:0 atIndex:1];[ce setBuffer:vb offset:0 atIndex:2];[ce setBuffer:ib offset:0 atIndex:3];[ce setBuffer:ao offset:0 atIndex:4];[ce setBytes:&N length:4 atIndex:5];[ce dispatchThreads:MTLSizeMake(24*32,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[ce setComputePipelineState:gatep];[ce setBuffer:ao offset:0 atIndex:0];[ce setBuffer:gb offset:0 atIndex:1];[ce dispatchThreads:MTLSizeMake(6144,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];uint32_t O=2560,I=6144;[ce setComputePipelineState:mm];[ce setBuffer:ov.weight offset:0 atIndex:0];[ce setBuffer:ov.scales offset:0 atIndex:1];[ce setBuffer:ov.biases offset:0 atIndex:2];[ce setBuffer:ao offset:0 atIndex:3];[ce setBuffer:oo offset:0 atIndex:4];[ce setBytes:&O length:4 atIndex:5];[ce setBytes:&I length:4 atIndex:6];[ce dispatchThreads:MTLSizeMake((NSUInteger)O*32,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[ce endEncoding];[cb commit];[cb waitUntilCompleted];if(cb.status==MTLCommandBufferStatusError){fprintf(stderr,"cmd %s\n",cb.error.description.UTF8String);return 1;}estat("attn_gated",ao.contents,ar,6144);estat("o_proj",oo.contents,or,2560);
 }
 free(q);free(k);free(v);free(g);free(ids);free(ar);free(or);qn_file_map_close(&fm);return 0;}
