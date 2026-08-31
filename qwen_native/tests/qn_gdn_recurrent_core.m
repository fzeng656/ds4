#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
static void*rd(const char*p,size_t n){FILE*f=fopen(p,"rb");if(!f){perror("fopen");exit(1);}void*x=malloc(n);if(fread(x,1,n,f)!=n){fprintf(stderr,"short %s\n",p);exit(2);}fclose(f);return x;}
static void statv(const char*n,const float*a,const float*b,size_t N){double mx=0,ma=0,se=0;for(size_t i=0;i<N;i++){double e=(double)a[i]-b[i],z=fabs(e);if(z>mx)mx=z;ma+=z;se+=e*e;}printf("%s max=%.7g mae=%.7g rmse=%.7g\n",n,mx,ma/N,sqrt(se/N));}
static NSString*src(void){return @R"METAL(
#include <metal_stdlib>
using namespace metal;
kernel void gdn_step(device const float*q0[[buffer(0)]],device const float*k0[[buffer(1)]],device const float*v[[buffer(2)]],device const float*b0[[buffer(3)]],device const float*g[[buffer(4)]],device float*S[[buffer(5)]],device float*out[[buffer(6)]],uint3 tg[[threadgroup_position_in_grid]],uint tid[[thread_index_in_threadgroup]]){
 uint h=tg.x;if(h>=48)return;threadgroup float q[128],k[128],delta[128];threadgroup float sq[256],sk[256];float qv=0,kv=0;if(tid<128){qv=q0[h*128+tid];kv=k0[h*128+tid];sq[tid]=qv*qv;sk[tid]=kv*kv;}else{sq[tid]=0;sk[tid]=0;}threadgroup_barrier(mem_flags::mem_threadgroup);for(uint s=128;s>0;s>>=1){if(tid<s){sq[tid]+=sq[tid+s];sk[tid]+=sk[tid+s];}threadgroup_barrier(mem_flags::mem_threadgroup);}if(tid<128){q[tid]=qv*rsqrt(sq[0]+1e-6f)*0.08838834764831845f;k[tid]=kv*rsqrt(sk[0]+1e-6f);}threadgroup_barrier(mem_flags::mem_threadgroup);
 // one thread owns each V column: predict, delta, then update all K rows
 if(tid<128){float decay=exp(g[h]),pred=0;uint base=h*16384;for(uint j=0;j<128;j++){float s=S[base+j*128+tid]*decay;S[base+j*128+tid]=s;pred+=s*k[j];}float beta=1.0f/(1.0f+exp(-b0[h]));float d=(v[h*128+tid]-pred)*beta;delta[tid]=d;for(uint j=0;j<128;j++)S[base+j*128+tid]+=k[j]*d;}
 threadgroup_barrier(mem_flags::mem_threadgroup);
 // one thread owns output V dim; q dot updated state column
 if(tid<128){uint base=h*16384;float a=0;for(uint j=0;j<128;j++)a+=q[j]*S[base+j*128+tid];out[h*128+tid]=a;}
}
)METAL";}
int main(){size_t hv=48*128,hs=48*128*128;float*q=rd("/tmp/qn_gdn_core/q.bin",hv*4),*k=rd("/tmp/qn_gdn_core/k.bin",hv*4),*v=rd("/tmp/qn_gdn_core/v.bin",hv*4),*b=rd("/tmp/qn_gdn_core/b.bin",48*4),*g=rd("/tmp/qn_gdn_core/g.bin",48*4),*S=rd("/tmp/qn_gdn_core/state.bin",hs*4),*oref=rd("/tmp/qn_gdn_core/out.bin",hv*4),*sref=rd("/tmp/qn_gdn_core/state_out.bin",hs*4);@autoreleasepool{id<MTLDevice>d=MTLCreateSystemDefaultDevice();NSError*e=nil;id<MTLLibrary>l=[d newLibraryWithSource:src() options:nil error:&e];if(!l){fprintf(stderr,"%s\n",e.description.UTF8String);return 1;}id<MTLComputePipelineState>p=[d newComputePipelineStateWithFunction:[l newFunctionWithName:@"gdn_step"] error:&e];id<MTLCommandQueue>cq=[d newCommandQueue];id<MTLBuffer>qb=[d newBufferWithBytes:q length:hv*4 options:MTLResourceStorageModeShared],kb=[d newBufferWithBytes:k length:hv*4 options:MTLResourceStorageModeShared],vb=[d newBufferWithBytes:v length:hv*4 options:MTLResourceStorageModeShared],bb=[d newBufferWithBytes:b length:48*4 options:MTLResourceStorageModeShared],gb=[d newBufferWithBytes:g length:48*4 options:MTLResourceStorageModeShared],sb=[d newBufferWithBytes:S length:hs*4 options:MTLResourceStorageModeShared],ob=[d newBufferWithLength:hv*4 options:MTLResourceStorageModeShared];id<MTLCommandBuffer>cb=[cq commandBuffer];id<MTLComputeCommandEncoder>ce=[cb computeCommandEncoder];[ce setComputePipelineState:p];[ce setBuffer:qb offset:0 atIndex:0];[ce setBuffer:kb offset:0 atIndex:1];[ce setBuffer:vb offset:0 atIndex:2];[ce setBuffer:bb offset:0 atIndex:3];[ce setBuffer:gb offset:0 atIndex:4];[ce setBuffer:sb offset:0 atIndex:5];[ce setBuffer:ob offset:0 atIndex:6];[ce dispatchThreadgroups:MTLSizeMake(48,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[ce endEncoding];[cb commit];[cb waitUntilCompleted];if(cb.status==MTLCommandBufferStatusError){fprintf(stderr,"cmd %s\n",cb.error.description.UTF8String);return 1;}statv("out",ob.contents,oref,hv);statv("state",sb.contents,sref,hs);}
return 0;}
