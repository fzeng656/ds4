#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "qn_qwen4_layer.h"
#include "qn_runtime.h"
#include <mach/mach_time.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

struct qn_qwen4_layer {
    uint32_t layer_index;
    qn_file_map shard8, shard9, shard10;
    __strong id<MTLDevice> device;
    __strong id<MTLCommandQueue> queue;
    __strong id<MTLLibrary> library;
    __strong NSMutableDictionary<NSString*, id<MTLComputePipelineState>> *pipelines;
    __strong NSMutableDictionary<NSString*, id<MTLBuffer>> *scratch;
    qn_affine_metal_view expert_gate[512];
    qn_affine_metal_view expert_up[512];
    qn_affine_metal_view expert_down[512];
    uint8_t expert_cached[512];
    qn_affine_metal_view router_view;
    qn_affine_metal_view shared_gate_proj;
    qn_affine_metal_view shared_up_proj;
    qn_affine_metal_view shared_down_proj;
    qn_bf16_metal_view shared_gate_weight;
};

static void seterr(char *e,size_t n,const char *s){if(e&&n)snprintf(e,n,"%s",s?s:"unknown error");}
static double now_ms(void){static mach_timebase_info_data_t tb;if(!tb.denom)mach_timebase_info(&tb);return (double)mach_absolute_time()*tb.numer/tb.denom/1e6;}

static NSString *kernel_source(void){return @R"METAL(
#include <metal_stdlib>
using namespace metal;
inline float bf16f(ushort v){return as_type<float>((uint)v<<16);} 
inline float sigm(float x){return 1.0f/(1.0f+exp(-x));}

kernel void q8_mv(device const uint*w[[buffer(0)]],device const ushort*sc[[buffer(1)]],device const ushort*bi[[buffer(2)]],device const float*x[[buffer(3)]],device float*y[[buffer(4)]],constant uint&O[[buffer(5)]],constant uint&I[[buffer(6)]],uint tid[[thread_position_in_grid]],uint lane[[thread_index_in_simdgroup]]){
 uint r=tid/32;if(r>=O)return;uint G=I/64,W=I/4;float a=0;
 for(uint g=0;g<G;g++){float s=bf16f(sc[r*G+g]),b=bf16f(bi[r*G+g]);for(uint j=lane;j<16;j+=32){uint p=w[r*W+g*16+j],k=g*64+j*4;float4 q=float4(p&255u,(p>>8)&255u,(p>>16)&255u,(p>>24)&255u);a+=dot(float4(x[k],x[k+1],x[k+2],x[k+3]),q*s+b);}}
 a=simd_sum(a);if(lane==0)y[r]=a;
}
kernel void q4_mv(device const uint*w[[buffer(0)]],device const ushort*sc[[buffer(1)]],device const ushort*bi[[buffer(2)]],device const float*x[[buffer(3)]],device float*y[[buffer(4)]],constant uint&O[[buffer(5)]],constant uint&I[[buffer(6)]],uint tid[[thread_position_in_grid]],uint lane[[thread_index_in_simdgroup]]){
 uint r=tid/32;if(r>=O)return;uint G=I/64,W=I/8;float a=0;
 for(uint g=0;g<G;g++){float s=bf16f(sc[r*G+g]),b=bf16f(bi[r*G+g]);for(uint j=lane;j<8;j+=32){uint p=w[r*W+g*8+j],k=g*64+j*8;for(uint t=0;t<8;t++)a+=x[k+t]*((float)((p>>(4*t))&15u)*s+b);}}
 a=simd_sum(a);if(lane==0)y[r]=a;
}
kernel void silu_mul(device const float*g[[buffer(0)]],device const float*u[[buffer(1)]],device float*h[[buffer(2)]],constant uint&N[[buffer(3)]],uint i[[thread_position_in_grid]]){if(i<N){float z=g[i];h[i]=(z*sigm(z))*u[i];}}
kernel void zero_f32(device float*x[[buffer(0)]],constant uint&N[[buffer(1)]],uint i[[thread_position_in_grid]]){if(i<N)x[i]=0;}
kernel void scale_add(device const float*x[[buffer(0)]],device float*y[[buffer(1)]],constant float&w[[buffer(2)]],constant uint&N[[buffer(3)]],uint i[[thread_position_in_grid]]){if(i<N)y[i]+=x[i]*w;}
kernel void bf16_dot_sigmoid(device const ushort*w[[buffer(0)]],device const float*x[[buffer(1)]],device float*y[[buffer(2)]],uint tid[[thread_position_in_grid]],uint lane[[thread_index_in_simdgroup]]){float a=0;for(uint j=lane;j<2560;j+=32)a+=bf16f(w[j])*x[j];a=simd_sum(a);if(lane==0)y[0]=sigm(a);}
kernel void mul_scalar(device float*x[[buffer(0)]],device const float*s[[buffer(1)]],constant uint&N[[buffer(2)]],uint i[[thread_position_in_grid]]){if(i<N)x[i]*=s[0];}
)METAL";}

static id<MTLComputePipelineState> pipeline_for(qn_qwen4_layer *l, NSString *name, NSError **err){
    id<MTLComputePipelineState> p=l->pipelines[name]; if(p)return p;
    id<MTLFunction> f=[l->library newFunctionWithName:name]; if(!f)return nil;
    p=[l->device newComputePipelineStateWithFunction:f error:err]; if(p)l->pipelines[name]=p; return p;
}
static id<MTLBuffer> scratch(qn_qwen4_layer *l,NSString *name,NSUInteger bytes){
    id<MTLBuffer>b=l->scratch[name]; if(!b || b.length<bytes){b=[l->device newBufferWithLength:bytes options:MTLResourceStorageModeShared]; if(b)l->scratch[name]=b;} return b;
}

int qn_qwen4_layer_open(qn_qwen4_layer **out,const qn_qwen4_layer_config *c,char *err,size_t errlen){
 if(!out||!c||!c->shard8_path||!c->shard9_path||!c->shard10_path){seterr(err,errlen,"invalid layer config");return -1;}
 if(c->layer_index!=11){seterr(err,errlen,"phase2c runtime currently supports layer 11 only");return -1;}
 qn_qwen4_layer*l=calloc(1,sizeof(*l)); if(!l){seterr(err,errlen,"calloc failed");return -1;} l->shard8.fd=l->shard9.fd=l->shard10.fd=-1;l->layer_index=c->layer_index;
 if(qn_file_map_open(&l->shard8,c->shard8_path,err,errlen)||qn_file_map_open(&l->shard9,c->shard9_path,err,errlen)||qn_file_map_open(&l->shard10,c->shard10_path,err,errlen)){qn_qwen4_layer_close(l);return -1;}
 @autoreleasepool{l->device=MTLCreateSystemDefaultDevice();if(!l->device){seterr(err,errlen,"no Metal device");qn_qwen4_layer_close(l);return -1;}l->queue=[l->device newCommandQueue];NSError*e=nil;l->library=[l->device newLibraryWithSource:kernel_source() options:nil error:&e];if(!l->library){seterr(err,errlen,e.description.UTF8String);qn_qwen4_layer_close(l);return -1;}l->pipelines=[NSMutableDictionary dictionary];l->scratch=[NSMutableDictionary dictionary];
   for(NSString*n in @[@"q8_mv",@"q4_mv",@"silu_mul",@"zero_f32",@"scale_add",@"bf16_dot_sigmoid",@"mul_scalar"]){if(!pipeline_for(l,n,&e)){seterr(err,errlen,e.description.UTF8String);qn_qwen4_layer_close(l);return -1;}}
   scratch(l,@"router_logits",512*4);scratch(l,@"expert_gate",640*4);scratch(l,@"expert_up",640*4);scratch(l,@"expert_hidden",640*4);scratch(l,@"expert_down",2560*4);scratch(l,@"routed",2560*4);scratch(l,@"shared",2560*4);scratch(l,@"moe_total",2560*4);scratch(l,@"shared_gate",4);
   qn_affine_desc rad={471869896,473180616,473221576,512,2560,64,8},sgd={475003336,476641736,476692936,640,2560,64,8},sud={476744136,478382536,478433736,640,2560,64,8},sdd={473262536,474900936,474952136,2560,640,64,8};
   if(qn_affine_make_view(l->device,&l->shard10,&rad,&l->router_view,err,errlen)||qn_affine_make_view(l->device,&l->shard10,&sgd,&l->shared_gate_proj,err,errlen)||qn_affine_make_view(l->device,&l->shard10,&sud,&l->shared_up_proj,err,errlen)||qn_affine_make_view(l->device,&l->shard10,&sdd,&l->shared_down_proj,err,errlen)||qn_bf16_make_view(l->device,&l->shard10,478484936,2560,&l->shared_gate_weight,err,errlen)){qn_qwen4_layer_close(l);return -1;}
 }
 *out=l;return 0;
}


static void encode_affine(id<MTLComputeCommandEncoder>ce,id<MTLComputePipelineState>ps,qn_affine_metal_view*v,id<MTLBuffer>x,id<MTLBuffer>y,uint32_t O,uint32_t I){
 [ce setComputePipelineState:ps];[ce setBuffer:v->weight offset:0 atIndex:0];[ce setBuffer:v->scales offset:0 atIndex:1];[ce setBuffer:v->biases offset:0 atIndex:2];[ce setBuffer:x offset:0 atIndex:3];[ce setBuffer:y offset:0 atIndex:4];[ce setBytes:&O length:4 atIndex:5];[ce setBytes:&I length:4 atIndex:6];[ce dispatchThreads:MTLSizeMake((NSUInteger)O*32,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
}
typedef struct{float p;uint32_t i;} qn_route_pair;
static int route_cmp(const void*a,const void*b){float x=((const qn_route_pair*)a)->p,y=((const qn_route_pair*)b)->p;return x<y?1:x>y?-1:0;}

int qn_qwen4_layer_forward_moe(qn_qwen4_layer*l,const float*hidden,float*output,uint32_t selected_ids[10],float selected_weights[10],double*elapsed_ms,char*err,size_t errlen){
 if(!l||!hidden||!output){seterr(err,errlen,"invalid moe arguments");return -1;}
 @autoreleasepool{
  NSError*e=nil;id<MTLComputePipelineState>q8=pipeline_for(l,@"q8_mv",&e),q4=pipeline_for(l,@"q4_mv",&e),sm=pipeline_for(l,@"silu_mul",&e),zp=pipeline_for(l,@"zero_f32",&e),sa=pipeline_for(l,@"scale_add",&e),bd=pipeline_for(l,@"bf16_dot_sigmoid",&e),ms=pipeline_for(l,@"mul_scalar",&e);
  if(!q8||!q4||!sm||!zp||!sa||!bd||!ms){seterr(err,errlen,e.description.UTF8String);return -1;}
  id<MTLBuffer>xb=scratch(l,@"moe_input",2560*4),logb=scratch(l,@"router_logits",512*4),gb=scratch(l,@"expert_gate",640*4),ub=scratch(l,@"expert_up",640*4),hb=scratch(l,@"expert_hidden",640*4),db=scratch(l,@"expert_down",2560*4),routed=scratch(l,@"routed",2560*4),shared=scratch(l,@"shared",2560*4),total=scratch(l,@"moe_total",2560*4),sgate=scratch(l,@"shared_gate",4);
  memcpy(xb.contents,hidden,2560*4);
  double t0=now_ms();id<MTLCommandBuffer>cb=[l->queue commandBuffer];id<MTLComputeCommandEncoder>ce=[cb computeCommandEncoder];encode_affine(ce,q8,&l->router_view,xb,logb,512,2560);[ce endEncoding];[cb commit];[cb waitUntilCompleted];if(cb.status==MTLCommandBufferStatusError){seterr(err,errlen,cb.error.description.UTF8String);return -1;}
  float*lg=logb.contents,vmax=-INFINITY;for(uint32_t i=0;i<512;i++)if(lg[i]>vmax)vmax=lg[i];double den=0;for(uint32_t i=0;i<512;i++)den+=exp((double)lg[i]-vmax);qn_route_pair p[512];for(uint32_t i=0;i<512;i++){p[i].p=(float)(exp((double)lg[i]-vmax)/den);p[i].i=i;}qsort(p,512,sizeof(*p),route_cmp);for(uint32_t j=0;j<10;j++){if(selected_ids)selected_ids[j]=p[j].i;if(selected_weights)selected_weights[j]=p[j].p;}
  uint32_t N2560=2560,N640=640;cb=[l->queue commandBuffer];ce=[cb computeCommandEncoder];[ce setComputePipelineState:zp];[ce setBuffer:routed offset:0 atIndex:0];[ce setBytes:&N2560 length:4 atIndex:1];[ce dispatchThreads:MTLSizeMake(2560,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
  for(uint32_t j=0;j<10;j++){uint64_t E=p[j].i,gw=E*(uint64_t)640*320*4,gs=E*(uint64_t)640*40*2,dw=E*(uint64_t)2560*80*4,ds=E*(uint64_t)2560*10*2;qn_affine_desc gd={896+gw,419431296+gs,445645696+gs,640,2560,64,4},ud={471860096+gw,891290496+gs,917504896+gs,640,2560,64,4},dd={10696+dw,419441096+ds,445655496+ds,2560,640,64,4};if(!l->expert_cached[E]){if(qn_affine_make_view(l->device,&l->shard9,&gd,&l->expert_gate[E],err,errlen)||qn_affine_make_view(l->device,&l->shard9,&ud,&l->expert_up[E],err,errlen)||qn_affine_make_view(l->device,&l->shard10,&dd,&l->expert_down[E],err,errlen))return -1;l->expert_cached[E]=1;}encode_affine(ce,q4,&l->expert_gate[E],xb,gb,640,2560);encode_affine(ce,q4,&l->expert_up[E],xb,ub,640,2560);[ce setComputePipelineState:sm];[ce setBuffer:gb offset:0 atIndex:0];[ce setBuffer:ub offset:0 atIndex:1];[ce setBuffer:hb offset:0 atIndex:2];[ce setBytes:&N640 length:4 atIndex:3];[ce dispatchThreads:MTLSizeMake(640,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];encode_affine(ce,q4,&l->expert_down[E],hb,db,2560,640);float w=p[j].p;[ce setComputePipelineState:sa];[ce setBuffer:db offset:0 atIndex:0];[ce setBuffer:routed offset:0 atIndex:1];[ce setBytes:&w length:4 atIndex:2];[ce setBytes:&N2560 length:4 atIndex:3];[ce dispatchThreads:MTLSizeMake(2560,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];}
  encode_affine(ce,q8,&l->shared_gate_proj,xb,gb,640,2560);encode_affine(ce,q8,&l->shared_up_proj,xb,ub,640,2560);[ce setComputePipelineState:sm];[ce setBuffer:gb offset:0 atIndex:0];[ce setBuffer:ub offset:0 atIndex:1];[ce setBuffer:hb offset:0 atIndex:2];[ce setBytes:&N640 length:4 atIndex:3];[ce dispatchThreads:MTLSizeMake(640,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];encode_affine(ce,q8,&l->shared_down_proj,hb,shared,2560,640);[ce setComputePipelineState:bd];[ce setBuffer:l->shared_gate_weight.buffer offset:0 atIndex:0];[ce setBuffer:xb offset:0 atIndex:1];[ce setBuffer:sgate offset:0 atIndex:2];[ce dispatchThreads:MTLSizeMake(32,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[ce setComputePipelineState:ms];[ce setBuffer:shared offset:0 atIndex:0];[ce setBuffer:sgate offset:0 atIndex:1];[ce setBytes:&N2560 length:4 atIndex:2];[ce dispatchThreads:MTLSizeMake(2560,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[ce setComputePipelineState:sa];float one=1.0f;[ce setBuffer:shared offset:0 atIndex:0];[ce setBuffer:routed offset:0 atIndex:1];[ce setBytes:&one length:4 atIndex:2];[ce setBytes:&N2560 length:4 atIndex:3];[ce dispatchThreads:MTLSizeMake(2560,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[ce endEncoding];[cb commit];[cb waitUntilCompleted];if(cb.status==MTLCommandBufferStatusError){seterr(err,errlen,cb.error.description.UTF8String);return -1;}memcpy(output,routed.contents,2560*4);if(elapsed_ms)*elapsed_ms=now_ms()-t0;
 }
 return 0;
}

/* Phase2C landing point: resource ownership and persistent scratch are now runtime-level.
 * Forward is wired incrementally below; returning an explicit error prevents accidental use
 * of the old test-only path as a production API. */
int qn_qwen4_layer_forward_decode(qn_qwen4_layer*l,const qn_qwen4_decode_input*in,qn_qwen4_decode_output*out,char*err,size_t errlen){
 (void)l;(void)in;(void)out;seterr(err,errlen,"forward_decode not wired yet; phase2c resource runtime only");return -1;
}
void qn_qwen4_layer_close(qn_qwen4_layer*l){if(!l)return;@autoreleasepool{l->scratch=nil;l->pipelines=nil;l->library=nil;l->queue=nil;l->device=nil;}qn_file_map_close(&l->shard8);qn_file_map_close(&l->shard9);qn_file_map_close(&l->shard10);free(l);}
