#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import "../qn_runtime.h"
#include <mach/mach_time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

typedef struct { const char *name; qn_affine_desc d; const char *xfile; const char *rfile; } Case;
static void *readall(const char*p,size_t n){FILE*f=fopen(p,"rb");if(!f){perror(p);exit(1);}void*x=malloc(n);if(fread(x,1,n,f)!=n){fprintf(stderr,"short %s\n",p);exit(1);}fclose(f);return x;}
static double nowms(void){static mach_timebase_info_data_t tb; if(!tb.denom) mach_timebase_info(&tb); return (double)mach_absolute_time()*tb.numer/tb.denom/1e6;}
static float bf16round(float f){union{float f;uint32_t u;}v={.f=f};uint32_t l=(v.u>>16)&1;v.u+=0x7fff+l;v.u&=0xffff0000u;return v.f;}
int main(int argc,char**argv){if(argc!=6){fprintf(stderr,"usage: %s shard x2560 x6144 refdir iters\n",argv[0]);return 2;}int iters=atoi(argv[5]);char rp[1024];
 Case cs[]={
  {"q",{505402312,536859592,537842632,12288,2560,64,8},argv[2],NULL},
  {"k",{487297480,488608200,488649160,512,2560,64,8},argv[2],NULL},
  {"v",{538825672,540136392,540177352,512,2560,64,8},argv[2],NULL},
  {"o",{488690120,504418760,504910280,2560,6144,64,8},argv[3],NULL},
 }; qn_file_map map;char err[256];if(qn_file_map_open(&map,argv[1],err,sizeof(err))){fprintf(stderr,"map: %s\n",err);return 1;}
 @autoreleasepool {id<MTLDevice>dev=MTLCreateSystemDefaultDevice();id<MTLCommandQueue>cq=[dev newCommandQueue];NSError*e=nil;
 NSString*src=@"#include <metal_stdlib>\nusing namespace metal;\ninline float bf(ushort v){return as_type<float>((uint)v<<16);}\nstruct P{uint outd;uint ind;uint groups;uint wcols;};\nkernel void q8mv(device const uint*w[[buffer(0)]],device const ushort*s[[buffer(1)]],device const ushort*b[[buffer(2)]],device const float*x[[buffer(3)]],device float*y[[buffer(4)]],constant P&p[[buffer(5)]],uint row[[thread_position_in_grid]]){if(row>=p.outd)return;float a=0;device const uint*wr=w+row*p.wcols;device const ushort*sr=s+row*p.groups;device const ushort*br=b+row*p.groups;for(uint g=0;g<p.groups;g++){float sc=bf(sr[g]),bi=bf(br[g]);uint xb=g*64,wb=g*16;for(uint j=0;j<16;j++){uint z=wr[wb+j],k=xb+j*4;float4 q=float4(z&255u,(z>>8)&255u,(z>>16)&255u,(z>>24)&255u);a+=dot(float4(x[k],x[k+1],x[k+2],x[k+3]),q*sc+bi);}}y[row]=a;}";
 id<MTLLibrary>lib=[dev newLibraryWithSource:src options:nil error:&e];if(!lib){fprintf(stderr,"compile %s\n",e.description.UTF8String);return 1;}id<MTLComputePipelineState>ps=[dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"q8mv"] error:&e];
 for(int ci=0;ci<4;ci++){Case*c=&cs[ci];snprintf(rp,sizeof(rp),"%s/%s.bin",argv[4],c->name);float*x=readall(c->xfile,(size_t)c->d.in_dim*4),*ref=readall(rp,(size_t)c->d.out_dim*4);qn_affine_metal_view vw;if(qn_affine_make_view(dev,&map,&c->d,&vw,err,sizeof(err))){fprintf(stderr,"%s view %s\n",c->name,err);return 1;}id<MTLBuffer>xb=[dev newBufferWithBytes:x length:c->d.in_dim*4 options:MTLResourceStorageModeShared],yb=[dev newBufferWithLength:c->d.out_dim*4 options:MTLResourceStorageModeShared];struct{uint32_t outd,ind,groups,wcols;}p={c->d.out_dim,c->d.in_dim,c->d.in_dim/64,c->d.in_dim/4};
   // warm
   for(int z=-1;z<iters;z++){double t0=nowms();id<MTLCommandBuffer>cb=[cq commandBuffer];id<MTLComputeCommandEncoder>ce=[cb computeCommandEncoder];[ce setComputePipelineState:ps];[ce setBuffer:vw.weight offset:0 atIndex:0];[ce setBuffer:vw.scales offset:0 atIndex:1];[ce setBuffer:vw.biases offset:0 atIndex:2];[ce setBuffer:xb offset:0 atIndex:3];[ce setBuffer:yb offset:0 atIndex:4];[ce setBytes:&p length:sizeof(p) atIndex:5];[ce dispatchThreads:MTLSizeMake(c->d.out_dim,1,1) threadsPerThreadgroup:MTLSizeMake(MIN((NSUInteger)256,ps.maxTotalThreadsPerThreadgroup),1,1)];[ce endEncoding];[cb commit];[cb waitUntilCompleted];double dt=nowms()-t0;if(z>=0){ static double dummy=0; dummy+=dt; if(z==iters-1){ (void)dummy; } } if(z==0) printf("%s first_ms=%.4f ",c->name,dt); if(z==iters-1){ /* recompute timing below */ }}
   // timing second loop for clean aggregate
   double t0=nowms();for(int z=0;z<iters;z++){id<MTLCommandBuffer>cb=[cq commandBuffer];id<MTLComputeCommandEncoder>ce=[cb computeCommandEncoder];[ce setComputePipelineState:ps];[ce setBuffer:vw.weight offset:0 atIndex:0];[ce setBuffer:vw.scales offset:0 atIndex:1];[ce setBuffer:vw.biases offset:0 atIndex:2];[ce setBuffer:xb offset:0 atIndex:3];[ce setBuffer:yb offset:0 atIndex:4];[ce setBytes:&p length:sizeof(p) atIndex:5];[ce dispatchThreads:MTLSizeMake(c->d.out_dim,1,1) threadsPerThreadgroup:MTLSizeMake(MIN((NSUInteger)256,ps.maxTotalThreadsPerThreadgroup),1,1)];[ce endEncoding];[cb commit];[cb waitUntilCompleted];}double avg=(nowms()-t0)/iters;
   float*y=(float*)yb.contents;double ma=0,me=0,se=0,dot=0,na=0,nb=0;uint32_t exact=0;for(uint32_t i=0;i<c->d.out_dim;i++){double d=(double)y[i]-ref[i],a=fabs(d);if(a>ma)ma=a;me+=a;se+=d*d;dot+=(double)y[i]*ref[i];na+=(double)y[i]*y[i];nb+=(double)ref[i]*ref[i];if(bf16round(y[i])==ref[i])exact++;}printf("avg_ms=%.4f exact_bf16=%u/%u max=%.6g mae=%.6g rmse=%.6g cos=%.12f\n",avg,exact,c->d.out_dim,ma,me/c->d.out_dim,sqrt(se/c->d.out_dim),dot/sqrt(na*nb));free(x);free(ref);}
 }
 qn_file_map_close(&map);return 0;}
