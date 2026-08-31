#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "qn_runtime.h"
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
static void*rd(const char*p,size_t n){FILE*f=fopen(p,"rb");if(!f){perror(p);exit(1);}void*x=malloc(n);fread(x,1,n,f);fclose(f);return x;}
static float bf(uint16_t x){uint32_t u=(uint32_t)x<<16;float f;memcpy(&f,&u,4);return f;}
int main(){int64_t*ids=rd("/tmp/qn_ple/ids.bin",16*8);float*ref=rd("/tmp/qn_ple/embed.bin",2560*4);const char*F="/Users/mikuru/qwen38fn-official-pipeline/output/Qwen3.8-Flash-Next-OurAblit-Mixed-MLX-Serve/ngram_table.bin";int fd=open(F,O_RDONLY);off_t sz=lseek(fd,0,SEEK_END);uint8_t*m=mmap(0,sz,PROT_READ,MAP_SHARED,fd,0);uint64_t hl=*(uint64_t*)m,base=8+hl;uint64_t rows=320001536,wo=0,so=25600122880ULL,bo=28800138240ULL;float out[2560];for(int h=0;h<16;h++){uint64_t r=(uint64_t)ids[h];uint32_t*w=(uint32_t*)(m+base+wo+r*80);uint16_t*s=(uint16_t*)(m+base+so+r*10),*b=(uint16_t*)(m+base+bo+r*10);for(int j=0;j<160;j++){uint32_t p=w[j/8];int q=(p>>(4*(j&7)))&15,g=j/32;out[h*160+j]=q*bf(s[g])+bf(b[g]);}}double mx=0,ma=0;for(int i=0;i<2560;i++){double z=fabs((double)out[i]-ref[i]);if(z>mx)mx=z;ma+=z;}printf("ple_lookup max=%g mae=%g first=%g ref=%g header=%llu base=%llu size=%lld\n",mx,ma/2560,out[0],ref[0],(unsigned long long)hl,(unsigned long long)base,(long long)sz);munmap(m,sz);close(fd);}
