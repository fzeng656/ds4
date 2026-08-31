#import <Foundation/Foundation.h>
#include "qn_manifest.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
struct qn_manifest{ __strong NSDictionary *tensors; __strong NSMutableArray *owned; };
static void er(char*e,size_t n,const char*s){if(e&&n)snprintf(e,n,"%s",s?s:"manifest error");}
int qn_manifest_open(qn_manifest **out,const char*path,char*e,size_t n){if(!out||!path){er(e,n,"invalid manifest args");return -1;}@autoreleasepool{NSData*d=[NSData dataWithContentsOfFile:[NSString stringWithUTF8String:path]];if(!d){er(e,n,"cannot read manifest");return -1;}NSError*x=nil;NSDictionary*j=[NSJSONSerialization JSONObjectWithData:d options:0 error:&x];NSDictionary*t=j[@"tensors"];if(!t){er(e,n,x?x.description.UTF8String:"manifest missing tensors");return -1;}qn_manifest*m=calloc(1,sizeof(*m));m->tensors=t;m->owned=[NSMutableArray array];*out=m;}return 0;}
int qn_manifest_tensor(qn_manifest*m,const char*name,qn_tensor_meta*out,char*e,size_t n){if(!m||!name||!out){er(e,n,"invalid tensor lookup");return -1;}@autoreleasepool{NSString*k=[NSString stringWithUTF8String:name];NSDictionary*v=m->tensors[k];if(!v){er(e,n,"tensor not found");return -1;}NSString*sh=v[@"shard"],*dt=v[@"dtype"];NSArray*shape=v[@"shape"];[m->owned addObject:k];[m->owned addObject:sh];[m->owned addObject:dt];memset(out,0,sizeof(*out));out->name=k.UTF8String;out->shard=sh.UTF8String;out->dtype=dt.UTF8String;out->file_offset=[v[@"file_offset"] unsignedLongLongValue];out->payload_bytes=[v[@"payload_bytes"] unsignedLongLongValue];out->ndim=(uint32_t)MIN((NSUInteger)4,shape.count);for(uint32_t i=0;i<out->ndim;i++)out->shape[i]=[shape[i] unsignedLongLongValue];}return 0;}
int qn_manifest_layer_tensor(qn_manifest*m,uint32_t layer,const char*suffix,qn_tensor_meta*out,char*e,size_t n){char k[512];snprintf(k,sizeof(k),"language_model.model.layers.%u.%s",layer,suffix);return qn_manifest_tensor(m,k,out,e,n);}
void qn_manifest_close(qn_manifest*m){if(!m)return;@autoreleasepool{m->tensors=nil;m->owned=nil;}free(m);}
