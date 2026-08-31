#import "qn_runtime.h"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

static void qn_err(char *err,size_t n,const char *msg){ if(err&&n) snprintf(err,n,"%s",msg); }
int qn_file_map_open(qn_file_map *m,const char *path,char *err,size_t errlen){
    memset(m,0,sizeof(*m)); m->fd=-1; m->fd=open(path,O_RDONLY); if(m->fd<0){qn_err(err,errlen,strerror(errno));return -1;}
    struct stat st; if(fstat(m->fd,&st)){qn_err(err,errlen,strerror(errno));close(m->fd);m->fd=-1;return -1;}
    m->bytes=(uint64_t)st.st_size; m->base=mmap(NULL,(size_t)m->bytes,PROT_READ,MAP_SHARED,m->fd,0);
    if(m->base==MAP_FAILED){m->base=NULL;qn_err(err,errlen,strerror(errno));close(m->fd);m->fd=-1;return -1;}
    snprintf(m->path,sizeof(m->path),"%s",path); return 0;
}
void qn_file_map_close(qn_file_map *m){ if(!m)return; if(m->base){munmap(m->base,(size_t)m->bytes);m->base=NULL;} if(m->fd>=0){close(m->fd);m->fd=-1;} }
int qn_file_map_advise(qn_file_map *m,int will_need){ if(!m||!m->base)return -1; return posix_madvise(m->base,(size_t)m->bytes,will_need?POSIX_MADV_WILLNEED:POSIX_MADV_DONTNEED); }
int qn_file_map_invalidate(qn_file_map *m){ if(!m||!m->base)return -1; return msync(m->base,(size_t)m->bytes,MS_INVALIDATE); }

int qn_affine_make_view(id<MTLDevice> dev,const qn_file_map *m,const qn_affine_desc *d,qn_affine_metal_view *out,char *err,size_t errlen){
    if(!dev||!m||!m->base||!d||!out){qn_err(err,errlen,"invalid argument");return -1;}
    if((d->bits!=4 && d->bits!=8) || d->group_size==0 || d->in_dim%d->group_size || (d->in_dim*d->bits)%32){qn_err(err,errlen,"unsupported affine layout");return -1;}
    uint64_t wcols=(uint64_t)d->in_dim*d->bits/32, groups=(uint64_t)d->in_dim/d->group_size;
    uint64_t wb=(uint64_t)d->out_dim*wcols*4, sb=(uint64_t)d->out_dim*groups*2;
    if(d->weight_offset+wb>m->bytes||d->scales_offset+sb>m->bytes||d->biases_offset+sb>m->bytes){qn_err(err,errlen,"tensor range outside shard");return -1;}
    out->weight=[dev newBufferWithBytesNoCopy:(uint8_t*)m->base+d->weight_offset length:(NSUInteger)wb options:MTLResourceStorageModeShared deallocator:nil];
    out->scales=[dev newBufferWithBytesNoCopy:(uint8_t*)m->base+d->scales_offset length:(NSUInteger)sb options:MTLResourceStorageModeShared deallocator:nil];
    out->biases=[dev newBufferWithBytesNoCopy:(uint8_t*)m->base+d->biases_offset length:(NSUInteger)sb options:MTLResourceStorageModeShared deallocator:nil];
    if(!out->weight||!out->scales||!out->biases){qn_err(err,errlen,"Metal no-copy view failed");return -1;} return 0;
}

int qn_bf16_make_view(id<MTLDevice> dev,const qn_file_map *m,uint64_t file_offset,uint64_t elements,qn_bf16_metal_view *out,char *err,size_t errlen){
    if(!dev||!m||!m->base||!out||elements==0){qn_err(err,errlen,"invalid bf16 view argument");return -1;}
    uint64_t bytes=elements*2;
    if(file_offset+bytes>m->bytes){qn_err(err,errlen,"bf16 tensor range outside shard");return -1;}
    out->buffer=[dev newBufferWithBytesNoCopy:(uint8_t*)m->base+file_offset length:(NSUInteger)bytes options:MTLResourceStorageModeShared deallocator:nil];
    out->elements=elements;
    if(!out->buffer){qn_err(err,errlen,"Metal bf16 no-copy view failed");return -1;}
    return 0;
}
