#import "qn_runtime.h"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <limits.h>
#include <stdlib.h>
#include <mach-o/dyld.h>

static void qn_err(char *err,size_t n,const char *msg){ if(err&&n) snprintf(err,n,"%s",msg); }
static int qn_set_default_env(const char *key,const char *value,char *err,size_t errlen){
    if(getenv(key)!=NULL)return 0;
    if(setenv(key,value,0)){char msg[256];snprintf(msg,sizeof(msg),"cannot configure stable default %s",key);qn_err(err,errlen,msg);return -1;}
    return 0;
}

static int qn_regular_readable(const char *p){
    struct stat st; return p&&*p&&stat(p,&st)==0&&S_ISREG(st.st_mode)&&access(p,R_OK)==0;
}
static int qn_copy_if_file(char out[1024],const char *p){
    if(!qn_regular_readable(p))return 0; char r[PATH_MAX];
    const char *src=realpath(p,r)?r:p; snprintf(out,1024,"%s",src); return 1;
}
static int qn_join_candidate(char out[1024],const char *base,const char *suffix){
    if(!base||!*base)return 0; char p[PATH_MAX];
    if(snprintf(p,sizeof(p),"%s/%s",base,suffix)>=(int)sizeof(p))return 0;
    return qn_copy_if_file(out,p);
}
static void qn_parent_dir(char *p){
    size_t n=strlen(p); while(n&&p[n-1]=='/')p[--n]=0;
    while(n&&p[n-1]!='/')p[--n]=0; while(n>1&&p[n-1]=='/')p[--n]=0;
    if(!n)snprintf(p,2,".");
}
static int qn_find_bundled_moe_metallib(char out[1024]){
    const char *explicit_path=getenv("QN_MOE_METALLIB");
    if(explicit_path&&*explicit_path)return qn_copy_if_file(out,explicit_path)?1:-1;
    const char *root=getenv("QN_RUNTIME_ROOT");
    if(root&&*root&&qn_join_candidate(out,root,"qwen_native/kernels/qn_gather_bm32.metallib"))return 1;

    uint32_t cap=PATH_MAX; char exe[PATH_MAX];
    if(_NSGetExecutablePath(exe,&cap)==0){
        char absbuf[PATH_MAX]; if(realpath(exe,absbuf))snprintf(exe,sizeof(exe),"%s",absbuf);
        qn_parent_dir(exe);
        if(qn_join_candidate(out,exe,"qwen_native/kernels/qn_gather_bm32.metallib"))return 1;
        if(qn_join_candidate(out,exe,"../share/qwen-native/qn_gather_bm32.metallib"))return 1;
        if(qn_join_candidate(out,exe,"../lib/qwen-native/qn_gather_bm32.metallib"))return 1;
    }
    char cwd[PATH_MAX];
    if(getcwd(cwd,sizeof(cwd))&&qn_join_candidate(out,cwd,"qwen_native/kernels/qn_gather_bm32.metallib"))return 1;

#ifndef QN_DISABLE_SOURCE_METALLIB_FALLBACK
    /* Development-only fallback. Production bundles disable this at compile time
       so a missing installed metallib cannot be hidden by a source checkout. */
    char src[PATH_MAX]; snprintf(src,sizeof(src),"%s",__FILE__);
    char srcabs[PATH_MAX]; if(realpath(src,srcabs))snprintf(src,sizeof(src),"%s",srcabs);
    qn_parent_dir(src); /* .../qwen_native */
    if(qn_join_candidate(out,src,"kernels/qn_gather_bm32.metallib"))return 1;
#endif
    return 0;
}
static int qn_find_bundled_prefill_metallib(char out[1024]){
    const char *explicit_path=getenv("QN_PREFILL_METALLIB");
    if(explicit_path&&*explicit_path)return qn_copy_if_file(out,explicit_path)?1:-1;
    const char *root=getenv("QN_RUNTIME_ROOT");
    if(root&&*root&&qn_join_candidate(out,root,"qwen_native/kernels/qn_prefill_nax.metallib"))return 1;
    uint32_t cap=PATH_MAX;char exe[PATH_MAX];
    if(_NSGetExecutablePath(exe,&cap)==0){char absbuf[PATH_MAX];if(realpath(exe,absbuf))snprintf(exe,sizeof(exe),"%s",absbuf);qn_parent_dir(exe);if(qn_join_candidate(out,exe,"qwen_native/kernels/qn_prefill_nax.metallib"))return 1;if(qn_join_candidate(out,exe,"../share/qwen-native/qn_prefill_nax.metallib"))return 1;if(qn_join_candidate(out,exe,"../lib/qwen-native/qn_prefill_nax.metallib"))return 1;}
    char cwd[PATH_MAX];if(getcwd(cwd,sizeof(cwd))&&qn_join_candidate(out,cwd,"qwen_native/kernels/qn_prefill_nax.metallib"))return 1;
#ifndef QN_DISABLE_SOURCE_METALLIB_FALLBACK
    char src[PATH_MAX];snprintf(src,sizeof(src),"%s",__FILE__);char srcabs[PATH_MAX];if(realpath(src,srcabs))snprintf(src,sizeof(src),"%s",srcabs);qn_parent_dir(src);if(qn_join_candidate(out,src,"kernels/qn_prefill_nax.metallib"))return 1;
#endif
    return 0;
}

int qn_runtime_configure_from_env(qn_runtime_config_status *status,char *err,size_t errlen){
    if(status)memset(status,0,sizeof(*status));
    const char *mode=getenv("QN_RUNTIME_MODE");
    if(!mode||strcmp(mode,"stable"))return 0;
    if(status)status->stable_mode=1;
    /* Phase6 validated production prefill defaults. Explicit caller settings
       always win, so every fast-path component remains independently rollbackable. */
    if(qn_set_default_env("QN_PREFILL_CHUNK","2048",err,errlen) ||
       qn_set_default_env("QN_GDN_SHARED_BATCH_POOL","1",err,errlen) ||
       qn_set_default_env("QN_P1_BF16_DENSE","1",err,errlen) ||
       qn_set_default_env("QN_P1_SORTED_MOE","1",err,errlen) ||
       qn_set_default_env("QN_PREFILL_GPU_FULL_TRUNK","1",err,errlen) ||
       qn_set_default_env("QN_QSA_BATCH_SELECTOR","1",err,errlen) ||
       qn_set_default_env("QN_QSA_VECTOR_BATCH","1",err,errlen) ||
       qn_set_default_env("QN_QSA_MSV_F32_MASKED","1",err,errlen) ||
       qn_set_default_env("QN_QSA_MSV_F32_BQ64","1",err,errlen))return -1;
    if(setenv("QN_PREFILL_MPS","1",1)){qn_err(err,errlen,"cannot enable stable MPS prefill");return -1;}
    if(status)status->prefill_mps_enabled=1;
    char mp[1024]={0}; int found=qn_find_bundled_moe_metallib(mp);
    if(found<0){qn_err(err,errlen,"QN_MOE_METALLIB is set but not a readable file");return -1;}
    if(!found){qn_err(err,errlen,"stable mode requires bundled qn_gather_bm32.metallib");return -1;}
    if(setenv("QN_MOE_METALLIB",mp,1)){qn_err(err,errlen,"cannot configure stable MoE metallib");return -1;}
    if(status){status->moe_metallib_configured=1;snprintf(status->moe_metallib_path,sizeof(status->moe_metallib_path),"%s",mp);}
    char pp[1024]={0};found=qn_find_bundled_prefill_metallib(pp);
    if(found<0){qn_err(err,errlen,"QN_PREFILL_METALLIB is set but not a readable file");return -1;}
    if(!found){qn_err(err,errlen,"stable mode requires bundled qn_prefill_nax.metallib");return -1;}
    if(setenv("QN_PREFILL_METALLIB",pp,1)){qn_err(err,errlen,"cannot configure stable prefill metallib");return -1;}
    if(status){status->prefill_metallib_configured=1;snprintf(status->prefill_metallib_path,sizeof(status->prefill_metallib_path),"%s",pp);}
    return 0;
}
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
