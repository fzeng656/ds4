/* Standalone filesystem round-trip using mocked GPU payload; never loads model. */
#include "ds4_vision_kvdisk.h"
#include <CommonCrypto/CommonDigest.h>
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <dirent.h>
#include <sys/stat.h>
struct ds4_engine { int model; };
struct ds4_session { int ctx,pos; ds4_vision_span image; };
int ds4_engine_model_id(ds4_engine *e){return e->model;}
int ds4_engine_routed_quant_bits(ds4_engine *e){(void)e;return 2;}
int ds4_session_ctx(ds4_session *s){return s->ctx;}
int ds4_session_pos(ds4_session *s){return s->pos;}
void ds4_session_invalidate(ds4_session *s){s->pos=0;memset(&s->image,0,sizeof(s->image));}
bool ds4_session_restore_vision_checkpoint(ds4_session *s,
                                         const ds4_vision_span *images,size_t n){
    if(n!=1||!images||!s->pos||images[0].token_start+images[0].embedding.token_count>(uint32_t)s->pos)return false;
    s->image=images[0];return true;
}
void ds4_kvstore_sha1_bytes_hex(const void *b,size_t n,char h[41]){
    unsigned char digest[CC_SHA1_DIGEST_LENGTH];CC_SHA1(b,(CC_LONG)n,digest);
    for(size_t i=0;i<CC_SHA1_DIGEST_LENGTH;i++)sprintf(h+i*2,"%02x",digest[i]);
    h[40]=0;
}
int ds4_session_stage_payload(ds4_session *s,ds4_session_payload_file *out,char *err,size_t len) {
    (void)s;(void)err;(void)len;
    char name[]="/tmp/ds4-vk-mock-XXXXXX";int fd=mkstemp(name);
    if(fd<0)return 1;
    if(write(fd,"MOCKKV0000",10)!=10){close(fd);unlink(name);return 1;}
    close(fd);out->path=strdup(name);out->bytes=10;return 0;
}
int ds4_session_write_staged_payload(const ds4_session_payload_file *p,FILE *dst,char *err,size_t len){
    (void)err;(void)len;
    FILE *src=fopen(p->path,"rb");if(!src)return 1;
    char b[10];int ok=fread(b,1,10,src)==10&&fwrite(b,1,10,dst)==10;
    fclose(src);return ok?0:1;
}
void ds4_session_payload_file_free(ds4_session_payload_file *p){
    if(p->path){unlink(p->path);free(p->path);}p->path=NULL;p->bytes=0;
}
int ds4_session_load_payload(ds4_session *s,FILE *fp,uint64_t len,char *err,size_t cap){
    (void)err;(void)cap;char b[10]={0};
    if(len!=10||fread(b,1,10,fp)!=10||memcmp(b,"MOCKKV0000",10))return 1;
    s->pos=1234;return 0;
}
static int vkd_count(const char *path){
    int n=0;DIR *d=opendir(path);if(!d)return 0;
    struct dirent *de;
    while((de=readdir(d)))if(strstr(de->d_name,".vkd"))n++;
    closedir(d);return n;
}
int main(void){
    char dir[]="/tmp/ds4-vkd-test-XXXXXX";assert(mkdtemp(dir));
    ds4_engine e={.model=4};
    ds4_session s={.ctx=393216,.pos=1234};
    const char text[]="model visible preface \036(image) and final answer";
    ds4_vision_kvmeta m={0};m.checkpoint_tokens=1234;m.visible_bytes=strlen(text);
    m.image_count=1;m.images[0].token_start=700;m.images[0].token_count=20;
    m.images[0].visible_marker_offset=22;memset(m.images[0].fingerprint,1,32);
    ds4_kvstore_sha1_bytes_hex(text,m.visible_bytes,m.visible_sha1);
    char err[160]={0};
    assert(ds4_vision_kvdisk_store(dir,text,&m,&e,&s,4096,err,sizeof(err)));
    assert(vkd_count(dir)==1);
    assert(ds4_vision_kvdisk_store(dir,text,&m,&e,&s,4096,err,sizeof(err)));
    assert(vkd_count(dir)==1);
    ds4_session_invalidate(&s);
    ds4_vision_kvdisk_hit hit={0};
    assert(ds4_vision_kvdisk_restore(dir,text,m.images,1,&e,&s,&hit,err,sizeof(err)));
    assert(hit.checkpoint_tokens==1234 && hit.visible_bytes==strlen(text) && s.image.token_start==700);
    ds4_vision_kvmeta_image changed=m.images[0];changed.fingerprint[0]^=1;
    assert(!ds4_vision_kvdisk_restore(dir,text,&changed,1,&e,&s,&hit,err,sizeof(err)));
    changed=m.images[0];changed.visible_marker_offset++;
    assert(!ds4_vision_kvdisk_restore(dir,text,&changed,1,&e,&s,&hit,err,sizeof(err)));
    assert(!ds4_vision_kvdisk_restore(dir,"different visible prefix",&m.images[0],1,&e,&s,&hit,err,sizeof(err)));
    assert(!ds4_vision_kvdisk_store(dir,text,&m,&e,&s,100,err,sizeof(err)));
    DIR *d=opendir(dir);struct dirent *de;
    while((de=readdir(d)))if(strstr(de->d_name,".vkd")){
        char p[1024];snprintf(p,sizeof(p),"%s/%s",dir,de->d_name);
        FILE *f=fopen(p,"r+b");assert(f);assert(fseek(f,50,SEEK_SET)==0);
        int c=fgetc(f);assert(c!=EOF);assert(fseek(f,50,SEEK_SET)==0);
        fputc(c^0x01,f);fclose(f);
        assert(!ds4_vision_kvdisk_restore(dir,text,m.images,1,&e,&s,&hit,err,sizeof(err)));
        unlink(p);break;
    }
    closedir(d);rmdir(dir);
    puts("vision disk mock: store/load/mismatch/corrupt/budget checks passed");
    return 0;
}
