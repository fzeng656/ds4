#include "ds4_vision_kvdisk.h"
#include "ds4_kvstore.h"
#include <dirent.h>
#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#define VKD_HEADER 48u
static const unsigned char vkd_magic[8]={'D','S','4','V','K','D','1',0};
static void put32(unsigned char *p,uint32_t v){for(int i=0;i<4;i++)p[i]=(unsigned char)(v>>(8*i));}
static uint32_t get32(const unsigned char *p){return (uint32_t)p[0]|((uint32_t)p[1]<<8)|((uint32_t)p[2]<<16)|((uint32_t)p[3]<<24);}
static void put64(unsigned char *p,uint64_t v){for(int i=0;i<8;i++)p[i]=(unsigned char)(v>>(8*i));}
static uint64_t get64(const unsigned char *p){uint64_t v=0;for(int i=7;i>=0;i--)v=(v<<8)|p[i];return v;}
static bool fname_is_vkd(const char *n) {
    size_t z=strlen(n);if(z!=44||strcmp(n+40,".vkd"))return false;
    for(size_t i=0;i<40;i++)if(!isxdigit((unsigned char)n[i]))return false;
    return true;
}
static bool dir_init(const char *path) {
    if(!path||!*path)return false;
    if(mkdir(path,0700)==0)return true;
    struct stat st;return errno==EEXIST&&stat(path,&st)==0&&S_ISDIR(st.st_mode);
}
static char *path_join(const char *a,const char *b) {
    size_t n=strlen(a),m=strlen(b);
    if(n>SIZE_MAX-m-2)return NULL;
    char *out=malloc(n+m+2);if(!out)return NULL;
    memcpy(out,a,n);out[n]='/';memcpy(out+n+1,b,m+1);return out;
}
/* Only touch .vkd files in this dedicated directory; never delete text .kv. */
static bool budget_room(const char *dir,uint64_t requested,uint64_t maxbytes,const char *protect) {
    if(requested>maxbytes)return false;
    for(int pass=0;pass<1024;pass++) {
        DIR *d=opendir(dir);if(!d)return false;
        struct dirent *e;uint64_t used=0;
        char *victim=NULL;time_t oldest=0;
        while((e=readdir(d))!=NULL) {
            if(!fname_is_vkd(e->d_name))continue;
            char *p=path_join(dir,e->d_name);if(!p)continue;
            struct stat st;
            if(stat(p,&st)!=0||!S_ISREG(st.st_mode)||st.st_size<0){free(p);continue;}
            uint64_t sz=(uint64_t)st.st_size;
            if(sz>UINT64_MAX-used)used=UINT64_MAX;else used+=sz;
            if(strcmp(p,protect)!=0 && (!victim||st.st_mtime<oldest)) {
                free(victim);victim=p;oldest=st.st_mtime;
            } else free(p);
        }
        closedir(d);
        if(used<=maxbytes-requested){free(victim);return true;}
        if(!victim)return false;
        const int rc=unlink(victim);free(victim);
        if(rc!=0)return false;
    }
    return false;
}
static bool make_header(unsigned char h[VKD_HEADER], ds4_engine *engine,
                        ds4_session *session,uint32_t meta_bytes,
                        uint32_t text_bytes,uint64_t payload_bytes) {
    int id=ds4_engine_model_id(engine),quant=ds4_engine_routed_quant_bits(engine),ctx=ds4_session_ctx(session);
    if(id<0||id>255||quant<=0||ctx<=0)return false;
    uint64_t total=VKD_HEADER+(uint64_t)meta_bytes+text_bytes;
    if(payload_bytes>UINT64_MAX-total)return false;total+=payload_bytes;
    memset(h,0,VKD_HEADER);memcpy(h,vkd_magic,8);
    put32(h+8,(uint32_t)id);put32(h+12,(uint32_t)quant);put32(h+16,(uint32_t)ctx);
    put32(h+20,meta_bytes);put32(h+24,text_bytes);
    put64(h+28,payload_bytes);put64(h+36,total);
    return true;
}
bool ds4_vision_kvdisk_store(const char *dir,const char *key,const ds4_vision_kvmeta *meta,
                            ds4_engine *engine,ds4_session *session,
                            uint64_t budget,char *err,size_t cap) {
    if(!dir||!key||!meta||!engine||!session||!budget)return false;
    uint8_t packed[DS4_VISION_KVMETA_MAX_BYTES];size_t meta_len=0;
    size_t text_len=strlen(key);
    if(text_len>UINT32_MAX||text_len!=meta->visible_bytes||
       !ds4_vision_kvmeta_encode(meta,packed,sizeof(packed),&meta_len))return false;
    char hash[41];ds4_kvstore_sha1_bytes_hex(key,text_len,hash);
    if(memcmp(hash,meta->visible_sha1,40)!=0)return false;
    if(!dir_init(dir))return false;
    /* Unique identity includes normalized text digest and exact image metadata. */
    char identity[41];ds4_kvstore_sha1_bytes_hex(packed,meta_len,identity);
    char name[45];snprintf(name,sizeof(name),"%s.vkd",identity);
    char *path=path_join(dir,name);if(!path)return false;
    ds4_session_payload_file staged={0};
    if(ds4_session_stage_payload(session,&staged,err,cap)!=0){free(path);return false;}
    unsigned char header[VKD_HEADER];
    bool ok=make_header(header,engine,session,(uint32_t)meta_len,(uint32_t)text_len,staged.bytes);
    const uint64_t total=ok?get64(header+36):0;
    if(!ok||total>budget){ds4_session_payload_file_free(&staged);free(path);return false;}
    struct stat present={0};
    if(stat(path,&present)==0 && S_ISREG(present.st_mode)&&
       (uint64_t)present.st_size==total){
        FILE *existing=fopen(path,"rb");
        if(existing){
            unsigned char eh[VKD_HEADER];uint8_t em[DS4_VISION_KVMETA_MAX_BYTES];
            bool valid=fread(eh,1,VKD_HEADER,existing)==VKD_HEADER&&
                memcmp(eh,header,VKD_HEADER)==0&&
                fread(em,1,meta_len,existing)==meta_len&&
                memcmp(em,packed,meta_len)==0;
            fclose(existing);
            if(valid){ds4_session_payload_file_free(&staged);free(path);return true;}
        }
    }
    if(!budget_room(dir,total,budget,path)) {
        ds4_session_payload_file_free(&staged);free(path);return false;
    }
    char tmp[PATH_MAX];int n=snprintf(tmp,sizeof(tmp),"%s.tmp.%ld",path,(long)getpid());
    if(n<=0||(size_t)n>=sizeof(tmp)){ds4_session_payload_file_free(&staged);free(path);return false;}
    FILE *fp=fopen(tmp,"wb");
    if(!fp){ds4_session_payload_file_free(&staged);free(path);return false;}
    (void)fchmod(fileno(fp),0600);
    ok=fwrite(header,1,VKD_HEADER,fp)==VKD_HEADER&&
       fwrite(packed,1,meta_len,fp)==meta_len&&
       fwrite(key,1,text_len,fp)==text_len&&
       ds4_session_write_staged_payload(&staged,fp,err,cap)==0&&
       fflush(fp)==0 && fsync(fileno(fp))==0;
    if(fclose(fp)!=0)ok=false;
    if(ok&&rename(tmp,path)!=0)ok=false;
    if(!ok)unlink(tmp);
    ds4_session_payload_file_free(&staged);free(path);
    return ok;
}
bool ds4_vision_kvdisk_restore(const char *dir,const char *current,
                              const ds4_vision_kvmeta_image *images,size_t num,
                              ds4_engine *engine,ds4_session *session,
                              ds4_vision_kvdisk_hit *hit,char *err,size_t cap) {
    if(hit)memset(hit,0,sizeof(*hit));
    if(!dir||!current||!images||!num||num>DS4_VISION_KVMETA_MAX_IMAGES||
       !engine||!session)return false;
    DIR *d=opendir(dir);if(!d)return false;
    bool restored=false;uint32_t best_tokens=0;
    char *best=NULL;uint32_t best_visible=0;
    struct dirent *de;size_t request_len=strlen(current);
    /* Scan only small headers, never load GPU state until a full match wins. */
    while((de=readdir(d))!=NULL) {
        if(!fname_is_vkd(de->d_name))continue;
        char *path=path_join(dir,de->d_name);if(!path)continue;
        FILE *fp=fopen(path,"rb");if(!fp){free(path);continue;}
        unsigned char hdr[VKD_HEADER];struct stat st={0};
        bool ok=fstat(fileno(fp),&st)==0 && S_ISREG(st.st_mode)&&
                fread(hdr,1,VKD_HEADER,fp)==VKD_HEADER&&
                memcmp(hdr,vkd_magic,8)==0&&
                get32(hdr+8)==(uint32_t)ds4_engine_model_id(engine)&&
                get32(hdr+12)==(uint32_t)ds4_engine_routed_quant_bits(engine)&&
                get32(hdr+16)==(uint32_t)ds4_session_ctx(session)&&
                get32(hdr+20)<=DS4_VISION_KVMETA_MAX_BYTES&&
                get32(hdr+20)>=108&&get32(hdr+24)>0&&
                (uint64_t)get32(hdr+24)<=request_len&&
                get64(hdr+36)==(uint64_t)st.st_size&&
                get64(hdr+36)==VKD_HEADER+(uint64_t)get32(hdr+20)+get32(hdr+24)+get64(hdr+28);
        ds4_vision_kvmeta meta={0};
        if(ok) {
            uint8_t packed[DS4_VISION_KVMETA_MAX_BYTES];
            ok=fread(packed,1,get32(hdr+20),fp)==get32(hdr+20)&&
               ds4_vision_kvmeta_decode(packed,get32(hdr+20),&meta)&&
               meta.visible_bytes==get32(hdr+24)&&
               meta.checkpoint_tokens>best_tokens;
            if(ok){
                char *key=malloc((size_t)meta.visible_bytes+1);
                if(!key)ok=false;
                else {
                    ok=fread(key,1,meta.visible_bytes,fp)==meta.visible_bytes;
                    if(ok) {
                        key[meta.visible_bytes]=0;
                        char sha[41];ds4_kvstore_sha1_bytes_hex(key,meta.visible_bytes,sha);
                        ok=!memcmp(key,current,meta.visible_bytes)&&
                           ds4_vision_kvmeta_matches(&meta,sha,meta.visible_bytes,images,num,false);
                    }
                    free(key);
                }
            }
        }
        fclose(fp);
        if(ok){free(best);best=path;best_tokens=meta.checkpoint_tokens;best_visible=meta.visible_bytes;}
        else free(path);
    }
    closedir(d);
    if(!best)return false;
    FILE *fp=fopen(best,"rb");
    if(fp) {
        unsigned char hdr[VKD_HEADER];uint8_t packed[DS4_VISION_KVMETA_MAX_BYTES];ds4_vision_kvmeta meta={0};
        if(fread(hdr,1,VKD_HEADER,fp)==VKD_HEADER&&
           fread(packed,1,get32(hdr+20),fp)==get32(hdr+20)&&
           ds4_vision_kvmeta_decode(packed,get32(hdr+20),&meta)&&
           fseeko(fp,(off_t)get32(hdr+24),SEEK_CUR)==0) {
            /* Never trust image-bearing tensors before validating the footer. */
            ds4_session_invalidate(session);
            if(ds4_session_load_payload(session,fp,get64(hdr+28),err,cap)==0 &&
               ds4_session_pos(session)==(int)meta.checkpoint_tokens) {
                ds4_vision_span spans[DS4_VISION_KVMETA_MAX_IMAGES]={0};
                for(size_t i=0;i<meta.image_count;i++){
                    spans[i].token_start=meta.images[i].token_start;
                    spans[i].embedding.token_count=meta.images[i].token_count;
                    memcpy(spans[i].embedding.fingerprint,meta.images[i].fingerprint,32);
                }
                restored=ds4_session_restore_vision_checkpoint(session,spans,meta.image_count);
                if(restored&&hit){hit->visible_bytes=best_visible;hit->checkpoint_tokens=(int)best_tokens;}
            }
            if(!restored)ds4_session_invalidate(session);
        }
        fclose(fp);
    }
    free(best);return restored;
}
