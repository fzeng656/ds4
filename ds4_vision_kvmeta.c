#include "ds4_vision_kvmeta.h"
#include <string.h>
static const uint8_t magic[8]={'D','S','4','V','K','V','1',0};
static void put32(uint8_t *p,uint32_t v) {for(int i=0;i<4;i++) p[i]=(uint8_t)(v>>(8*i));}
static uint32_t get32(const uint8_t *p) {return (uint32_t)p[0]|((uint32_t)p[1]<<8)|((uint32_t)p[2]<<16)|((uint32_t)p[3]<<24);}
static uint32_t crc32(const uint8_t *b,size_t n) {
    uint32_t c=~0u;
    for(size_t i=0;i<n;i++) {c^=b[i];for(int j=0;j<8;j++) c=(c>>1)^((0u-(c&1u))&0xedb88320u);}
    return ~c;
}
static bool sha_valid(const char *s) {
    if(!s)return false;
    for(size_t i=0;i<40;i++)if(!((s[i]>='0'&&s[i]<='9')||(s[i]>='a'&&s[i]<='f')))return false;
    return s[40]==0;
}
static bool valid(const ds4_vision_kvmeta *m) {
    if(!m||!m->checkpoint_tokens||!m->visible_bytes||
       !m->image_count||m->image_count>DS4_VISION_KVMETA_MAX_IMAGES||
       !sha_valid(m->visible_sha1))return false;
    uint64_t prev_end=0;
    uint32_t prev_offset=0;
    for(size_t i=0;i<m->image_count;i++) {
        const ds4_vision_kvmeta_image *v=&m->images[i];
        const uint64_t end=(uint64_t)v->token_start+v->token_count;
        if(!v->token_count||v->token_start<prev_end||end>m->checkpoint_tokens||
           v->visible_marker_offset>=m->visible_bytes||
           (i&&v->visible_marker_offset<=prev_offset))return false;
        unsigned nonzero=0;for(size_t j=0;j<32;j++)nonzero|=v->fingerprint[j];
        if(!nonzero)return false;
        prev_end=end;prev_offset=v->visible_marker_offset;
    }
    return true;
}
bool ds4_vision_kvmeta_encode(const ds4_vision_kvmeta *m,uint8_t *out,
                              size_t cap,size_t *written) {
    if(written)*written=0;
    if(!out||!valid(m))return false;
    size_t n=68u+44u*m->image_count;
    if(cap<n)return false;
    memcpy(out,magic,8);put32(out+8,m->checkpoint_tokens);
    put32(out+12,m->visible_bytes);put32(out+16,m->image_count);put32(out+20,0);
    memcpy(out+24,m->visible_sha1,40);
    for(size_t i=0;i<m->image_count;i++) {
        size_t off=64u+44u*i;const ds4_vision_kvmeta_image *e=&m->images[i];
        put32(out+off,e->token_start);put32(out+off+4,e->token_count);
        put32(out+off+8,e->visible_marker_offset);
        memcpy(out+off+12,e->fingerprint,32);
    }
    put32(out+n-4,crc32(out,n-4));
    if(written)*written=n;
    return true;
}
bool ds4_vision_kvmeta_decode(const uint8_t *b,size_t n,ds4_vision_kvmeta *out) {
    if(!b||!out||n<108||n>DS4_VISION_KVMETA_MAX_BYTES||memcmp(b,magic,8))return false;
    const uint32_t count=get32(b+16);
    if(!count||count>DS4_VISION_KVMETA_MAX_IMAGES||
       n!=68u+44u*count||get32(b+20)!=0||
       get32(b+n-4)!=crc32(b,n-4))return false;
    ds4_vision_kvmeta m={0};
    m.checkpoint_tokens=get32(b+8);m.visible_bytes=get32(b+12);m.image_count=count;
    memcpy(m.visible_sha1,b+24,40);m.visible_sha1[40]=0;
    for(size_t i=0;i<count;i++) {
        size_t off=64u+44u*i;
        m.images[i].token_start=get32(b+off);
        m.images[i].token_count=get32(b+off+4);
        m.images[i].visible_marker_offset=get32(b+off+8);
        memcpy(m.images[i].fingerprint,b+off+12,32);
    }
    if(!valid(&m))return false;
    *out=m;return true;
}
bool ds4_vision_kvmeta_matches(const ds4_vision_kvmeta *saved,
                               const char *sha,uint32_t bytes,
                               const ds4_vision_kvmeta_image *images,
                               size_t count,bool require_positions) {
    if(!valid(saved)||!sha_valid(sha)||!images||
       saved->visible_bytes!=bytes||
       memcmp(saved->visible_sha1,sha,41)||
       saved->image_count>count||count>DS4_VISION_KVMETA_MAX_IMAGES)return false;
    for(size_t i=0;i<saved->image_count;i++) {
        const ds4_vision_kvmeta_image *x=&saved->images[i],*y=&images[i];
        if(x->token_count!=y->token_count||
           x->visible_marker_offset!=y->visible_marker_offset||
           memcmp(x->fingerprint,y->fingerprint,32)||
           (require_positions&&x->token_start!=y->token_start))return false;
    }
    return true;
}
