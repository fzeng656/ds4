#include "ds4_vision_kvmeta.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>
static ds4_vision_kvmeta example(void) {
    ds4_vision_kvmeta m={0};
    m.checkpoint_tokens=280471;m.visible_bytes=896233;
    memcpy(m.visible_sha1,"0123456789abcdef0123456789abcdef01234567",41);
    m.image_count=2;
    m.images[0].token_start=265761;m.images[0].token_count=576;
    m.images[0].visible_marker_offset=855201;
    m.images[1].token_start=272000;m.images[1].token_count=777;
    m.images[1].visible_marker_offset=879100;
    memset(m.images[0].fingerprint,0x12,32);
    memset(m.images[1].fingerprint,0x34,32);
    return m;
}
int main(void) {
    ds4_vision_kvmeta m=example(),out={0};
    uint8_t b[DS4_VISION_KVMETA_MAX_BYTES],tmp[sizeof(b)];
    size_t n=0;
    assert(ds4_vision_kvmeta_encode(&m,b,sizeof(b),&n));
    assert(n==156 && ds4_vision_kvmeta_decode(b,n,&out));
    assert(out.checkpoint_tokens==m.checkpoint_tokens);
    assert(ds4_vision_kvmeta_matches(&out,m.visible_sha1,m.visible_bytes,m.images,2,true));
    assert(!ds4_vision_kvmeta_matches(&out,m.visible_sha1,m.visible_bytes-1,m.images,2,true));
    assert(!ds4_vision_kvmeta_matches(&out,"0000000000000000000000000000000000000000",m.visible_bytes,m.images,2,true));
    assert(!ds4_vision_kvmeta_matches(&out,m.visible_sha1,m.visible_bytes,m.images,1,true));
    ds4_vision_kvmeta_image changed[2];memcpy(changed,m.images,sizeof(changed));
    changed[0].fingerprint[0]^=1;
    assert(!ds4_vision_kvmeta_matches(&out,m.visible_sha1,m.visible_bytes,changed,2,false));
    changed[0]=m.images[0];changed[0].token_start++;
    assert(!ds4_vision_kvmeta_matches(&out,m.visible_sha1,m.visible_bytes,changed,2,true));
    assert(ds4_vision_kvmeta_matches(&out,m.visible_sha1,m.visible_bytes,changed,2,false));
    changed[0]=m.images[0];changed[0].visible_marker_offset++;
    assert(!ds4_vision_kvmeta_matches(&out,m.visible_sha1,m.visible_bytes,changed,2,false));
    assert(!ds4_vision_kvmeta_decode(b,n-1,&out));
    memcpy(tmp,b,n);tmp[39]^=1;assert(!ds4_vision_kvmeta_decode(tmp,n,&out));
    memcpy(tmp,b,n);tmp[16]=17;assert(!ds4_vision_kvmeta_decode(tmp,n,&out));
    memcpy(tmp,b,n);tmp[0]=0;assert(!ds4_vision_kvmeta_decode(tmp,n,&out));
    assert(!ds4_vision_kvmeta_encode(&m,b,n-1,&n));
    ds4_vision_kvmeta v=example();v.images[1].token_start=265900;
    assert(!ds4_vision_kvmeta_encode(&v,b,sizeof(b),&n));
    v=example();v.images[1].token_count=50000;
    assert(!ds4_vision_kvmeta_encode(&v,b,sizeof(b),&n));
    v=example();v.images[1].visible_marker_offset=v.images[0].visible_marker_offset;
    assert(!ds4_vision_kvmeta_encode(&v,b,sizeof(b),&n));
    v=example();memset(v.images[0].fingerprint,0,32);
    assert(!ds4_vision_kvmeta_encode(&v,b,sizeof(b),&n));
    puts("vision kv metadata: 19 checks passed");return 0;
}
