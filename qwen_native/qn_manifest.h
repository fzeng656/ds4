#ifndef QN_MANIFEST_H
#define QN_MANIFEST_H
#include <stddef.h>
#include <stdint.h>
typedef struct qn_manifest qn_manifest;
typedef struct {
    const char *name;
    const char *shard;
    const char *dtype;
    uint64_t file_offset;
    uint64_t payload_bytes;
    uint32_t ndim;
    uint64_t shape[4];
} qn_tensor_meta;
int qn_manifest_open(qn_manifest **out,const char *path,char *err,size_t errlen);
int qn_manifest_tensor(qn_manifest *m,const char *name,qn_tensor_meta *out,char *err,size_t errlen);
int qn_manifest_layer_tensor(qn_manifest *m,uint32_t layer,const char *suffix,qn_tensor_meta *out,char *err,size_t errlen);
void qn_manifest_close(qn_manifest *m);
#endif
