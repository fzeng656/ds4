#ifndef QN_RUNTIME_H
#define QN_RUNTIME_H
#include <stddef.h>
#include <stdint.h>
#ifdef __OBJC__
#import <Metal/Metal.h>
#endif

typedef struct {
    void *base;
    uint64_t bytes;
    int fd;
    char path[1024];
} qn_file_map;

typedef struct {
    uint64_t weight_offset;
    uint64_t scales_offset;
    uint64_t biases_offset;
    uint32_t out_dim;
    uint32_t in_dim;
    uint32_t group_size;
    uint32_t bits;
} qn_affine_desc;

int qn_file_map_open(qn_file_map *m, const char *path, char *err, size_t errlen);
void qn_file_map_close(qn_file_map *m);
int qn_file_map_advise(qn_file_map *m, int will_need);
int qn_file_map_invalidate(qn_file_map *m);

typedef struct {
    int stable_mode;
    int prefill_mps_enabled;
    int moe_metallib_configured;
    char moe_metallib_path[1024];
} qn_runtime_config_status;

/* Production configuration is opt-in through QN_RUNTIME_MODE=stable.
   Stable mode forces the verified MPS prefill path and resolves the bundled
   BM32 sorted-MoE metallib. It fails closed if that metallib cannot be found. */
int qn_runtime_configure_from_env(qn_runtime_config_status *status, char *err, size_t errlen);

#ifdef __OBJC__
typedef struct {
    __strong id<MTLBuffer> weight;
    __strong id<MTLBuffer> scales;
    __strong id<MTLBuffer> biases;
} qn_affine_metal_view;

typedef struct {
    __strong id<MTLBuffer> buffer;
    uint64_t elements;
} qn_bf16_metal_view;

int qn_affine_make_view(id<MTLDevice> dev, const qn_file_map *m,
                        const qn_affine_desc *d,
                        qn_affine_metal_view *out,
                        char *err, size_t errlen);
int qn_bf16_make_view(id<MTLDevice> dev, const qn_file_map *m,
                      uint64_t file_offset, uint64_t elements,
                      qn_bf16_metal_view *out,
                      char *err, size_t errlen);
#endif
#endif
