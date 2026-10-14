#ifndef DS4_VISION_KVDISK_H
#define DS4_VISION_KVDISK_H
/* EXPERIMENTAL. Persistent completed-image checkpoints, separate from text .kv.
 * Caller must hold the inference lock while staging/restoring Metal state. */
#include "ds4.h"
#include "ds4_vision_kvmeta.h"
#include <stdint.h>
#include <stdbool.h>
typedef struct {
    uint32_t visible_bytes;
    int checkpoint_tokens;
} ds4_vision_kvdisk_hit;
bool ds4_vision_kvdisk_store(const char *cache_dir, const char *visible_key,
                            const ds4_vision_kvmeta *meta,
                            ds4_engine *engine, ds4_session *session,
                            uint64_t budget_bytes, char *err, size_t errlen);
bool ds4_vision_kvdisk_restore(const char *cache_dir, const char *request_visible_key,
                              const ds4_vision_kvmeta_image *images, size_t image_count,
                              ds4_engine *engine, ds4_session *session,
                              ds4_vision_kvdisk_hit *hit, char *err, size_t errlen);
#endif
