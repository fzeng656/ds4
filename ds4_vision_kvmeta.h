#ifndef DS4_VISION_KVMETA_H
#define DS4_VISION_KVMETA_H
/* Versioned, fail-closed image identity metadata for experimental disk KV.
 * This does not authorize a cache hit: callers must ALSO verify model ABI,
 * actual KV payload, full normalized visible history and tool-call replay. */
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#define DS4_VISION_KVMETA_MAX_IMAGES 16u
#define DS4_VISION_KVMETA_MAX_BYTES (68u + 44u * DS4_VISION_KVMETA_MAX_IMAGES)
typedef struct {
    uint32_t token_start;
    uint32_t token_count;
    uint32_t visible_marker_offset; /* normalized visible-history byte offset */
    uint8_t fingerprint[32];
} ds4_vision_kvmeta_image;
typedef struct {
    uint32_t checkpoint_tokens;
    uint32_t visible_bytes;
    char visible_sha1[41]; /* SHA1 hex of nonce-normalized visible prefix */
    uint32_t image_count;
    ds4_vision_kvmeta_image images[DS4_VISION_KVMETA_MAX_IMAGES];
} ds4_vision_kvmeta;
bool ds4_vision_kvmeta_encode(const ds4_vision_kvmeta *m, uint8_t *out,
                              size_t cap, size_t *written);
bool ds4_vision_kvmeta_decode(const uint8_t *bytes, size_t len,
                              ds4_vision_kvmeta *out);
bool ds4_vision_kvmeta_matches(const ds4_vision_kvmeta *saved,
                               const char *visible_sha1, uint32_t visible_bytes,
                               const ds4_vision_kvmeta_image *current,
                               size_t current_count, bool require_positions);
#endif
