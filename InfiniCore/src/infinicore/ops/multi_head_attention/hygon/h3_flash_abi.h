#ifndef H3_FLASH_ABI_H
#define H3_FLASH_ABI_H
#include <stdint.h>

#define H3_FLASH_ABI_VERSION 1
#ifdef __cplusplus
extern "C" {
#endif
// Dense B,S,H,D dimensions; each tensor's positive strides are in elements.
// Only BF16, D=128, equal Q/KV heads, no mask/dropout/causal/ALiBi.
typedef struct {
    int64_t batch, query_tokens, key_tokens, heads;
    int64_t strides[4][4]; // Q,K,V,O, each B,S,H,D
} H3FlashSpec;
typedef struct {
    const void *q;
    const void *k;
    const void *v;
    void *out;
    float scale;
} H3FlashCall;
typedef struct {
    uint32_t abi_version, struct_bytes;
    const char *arch; // Exactly gfx936.
    const char *build_id;
    int (*prepare)(const H3FlashSpec *, void **plan);
    int (*launch)(void *plan, const H3FlashCall *, void *stream);
    void (*destroy)(void *plan);
} H3FlashApi;
typedef const H3FlashApi *(*H3FlashGetApi)(void);
#ifdef __cplusplus
}
#endif
#endif
