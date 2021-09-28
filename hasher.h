#include <stdint.h>
#include <nmmintrin.h>


uint64_t hasher(uint64_t v) {
    return _mm_crc32_u64(v, 0);
}
