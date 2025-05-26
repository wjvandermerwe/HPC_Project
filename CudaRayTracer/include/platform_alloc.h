/* platform_alloc.h  ── include this before allocator.c or in a common header */
#pragma once
#include <stdlib.h>     /* size_t */

/* Windows / MSVC --------------------------------------------------------- */
#ifdef _MSC_VER
    #include <malloc.h>                 /* _aligned_malloc, _aligned_free */

    /* C11-style wrapper */
    static inline void* aligned_alloc(size_t alignment, size_t size)
    {
        /* C11 requires size to be a multiple of alignment; _aligned_malloc
           doesn’t care, but we keep the check for portability. */
        if (size % alignment) size += alignment - (size % alignment);
        return _aligned_malloc(size, alignment);
    }

static inline void  aligned_free(void* p) { _aligned_free(p); }

/* Other compilers (GCC/Clang, etc.) ------------------------------------- */
#else
#include <stdlib.h>   /* the real aligned_alloc */
#define aligned_free free
#endif
