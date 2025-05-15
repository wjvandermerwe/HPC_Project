//
// Created by johan on 15/05/2025.
//

#include <cassert>

#include "allocator.h"
#include "sphere.h"

__device__ __host__
void * alloc_linearAllocFCAllocate(LinearAllocFC * lafc){
    if((lafc->curOffset + lafc->chunkSize) > lafc->totalSize){
        assert(0 && "Linear allocator is full");
        return NULL;
    }

    void * outAddr = lafc->bufptr + lafc->curOffset;
    lafc->curOffset += lafc->chunkSize;

    return memset(outAddr, 0, lafc->chunkSize);
}
