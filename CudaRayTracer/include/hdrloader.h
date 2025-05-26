//
// Created by johan on 25/05/2025.
//

#ifndef HDRLOADER_H
#define HDRLOADER_H
#pragma once

#include <vector>
#include <cuda_runtime.h>   // for float4

// load an HDR image into a float4 buffer (RGB → xyz, w=1.0f)
bool loadHDR(const char* path,
             std::vector<float4>& outPixels,
             int& width,
             int& height);

#endif //HDRLOADER_H
