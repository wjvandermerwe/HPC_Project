#ifndef OUTFILE_H
#define OUTFILE_H

#include "hypatiaINC.h"
#include "color.h"

#ifdef __cplusplus
extern "C" {
#endif
extern void writeToPPM(const char * filename, int width, int height, const RGBColorU8* arr);
#ifdef __cplusplus
}
#endif
#endif

