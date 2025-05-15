#include "texture.h"

#include "util.h"

#include <tgmath.h>
#include <stdio.h>

#include "stb_imageINC.h"







void tex_loadImage(Image * restrict img, const char* filename){
    img->data = stbi_load(filename, &img->width, &img->height, &img->compsPerPixel, 0);

    if(!img->data){
        printf("%s\n", filename);
        return;
    }

    img->bytesPerScanLine = img->compsPerPixel * img->width;
}