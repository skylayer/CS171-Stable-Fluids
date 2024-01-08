//
// Created by condo on 2024/1/8.
//

#ifndef RENDER_CUH
#define RENDER_CUH

#include "params.h"

void render_density(const float view[3][3], const float pos[3], float focal, float **field, float *output);

#endif // RENDER_CUH
