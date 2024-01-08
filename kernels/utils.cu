//
// Created by condo on 2024/1/8.
//

#include "utils.cuh"
#include "params.h"

__device__ float lin_interp(float3 pos, const float *field) {
    auto [x, y, z] = pos;

    const int zfloor = static_cast<int>(z - 0.5f);
    const int yfloor = static_cast<int>(y - 0.5f);
    const int xfloor = static_cast<int>(x - 0.5f);

    const float zdiff = z - 0.5f - static_cast<float>(zfloor);
    const float ydiff = y - 0.5f - static_cast<float>(yfloor);
    const float xdiff = x - 0.5f - static_cast<float>(xfloor);

    const float ftl = field[idx3d(zfloor, yfloor, xfloor)];
    const float fbl = field[idx3d(zfloor, yfloor + 1, xfloor)];
    const float ftr = field[idx3d(zfloor, yfloor, xfloor + 1)];
    const float fbr = field[idx3d(zfloor, yfloor + 1, xfloor + 1)];
    const float btl = field[idx3d(zfloor + 1, yfloor, xfloor)];
    const float bbl = field[idx3d(zfloor + 1, yfloor + 1, xfloor)];
    const float btr = field[idx3d(zfloor + 1, yfloor, xfloor + 1)];
    const float bbr = field[idx3d(zfloor + 1, yfloor + 1, xfloor + 1)];

    const float vfl = (1.0f - ydiff) * ftl + ydiff * fbl;
    const float vfr = (1.0f - ydiff) * ftr + ydiff * fbr;
    const float vbl = (1.0f - ydiff) * btl + ydiff * bbl;
    const float vbr = (1.0f - ydiff) * btr + ydiff * bbr;

    const float ff = (1.0f - xdiff) * vfl + xdiff * vfr;
    const float fb = (1.0f - xdiff) * vbl + xdiff * vbr;

    return (1.0f - zdiff) * ff + zdiff * fb;
}
