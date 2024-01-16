//
// Created by condo on 2024/1/8.
//

#include "utils.cuh"
#include "params.h"

__device__ float lin_interp(float3 pos, const float *field) {
    auto [x, y, z] = pos;

    const int zfloor = min(static_cast<int>(z), CELLS_Z - 2);
    const int yfloor = min(static_cast<int>(y), CELLS_Y - 2);
    const int xfloor = min(static_cast<int>(x), CELLS_X - 2);

    const float zdiff = z - static_cast<float>(zfloor);
    const float ydiff = y - static_cast<float>(yfloor);
    const float xdiff = x - static_cast<float>(xfloor);

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
