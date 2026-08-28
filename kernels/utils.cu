//
// Created by condo on 2024/1/8.
//

#include "params.h"
#include "utils.cuh"

namespace {
    /* Clamping the position here rather than at every call site: a back-trace
     * that leaves the grid used to index one cell past the end, and a negative
     * coordinate truncated towards zero and then extrapolated. */
    __device__ inline void corner(float3 pos, int &zf, int &yf, int &xf, float &zd, float &yd, float &xd) {
        const float x = fminf(fmaxf(pos.x, 0.0f), CELLS_X - 1.0f);
        const float y = fminf(fmaxf(pos.y, 0.0f), CELLS_Y - 1.0f);
        const float z = fminf(fmaxf(pos.z, 0.0f), CELLS_Z - 1.0f);

        zf = min(static_cast<int>(z), CELLS_Z - 2);
        yf = min(static_cast<int>(y), CELLS_Y - 2);
        xf = min(static_cast<int>(x), CELLS_X - 2);

        zd = z - static_cast<float>(zf);
        yd = y - static_cast<float>(yf);
        xd = x - static_cast<float>(xf);
    }
} // namespace

__device__ float lin_interp(float3 pos, const float *field) {
    int   zfloor, yfloor, xfloor;
    float zdiff, ydiff, xdiff;
    corner(pos, zfloor, yfloor, xfloor, zdiff, ydiff, xdiff);

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

__device__ void lin_interp_bounds(float3 pos, const float *field, float &lo, float &hi) {
    int   zfloor, yfloor, xfloor;
    float zdiff, ydiff, xdiff;
    corner(pos, zfloor, yfloor, xfloor, zdiff, ydiff, xdiff);

    lo = field[idx3d(zfloor, yfloor, xfloor)];
    hi = lo;
    for (int dz = 0; dz <= 1; dz++)
        for (int dy = 0; dy <= 1; dy++)
            for (int dx = 0; dx <= 1; dx++) {
                const float v = field[idx3d(zfloor + dz, yfloor + dy, xfloor + dx)];
                lo            = fminf(lo, v);
                hi            = fmaxf(hi, v);
            }
}
