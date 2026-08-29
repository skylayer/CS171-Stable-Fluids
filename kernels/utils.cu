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

namespace {
    /* One-dimensional Catmull-Rom between f1 and f2, with the slopes clamped so
     * the result cannot overshoot the data it came from. Without that clamp a
     * cubic rings around sharp fronts and the advected field goes negative. */
    __device__ inline float cubic_1d(float f0, float f1, float f2, float f3, float t) {
        const float delta = f2 - f1;
        float       d1    = 0.5f * (f2 - f0);
        float       d2    = 0.5f * (f3 - f1);

        if (delta * d1 <= 0.0f)
            d1 = 0.0f;
        if (delta * d2 <= 0.0f)
            d2 = 0.0f;

        const float a2 = 3.0f * delta - 2.0f * d1 - d2;
        const float a3 = -2.0f * delta + d1 + d2;
        const float v  = f1 + t * (d1 + t * (a2 + t * a3));

        return fminf(fmaxf(v, fminf(f1, f2)), fmaxf(f1, f2));
    }
} // namespace

__device__ float cubic_interp(float3 pos, const float *field) {
    const float x = fminf(fmaxf(pos.x, 0.0f), CELLS_X - 1.0f);
    const float y = fminf(fmaxf(pos.y, 0.0f), CELLS_Y - 1.0f);
    const float z = fminf(fmaxf(pos.z, 0.0f), CELLS_Z - 1.0f);

    const int xf = min(static_cast<int>(x), CELLS_X - 2);
    const int yf = min(static_cast<int>(y), CELLS_Y - 2);
    const int zf = min(static_cast<int>(z), CELLS_Z - 2);

    // the four-point stencil needs one cell of margin on each side
    if (xf < 1 || xf > CELLS_X - 3 || yf < 1 || yf > CELLS_Y - 3 || zf < 1 || zf > CELLS_Z - 3)
        return lin_interp(pos, field);

    const float tx = x - static_cast<float>(xf);
    const float ty = y - static_cast<float>(yf);
    const float tz = z - static_cast<float>(zf);

    float along_z[4];
    for (int dz = -1; dz <= 2; dz++) {
        float along_y[4];
        for (int dy = -1; dy <= 2; dy++) {
            const int row   = idx3d(zf + dz, yf + dy, xf);
            along_y[dy + 1] = cubic_1d(field[row - 1], field[row], field[row + 1], field[row + 2], tx);
        }
        along_z[dz + 1] = cubic_1d(along_y[0], along_y[1], along_y[2], along_y[3], ty);
    }
    return cubic_1d(along_z[0], along_z[1], along_z[2], along_z[3], tz);
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
