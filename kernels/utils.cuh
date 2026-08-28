//
// Created by condo on 2024/1/8.
//

#ifndef UTILS_CUH
#define UTILS_CUH

/* Trilinear sample of a cell-centred field. `pos` is in cell-index coordinates
 * (the centre of cell i sits at i) and is clamped into the grid, so callers do
 * not have to guard the trace themselves. */
__device__ float lin_interp(float3 pos, const float *field);

/* Same sample point, but returns the smallest and largest of the eight corner
 * values instead of the interpolant. MacCormack needs these to limit its
 * correction to the range the source data actually spans. */
__device__ void lin_interp_bounds(float3 pos, const float *field, float &lo, float &hi);

#endif // UTILS_CUH
