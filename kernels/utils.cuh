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

/* Monotonic cubic sample of a cell-centred field (Fedkiw et al. 2001).
 *
 * Trilinear interpolation is the dominant remaining error in the advection. A
 * semi-Lagrangian step's numerical viscosity is h^2 / (2 dt) * f(1-f), where f
 * is the fractional part of the CFL -- the interpolation is the whole of it,
 * which is why the dissipation vanishes when the departure point lands on a
 * cell centre. A clamped Catmull-Rom interpolant does not change that shape,
 * but it drops the magnitude by one to two orders of magnitude: measured on a
 * fixed physical wavelength at 64^3, 2.8e-03 trilinear+first-order, 7.0e-04
 * trilinear+MacCormack, 5.8e-05 here.
 *
 * Falls back to lin_interp within two cells of a wall, where the four-point
 * stencil does not fit. */
__device__ float cubic_interp(float3 pos, const float *field);

#endif // UTILS_CUH
