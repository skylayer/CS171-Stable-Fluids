//
// Created by condo on 2024/1/5.
//

#ifndef SOLVER_CUH
#define SOLVER_CUH

#include "params.h"

namespace cuda_solver {
    /* Velocity lives on a staggered (MAC) grid:
     *
     *     u_x[idx3d(z, y, x)]  is the x-velocity on the face at  x - 1/2
     *     u_y[idx3d(z, y, x)]  is the y-velocity on the face at  y - 1/2
     *     u_z[idx3d(z, y, x)]  is the z-velocity on the face at  z - 1/2
     *
     * Scalars -- density, temperature, pressure -- stay at cell centres. Cells
     * 1..N-2 are fluid; 0 and N-1 are ghosts.
     *
     * The staggering exists so that the divergence and the pressure gradient are
     * compact and exactly adjoint: div(grad p) is then precisely the 7-point
     * Laplacian lin_solve inverts. The collocated layout this replaces used
     * central differences over 2h for both while solving the compact operator
     * over h, so the two did not compose. A grid-scale checkerboard velocity had
     * exactly zero measured divergence and came through projection untouched --
     * which is the scale turbulence lives at.
     */

    enum stagger_axis { CENTRED = -1, STAGGER_X = 0, STAGGER_Y = 1, STAGGER_Z = 2 };

    /* The velocity pointers are taken by reference: v_step ping-pongs the two
     * workspaces internally and the caller has to follow along. On return the
     * new velocity is in U0_* and U1_* is scratch. */
    void v_step(float *&U1_z, float *&U1_y, float *&U1_x, float *&U0_z, float *&U0_y, float *&U0_x, float **S0, const float *T0);
    void s_step(float *S1, const float *S0, const float *U_z, const float *U_y, const float *U_x);
    void t_step(float *T1, const float *T0, const float *U_z, const float *U_y, const float *U_x);

    void set_velocity_boundary(float *U_z, float *U_y, float *U_x);
    void set_scalar_boundary(float *field);
    void release_scratch(void);

    // block size (1024 threads)
    constexpr unsigned BLOCK_X = 8;
    constexpr unsigned BLOCK_Y = 8;
    constexpr unsigned BLOCK_Z = 16;

    // interior_sum_kernel halves a shared-memory array down to one value.
    static_assert((BLOCK_X * BLOCK_Y * BLOCK_Z & (BLOCK_X * BLOCK_Y * BLOCK_Z - 1)) == 0, "block must hold a power-of-two number of threads");

    inline dim3        block_size(BLOCK_X, BLOCK_Y, BLOCK_Z);
    constexpr unsigned NUM_BLOCKS = ((CELLS_X + BLOCK_X - 1) / BLOCK_X) * ((CELLS_Y + BLOCK_Y - 1) / BLOCK_Y) * ((CELLS_Z + BLOCK_Z - 1) / BLOCK_Z);

    inline dim3 grid_size((CELLS_X + BLOCK_X - 1) / BLOCK_X, (CELLS_Y + BLOCK_Y - 1) / BLOCK_Y, (CELLS_Z + BLOCK_Z - 1) / BLOCK_Z);

    // The boundary kernels only walk faces of the cube, so a 2D launch large
    // enough to cover the biggest one is all they need.
    constexpr unsigned CELLS_MAX      = CELLS_X > CELLS_Y ? (CELLS_X > CELLS_Z ? CELLS_X : CELLS_Z) : (CELLS_Y > CELLS_Z ? CELLS_Y : CELLS_Z);
    constexpr unsigned BOUNDARY_BLOCK = 16;

    inline dim3 boundary_block_size(BOUNDARY_BLOCK, BOUNDARY_BLOCK);
    inline dim3 boundary_grid_size((CELLS_MAX + BOUNDARY_BLOCK - 1) / BOUNDARY_BLOCK, (CELLS_MAX + BOUNDARY_BLOCK - 1) / BOUNDARY_BLOCK);
} // namespace cuda_solver

#endif // SOLVER_CUH
