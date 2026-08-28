//
// Created by condo on 2024/1/5.
//

#ifndef SOLVER_CUH
#define SOLVER_CUH

#include "params.h"

namespace cuda_solver {
    enum boundary_type { BOUNDARY_SCALAR, BOUNDARY_Z, BOUNDARY_Y, BOUNDARY_X };

    /* The velocity pointers are taken by reference: v_step ping-pongs the two
     * workspaces internally and the caller has to follow along.  On return the
     * new velocity field is in U0_* and U1_* is scratch, which is what the
     * closing `project(U0_*, U1_*)` writes.  Passing them by value -- as this
     * used to -- left the caller pointing at the pre-projection field. */
    void v_step(float *&U1_z, float *&U1_y, float *&U1_x, float *&U0_z, float *&U0_y, float *&U0_x, float **S0);
    void s_step(float *S1, float *S0, float *U_z, float *U_y, float *U_x);


    // block size (1024 threads)
    constexpr unsigned BLOCK_X = 8;
    constexpr unsigned BLOCK_Y = 8;
    constexpr unsigned BLOCK_Z = 16;

    // interior_sum_kernel halves a shared-memory array down to one value.
    static_assert((BLOCK_X * BLOCK_Y * BLOCK_Z & (BLOCK_X * BLOCK_Y * BLOCK_Z - 1)) == 0, "block must hold a power-of-two number of threads");

    inline dim3 block_size(BLOCK_X, BLOCK_Y, BLOCK_Z);
    // grid size, rounded up -- `N / B + 1` launched a whole redundant block slab
    // whenever N divided evenly by B.
    inline dim3 grid_size((CELLS_X + BLOCK_X - 1) / BLOCK_X, (CELLS_Y + BLOCK_Y - 1) / BLOCK_Y, (CELLS_Z + BLOCK_Z - 1) / BLOCK_Z);

    // The boundary kernel only walks the six faces of the cube, so a 2D launch
    // large enough to cover the biggest face is all it needs.
    constexpr unsigned CELLS_MAX      = CELLS_X > CELLS_Y ? (CELLS_X > CELLS_Z ? CELLS_X : CELLS_Z) : (CELLS_Y > CELLS_Z ? CELLS_Y : CELLS_Z);
    constexpr unsigned BOUNDARY_BLOCK = 16;

    inline dim3 boundary_block_size(BOUNDARY_BLOCK, BOUNDARY_BLOCK);
    inline dim3 boundary_grid_size((CELLS_MAX + BOUNDARY_BLOCK - 1) / BOUNDARY_BLOCK, (CELLS_MAX + BOUNDARY_BLOCK - 1) / BOUNDARY_BLOCK);
} // namespace cuda_solver

#endif // SOLVER_CUH
