//
// Created by condo on 2024/1/5.
//

#ifndef SOLVER_CUH
#define SOLVER_CUH

#include "cuda_runtime.h"
#include "params.h"

namespace cuda_solver {
    enum boundary_type {
        BOUNDARY_SCALAR,
        BOUNDARY_Z,
        BOUNDARY_Y,
        BOUNDARY_X
    };

    void v_step(float *U1_z, float *U1_y, float *U1_x, float *U0_z, float *U0_y, float *U0_x);


    // block size
    inline dim3 block_size(8, 8, 8);
    // grid size
    inline dim3 grid_size(CELLS_X / block_size.x + 1, CELLS_Y / block_size.y + 1, CELLS_Z / block_size.z + 1);
}

#endif //SOLVER_CUH
