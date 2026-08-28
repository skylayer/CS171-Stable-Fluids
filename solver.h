//
// Created by condo on 2024/1/2.
//

#ifndef SOLVER_H
#define SOLVER_H

#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <utility>

#include "params.h"

namespace solver {
    /* The velocity pointers are taken by reference: v_step ping-pongs the two
     * workspaces internally and the caller has to follow along.  On return the
     * new velocity field is in U0_* and U1_* is scratch, which is what the
     * closing `project(U0_*, U1_*)` writes.  Passing them by value -- as this
     * used to -- left the caller pointing at the pre-projection field. */
    void v_step(float*& U1_z, float*& U1_y, float*& U1_x, float*& U0_z, float*& U0_y, float*& U0_x);
    void s_step(float* S1, float* S0, float* U1_z, float* U1_y, float* U1_x);
}

#endif //SOLVER_H
