//
// Created by condo on 2024/1/8.
//

#ifndef RENDER_CUH
#define RENDER_CUH

#include "Eigen/Core"
#include "params.h"

void render_density(const Eigen::Matrix3f &view, const Eigen::Vector3f &pos, float focal, const float **field, float *output);

#endif // RENDER_CUH
