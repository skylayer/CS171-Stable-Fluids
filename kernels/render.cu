//
// Created by condo on 2024/1/8.
//

#include "render.cuh"
#include "utils.cuh"


__device__ bool intersect(const Eigen::Vector3f &pos, const Eigen::Vector3f &dir, Eigen::Vector3f &hit) {
    const Eigen::Vector3f min      = Eigen::Vector3f(0, 0, 0);
    const Eigen::Vector3f max      = Eigen::Vector3f(1, 1, 1);
    const Eigen::Vector3f inv_dir  = Eigen::Vector3f(1, 1, 1).cwiseQuotient(dir);
    const Eigen::Vector3f t0s      = (min - pos).cwiseProduct(inv_dir);
    const Eigen::Vector3f t1s      = (max - pos).cwiseProduct(inv_dir);
    const Eigen::Vector3f tsmaller = t0s.cwiseMin(t1s);
    const Eigen::Vector3f tbigger  = t0s.cwiseMax(t1s);
    const float           tmin     = tsmaller.maxCoeff();
    const float           tmax     = tbigger.minCoeff();

    if (tmin > tmax) {
        return false;
    }

    hit = pos + tmin * dir;
    return true;
}


__device__ float density(const float *field, const Eigen::Vector3f &pos) {
    if (pos.x() < 0 || pos.x() > 1 || pos.y() < 0 || pos.y() > 1 || pos.z() < 0 || pos.z() > 1) {
        return 0;
    }

    auto coord = (pos.cwiseProduct(Eigen::Vector3f(CELLS_X - 2, CELLS_Y - 2, CELLS_Z - 2)) + Eigen::Vector3f::Constant(1)).eval();
    return lin_interp({coord.x(), coord.y(), coord.z()}, field);
}

__global__ void density_renderer(const Eigen::Matrix3f &view, const Eigen::Vector3f &pos, const float focal, const float **field, float *output) {
    const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;

    const Eigen::Vector3f colors[7] = ALL_COLORS;

    if (x < WINDOW_WIDTH && y < WINDOW_HEIGHT) {
        const auto dir = (view * Eigen::Vector3f(x - WINDOW_WIDTH / 2.0f, y - WINDOW_HEIGHT / 2.0f, -focal)).normalized();
        if (Eigen::Vector3f hit; intersect(pos, dir, hit)) {
            float           accumlatedOpacity = 0;
            Eigen::Vector3f color             = Eigen::Vector3f::Zero();
            float           step              = 0.01;
            int             maxIter           = 1000;

            while (accumlatedOpacity < 1 && maxIter--) {
                float maxDensity = 0;
                for (int i = 0; i < NUM_FLUIDS; i++) {
                    const float d = density(field[i], hit);
                    maxDensity    = fmaxf(maxDensity, d);

                    if (d > 0) {
                        const float opacity = 1 - exp(-d * step);
                        const float weight  = opacity * (1 - accumlatedOpacity);

                        color += weight * colors[i];
                        accumlatedOpacity += weight;
                    }
                }

                hit += step * dir;

                if (hit.x() < 0 || hit.x() > 1 || hit.y() < 0 || hit.y() > 1 || hit.z() < 0 || hit.z() > 1) {
                    break;
                }
            }

            output[3 * (y * WINDOW_WIDTH + x) + 0] = color.x();
            output[3 * (y * WINDOW_WIDTH + x) + 1] = color.y();
            output[3 * (y * WINDOW_WIDTH + x) + 2] = color.z();
        }
    }
}

__host__ void render_density(const Eigen::Matrix3f &view, const Eigen::Vector3f &pos, const float focal, const float **field, float3 *output) {
    const dim3 block_size(32, 32);
    const dim3 grid_size(WINDOW_WIDTH / block_size.x + 1, WINDOW_HEIGHT / block_size.y + 1);

    density_renderer<<<grid_size, block_size>>>(view, pos, focal, field, output);
    cudaDeviceSynchronize();
}
