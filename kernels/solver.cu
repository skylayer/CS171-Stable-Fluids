#include <utility>

#include <curand_kernel.h>

#include "solver.cuh"
#include "utils.cuh"

using namespace cuda_solver;

__device__ unsigned adj(const unsigned id, boundary_type type) {
    if (id == 0)
        return 1;
    if (id == CELLS_X - 1 && type == BOUNDARY_X)
        return CELLS_X - 2;
    if (id == CELLS_Y - 1 && type == BOUNDARY_Y)
        return CELLS_Y - 2;
    if (id == CELLS_Z - 1 && type == BOUNDARY_Z)
        return CELLS_Z - 2;
    return id;
}

namespace cuda_solver {
    __global__ void set_boundary_face_kernel(float *field, boundary_type type) {
        const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
        const unsigned z = blockIdx.z * blockDim.z + threadIdx.z;

        if (x >= CELLS_X || y >= CELLS_Y || z >= CELLS_Z) return;

        const bool on_x = (x == 0 || x == CELLS_X - 1);
        const bool on_y = (y == 0 || y == CELLS_Y - 1);
        const bool on_z = (z == 0 || z == CELLS_Z - 1);
        if (!on_x && !on_y && !on_z) return;

        // For a boundary cell, mirror from the inward neighbor on whichever axis
        // is on the boundary. Pick a single axis (most-outer wins) so corners are
        // deterministic.
        if (on_x) {
            field[idx3d(z, y, x)] = (type == BOUNDARY_X ? -1.0f : 1.0f) * field[idx3d(z, y, adj(x, BOUNDARY_X))];
        } else if (on_y) {
            field[idx3d(z, y, x)] = (type == BOUNDARY_Y ? -1.0f : 1.0f) * field[idx3d(z, adj(y, BOUNDARY_Y), x)];
        } else {
            field[idx3d(z, y, x)] = (type == BOUNDARY_Z ? -1.0f : 1.0f) * field[idx3d(adj(z, BOUNDARY_Z), y, x)];
        }
    }

    // Jacobi iteration: writes to dst, reads only from src (and the constant rhs S0).
    __global__ void lin_solve_kernel(float *dst, const float *src, const float *S0, const float a, const float b) {
        const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
        const unsigned z = blockIdx.z * blockDim.z + threadIdx.z;

        if (x >= 1 && x < CELLS_X - 1 && y >= 1 && y < CELLS_Y - 1 && z >= 1 && z < CELLS_Z - 1) {
            const int index = idx3d(z, y, x);
            dst[index] = (S0[index] + a * (src[index + 1] + src[index - 1] +
                                           src[index + CELLS_X] + src[index - CELLS_X] +
                                           src[index + CELLS_X * CELLS_Y] + src[index - CELLS_X * CELLS_Y])) / b;
        }
    }

    __global__ void transport_kernel(float *S1, const float *S0, const float *U_z, const float *U_y, const float *U_x) {
        const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
        const unsigned z = blockIdx.z * blockDim.z + threadIdx.z;

        if (x >= 1 && x < CELLS_X - 1 && y >= 1 && y < CELLS_Y - 1 && z >= 1 && z < CELLS_Z - 1) {
            float z0 = static_cast<float>(z) - DT * U_z[idx3d(z, y, x)] * CELLS_Z;
            float y0 = static_cast<float>(y) - DT * U_y[idx3d(z, y, x)] * CELLS_Y;
            float x0 = static_cast<float>(x) - DT * U_x[idx3d(z, y, x)] * CELLS_X;

            z0 = fmaxf(0.0f, fminf(static_cast<float>(CELLS_Z - 1), z0));
            y0 = fmaxf(0.0f, fminf(static_cast<float>(CELLS_Y - 1), y0));
            x0 = fmaxf(0.0f, fminf(static_cast<float>(CELLS_X - 1), x0));

            S1[idx3d(z, y, x)] = lin_interp({x0, y0, z0}, S0);
        }
    }

    template <bool negate>
    __global__ void divergence_kernel(float *div, const float *U_z, const float *U_y, const float *U_x) {
        const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
        const unsigned z = blockIdx.z * blockDim.z + threadIdx.z;

        if (x >= 1 && x < CELLS_X - 1 && y >= 1 && y < CELLS_Y - 1 && z >= 1 && z < CELLS_Z - 1) {
            float d = (U_z[idx3d(z + 1, y, x)] - U_z[idx3d(z - 1, y, x)]) * CELLS_Z
                    + (U_y[idx3d(z, y + 1, x)] - U_y[idx3d(z, y - 1, x)]) * CELLS_Y
                    + (U_x[idx3d(z, y, x + 1)] - U_x[idx3d(z, y, x - 1)]) * CELLS_X;
            div[idx3d(z, y, x)] = d / (negate ? -2.0f : 2.0f);
        }
    }

    __global__ void project_kernel_(float *U1_z, float *U1_y, float *U1_x, const float *U0_z, const float *U0_y, const float *U0_x, const float *pressure) {
        const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
        const unsigned z = blockIdx.z * blockDim.z + threadIdx.z;

        if (x >= 1 && x < CELLS_X - 1 && y >= 1 && y < CELLS_Y - 1 && z >= 1 && z < CELLS_Z - 1) {
            U1_z[idx3d(z, y, x)] = U0_z[idx3d(z, y, x)] - (pressure[idx3d(z + 1, y, x)] - pressure[idx3d(z - 1, y, x)]) * CELLS_Z / 2.0f;
            U1_y[idx3d(z, y, x)] = U0_y[idx3d(z, y, x)] - (pressure[idx3d(z, y + 1, x)] - pressure[idx3d(z, y - 1, x)]) * CELLS_Y / 2.0f;
            U1_x[idx3d(z, y, x)] = U0_x[idx3d(z, y, x)] - (pressure[idx3d(z, y, x + 1)] - pressure[idx3d(z, y, x - 1)]) * CELLS_X / 2.0f;
        }
    }

    __global__ void reflect_kernel(float *U1_z, float *U1_y, float *U1_x, const float *U0_z, const float *U0_y, const float *U0_x) {
        const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
        const unsigned z = blockIdx.z * blockDim.z + threadIdx.z;

        if (x >= 1 && x < CELLS_X - 1 && y >= 1 && y < CELLS_Y - 1 && z >= 1 && z < CELLS_Z - 1) {
            U1_z[idx3d(z, y, x)] = 2.0f * U1_z[idx3d(z, y, x)] - U0_z[idx3d(z, y, x)];
            U1_y[idx3d(z, y, x)] = 2.0f * U1_y[idx3d(z, y, x)] - U0_y[idx3d(z, y, x)];
            U1_x[idx3d(z, y, x)] = 2.0f * U1_x[idx3d(z, y, x)] - U0_x[idx3d(z, y, x)];
        }
    }

    __global__ void buoyancy_kernel(float *U1_z, float *U1_y, float *U1_x, const float *U0_z, const float *U0_y, const float *U0_x, float **S0, unsigned long long frame) {
        const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
        const unsigned z = blockIdx.z * blockDim.z + threadIdx.z;

        if (x >= 1 && x < CELLS_X - 1 && y >= 1 && y < CELLS_Y - 1 && z >= 1 && z < CELLS_Z - 1) {
            curandStatePhilox4_32_10_t state;
            curand_init(frame, idx3d(z, y, x), 0, &state);

            float buoy = 0.0f;
            for (int i = 0; i < NUM_FLUIDS; i++) {
                const float r = 5.0f * (curand_uniform(&state) - 0.5f) + 1.f;
                buoy += r * DT * BUOYANCY * powf(0.5f * (S0[i][idx3d(z, y, x)] + S0[i][idx3d(z, y - 1, x)]), 1.3f) * CELLS_Y;
            }
            U1_y[idx3d(z, y, x)] = U0_y[idx3d(z, y, x)] + buoy;
            U1_x[idx3d(z, y, x)] = U0_x[idx3d(z, y, x)];
            U1_z[idx3d(z, y, x)] = U0_z[idx3d(z, y, x)];
        }
    }

    __host__ void set_boundary(float *field, boundary_type type) {
        set_boundary_face_kernel<<<grid_size, block_size>>>(field, type);
    }

    __host__ void lin_solve(float *S1, const float *S0, const float a, const float b, const boundary_type type) {
        // Ping-pong scratch buffer so each iteration is a proper Jacobi update
        // (no in-place reads/writes from neighbor threads).
        static float *scratch = nullptr;
        if (!scratch) {
            cudaMalloc(&scratch, num_cells * sizeof(float));
            cudaMemset(scratch, 0, num_cells * sizeof(float));
        }
        cudaMemcpy(scratch, S1, num_cells * sizeof(float), cudaMemcpyDeviceToDevice);

        float *src = scratch;
        float *dst = S1;
        for (int iter = 0; iter < NUM_ITER; ++iter) {
            lin_solve_kernel<<<grid_size, block_size>>>(dst, src, S0, a, b);
            set_boundary_face_kernel<<<grid_size, block_size>>>(dst, type);
            std::swap(src, dst);
        }
        // src now holds the latest result (we swapped after the last write).
        if (src != S1) {
            cudaMemcpy(S1, src, num_cells * sizeof(float), cudaMemcpyDeviceToDevice);
        }
    }

    __host__ void diffuse(float *S1, const float *S0, boundary_type type) {
        constexpr float a = DT * VISCOSITY * CELLS_X * CELLS_X;
        constexpr float b = 1 + 6 * a;
        lin_solve(S1, S0, a, b, type);
    }

    __host__ void transport(float *S1, const float *S0, const float *U_z, const float *U_y, const float *U_x, const boundary_type type) {
        transport_kernel<<<grid_size, block_size>>>(S1, S0, U_z, U_y, U_x);
        set_boundary_face_kernel<<<grid_size, block_size>>>(S1, type);
    }

    template <bool negate>
    __host__ void divergence(float *div, const float *U_z, const float *U_y, const float *U_x) {
        divergence_kernel<negate><<<grid_size, block_size>>>(div, U_z, U_y, U_x);
        set_boundary_face_kernel<<<grid_size, block_size>>>(div, BOUNDARY_SCALAR);
    }

    __host__ void project(float *U1_z, float *U1_y, float *U1_x, const float *U0_z, const float *U0_y, const float *U0_x) {
        static float *div = nullptr, *pressure = nullptr;
        if (!div) {
            cudaMalloc(&div, num_cells * sizeof(float));
            cudaMalloc(&pressure, num_cells * sizeof(float));
            cudaMemset(div, 0, num_cells * sizeof(float));
            cudaMemset(pressure, 0, num_cells * sizeof(float));
        }

        divergence<true>(div, U0_z, U0_y, U0_x);

        constexpr float a = CELLS_X * CELLS_Y;
        constexpr float b = 6.0f * a;
        lin_solve(pressure, div, a, b, BOUNDARY_SCALAR);

        project_kernel_<<<grid_size, block_size>>>(U1_z, U1_y, U1_x, U0_z, U0_y, U0_x, pressure);
        set_boundary_face_kernel<<<grid_size, block_size>>>(U1_z, BOUNDARY_Z);
        set_boundary_face_kernel<<<grid_size, block_size>>>(U1_y, BOUNDARY_Y);
        set_boundary_face_kernel<<<grid_size, block_size>>>(U1_x, BOUNDARY_X);
    }

    __host__ void reflect(float *U1_z, float *U1_y, float *U1_x, const float *U0_z, const float *U0_y, const float *U0_x) {
        project(U1_z, U1_y, U1_x, U0_z, U0_y, U0_x);
        reflect_kernel<<<grid_size, block_size>>>(U1_z, U1_y, U1_x, U0_z, U0_y, U0_x);
        set_boundary_face_kernel<<<grid_size, block_size>>>(U1_z, BOUNDARY_Z);
        set_boundary_face_kernel<<<grid_size, block_size>>>(U1_y, BOUNDARY_Y);
        set_boundary_face_kernel<<<grid_size, block_size>>>(U1_x, BOUNDARY_X);
    }

    __host__ void buoyancy(float *U1_z, float *U1_y, float *U1_x, const float *U0_z, const float *U0_y, const float *U0_x, float **S0) {
        static unsigned long long frame = 0;
        buoyancy_kernel<<<grid_size, block_size>>>(U1_z, U1_y, U1_x, U0_z, U0_y, U0_x, S0, frame++);
        set_boundary_face_kernel<<<grid_size, block_size>>>(U1_z, BOUNDARY_Z);
        set_boundary_face_kernel<<<grid_size, block_size>>>(U1_y, BOUNDARY_Y);
        set_boundary_face_kernel<<<grid_size, block_size>>>(U1_x, BOUNDARY_X);
    }

    __host__ void swap_workspace(float *&U0_z, float *&U0_y, float *&U0_x, float *&U1_z, float *&U1_y, float *&U1_x) {
        std::swap(U0_z, U1_z);
        std::swap(U0_y, U1_y);
        std::swap(U0_x, U1_x);
    }

    __host__ void v_step(float *U1_z, float *U1_y, float *U1_x, float *U0_z, float *U0_y, float *U0_x, float **S0) {

        buoyancy(U1_z, U1_y, U1_x, U0_z, U0_y, U0_x, S0);
        swap_workspace(U0_z, U0_y, U0_x, U1_z, U1_y, U1_x);

        transport(U1_z, U0_z, U0_z, U0_y, U0_x, BOUNDARY_Z);
        transport(U1_y, U0_y, U0_z, U0_y, U0_x, BOUNDARY_Y);
        transport(U1_x, U0_x, U0_z, U0_y, U0_x, BOUNDARY_X);
        swap_workspace(U0_z, U0_y, U0_x, U1_z, U1_y, U1_x);

        diffuse(U1_z, U0_z, BOUNDARY_Z);
        diffuse(U1_y, U0_y, BOUNDARY_Y);
        diffuse(U1_x, U0_x, BOUNDARY_X);
        swap_workspace(U0_z, U0_y, U0_x, U1_z, U1_y, U1_x);

        reflect(U1_z, U1_y, U1_x, U0_z, U0_y, U0_x);
        swap_workspace(U0_z, U0_y, U0_x, U1_z, U1_y, U1_x);

        transport(U1_z, U0_z, U0_z, U0_y, U0_x, BOUNDARY_Z);
        transport(U1_y, U0_y, U0_z, U0_y, U0_x, BOUNDARY_Y);
        transport(U1_x, U0_x, U0_z, U0_y, U0_x, BOUNDARY_X);
        swap_workspace(U0_z, U0_y, U0_x, U1_z, U1_y, U1_x);

        diffuse(U1_z, U0_z, BOUNDARY_Z);
        diffuse(U1_y, U0_y, BOUNDARY_Y);
        diffuse(U1_x, U0_x, BOUNDARY_X);

        project(U0_z, U0_y, U0_x, U1_z, U1_y, U1_x);
    }

    __host__ void s_step(float *S1, float *S0, float *U_z, float *U_y, float *U_x) {
        set_boundary(S1, BOUNDARY_SCALAR);

        std::swap(S0, S1);

        transport(S1, S0, U_z, U_y, U_x, BOUNDARY_SCALAR);
        std::swap(S0, S1);
        transport(S1, S0, U_z, U_y, U_x, BOUNDARY_SCALAR);

        std::swap(S0, S1);
    }
}
