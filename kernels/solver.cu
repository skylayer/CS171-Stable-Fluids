#include "solver.cuh"
#include "cooperative_groups.h"

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

namespace cuda_solver_single {
    __device__ void set_boundary_face_kernel(volatile float *field, boundary_type type) {
        const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
        const unsigned z = blockIdx.z * blockDim.z + threadIdx.z;

        if (x == 0 || x == CELLS_X - 1)
            field[idx3d(z, y, x)] = (type == BOUNDARY_X ? -1.0f : 1.0f) * field[idx3d(z, y, adj(x, BOUNDARY_X))];
        if (y == 0 || y == CELLS_Y - 1)
            field[idx3d(z, y, x)] = (type == BOUNDARY_Y ? -1.0f : 1.0f) * field[idx3d(z, adj(y, BOUNDARY_Y), x)];
        if (z == 0 || z == CELLS_Z - 1)
            field[idx3d(z, y, x)] = (type == BOUNDARY_Z ? -1.0f : 1.0f) * field[idx3d(adj(z, BOUNDARY_Z), y, x)];
    }

    // __global__ void set_boundary_edge_kernel(float *field, boundary_type type) {
    //     const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
    //     const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
    //     const unsigned z = blockIdx.z * blockDim.z + threadIdx.z;
    //
    //     if (x != 0 && x != CELLS_X - 1)
    //         field[idx3d(z, y, x)] = (field[idx3d(adj(z), y, x)] + field[idx3d(z, adj(y), x)]) / 2;
    //     if (y != 0 && y != CELLS_Y - 1)
    //         field[idx3d(z, y, x)] = (field[idx3d(z, y, adj(x))] + field[idx3d(adj(z), y, x)]) / 2;
    //     if (z != 0 && z != CELLS_Z - 1)
    //         field[idx3d(z, y, x)] = (field[idx3d(z, y, adj(x))] + field[idx3d(z, adj(y), x)]) / 2;
    // }

    __device__ void lin_solve_kernel(volatile float *S1, const float *S0, const float a, const float b) {
        const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
        const unsigned z = blockIdx.z * blockDim.z + threadIdx.z;

        const int index = idx3d(z, y, x);
        if (x >= 1 && x < CELLS_X - 1 && y >= 1 && y < CELLS_Y - 1 && z >= 1 && z < CELLS_Z - 1) {
            // Calculate the linear index from the 3D coordinates
            // Update the cell value based on neighboring cells
            S1[index] = (S0[index] + a * (S1[index + 1] + S1[index - 1] + S1[index + CELLS_X] + S1[index - CELLS_X] +
                                          S1[index + CELLS_X * CELLS_Y] + S1[index - CELLS_X * CELLS_Y])) / b;
        }
        // Wait for all threads to finish updating their values
        __threadfence();
        __syncthreads();
        // cooperative_groups::this_grid().sync();

        set_boundary_face_kernel(S1, BOUNDARY_SCALAR);
    }

    __device__ float lin_interp(float3 pos, const float *field) {
        auto [x, y, z] = pos;

        const int zfloor = static_cast<int>(z - 0.5f);
        const int yfloor = static_cast<int>(y - 0.5f);
        const int xfloor = static_cast<int>(x - 0.5f);

        const float zdiff = z - 0.5f - static_cast<float>(zfloor);
        const float ydiff = y - 0.5f - static_cast<float>(yfloor);
        const float xdiff = x - 0.5f - static_cast<float>(xfloor);

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

    __device__ void transport_kernel(float *S1, const float *S0, const float *U_z, const float *U_y, const float *U_x) {
        const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
        const unsigned z = blockIdx.z * blockDim.z + threadIdx.z;

        if (x >= 1 && x < CELLS_X - 1 && y >= 1 && y < CELLS_Y - 1 && z >= 1 && z < CELLS_Z - 1) {
            float z0 = static_cast<float>(z) + 0.5f - DT * U_z[idx3d(z, y, x)] * CELLS_Z;
            float y0 = static_cast<float>(y) + 0.5f - DT * U_y[idx3d(z, y, x)] * CELLS_Y;
            float x0 = static_cast<float>(x) + 0.5f - DT * U_x[idx3d(z, y, x)] * CELLS_X;

            z0 = fmax(1.0f, fmin(static_cast<float>(CELLS_Z) - 1.0f, z0));
            y0 = fmax(1.0f, fmin(static_cast<float>(CELLS_Y) - 1.0f, y0));
            x0 = fmax(1.0f, fmin(static_cast<float>(CELLS_X) - 1.0f, x0));

            S1[idx3d(z, y, x)] = lin_interp({x0, y0, z0}, S0);
        }

        set_boundary_face_kernel(S1, BOUNDARY_SCALAR);
    }

    template <bool negate>
    __device__ void divergence_kernel(float *div, const float *U_z, const float *U_y, const float *U_x) {
        const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
        const unsigned z = blockIdx.z * blockDim.z + threadIdx.z;

        if (x >= 1 && x < CELLS_X - 1 && y >= 1 && y < CELLS_Y - 1 && z >= 1 && z < CELLS_Z - 1) {
            div[idx3d(z, y, x)] =
                (U_z[idx3d(z + 1, y, x)] - U_z[idx3d(z - 1, y, x)]) * CELLS_Z
                + (U_y[idx3d(z, y + 1, x)] - U_y[idx3d(z, y - 1, x)]) * CELLS_Y
                + (U_x[idx3d(z, y, x + 1)] - U_x[idx3d(z, y, x - 1)]) * CELLS_X;

            div[idx3d(z, y, x)] /= negate ? -2.0f : 2.0f;
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

    __device__ void project_kernel(float *U1_z, float *U1_y, float *U1_x, const float *U0_z, const float *U0_y, const float *U0_x) {
        const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
        const unsigned z = blockIdx.z * blockDim.z + threadIdx.z;

        static float S[num_cells], divergence[num_cells];

        S[idx3d(z, y, x)] = 0;
        if (x >= 1 && x < CELLS_X - 1 && y >= 1 && y < CELLS_Y - 1 && z >= 1 && z < CELLS_Z - 1) {
            divergence[idx3d(z, y, x)] =
                (U0_z[idx3d(z + 1, y, x)] - U0_z[idx3d(z - 1, y, x)]) * CELLS_Z / 2.0f
                + (U0_y[idx3d(z, y + 1, x)] - U0_y[idx3d(z, y - 1, x)]) * CELLS_Y / 2.0f
                + (U0_x[idx3d(z, y, x + 1)] - U0_x[idx3d(z, y, x - 1)]) * CELLS_X / 2.0f;
        }
        divergence[idx3d(z, y, x)] *= -1.0f;
        __syncthreads();

        set_boundary_face_kernel(divergence, BOUNDARY_SCALAR);
        __syncthreads();

        constexpr float a = CELLS_X * CELLS_X;
        constexpr float b = 6.0f * a;
        lin_solve_kernel(S, divergence, a, b);
        __syncthreads();

        if (x >= 1 && x < CELLS_X - 1 && y >= 1 && y < CELLS_Y - 1 && z >= 1 && z < CELLS_Z - 1) {
            U1_z[idx3d(z, y, x)] = U0_z[idx3d(z, y, x)] - (S[idx3d(z + 1, y, x)] - S[idx3d(z - 1, y, x)]) * CELLS_Z / 2.0f;
            U1_y[idx3d(z, y, x)] = U0_y[idx3d(z, y, x)] - (S[idx3d(z, y + 1, x)] - S[idx3d(z, y - 1, x)]) * CELLS_Y / 2.0f;
            U1_x[idx3d(z, y, x)] = U0_x[idx3d(z, y, x)] - (S[idx3d(z, y, x + 1)] - S[idx3d(z, y, x - 1)]) * CELLS_X / 2.0f;
        }
        __syncthreads();

        set_boundary_face_kernel(U1_z, BOUNDARY_Z);
        set_boundary_face_kernel(U1_y, BOUNDARY_Y);
        set_boundary_face_kernel(U1_x, BOUNDARY_X);
    }

    __device__ void reflect_kernel(float *U1_z, float *U1_y, float *U1_x, const float *U0_z, const float *U0_y, const float *U0_x) {
        const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
        const unsigned z = blockIdx.z * blockDim.z + threadIdx.z;

        project_kernel(U1_z, U1_y, U1_x, U0_z, U0_y, U0_x);

        if (x >= 1 && x < CELLS_X - 1 && y >= 1 && y < CELLS_Y - 1 && z >= 1 && z < CELLS_Z - 1) {
            U1_z[idx3d(z, y, x)] *= 2.0f;
            U1_y[idx3d(z, y, x)] *= 2.0f;
            U1_x[idx3d(z, y, x)] *= 2.0f;

            U1_z[idx3d(z, y, x)] -= U0_z[idx3d(z, y, x)];
            U1_y[idx3d(z, y, x)] -= U0_y[idx3d(z, y, x)];
            U1_x[idx3d(z, y, x)] -= U0_x[idx3d(z, y, x)];
        }
    }

    __device__ void commit(float *&U1_z, float *&U1_y, float *&U1_x, float *&U0_z, float *&U0_y, float *&U0_x) {
#define SWAP(a, b) { auto tmp = a; a = b; b = tmp; }
        SWAP(U1_z, U0_z);
        SWAP(U1_y, U0_y);
        SWAP(U1_x, U0_x);
    }

    __device__ void diffuse_kernel(float *S1, const float *S0) {
        constexpr float a = DT * VISCOSITY * CELLS_X * CELLS_X;
        constexpr float b = 1 + 6 * a;

        lin_solve_kernel(S1, S0, a, b);
    }

    __global__ void step_kernel(float *U1_z, float *U1_y, float *U1_x, float *U0_z, float *U0_y, float *U0_x) {
        set_boundary_face_kernel(U1_z, BOUNDARY_Z);
        set_boundary_face_kernel(U1_y, BOUNDARY_Y);
        set_boundary_face_kernel(U1_x, BOUNDARY_X);
        commit(U0_z, U0_y, U0_x, U1_z, U1_y, U1_x);
        __syncthreads();

        transport_kernel(U1_z, U0_z, U0_z, U0_y, U0_x);
        transport_kernel(U1_y, U0_y, U0_z, U0_y, U0_x);
        transport_kernel(U1_x, U0_x, U0_z, U0_y, U0_x);
        commit(U0_z, U0_y, U0_x, U1_z, U1_y, U1_x);
        __syncthreads();

        diffuse_kernel(U1_z, U0_z);
        diffuse_kernel(U1_y, U0_y);
        diffuse_kernel(U1_x, U0_x);
        commit(U0_z, U0_y, U0_x, U1_z, U1_y, U1_x);
        __syncthreads();

        project_kernel(U1_z, U1_y, U1_x, U0_z, U0_y, U0_x);
        commit(U0_z, U0_y, U0_x, U1_z, U1_y, U1_x);
        __syncthreads();

        transport_kernel(U1_z, U0_z, U0_z, U0_y, U0_x);
        transport_kernel(U1_y, U0_y, U0_z, U0_y, U0_x);
        transport_kernel(U1_x, U0_x, U0_z, U0_y, U0_x);
        commit(U0_z, U0_y, U0_x, U1_z, U1_y, U1_x);
        __syncthreads();

        diffuse_kernel(U1_z, U0_z);
        diffuse_kernel(U1_y, U0_y);
        diffuse_kernel(U1_x, U0_x);
        commit(U0_z, U0_y, U0_x, U1_z, U1_y, U1_x);
        __syncthreads();

        project_kernel(U0_z, U0_y, U0_x, U1_z, U1_y, U1_x);
        __syncthreads();
    }

    __host__ void v_step(float *U1_z, float *U1_y, float *U1_x, float *U0_z, float *U0_y, float *U0_x) {
        step_kernel<<<grid_size, block_size>>>(U1_z, U1_y, U1_x, U0_z, U0_y, U0_x);
    }
}

namespace cuda_solver {
    __global__ void set_boundary_face_kernel(volatile float *field, boundary_type type) {
        const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
        const unsigned z = blockIdx.z * blockDim.z + threadIdx.z;

        if (x == 0 || x == CELLS_X - 1)
            field[idx3d(z, y, x)] = (type == BOUNDARY_X ? -1.0f : 1.0f) * field[idx3d(z, y, adj(x, BOUNDARY_X))];
        if (y == 0 || y == CELLS_Y - 1)
            field[idx3d(z, y, x)] = (type == BOUNDARY_Y ? -1.0f : 1.0f) * field[idx3d(z, adj(y, BOUNDARY_Y), x)];
        if (z == 0 || z == CELLS_Z - 1)
            field[idx3d(z, y, x)] = (type == BOUNDARY_Z ? -1.0f : 1.0f) * field[idx3d(adj(z, BOUNDARY_Z), y, x)];
    }

    // __global__ void set_boundary_edge_kernel(float *field, boundary_type type) {
    //     const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
    //     const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
    //     const unsigned z = blockIdx.z * blockDim.z + threadIdx.z;
    //
    //     if (x != 0 && x != CELLS_X - 1)
    //         field[idx3d(z, y, x)] = (field[idx3d(adj(z), y, x)] + field[idx3d(z, adj(y), x)]) / 2;
    //     if (y != 0 && y != CELLS_Y - 1)
    //         field[idx3d(z, y, x)] = (field[idx3d(z, y, adj(x))] + field[idx3d(adj(z), y, x)]) / 2;
    //     if (z != 0 && z != CELLS_Z - 1)
    //         field[idx3d(z, y, x)] = (field[idx3d(z, y, adj(x))] + field[idx3d(z, adj(y), x)]) / 2;
    // }

    __global__ void lin_solve_kernel(volatile float *S1, const float *S0, const float a, const float b) {
        const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
        const unsigned z = blockIdx.z * blockDim.z + threadIdx.z;

        const int index = idx3d(z, y, x);
        if (x >= 1 && x < CELLS_X - 1 && y >= 1 && y < CELLS_Y - 1 && z >= 1 && z < CELLS_Z - 1) {
            // Calculate the linear index from the 3D coordinates
            // Update the cell value based on neighboring cells
            S1[index] = (S0[index] + a * (S1[index + 1] + S1[index - 1] + S1[index + CELLS_X] + S1[index - CELLS_X] +
                                          S1[index + CELLS_X * CELLS_Y] + S1[index - CELLS_X * CELLS_Y])) / b;

            // cooperative_groups::this_grid().sync();
        }
    }

    __device__ float lin_interp(float3 pos, const float *field) {
        auto [x, y, z] = pos;

        const int zfloor = static_cast<int>(z - 0.5f);
        const int yfloor = static_cast<int>(y - 0.5f);
        const int xfloor = static_cast<int>(x - 0.5f);

        const float zdiff = z - 0.5f - static_cast<float>(zfloor);
        const float ydiff = y - 0.5f - static_cast<float>(yfloor);
        const float xdiff = x - 0.5f - static_cast<float>(xfloor);

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

    __global__ void transport_kernel(float *S1, const float *S0, const float *U_z, const float *U_y, const float *U_x) {
        const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
        const unsigned z = blockIdx.z * blockDim.z + threadIdx.z;

        if (x >= 1 && x < CELLS_X - 1 && y >= 1 && y < CELLS_Y - 1 && z >= 1 && z < CELLS_Z - 1) {
            float z0 = static_cast<float>(z) + 0.5f - DT * U_z[idx3d(z, y, x)] * CELLS_Z;
            float y0 = static_cast<float>(y) + 0.5f - DT * U_y[idx3d(z, y, x)] * CELLS_Y;
            float x0 = static_cast<float>(x) + 0.5f - DT * U_x[idx3d(z, y, x)] * CELLS_X;

            z0 = fmax(1.0f, fmin(static_cast<float>(CELLS_Z) - 1.0f, z0));
            y0 = fmax(1.0f, fmin(static_cast<float>(CELLS_Y) - 1.0f, y0));
            x0 = fmax(1.0f, fmin(static_cast<float>(CELLS_X) - 1.0f, x0));

            S1[idx3d(z, y, x)] = lin_interp({x0, y0, z0}, S0);
        }
    }

    template <bool negate>
    __global__ void divergence_kernel(float *div, const float *U_z, const float *U_y, const float *U_x) {
        const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
        const unsigned z = blockIdx.z * blockDim.z + threadIdx.z;

        if (x >= 1 && x < CELLS_X - 1 && y >= 1 && y < CELLS_Y - 1 && z >= 1 && z < CELLS_Z - 1) {
            div[idx3d(z, y, x)] =
                (U_z[idx3d(z + 1, y, x)] - U_z[idx3d(z - 1, y, x)]) * CELLS_Z
                + (U_y[idx3d(z, y + 1, x)] - U_y[idx3d(z, y - 1, x)]) * CELLS_Y
                + (U_x[idx3d(z, y, x + 1)] - U_x[idx3d(z, y, x - 1)]) * CELLS_X;

            div[idx3d(z, y, x)] /= negate ? -2.0f : 2.0f;
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
            U1_z[idx3d(z, y, x)] *= 2.0f;
            U1_y[idx3d(z, y, x)] *= 2.0f;
            U1_x[idx3d(z, y, x)] *= 2.0f;

            U1_z[idx3d(z, y, x)] -= U0_z[idx3d(z, y, x)];
            U1_y[idx3d(z, y, x)] -= U0_y[idx3d(z, y, x)];
            U1_x[idx3d(z, y, x)] -= U0_x[idx3d(z, y, x)];
        }
    }

    __host__ void lin_solve(float *S1, const float *S0, const float a, const float b) {
        // kernel
        for (int iter = 0; iter < NUM_ITER; ++iter)
            lin_solve_kernel<<<grid_size, block_size>>>(S1, S0, a, b);
    }

    __host__ void diffuse(float *S1, const float *S0, boundary_type type) {
        constexpr float a = DT * VISCOSITY * CELLS_X * CELLS_X;
        constexpr float b = 1 + 6 * a;

        lin_solve(S1, S0, a, b);
    }

    __host__ void set_boundary(float *field, boundary_type type) {
        // kernel
        set_boundary_face_kernel<<<grid_size, block_size>>>(field, type);
    }

    __host__ void transport(float *S1, const float *S0, const float *U_z, const float *U_y, const float *U_x) {
        // kernel
        transport_kernel<<<grid_size, block_size>>>(S1, S0, U_z, U_y, U_x);
    }

    template <bool negate>
    __host__ void divergence(float *div, const float *U_z, const float *U_y, const float *U_x) {
        // kernel
        divergence_kernel<negate><<<grid_size, block_size>>>(div, U_z, U_y, U_x);
    }

    __host__ void project(float *U1_z, float *U1_y, float *U1_x, const float *U0_z, const float *U0_y, const float *U0_x) {
        static float *div = nullptr, *pressure = nullptr;
        if (!div) {
            cudaMalloc(&div, num_cells * sizeof(float));
            cudaMalloc(&pressure, num_cells * sizeof(float));
        }

        divergence<true>(div, U0_z, U0_y, U0_x);
        cudaDeviceSynchronize();

        set_boundary(div, BOUNDARY_SCALAR);
        cudaDeviceSynchronize();

        constexpr float a = CELLS_X * CELLS_Y;
        constexpr float b = 6.0f * a;
        lin_solve(pressure, div, a, b);
        cudaDeviceSynchronize();

        set_boundary(pressure, BOUNDARY_SCALAR);
        cudaDeviceSynchronize();

        project_kernel_<<<grid_size, block_size>>>(U1_z, U1_y, U1_x, U0_z, U0_y, U0_x, pressure);
        cudaDeviceSynchronize();

        set_boundary(U1_z, BOUNDARY_Z);
        set_boundary(U1_y, BOUNDARY_Y);
        set_boundary(U1_x, BOUNDARY_X);
        cudaDeviceSynchronize();
    }

    __host__ void reflect(float *U1_z, float *U1_y, float *U1_x, const float *U0_z, const float *U0_y, const float *U0_x) {
        project(U1_z, U1_y, U1_x, U0_z, U0_y, U0_x);
        cudaDeviceSynchronize();

        reflect_kernel<<<grid_size, block_size>>>(U1_z, U1_y, U1_x, U0_z, U0_y, U0_x);
    }

    __host__ void swap_workspace(float *&U0_z, float *&U0_y, float *&U0_x, float *&U1_z, float *&U1_y, float *&U1_x) {
        using namespace std;
        swap(U0_z, U1_z);
        swap(U0_y, U1_y);
        swap(U0_x, U1_x);
    }

    __host__ void cuda_solver::v_step(float *U1_z, float *U1_y, float *U1_x, float *U0_z, float *U0_y, float *U0_x) {
        set_boundary(U1_z, BOUNDARY_Z);
        set_boundary(U1_y, BOUNDARY_Y);
        set_boundary(U1_x, BOUNDARY_X);
        swap_workspace(U0_z, U0_y, U0_x, U1_z, U1_y, U1_x);
        // cudaDeviceSynchronize();

        transport(U1_z, U0_z, U0_z, U0_y, U0_x);
        transport(U1_y, U0_y, U0_z, U0_y, U0_x);
        transport(U1_x, U0_x, U0_z, U0_y, U0_x);
        swap_workspace(U0_z, U0_y, U0_x, U1_z, U1_y, U1_x);
        // cudaDeviceSynchronize();

        diffuse(U1_z, U0_z, BOUNDARY_Z);
        diffuse(U1_y, U0_y, BOUNDARY_Y);
        diffuse(U1_x, U0_x, BOUNDARY_X);
        swap_workspace(U0_z, U0_y, U0_x, U1_z, U1_y, U1_x);
        // cudaDeviceSynchronize();

        reflect(U1_z, U1_y, U1_x, U0_z, U0_y, U0_x);
        swap_workspace(U0_z, U0_y, U0_x, U1_z, U1_y, U1_x);
        // cudaDeviceSynchronize();

        transport(U1_z, U0_z, U0_z, U0_y, U0_x);
        transport(U1_y, U0_y, U0_z, U0_y, U0_x);
        transport(U1_x, U0_x, U0_z, U0_y, U0_x);
        swap_workspace(U0_z, U0_y, U0_x, U1_z, U1_y, U1_x);
        // cudaDeviceSynchronize();

        diffuse(U1_z, U0_z, BOUNDARY_Z);
        diffuse(U1_y, U0_y, BOUNDARY_Y);
        diffuse(U1_x, U0_x, BOUNDARY_X);
        // cudaDeviceSynchronize();

        project(U0_z, U0_y, U0_x, U1_z, U1_y, U1_x);
        // cudaDeviceSynchronize();
    }
}
