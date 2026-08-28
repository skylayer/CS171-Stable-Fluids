#include "solver.cuh"
#include "cuda_check.cuh"
#include "utils.cuh"

#include <utility>

using namespace cuda_solver;

namespace {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
    /* atomicAdd(double *) is native only from SM 6.0 onwards. */
    __device__ inline double atomicAdd(double *address, double val) {
        auto             *ptr = reinterpret_cast<unsigned long long *>(address);
        unsigned long long old = *ptr, assumed;
        do {
            assumed = old;
            old     = atomicCAS(ptr, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
        } while (assumed != old);
        return __longlong_as_double(old);
    }
#endif

    /* Cheap stateless per-cell hash (Wang mix) folded into [0, 1).
     *
     * The previous buoyancy kernel kept a `static curandState` inside the kernel
     * body.  A `static` in device code is a single object in global memory shared
     * by the whole grid, so every one of the ~8M threads was initialising and
     * drawing from the same state concurrently.  A stateless hash gives each cell
     * an independent value with no shared state, so there is nothing to race on.
     */
    __device__ inline float hash01(unsigned seed) {
        seed = (seed ^ 61u) ^ (seed >> 16);
        seed += seed << 3;
        seed ^= seed >> 4;
        seed *= 0x27d4eb2du;
        seed ^= seed >> 15;
        return static_cast<float>(seed) * (1.0f / 4294967296.0f);
    }
}

namespace cuda_solver {
    /* Boundary conditions.
     *
     * This runs as its own kernel launch and is never fused into a compute
     * kernel.  A boundary cell is defined by interior cells that other *blocks*
     * write, and a kernel launch boundary is the only grid-wide barrier we have;
     * calling this at the tail of `transport_kernel` / `lin_solve_kernel` (as the
     * code used to) read neighbours that other blocks had not written yet.
     *
     * The writes are partitioned so that nothing this kernel writes is also read
     * by this kernel:
     *   - face cells   (exactly one coordinate on the boundary) read the strictly
     *                  interior neighbour one step in along that axis;
     *   - edge cells   (exactly two) read the interior cell one step in along both;
     *   - corner cells (all three) read the interior diagonal.
     * Every read therefore lands on an interior cell, which this kernel never
     * touches, so a single launch is race free.
     */
    __global__ void set_boundary_kernel(float *field, const boundary_type type) {
        const unsigned a = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned b = blockIdx.y * blockDim.y + threadIdx.y;

        // A velocity component is mirrored on the two faces it is normal to and
        // copied on the other four.
        const float sx = type == BOUNDARY_X ? -1.0f : 1.0f;
        const float sy = type == BOUNDARY_Y ? -1.0f : 1.0f;
        const float sz = type == BOUNDARY_Z ? -1.0f : 1.0f;

        /* faces -- the in-face coordinates stay strictly interior so the six
         * faces write disjoint sets of cells and leave the edges to the block
         * below. */
        if (a >= 1 && a < CELLS_Y - 1 && b >= 1 && b < CELLS_Z - 1) {
            field[idx3d(b, a, 0)]           = sx * field[idx3d(b, a, 1)];
            field[idx3d(b, a, CELLS_X - 1)] = sx * field[idx3d(b, a, CELLS_X - 2)];
        }
        if (a >= 1 && a < CELLS_X - 1 && b >= 1 && b < CELLS_Z - 1) {
            field[idx3d(b, 0, a)]           = sy * field[idx3d(b, 1, a)];
            field[idx3d(b, CELLS_Y - 1, a)] = sy * field[idx3d(b, CELLS_Y - 2, a)];
        }
        if (a >= 1 && a < CELLS_X - 1 && b >= 1 && b < CELLS_Y - 1) {
            field[idx3d(0, b, a)]           = sz * field[idx3d(1, b, a)];
            field[idx3d(CELLS_Z - 1, b, a)] = sz * field[idx3d(CELLS_Z - 2, b, a)];
        }

        /* edges -- row b == 0 does no face work, so reuse it here with `a` as the
         * free coordinate.  Each edge takes the diagonally interior cell. */
        if (b == 0) {
            if (a >= 1 && a < CELLS_Z - 1) { // the four edges running along z
                field[idx3d(a, 0, 0)]                     = field[idx3d(a, 1, 1)];
                field[idx3d(a, 0, CELLS_X - 1)]           = field[idx3d(a, 1, CELLS_X - 2)];
                field[idx3d(a, CELLS_Y - 1, 0)]           = field[idx3d(a, CELLS_Y - 2, 1)];
                field[idx3d(a, CELLS_Y - 1, CELLS_X - 1)] = field[idx3d(a, CELLS_Y - 2, CELLS_X - 2)];
            }
            if (a >= 1 && a < CELLS_Y - 1) { // the four edges running along y
                field[idx3d(0, a, 0)]                     = field[idx3d(1, a, 1)];
                field[idx3d(0, a, CELLS_X - 1)]           = field[idx3d(1, a, CELLS_X - 2)];
                field[idx3d(CELLS_Z - 1, a, 0)]           = field[idx3d(CELLS_Z - 2, a, 1)];
                field[idx3d(CELLS_Z - 1, a, CELLS_X - 1)] = field[idx3d(CELLS_Z - 2, a, CELLS_X - 2)];
            }
            if (a >= 1 && a < CELLS_X - 1) { // the four edges running along x
                field[idx3d(0, 0, a)]                     = field[idx3d(1, 1, a)];
                field[idx3d(0, CELLS_Y - 1, a)]           = field[idx3d(1, CELLS_Y - 2, a)];
                field[idx3d(CELLS_Z - 1, 0, a)]           = field[idx3d(CELLS_Z - 2, 1, a)];
                field[idx3d(CELLS_Z - 1, CELLS_Y - 1, a)] = field[idx3d(CELLS_Z - 2, CELLS_Y - 2, a)];
            }
        }

        /* corners -- the CPU reference averages the three adjoining edge cells,
         * which are written by this same launch.  Reading the interior diagonal
         * instead keeps the kernel race free and lands on the same value to
         * within one cell. */
        if (a == 0 && b == 0) {
            field[idx3d(0, 0, 0)]                                   = field[idx3d(1, 1, 1)];
            field[idx3d(0, 0, CELLS_X - 1)]                         = field[idx3d(1, 1, CELLS_X - 2)];
            field[idx3d(0, CELLS_Y - 1, 0)]                         = field[idx3d(1, CELLS_Y - 2, 1)];
            field[idx3d(0, CELLS_Y - 1, CELLS_X - 1)]               = field[idx3d(1, CELLS_Y - 2, CELLS_X - 2)];
            field[idx3d(CELLS_Z - 1, 0, 0)]                         = field[idx3d(CELLS_Z - 2, 1, 1)];
            field[idx3d(CELLS_Z - 1, 0, CELLS_X - 1)]               = field[idx3d(CELLS_Z - 2, 1, CELLS_X - 2)];
            field[idx3d(CELLS_Z - 1, CELLS_Y - 1, 0)]               = field[idx3d(CELLS_Z - 2, CELLS_Y - 2, 1)];
            field[idx3d(CELLS_Z - 1, CELLS_Y - 1, CELLS_X - 1)]     = field[idx3d(CELLS_Z - 2, CELLS_Y - 2, CELLS_X - 2)];
        }
    }

    /* One colour of a red-black (checkerboard) Gauss-Seidel sweep.
     *
     * A cell at (z, y, x) is coloured by the parity of z + y + x.  Every one of
     * its six face neighbours flips exactly one coordinate, so all six carry the
     * opposite colour and none of them is written by this launch.  The sweep is
     * therefore a genuine Gauss-Seidel update -- deterministic, and reading
     * values that are actually finished -- instead of the in-place free-for-all
     * the single-kernel version performed.
     *
     * omega is the SOR factor; omega == 1 reduces to plain Gauss-Seidel.
     */
    template <unsigned color>
    __global__ void lin_solve_kernel(float *S1, const float *S0, const float a, const float b, const float omega) {
        const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
        const unsigned z = blockIdx.z * blockDim.z + threadIdx.z;

        if (x < 1 || x >= CELLS_X - 1 || y < 1 || y >= CELLS_Y - 1 || z < 1 || z >= CELLS_Z - 1)
            return;
        if (((x + y + z) & 1u) != color)
            return;

        const int index = idx3d(z, y, x);

        const float relaxed = (S0[index] + a * (S1[index + 1] + S1[index - 1]
                                                + S1[index + CELLS_X] + S1[index - CELLS_X]
                                                + S1[index + CELLS_X * CELLS_Y] + S1[index - CELLS_X * CELLS_Y])) / b;

        S1[index] += omega * (relaxed - S1[index]);
    }

    __global__ void transport_kernel(float *S1, const float *S0, const float *U_z, const float *U_y, const float *U_x) {
        const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
        const unsigned z = blockIdx.z * blockDim.z + threadIdx.z;

        if (x >= 1 && x < CELLS_X - 1 && y >= 1 && y < CELLS_Y - 1 && z >= 1 && z < CELLS_Z - 1) {
            float z0 = static_cast<float>(z) - DT * U_z[idx3d(z, y, x)] * CELLS_Z;
            float y0 = static_cast<float>(y) - DT * U_y[idx3d(z, y, x)] * CELLS_Y;
            float x0 = static_cast<float>(x) - DT * U_x[idx3d(z, y, x)] * CELLS_X;

            z0 = fmax(0.0f, fmin(static_cast<float>(CELLS_Z - 1), z0));
            y0 = fmax(0.0f, fmin(static_cast<float>(CELLS_Y - 1), y0));
            x0 = fmax(0.0f, fmin(static_cast<float>(CELLS_X - 1), x0));

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

    __global__ void buoyancy_kernel(float *U1_z, float *U1_y, float *U1_x, const float *U0_z, const float *U0_y, const float *U0_x, float **S0, const unsigned frame) {
        const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
        const unsigned z = blockIdx.z * blockDim.z + threadIdx.z;

        if (x >= 1 && x < CELLS_X - 1 && y >= 1 && y < CELLS_Y - 1 && z >= 1 && z < CELLS_Z - 1) {
            const int index = idx3d(z, y, x);
            const float r   = 5.0f * (hash01(index * 2654435761u + frame) - 0.5f) + 1.0f;

            float lift = 0.0f;
            for (int i = 0; i < NUM_FLUIDS; i++)
                lift += r * DT * BUOYANCY * powf(0.5f * (S0[i][index] + S0[i][idx3d(z, y - 1, x)]), 1.3f) * CELLS_Y;

            // U1 is the whole new velocity field, not just the component buoyancy
            // touches: z and x used to be left holding whatever the scratch buffer
            // had from two frames ago, which then flowed on into transport.
            U1_z[index] = U0_z[index];
            U1_y[index] = U0_y[index] + lift;
            U1_x[index] = U0_x[index];
        }
    }

    /* Sum the interior of a field.  Each block reduces in shared memory and adds
     * one value to a double accumulator, so the result does not depend on block
     * order beyond double rounding. */
    __global__ void interior_sum_kernel(const float *field, double *accum) {
        __shared__ float partial[BLOCK_X * BLOCK_Y * BLOCK_Z];

        const unsigned x   = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned y   = blockIdx.y * blockDim.y + threadIdx.y;
        const unsigned z   = blockIdx.z * blockDim.z + threadIdx.z;
        const unsigned tid = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;

        partial[tid] = x >= 1 && x < CELLS_X - 1 && y >= 1 && y < CELLS_Y - 1 && z >= 1 && z < CELLS_Z - 1
                           ? field[idx3d(z, y, x)]
                           : 0.0f;
        __syncthreads();

        for (unsigned stride = (BLOCK_X * BLOCK_Y * BLOCK_Z) / 2; stride > 0; stride >>= 1) {
            if (tid < stride)
                partial[tid] += partial[tid + stride];
            __syncthreads();
        }

        if (tid == 0)
            atomicAdd(accum, static_cast<double>(partial[0]));
    }

    __global__ void subtract_mean_kernel(float *field, const double *accum, const double count) {
        const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
        const unsigned z = blockIdx.z * blockDim.z + threadIdx.z;

        if (x < CELLS_X && y < CELLS_Y && z < CELLS_Z)
            field[idx3d(z, y, x)] -= static_cast<float>(*accum / count);
    }

    __host__ void set_boundary(float *field, boundary_type type) {
        // kernel
        set_boundary_kernel<<<boundary_grid_size, boundary_block_size>>>(field, type);
    }

    __host__ void lin_solve(float *S1, const float *S0, const float a, const float b, const boundary_type type, const float omega = 1.0f) {
        // Red sweep, black sweep, then the boundary -- three separate launches.
        // Launches on the same stream are ordered and carry an implicit
        // grid-wide memory barrier, which is the synchronisation this solver
        // needs and the one a single fused kernel cannot provide.
        for (int iter = 0; iter < NUM_ITER; ++iter) {
            lin_solve_kernel<0><<<grid_size, block_size>>>(S1, S0, a, b, omega);
            lin_solve_kernel<1><<<grid_size, block_size>>>(S1, S0, a, b, omega);
            set_boundary(S1, type);
        }
    }

    __host__ void diffuse(float *S1, const float *S0, boundary_type type) {
        constexpr float a = DT * VISCOSITY * CELLS_X * CELLS_X;
        constexpr float b = 1 + 6 * a;

        // With the shipped VISCOSITY the off-diagonal weight is ~4e-7, so NUM_ITER
        // sweeps move the field by ~1e-5 relative -- the solve is the identity to
        // well inside what a float can represent, but it still costs 3 launches
        // per iteration per component. Copy instead, and let a real viscosity
        // switch the solve back on.
        if constexpr (6.0f * a < 1e-5f) {
            cudaMemcpyAsync(S1, S0, num_cells * sizeof(float), cudaMemcpyDeviceToDevice);
            set_boundary(S1, type);
            return;
        }

        // Strongly diagonally dominant (b >> a), so over-relaxation buys nothing.
        lin_solve(S1, S0, a, b, type);
    }

    __host__ void transport(float *S1, const float *S0, const float *U_z, const float *U_y, const float *U_x, const boundary_type type) {
        // kernel
        transport_kernel<<<grid_size, block_size>>>(S1, S0, U_z, U_y, U_x);
        set_boundary(S1, type);
    }

    template <bool negate>
    __host__ void divergence(float *div, const float *U_z, const float *U_y, const float *U_x) {
        // kernel
        divergence_kernel<negate><<<grid_size, block_size>>>(div, U_z, U_y, U_x);
        set_boundary(div, BOUNDARY_SCALAR);
    }

    __host__ void project(float *U1_z, float *U1_y, float *U1_x, const float *U0_z, const float *U0_y, const float *U0_x) {
        static float  *div = nullptr, *pressure = nullptr;
        static double *div_sum = nullptr;
        if (!div) {
            CUDA_CHECK(cudaMalloc(&div, num_cells * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&pressure, num_cells * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&div_sum, sizeof(double)));
            // cudaMalloc hands back uninitialised memory and the first relaxation
            // sweep reads `pressure` before it writes it, so it has to be cleared
            // at least once.  Later frames deliberately keep the previous
            // solution as a warm start.
            CUDA_CHECK(cudaMemset(div, 0, num_cells * sizeof(float)));
            CUDA_CHECK(cudaMemset(pressure, 0, num_cells * sizeof(float)));
        }

        divergence<true>(div, U0_z, U0_y, U0_x);

        /* Every wall is Neumann, so the discrete Laplacian is singular with the
         * constant vector in its null space and the system is only solvable for a
         * right-hand side that sums to zero over the interior.  The divergence of
         * a numerical velocity field never quite does, and the incompatible part
         * is exactly what the relaxation cannot remove: without this projection
         * the residual stalls on a plateau no matter how many sweeps it gets.
         * Subtracting over the whole field keeps the boundary layer consistent
         * with the interior it mirrors. */
        constexpr double interior = static_cast<double>(CELLS_X - 2) * (CELLS_Y - 2) * (CELLS_Z - 2);
        CUDA_CHECK(cudaMemsetAsync(div_sum, 0, sizeof(double)));
        interior_sum_kernel<<<grid_size, block_size>>>(div, div_sum);
        subtract_mean_kernel<<<grid_size, block_size>>>(div, div_sum, interior);

        constexpr float a = CELLS_X * CELLS_Y;
        constexpr float b = 6.0f * a;
        lin_solve(pressure, div, a, b, BOUNDARY_SCALAR, SOR_OMEGA);

        project_kernel_<<<grid_size, block_size>>>(U1_z, U1_y, U1_x, U0_z, U0_y, U0_x, pressure);
        set_boundary(U1_z, BOUNDARY_Z);
        set_boundary(U1_y, BOUNDARY_Y);
        set_boundary(U1_x, BOUNDARY_X);
    }

    __host__ void reflect(float *U1_z, float *U1_y, float *U1_x, const float *U0_z, const float *U0_y, const float *U0_x) {
        project(U1_z, U1_y, U1_x, U0_z, U0_y, U0_x);

        reflect_kernel<<<grid_size, block_size>>>(U1_z, U1_y, U1_x, U0_z, U0_y, U0_x);
    }

    __host__ void buoyancy(float *U1_z, float *U1_y, float *U1_x, const float *U0_z, const float *U0_y, const float *U0_x, float **S0) {
        static unsigned frame = 0;

        buoyancy_kernel<<<grid_size, block_size>>>(U1_z, U1_y, U1_x, U0_z, U0_y, U0_x, S0, frame++);
        set_boundary(U1_z, BOUNDARY_Z);
        set_boundary(U1_y, BOUNDARY_Y);
        set_boundary(U1_x, BOUNDARY_X);
    }

    __host__ void swap_workspace(float *&U0_z, float *&U0_y, float *&U0_x, float *&U1_z, float *&U1_y, float *&U1_x) {
        using namespace std;
        swap(U0_z, U1_z);
        swap(U0_y, U1_y);
        swap(U0_x, U1_x);
    }

    __host__ void v_step(float *&U1_z, float *&U1_y, float *&U1_x, float *&U0_z, float *&U0_y, float *&U0_x, float **S0) {

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
