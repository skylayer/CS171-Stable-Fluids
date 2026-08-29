#include "cuda_check.cuh"
#include "solver.cuh"
#include "utils.cuh"

#include <utility>

using namespace cuda_solver;

namespace {
    constexpr int STRIDE_X = 1;
    constexpr int STRIDE_Y = CELLS_X;
    constexpr int STRIDE_Z = CELLS_X * CELLS_Y;

    /* Where element (x, y, z) of a field with the given stagger actually sits,
     * in cell-index coordinates. */
    template <int S>
    __device__ inline float3 element_position(unsigned x, unsigned y, unsigned z) {
        float3 p = {static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)};
        if (S == STAGGER_X)
            p.x -= 0.5f;
        if (S == STAGGER_Y)
            p.y -= 0.5f;
        if (S == STAGGER_Z)
            p.z -= 0.5f;
        return p;
    }

    /* ...and the inverse: the array coordinate that samples such a field at p. */
    template <int S>
    __device__ inline float3 array_coord(float3 p) {
        if (S == STAGGER_X)
            p.x += 0.5f;
        if (S == STAGGER_Y)
            p.y += 0.5f;
        if (S == STAGGER_Z)
            p.z += 0.5f;
        return p;
    }

    /* Faces run one further than cells along the staggered axis. */
    template <int S>
    __device__ inline bool in_range(unsigned x, unsigned y, unsigned z) {
        const unsigned hx = S == STAGGER_X ? CELLS_X - 1 : CELLS_X - 2;
        const unsigned hy = S == STAGGER_Y ? CELLS_Y - 1 : CELLS_Y - 2;
        const unsigned hz = S == STAGGER_Z ? CELLS_Z - 1 : CELLS_Z - 2;
        return x >= 1 && x <= hx && y >= 1 && y <= hy && z >= 1 && z <= hz;
    }

    /* Each component is interpolated from its own array, with its own half-cell
     * offset undone -- that is the whole point of keeping them separate. */
    __device__ inline float3 sample_velocity(const float *ux, const float *uy, const float *uz, float3 p) {
        return {lin_interp(array_coord<STAGGER_X>(p), ux), lin_interp(array_coord<STAGGER_Y>(p), uy), lin_interp(array_coord<STAGGER_Z>(p), uz)};
    }

    /* Second-order (midpoint) back-trace. The first-order trace this replaces
     * put the departure point half a velocity-gradient off, which shows up as
     * extra smearing on every advected field. */
    __device__ inline float3 trace(const float *ux, const float *uy, const float *uz, float3 p, float dt) {
        const float3 v0  = sample_velocity(ux, uy, uz, p);
        const float3 mid = {p.x - 0.5f * dt * v0.x * CELLS_X, p.y - 0.5f * dt * v0.y * CELLS_Y, p.z - 0.5f * dt * v0.z * CELLS_Z};
        const float3 v1  = sample_velocity(ux, uy, uz, mid);
        return {p.x - dt * v1.x * CELLS_X, p.y - dt * v1.y * CELLS_Y, p.z - dt * v1.z * CELLS_Z};
    }

    __device__ inline float3 centred_velocity(const float *ux, const float *uy, const float *uz, int i) {
        return {0.5f * (ux[i] + ux[i + STRIDE_X]), 0.5f * (uy[i] + uy[i + STRIDE_Y]), 0.5f * (uz[i] + uz[i + STRIDE_Z])};
    }

    /* Curl at a cell centre, from the face values averaged to centres.
     * Well defined for 1 <= x,y,z <= N-3: the neighbours it reads must themselves
     * have both of their faces in range. */
    __device__ inline float3 curl_at(const float *ux, const float *uy, const float *uz, int i) {
        const float3 xp = centred_velocity(ux, uy, uz, i + STRIDE_X);
        const float3 xm = centred_velocity(ux, uy, uz, i - STRIDE_X);
        const float3 yp = centred_velocity(ux, uy, uz, i + STRIDE_Y);
        const float3 ym = centred_velocity(ux, uy, uz, i - STRIDE_Y);
        const float3 zp = centred_velocity(ux, uy, uz, i + STRIDE_Z);
        const float3 zm = centred_velocity(ux, uy, uz, i - STRIDE_Z);

        return {
            (yp.z - ym.z) * CELLS_Y * 0.5f - (zp.y - zm.y) * CELLS_Z * 0.5f,
            (zp.x - zm.x) * CELLS_Z * 0.5f - (xp.z - xm.z) * CELLS_X * 0.5f,
            (xp.y - xm.y) * CELLS_X * 0.5f - (yp.x - ym.x) * CELLS_Y * 0.5f};
    }

    __device__ inline bool curl_defined(unsigned x, unsigned y, unsigned z) {
        return x >= 1 && x <= CELLS_X - 3 && y >= 1 && y <= CELLS_Y - 3 && z >= 1 && z <= CELLS_Z - 3;
    }
} // namespace

namespace cuda_solver {

    /* ------------------------------------------------------------------ *
     *  Boundary conditions
     *
     *  Each runs as its own launch, never fused into a compute kernel: the
     *  values depend on interior cells other blocks write, and a launch
     *  boundary is the only grid-wide barrier available. Within a launch the
     *  writes and reads are disjoint, so no schedule can change the result.
     * ------------------------------------------------------------------ */

    /* Zero gradient on all six walls, for density, temperature and pressure. */
    __global__ void scalar_boundary_kernel(float *field) {
        const unsigned a = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned b = blockIdx.y * blockDim.y + threadIdx.y;

        if (a >= 1 && a < CELLS_Y - 1 && b >= 1 && b < CELLS_Z - 1) {
            field[idx3d(b, a, 0)]           = field[idx3d(b, a, 1)];
            field[idx3d(b, a, CELLS_X - 1)] = field[idx3d(b, a, CELLS_X - 2)];
        }
        if (a >= 1 && a < CELLS_X - 1 && b >= 1 && b < CELLS_Z - 1) {
            field[idx3d(b, 0, a)]           = field[idx3d(b, 1, a)];
            field[idx3d(b, CELLS_Y - 1, a)] = field[idx3d(b, CELLS_Y - 2, a)];
        }
        if (a >= 1 && a < CELLS_X - 1 && b >= 1 && b < CELLS_Y - 1) {
            field[idx3d(0, b, a)]           = field[idx3d(1, b, a)];
            field[idx3d(CELLS_Z - 1, b, a)] = field[idx3d(CELLS_Z - 2, b, a)];
        }

        // row b == 0 does no face work, so the twelve edges go here
        if (b == 0) {
            if (a >= 1 && a < CELLS_Z - 1) {
                field[idx3d(a, 0, 0)]                     = field[idx3d(a, 1, 1)];
                field[idx3d(a, 0, CELLS_X - 1)]           = field[idx3d(a, 1, CELLS_X - 2)];
                field[idx3d(a, CELLS_Y - 1, 0)]           = field[idx3d(a, CELLS_Y - 2, 1)];
                field[idx3d(a, CELLS_Y - 1, CELLS_X - 1)] = field[idx3d(a, CELLS_Y - 2, CELLS_X - 2)];
            }
            if (a >= 1 && a < CELLS_Y - 1) {
                field[idx3d(0, a, 0)]                     = field[idx3d(1, a, 1)];
                field[idx3d(0, a, CELLS_X - 1)]           = field[idx3d(1, a, CELLS_X - 2)];
                field[idx3d(CELLS_Z - 1, a, 0)]           = field[idx3d(CELLS_Z - 2, a, 1)];
                field[idx3d(CELLS_Z - 1, a, CELLS_X - 1)] = field[idx3d(CELLS_Z - 2, a, CELLS_X - 2)];
            }
            if (a >= 1 && a < CELLS_X - 1) {
                field[idx3d(0, 0, a)]                     = field[idx3d(1, 1, a)];
                field[idx3d(0, CELLS_Y - 1, a)]           = field[idx3d(1, CELLS_Y - 2, a)];
                field[idx3d(CELLS_Z - 1, 0, a)]           = field[idx3d(CELLS_Z - 2, 1, a)];
                field[idx3d(CELLS_Z - 1, CELLS_Y - 1, a)] = field[idx3d(CELLS_Z - 2, CELLS_Y - 2, a)];
            }
        }

        if (a == 0 && b == 0) {
            field[idx3d(0, 0, 0)]                               = field[idx3d(1, 1, 1)];
            field[idx3d(0, 0, CELLS_X - 1)]                     = field[idx3d(1, 1, CELLS_X - 2)];
            field[idx3d(0, CELLS_Y - 1, 0)]                     = field[idx3d(1, CELLS_Y - 2, 1)];
            field[idx3d(0, CELLS_Y - 1, CELLS_X - 1)]           = field[idx3d(1, CELLS_Y - 2, CELLS_X - 2)];
            field[idx3d(CELLS_Z - 1, 0, 0)]                     = field[idx3d(CELLS_Z - 2, 1, 1)];
            field[idx3d(CELLS_Z - 1, 0, CELLS_X - 1)]           = field[idx3d(CELLS_Z - 2, 1, CELLS_X - 2)];
            field[idx3d(CELLS_Z - 1, CELLS_Y - 1, 0)]           = field[idx3d(CELLS_Z - 2, CELLS_Y - 2, 1)];
            field[idx3d(CELLS_Z - 1, CELLS_Y - 1, CELLS_X - 1)] = field[idx3d(CELLS_Z - 2, CELLS_Y - 2, CELLS_X - 2)];
        }
    }

    /* MAC velocity walls, in four ordered passes.
     *
     *   pass 0  the wall-normal face of each component carries no flux, so it is
     *           exactly zero -- no mirroring, no half-cell guessing about where
     *           the wall is. Index 0 along the staggered axis is unused; zero it
     *           too so interpolation near the wall reads something sane.
     *   pass 1-3  free slip for the two components tangential to each wall pair.
     *
     * Splitting by axis keeps every pass's writes disjoint from its reads; the
     * ordering between passes is what fills the edges and corners correctly.
     */
    template <int PASS>
    __global__ void velocity_boundary_kernel(float *ux, float *uy, float *uz) {
        const unsigned a = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned b = blockIdx.y * blockDim.y + threadIdx.y;

        if (PASS == 0) {
            if (a < CELLS_Y && b < CELLS_Z) {
                ux[idx3d(b, a, 0)]           = 0.0f;
                ux[idx3d(b, a, 1)]           = 0.0f;
                ux[idx3d(b, a, CELLS_X - 1)] = 0.0f;
            }
            if (a < CELLS_X && b < CELLS_Z) {
                uy[idx3d(b, 0, a)]           = 0.0f;
                uy[idx3d(b, 1, a)]           = 0.0f;
                uy[idx3d(b, CELLS_Y - 1, a)] = 0.0f;
            }
            if (a < CELLS_X && b < CELLS_Y) {
                uz[idx3d(0, b, a)]           = 0.0f;
                uz[idx3d(1, b, a)]           = 0.0f;
                uz[idx3d(CELLS_Z - 1, b, a)] = 0.0f;
            }
        }
        else if (PASS == 1) { // the two x walls
            if (a < CELLS_Y && b < CELLS_Z) {
                uy[idx3d(b, a, 0)]           = uy[idx3d(b, a, 1)];
                uy[idx3d(b, a, CELLS_X - 1)] = uy[idx3d(b, a, CELLS_X - 2)];
                uz[idx3d(b, a, 0)]           = uz[idx3d(b, a, 1)];
                uz[idx3d(b, a, CELLS_X - 1)] = uz[idx3d(b, a, CELLS_X - 2)];
            }
        }
        else if (PASS == 2) { // the two y walls
            if (a < CELLS_X && b < CELLS_Z) {
                ux[idx3d(b, 0, a)]           = ux[idx3d(b, 1, a)];
                ux[idx3d(b, CELLS_Y - 1, a)] = ux[idx3d(b, CELLS_Y - 2, a)];
                uz[idx3d(b, 0, a)]           = uz[idx3d(b, 1, a)];
                uz[idx3d(b, CELLS_Y - 1, a)] = uz[idx3d(b, CELLS_Y - 2, a)];
            }
        }
        else { // the two z walls
            if (a < CELLS_X && b < CELLS_Y) {
                ux[idx3d(0, b, a)]           = ux[idx3d(1, b, a)];
                ux[idx3d(CELLS_Z - 1, b, a)] = ux[idx3d(CELLS_Z - 2, b, a)];
                uy[idx3d(0, b, a)]           = uy[idx3d(1, b, a)];
                uy[idx3d(CELLS_Z - 1, b, a)] = uy[idx3d(CELLS_Z - 2, b, a)];
            }
        }
    }

    /* ------------------------------------------------------------------ *
     *  Relaxation
     * ------------------------------------------------------------------ */

    /* One colour of a red-black (checkerboard) Gauss-Seidel sweep. Every face
     * neighbour of a cell flips exactly one coordinate, so all six carry the
     * opposite colour and none is written by this launch. */
    template <unsigned color>
    __global__ void lin_solve_kernel(float *S1, const float *S0, const float a, const float b, const float omega) {
        const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
        const unsigned z = blockIdx.z * blockDim.z + threadIdx.z;

        if (!in_range<CENTRED>(x, y, z))
            return;
        if (((x + y + z) & 1u) != color)
            return;

        const int   index   = idx3d(z, y, x);
        const float relaxed = (S0[index] + a * (S1[index + STRIDE_X] + S1[index - STRIDE_X] + S1[index + STRIDE_Y] + S1[index - STRIDE_Y] +
                                                S1[index + STRIDE_Z] + S1[index - STRIDE_Z])) /
                              b;

        S1[index] += omega * (relaxed - S1[index]);
    }

    /* ------------------------------------------------------------------ *
     *  Advection: MacCormack on top of a second-order semi-Lagrangian trace
     * ------------------------------------------------------------------ */

    template <int S>
    __global__ void advect_kernel(float *dst, const float *src, const float *ux, const float *uy, const float *uz, const float dt) {
        const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
        const unsigned z = blockIdx.z * blockDim.z + threadIdx.z;

        if (!in_range<S>(x, y, z))
            return;

        const float3 q = trace(ux, uy, uz, element_position<S>(x, y, z), dt);

        /* Cubic for the value, linear for the trace: the trace only has to land
         * the departure point, and sampling the velocity cubically costs another
         * three times 64 gathers for error that RK2 already handles. */
#if CUBIC_ADVECTION
        dst[idx3d(z, y, x)] = cubic_interp(array_coord<S>(q), src);
#else
        dst[idx3d(z, y, x)] = lin_interp(array_coord<S>(q), src);
#endif
    }

    /* phi^{n+1} = phi_forward + (phi^n - phi_backward) / 2
     *
     * The correction cancels the leading dissipation term of the forward pass.
     * It is only trustworthy where it stays inside the range the source data
     * actually spans -- outside it the scheme is extrapolating and will grow
     * without bound -- so those cells fall back to plain semi-Lagrangian. */
    template <int S>
    __global__ void maccormack_kernel(
        float *dst, const float *forward, const float *backward, const float *src, const float *ux, const float *uy, const float *uz, const float dt) {
        const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
        const unsigned z = blockIdx.z * blockDim.z + threadIdx.z;

        if (!in_range<S>(x, y, z))
            return;

        const int    index = idx3d(z, y, x);
        const float3 q     = trace(ux, uy, uz, element_position<S>(x, y, z), dt);

        float lo, hi;
        lin_interp_bounds(array_coord<S>(q), src, lo, hi);

        float value = forward[index] + 0.5f * (src[index] - backward[index]);
        if (value < lo || value > hi)
            value = forward[index];

        dst[index] = value;
    }

    /* ------------------------------------------------------------------ *
     *  Incompressibility
     * ------------------------------------------------------------------ */

    /* Compact divergence: the flux difference across the cell's own two faces.
     * Stores -div, which is the right-hand side lin_solve's arrangement wants. */
    __global__ void divergence_kernel(float *div, const float *ux, const float *uy, const float *uz) {
        const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
        const unsigned z = blockIdx.z * blockDim.z + threadIdx.z;

        if (!in_range<CENTRED>(x, y, z))
            return;

        const int index = idx3d(z, y, x);
        div[index] =
            -((ux[index + STRIDE_X] - ux[index]) * CELLS_X + (uy[index + STRIDE_Y] - uy[index]) * CELLS_Y + (uz[index + STRIDE_Z] - uz[index]) * CELLS_Z);
    }

    /* Compact gradient, the exact adjoint of the divergence above, so that
     * div(grad p) is the same 7-point Laplacian lin_solve inverts. */
    __global__ void project_kernel(float *u1z, float *u1y, float *u1x, const float *u0z, const float *u0y, const float *u0x, const float *p) {
        const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
        const unsigned z = blockIdx.z * blockDim.z + threadIdx.z;

        const int index = idx3d(z, y, x);

        if (x >= 2 && x <= CELLS_X - 2 && y >= 1 && y <= CELLS_Y - 2 && z >= 1 && z <= CELLS_Z - 2)
            u1x[index] = u0x[index] - (p[index] - p[index - STRIDE_X]) * CELLS_X;
        if (y >= 2 && y <= CELLS_Y - 2 && x >= 1 && x <= CELLS_X - 2 && z >= 1 && z <= CELLS_Z - 2)
            u1y[index] = u0y[index] - (p[index] - p[index - STRIDE_Y]) * CELLS_Y;
        if (z >= 2 && z <= CELLS_Z - 2 && x >= 1 && x <= CELLS_X - 2 && y >= 1 && y <= CELLS_Y - 2)
            u1z[index] = u0z[index] - (p[index] - p[index - STRIDE_Z]) * CELLS_Z;
    }

    __global__ void reflect_kernel(float *u1z, float *u1y, float *u1x, const float *u0z, const float *u0y, const float *u0x) {
        const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
        const unsigned z = blockIdx.z * blockDim.z + threadIdx.z;

        const int index = idx3d(z, y, x);

        if (in_range<STAGGER_X>(x, y, z))
            u1x[index] = 2.0f * u1x[index] - u0x[index];
        if (in_range<STAGGER_Y>(x, y, z))
            u1y[index] = 2.0f * u1y[index] - u0y[index];
        if (in_range<STAGGER_Z>(x, y, z))
            u1z[index] = 2.0f * u1z[index] - u0z[index];
    }

    /* The Poisson operator is singular under all-Neumann walls, so its
     * right-hand side has to sum to zero over the interior. With no-flux walls
     * the discrete divergence telescopes to exactly that, but float rounding
     * still leaves a residue worth removing. */
    __global__ void interior_sum_kernel(const float *field, double *partials) {
        __shared__ float partial[BLOCK_X * BLOCK_Y * BLOCK_Z];

        const unsigned x   = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned y   = blockIdx.y * blockDim.y + threadIdx.y;
        const unsigned z   = blockIdx.z * blockDim.z + threadIdx.z;
        const unsigned tid = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;

        partial[tid] = in_range<CENTRED>(x, y, z) ? field[idx3d(z, y, x)] : 0.0f;
        __syncthreads();

        for (unsigned stride = (BLOCK_X * BLOCK_Y * BLOCK_Z) / 2; stride > 0; stride >>= 1) {
            if (tid < stride)
                partial[tid] += partial[tid + stride];
            __syncthreads();
        }

        /* One value per block, indexed by block, rather than an atomicAdd into a
         * single accumulator: float addition is not associative, so an atomic
         * would make the result depend on the order blocks happen to finish and
         * cost the solver its bit-for-bit reproducibility. */
        if (tid == 0)
            partials[(blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x] = partial[0];
    }

    __global__ void reduce_partials_kernel(const double *partials, double *accum) {
        double total = 0.0;
        for (unsigned i = 0; i < NUM_BLOCKS; i++)
            total += partials[i];
        *accum = total;
    }

    __global__ void subtract_mean_kernel(float *field, const double *accum, const double count) {
        const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
        const unsigned z = blockIdx.z * blockDim.z + threadIdx.z;

        if (x < CELLS_X && y < CELLS_Y && z < CELLS_Z)
            field[idx3d(z, y, x)] -= static_cast<float>(*accum / count);
    }

    /* ------------------------------------------------------------------ *
     *  Forces
     * ------------------------------------------------------------------ */

    /* Boussinesq buoyancy on the y faces:  -ALPHA_SMOKE * s + BETA_TEMP * (T - T_ambient)
     *
     * u_y sits at y - 1/2, between cells y-1 and y, so the scalars it needs are
     * the average of that pair. The other two components are copied through --
     * v_step's contract is that U1 is the whole new field, not just the part a
     * force touched. */
    __global__ void
    buoyancy_kernel(float *u1z, float *u1y, float *u1x, const float *u0z, const float *u0y, const float *u0x, float **S0, const float *T0, const float dt) {
        const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
        const unsigned z = blockIdx.z * blockDim.z + threadIdx.z;

        const int index = idx3d(z, y, x);

        if (in_range<STAGGER_X>(x, y, z))
            u1x[index] = u0x[index];
        if (in_range<STAGGER_Z>(x, y, z))
            u1z[index] = u0z[index];

        if (in_range<STAGGER_Y>(x, y, z)) {
            float smoke = 0.0f;
            for (int f = 0; f < NUM_FLUIDS; f++)
                smoke += 0.5f * (S0[f][index] + S0[f][index - STRIDE_Y]);

            const float temperature = 0.5f * (T0[index] + T0[index - STRIDE_Y]);

            u1y[index] = u0y[index] + dt * (-ALPHA_SMOKE * smoke + BETA_TEMP * (temperature - T_AMBIENT));
        }
    }

    __global__ void vorticity_magnitude_kernel(float *mag, const float *ux, const float *uy, const float *uz) {
        const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
        const unsigned z = blockIdx.z * blockDim.z + threadIdx.z;

        if (x >= CELLS_X || y >= CELLS_Y || z >= CELLS_Z)
            return;

        const int index = idx3d(z, y, x);
        if (!curl_defined(x, y, z)) {
            mag[index] = 0.0f;
            return;
        }

        const float3 w = curl_at(ux, uy, uz, index);
        mag[index]     = sqrtf(w.x * w.x + w.y * w.y + w.z * w.z);
    }

    /* Vorticity confinement (Fedkiw et al. 2001).
     *
     *     N = grad|omega| / |grad|omega||        f = eps * h * (N x omega)
     *
     * N points from where vorticity is weak towards where it is strong, so the
     * force pushes rotation back into the vortex cores that the advection scheme
     * keeps smearing out. The h factor makes it vanish under refinement: it is a
     * stand-in for detail the grid cannot hold, not a permanent forcing. */
    __global__ void confinement_kernel(float *fz, float *fy, float *fx, const float *mag, const float *ux, const float *uy, const float *uz) {
        const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
        const unsigned z = blockIdx.z * blockDim.z + threadIdx.z;

        if (x >= CELLS_X || y >= CELLS_Y || z >= CELLS_Z)
            return;

        const int index = idx3d(z, y, x);
        fx[index] = fy[index] = fz[index] = 0.0f;

        if (x < 2 || x > CELLS_X - 4 || y < 2 || y > CELLS_Y - 4 || z < 2 || z > CELLS_Z - 4)
            return;

        const float3 eta = {
            (mag[index + STRIDE_X] - mag[index - STRIDE_X]) * CELLS_X * 0.5f,
            (mag[index + STRIDE_Y] - mag[index - STRIDE_Y]) * CELLS_Y * 0.5f,
            (mag[index + STRIDE_Z] - mag[index - STRIDE_Z]) * CELLS_Z * 0.5f};

        const float len = sqrtf(eta.x * eta.x + eta.y * eta.y + eta.z * eta.z);
        if (len < 1e-12f)
            return;

        const float3 n = {eta.x / len, eta.y / len, eta.z / len};
        const float3 w = curl_at(ux, uy, uz, index);
        const float  s = VORT_EPS / CELLS_X; // eps * h

        fx[index] = s * (n.y * w.z - n.z * w.y);
        fy[index] = s * (n.z * w.x - n.x * w.z);
        fz[index] = s * (n.x * w.y - n.y * w.x);
    }

    /* Cell-centred force onto the MAC faces. */
    __global__ void apply_force_kernel(float *uz, float *uy, float *ux, const float *fz, const float *fy, const float *fx, const float dt) {
        const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
        const unsigned z = blockIdx.z * blockDim.z + threadIdx.z;

        const int index = idx3d(z, y, x);

        if (in_range<STAGGER_X>(x, y, z))
            ux[index] += dt * 0.5f * (fx[index] + fx[index - STRIDE_X]);
        if (in_range<STAGGER_Y>(x, y, z))
            uy[index] += dt * 0.5f * (fy[index] + fy[index - STRIDE_Y]);
        if (in_range<STAGGER_Z>(x, y, z))
            uz[index] += dt * 0.5f * (fz[index] + fz[index - STRIDE_Z]);
    }

    /* Implicit relaxation towards a target: smoke towards zero, heat towards
     * ambient. Unconditionally stable, unlike the explicit form. */
    __global__ void decay_kernel(float *field, const float rate, const float target, const float dt) {
        const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
        const unsigned z = blockIdx.z * blockDim.z + threadIdx.z;

        if (x < CELLS_X && y < CELLS_Y && z < CELLS_Z) {
            const int index = idx3d(z, y, x);
            field[index]    = target + (field[index] - target) / (1.0f + dt * rate);
        }
    }

    /* ================================================================== *
     *  Host side
     * ================================================================== */

    namespace {
        float  *s_div = nullptr, *s_pressure = nullptr;
        float  *s_tmp_a = nullptr, *s_tmp_b = nullptr;
        float  *s_mag = nullptr, *s_fx = nullptr, *s_fy = nullptr, *s_fz = nullptr;
        double *s_sum = nullptr, *s_partials = nullptr;

        void ensure_scratch() {
            if (s_div)
                return;
            const size_t bytes = num_cells * sizeof(float);
            for (float **p : {&s_div, &s_pressure, &s_tmp_a, &s_tmp_b, &s_mag, &s_fx, &s_fy, &s_fz}) {
                CUDA_CHECK(cudaMalloc(p, bytes));
                CUDA_CHECK(cudaMemset(*p, 0, bytes));
            }
            CUDA_CHECK(cudaMalloc(&s_sum, sizeof(double)));
            CUDA_CHECK(cudaMalloc(&s_partials, NUM_BLOCKS * sizeof(double)));
        }
    } // namespace

    __host__ void release_scratch(void) {
        for (float **p : {&s_div, &s_pressure, &s_tmp_a, &s_tmp_b, &s_mag, &s_fx, &s_fy, &s_fz}) {
            cudaFree(*p);
            *p = nullptr;
        }
        cudaFree(s_sum);
        s_sum = nullptr;
        cudaFree(s_partials);
        s_partials = nullptr;
    }

    __host__ void set_scalar_boundary(float *field) { scalar_boundary_kernel<<<boundary_grid_size, boundary_block_size>>>(field); }

    __host__ void set_velocity_boundary(float *uz, float *uy, float *ux) {
        velocity_boundary_kernel<0><<<boundary_grid_size, boundary_block_size>>>(ux, uy, uz);
        velocity_boundary_kernel<1><<<boundary_grid_size, boundary_block_size>>>(ux, uy, uz);
        velocity_boundary_kernel<2><<<boundary_grid_size, boundary_block_size>>>(ux, uy, uz);
        velocity_boundary_kernel<3><<<boundary_grid_size, boundary_block_size>>>(ux, uy, uz);
    }

    __host__ void lin_solve(float *S1, const float *S0, const float a, const float b, const float omega = 1.0f) {
        // Red sweep, black sweep, boundary -- three launches. Launches on one
        // stream are ordered and carry an implicit grid-wide barrier, which is
        // the synchronisation the sweep needs and a fused kernel cannot give.
        for (int iter = 0; iter < NUM_ITER; ++iter) {
            lin_solve_kernel<0><<<grid_size, block_size>>>(S1, S0, a, b, omega);
            lin_solve_kernel<1><<<grid_size, block_size>>>(S1, S0, a, b, omega);
            set_scalar_boundary(S1);
        }
    }

    template <int S>
    __host__ void advect(float *dst, const float *src, const float *uz, const float *uy, const float *ux, const float dt) {
        advect_kernel<S><<<grid_size, block_size>>>(s_tmp_a, src, ux, uy, uz, dt);
        // The intermediate only feeds the error estimate, which is then limited,
        // so a zero-gradient wall is accurate enough for it.
        set_scalar_boundary(s_tmp_a);
        advect_kernel<S><<<grid_size, block_size>>>(s_tmp_b, s_tmp_a, ux, uy, uz, -dt);
        maccormack_kernel<S><<<grid_size, block_size>>>(dst, s_tmp_a, s_tmp_b, src, ux, uy, uz, dt);
    }

    __host__ void advect_velocity(float *dz, float *dy, float *dx, const float *sz, const float *sy, const float *sx, const float dt) {
        advect<STAGGER_X>(dx, sx, sz, sy, sx, dt);
        advect<STAGGER_Y>(dy, sy, sz, sy, sx, dt);
        advect<STAGGER_Z>(dz, sz, sz, sy, sx, dt);
        set_velocity_boundary(dz, dy, dx);
    }

    __host__ void diffuse_velocity(float *dz, float *dy, float *dx, const float *sz, const float *sy, const float *sx) {
        constexpr float a = DT * VISCOSITY * CELLS_X * CELLS_X;
        constexpr float b = 1 + 6 * a;

        // At the shipped VISCOSITY the off-diagonal weight is ~4e-7, so this
        // solve is the identity to well inside float precision while costing
        // three launches per iteration per component. The numerical viscosity of
        // the advection scheme is five orders of magnitude larger than the
        // physical one anyway; raising VISCOSITY switches the solve back on.
        if constexpr (6.0f * a < 1e-5f) {
            const size_t bytes = num_cells * sizeof(float);
            CUDA_CHECK(cudaMemcpyAsync(dx, sx, bytes, cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpyAsync(dy, sy, bytes, cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpyAsync(dz, sz, bytes, cudaMemcpyDeviceToDevice));
            return;
        }

        lin_solve(dx, sx, a, b);
        lin_solve(dy, sy, a, b);
        lin_solve(dz, sz, a, b);
        set_velocity_boundary(dz, dy, dx);
    }

    __host__ void project(float *dz, float *dy, float *dx, const float *sz, const float *sy, const float *sx) {
        ensure_scratch();

        divergence_kernel<<<grid_size, block_size>>>(s_div, sx, sy, sz);

        constexpr double interior = static_cast<double>(CELLS_X - 2) * (CELLS_Y - 2) * (CELLS_Z - 2);
        interior_sum_kernel<<<grid_size, block_size>>>(s_div, s_partials);
        reduce_partials_kernel<<<1, 1>>>(s_partials, s_sum);
        subtract_mean_kernel<<<grid_size, block_size>>>(s_div, s_sum, interior);

        /* Clear the guess every call. Keeping the previous frame's pressure
         * looks free but carries the solver's transient across frames, where it
         * compounds -- at omega 1.95 and above that alone makes the run
         * diverge. */
        CUDA_CHECK(cudaMemset(s_pressure, 0, num_cells * sizeof(float)));

        constexpr float a = CELLS_X * CELLS_Y;
        constexpr float b = 6.0f * a;
        lin_solve(s_pressure, s_div, a, b, SOR_OMEGA);

        project_kernel<<<grid_size, block_size>>>(dz, dy, dx, sz, sy, sx, s_pressure);
        set_velocity_boundary(dz, dy, dx);
    }

    __host__ void reflect(float *dz, float *dy, float *dx, const float *sz, const float *sy, const float *sx) {
        project(dz, dy, dx, sz, sy, sx);
        reflect_kernel<<<grid_size, block_size>>>(dz, dy, dx, sz, sy, sx);
        // The reflection changes the interior, so the walls have to follow; the
        // previous version left them holding the un-reflected projection.
        set_velocity_boundary(dz, dy, dx);
    }

    __host__ void confine_vorticity(float *uz, float *uy, float *ux, const float dt) {
        if constexpr (VORT_EPS <= 0.0f)
            return;

        ensure_scratch();
        vorticity_magnitude_kernel<<<grid_size, block_size>>>(s_mag, ux, uy, uz);
        confinement_kernel<<<grid_size, block_size>>>(s_fz, s_fy, s_fx, s_mag, ux, uy, uz);
        apply_force_kernel<<<grid_size, block_size>>>(uz, uy, ux, s_fz, s_fy, s_fx, dt);
        set_velocity_boundary(uz, uy, ux);
    }

    __host__ void swap_workspace(float *&U0_z, float *&U0_y, float *&U0_x, float *&U1_z, float *&U1_y, float *&U1_x) {
        using namespace std;
        swap(U0_z, U1_z);
        swap(U0_y, U1_y);
        swap(U0_x, U1_x);
    }

    /* Advection-reflection (Zehnder et al. 2018): two half steps with a
     * reflection between them, which is why one v_step advances 2 * DT. */
    __host__ void v_step(float *&U1_z, float *&U1_y, float *&U1_x, float *&U0_z, float *&U0_y, float *&U0_x, float **S0, const float *T0) {
        ensure_scratch();

        buoyancy_kernel<<<grid_size, block_size>>>(U1_z, U1_y, U1_x, U0_z, U0_y, U0_x, S0, T0, 2.0f * DT);
        set_velocity_boundary(U1_z, U1_y, U1_x);
        confine_vorticity(U1_z, U1_y, U1_x, 2.0f * DT);
        swap_workspace(U0_z, U0_y, U0_x, U1_z, U1_y, U1_x);

        advect_velocity(U1_z, U1_y, U1_x, U0_z, U0_y, U0_x, DT);
        swap_workspace(U0_z, U0_y, U0_x, U1_z, U1_y, U1_x);

        diffuse_velocity(U1_z, U1_y, U1_x, U0_z, U0_y, U0_x);
        swap_workspace(U0_z, U0_y, U0_x, U1_z, U1_y, U1_x);

        reflect(U1_z, U1_y, U1_x, U0_z, U0_y, U0_x);
        swap_workspace(U0_z, U0_y, U0_x, U1_z, U1_y, U1_x);

        advect_velocity(U1_z, U1_y, U1_x, U0_z, U0_y, U0_x, DT);
        swap_workspace(U0_z, U0_y, U0_x, U1_z, U1_y, U1_x);

        diffuse_velocity(U1_z, U1_y, U1_x, U0_z, U0_y, U0_x);

        project(U0_z, U0_y, U0_x, U1_z, U1_y, U1_x);
    }

    __host__ void s_step(float *S1, const float *S0, const float *U_z, const float *U_y, const float *U_x) {
        ensure_scratch();
        advect<CENTRED>(S1, S0, U_z, U_y, U_x, 2.0f * DT);
        set_scalar_boundary(S1);
        decay_kernel<<<grid_size, block_size>>>(S1, DISSIPATION, 0.0f, 2.0f * DT);
    }

    __host__ void t_step(float *T1, const float *T0, const float *U_z, const float *U_y, const float *U_x) {
        ensure_scratch();
        advect<CENTRED>(T1, T0, U_z, U_y, U_x, 2.0f * DT);
        set_scalar_boundary(T1);
        decay_kernel<<<grid_size, block_size>>>(T1, COOLING, T_AMBIENT, 2.0f * DT);
    }
} // namespace cuda_solver
