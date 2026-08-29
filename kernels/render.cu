//
// Created by condo on 2024/1/8.
//

#include "cuda_check.cuh"
#include "render.cuh"
#include "utils.cuh"

namespace {
    constexpr float PI = 3.14159265358979323846f;

    /* Scattering albedo per fluid. These used to be added as emission; they are
     * now what fraction of each wavelength survives a scattering event. */
    __constant__ float fluid_colors[7][3] = ALL_COLORS;

    /* Transmittance from the light source to every cell, rebuilt each frame.
     * Marching a shadow ray per sample would cost RENDER_SAMPLES times as much
     * again; sweeping the grid once in the light direction costs one pass. */
    float *s_light_T = nullptr;

    /* Henyey-Greenstein, normalised over the sphere. cos_theta is between the
     * direction the light travels and the direction the scattered radiance
     * travels, so it is +1 when the camera looks into the light. */
    __device__ inline float henyey_greenstein(float cos_theta) {
        const float g2 = HG_G * HG_G;
        const float d  = fmaxf(1.0f + g2 - 2.0f * HG_G * cos_theta, 1e-6f);
        return (1.0f - g2) / (4.0f * PI * d * sqrtf(d));
    }

    /* Extinction of the mixture at a cell. Co-located media add their
     * extinction coefficients -- they are not stacked slabs. */
    __device__ inline float extinction_at(float *const *field, unsigned active, int index) {
        float sigma = 0.0f;
        for (int f = 0; f < NUM_FLUIDS; f++)
            if (active & 1u << f)
                sigma += field[f][index];
        return sigma * ALPHA_OPTION;
    }

    /* Simulation cells 1..N-2 are fluid; 0 and N-1 are ghosts holding a copy of
     * the boundary layer. The visible box is the fluid part. */
    __device__ inline float3 volume_sample_point(const float p[3]) {
        return {1.0f + p[0] * (CELLS_X - 3), 1.0f + p[1] * (CELLS_Y - 3), 1.0f + p[2] * (CELLS_Z - 3)};
    }

    __device__ inline unsigned hash_u32(unsigned v) {
        v ^= v >> 16;
        v *= 0x7feb352du;
        v ^= v >> 15;
        v *= 0x846ca68bu;
        v ^= v >> 16;
        return v;
    }
} // namespace

__device__ bool intersect(const float pos[3], const float dir[3], float &t_in, float &t_out) {
    /* A large finite value rather than an infinity, deliberately: with a true
     * infinity a ray whose origin lies exactly on a slab plane computes 0 * inf
     * and the whole test collapses to NaN. At 1e32 the parallel-ray cases still
     * come out right -- inside the slab the bounds straddle zero and constrain
     * nothing, outside it they are both the same sign and reject. */
    const float big        = 1.0e32f;
    const float dir_frac_x = dir[0] == 0.0f ? big : 1.0f / dir[0];
    const float dir_frac_y = dir[1] == 0.0f ? big : 1.0f / dir[1];
    const float dir_frac_z = dir[2] == 0.0f ? big : 1.0f / dir[2];

    const float tx1 = (0 - pos[0]) * dir_frac_x;
    const float tx2 = (1 - pos[0]) * dir_frac_x;
    const float ty1 = (0 - pos[1]) * dir_frac_y;
    const float ty2 = (1 - pos[1]) * dir_frac_y;
    const float tz1 = (0 - pos[2]) * dir_frac_z;
    const float tz2 = (1 - pos[2]) * dir_frac_z;

    t_in  = fmaxf(fmaxf(fminf(tx1, tx2), fminf(ty1, ty2)), fminf(tz1, tz2));
    t_out = fminf(fminf(fmaxf(tx1, tx2), fmaxf(ty1, ty2)), fmaxf(tz1, tz2));

    /* When t_out < 0 and the ray is intersecting with AABB, the whole AABB is
     * behind us */
    if (t_out < 0) {
        return false;
    }

    return t_out >= t_in;
}

__global__ void light_fill_kernel(float *T) {
    const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x < CELLS_X && y < CELLS_Y && z < CELLS_Z)
        T[idx3d(z, y, x)] = 1.0f;
}

/* One slice of the light sweep.
 *
 *     T(p) = T(p - step) * exp(-sigma(p) * ds)
 *
 * The step advances exactly one cell along the light's dominant axis, so the
 * previous sample always lands in the slice this launch does not write and no
 * schedule can change the answer. AXIS names that dominant axis. */
template <int AXIS>
__global__ void light_slice_kernel(float *T, float *const *field, const unsigned active, const float3 back, const int slice, const float ds) {
    const unsigned a = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned b = blockIdx.y * blockDim.y + threadIdx.y;

    int x, y, z;
    if (AXIS == 0) {
        if (a >= CELLS_Y || b >= CELLS_Z)
            return;
        x = slice;
        y = (int)a;
        z = (int)b;
    }
    else if (AXIS == 1) {
        if (a >= CELLS_X || b >= CELLS_Z)
            return;
        y = slice;
        x = (int)a;
        z = (int)b;
    }
    else {
        if (a >= CELLS_X || b >= CELLS_Y)
            return;
        z = slice;
        x = (int)a;
        y = (int)b;
    }

    const int    index = idx3d(z, y, x);
    const float3 prev  = {x + back.x, y + back.y, z + back.z};

    T[index] = lin_interp(prev, T) * __expf(-extinction_at(field, active, index) * ds);
}

__global__ void density_renderer(
    const float    view[3][3],
    const float    origin[3],
    const float    focal,
    float        **field,
    const unsigned active_fluids,
    const float   *light_T,
    const float3   light_dir,
    float         *output) {
    const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= WINDOW_WIDTH || y >= WINDOW_HEIGHT)
        return;

    const int pixel   = 3 * (y * WINDOW_WIDTH + x);
    output[pixel + 0] = 0;
    output[pixel + 1] = 0;
    output[pixel + 2] = 0;

    // Pixel centres, not pixel corners: the half cell used to shift the image.
    const float px = x + 0.5f - WINDOW_WIDTH / 2.0f;
    const float py = y + 0.5f - WINDOW_HEIGHT / 2.0f;

    float dir[3];
    dir[0] = view[0][0] * px + view[1][0] * py + view[2][0] * (-focal);
    dir[1] = view[0][1] * px + view[1][1] * py + view[2][1] * (-focal);
    dir[2] = view[0][2] * px + view[1][2] * py + view[2][2] * (-focal);

    const float norm = sqrtf(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
    dir[0] /= norm;
    dir[1] /= norm;
    dir[2] /= norm;

    float t_in, t_out;
    if (!intersect(origin, dir, t_in, t_out))
        return;

    // The camera may sit inside the volume, in which case t_in is negative and
    // the march would start behind the eye.
    t_in = fmaxf(t_in, 0.0f);

    const float step = (t_out - t_in) / RENDER_SAMPLES;

    /* Starting every ray at the same offset puts the sampling error in phase
     * across neighbouring pixels, which shows up as wood-grain banding. A per
     * pixel offset turns it into noise. */
    const float jitter = hash_u32(y * WINDOW_WIDTH + x) * (1.0f / 4294967296.0f);

    // cos of the angle between the light's travel direction and the direction
    // the scattered radiance travels, which is back towards the camera
    const float cos_theta = -(light_dir.x * dir[0] + light_dir.y * dir[1] + light_dir.z * dir[2]);
    const float phase     = henyey_greenstein(cos_theta);
    const float lc[3]     = LIGHT_COLOR;

    float transmittance = 1.0f;
    float color[3]      = {0, 0, 0};

    for (int i = 0; i < RENDER_SAMPLES; i++) {
        // Nothing behind an all but opaque column can still change this pixel.
        if (transmittance < 0.005f)
            break;

        const float t        = t_in + (i + jitter) * step;
        const float point[3] = {origin[0] + t * dir[0], origin[1] + t * dir[1], origin[2] + t * dir[2]};

        if (point[0] < 0 || point[0] > 1 || point[1] < 0 || point[1] > 1 || point[2] < 0 || point[2] > 1)
            continue;

        const float3 sample = volume_sample_point(point);

        /* Co-located media add their extinction; the colour that scatters is the
         * density-weighted mix of their albedos. Compositing them one after
         * another, as this used to, makes whichever fluid the loop reaches first
         * occlude the rest within the same sample. */
        float sigma     = 0.0f;
        float albedo[3] = {0, 0, 0};
        for (int f = 0; f < NUM_FLUIDS; f++) {
            if (!(active_fluids & 1u << f))
                continue;
            const float d = lin_interp(sample, field[f]);
            if (d > 0) {
                sigma += d;
                albedo[0] += d * fluid_colors[f][0];
                albedo[1] += d * fluid_colors[f][1];
                albedo[2] += d * fluid_colors[f][2];
            }
        }
        if (sigma <= 0)
            continue;

        albedo[0] /= sigma;
        albedo[1] /= sigma;
        albedo[2] /= sigma;
        sigma *= ALPHA_OPTION;

        const float alpha  = 1.0f - __expf(-sigma * step);
        const float weight = transmittance * alpha * SCATTER_ALBEDO;
        const float direct = LIGHT_INTENSITY * phase * lin_interp(sample, light_T);

        color[0] += weight * albedo[0] * (direct * lc[0] + AMBIENT_LIGHT);
        color[1] += weight * albedo[1] * (direct * lc[1] + AMBIENT_LIGHT);
        color[2] += weight * albedo[2] * (direct * lc[2] + AMBIENT_LIGHT);

        transmittance *= 1.0f - alpha;
    }

    output[pixel + 0] = color[0];
    output[pixel + 1] = color[1];
    output[pixel + 2] = color[2];
}

__host__ void release_render_scratch(void) {
    cudaFree(s_light_T);
    s_light_T = nullptr;
}

__host__ void render_density(const float view[3][3], const float pos[3], const float focal, float **field, const unsigned active_fluids, float *output) {
    constexpr unsigned BLOCK = 16;

    if (!s_light_T)
        CUDA_CHECK(cudaMalloc(&s_light_T, num_cells * sizeof(float)));

    const float  ld[3] = LIGHT_DIR;
    const float  len   = sqrtf(ld[0] * ld[0] + ld[1] * ld[1] + ld[2] * ld[2]);
    const float3 L     = {ld[0] / len, ld[1] / len, ld[2] / len};

    // sweep along whichever axis the light runs most steeply down
    const float ax = fabsf(L.x), ay = fabsf(L.y), az = fabsf(L.z);
    const int   axis = ax >= ay && ax >= az ? 0 : (ay >= az ? 1 : 2);
    const float dom  = axis == 0 ? L.x : (axis == 1 ? L.y : L.z);
    const int   sign = dom > 0 ? 1 : -1;

    // one cell along the dominant axis per step; the other two follow the ray
    const float3 back = {-L.x / fabsf(dom), -L.y / fabsf(dom), -L.z / fabsf(dom)};
    const int    span = axis == 0 ? CELLS_X : (axis == 1 ? CELLS_Y : CELLS_Z);
    const float  ds   = 1.0f / (CELLS_X * fabsf(dom)); // world length of one step

    {
        const dim3 b(8, 8, 16);
        const dim3 g((CELLS_X + b.x - 1) / b.x, (CELLS_Y + b.y - 1) / b.y, (CELLS_Z + b.z - 1) / b.z);
        light_fill_kernel<<<g, b>>>(s_light_T);
    }

    const dim3 sb(BLOCK, BLOCK);
    const dim3 sg((CELLS_X + BLOCK - 1) / BLOCK, (CELLS_Y + BLOCK - 1) / BLOCK);
    for (int i = 1; i < span; i++) {
        const int slice = sign > 0 ? i : span - 1 - i;
        switch (axis) {
        case 0:
            light_slice_kernel<0><<<sg, sb>>>(s_light_T, field, active_fluids, back, slice, ds);
            break;
        case 1:
            light_slice_kernel<1><<<sg, sb>>>(s_light_T, field, active_fluids, back, slice, ds);
            break;
        default:
            light_slice_kernel<2><<<sg, sb>>>(s_light_T, field, active_fluids, back, slice, ds);
            break;
        }
    }

    const dim3 block_size(BLOCK, BLOCK);
    const dim3 grid_size((WINDOW_WIDTH + BLOCK - 1) / BLOCK, (WINDOW_HEIGHT + BLOCK - 1) / BLOCK);

    density_renderer<<<grid_size, block_size>>>(view, pos, focal, field, active_fluids, s_light_T, L, output);
    CUDA_CHECK_KERNEL();
}
