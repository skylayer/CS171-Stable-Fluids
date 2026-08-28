//
// Created by condo on 2024/1/8.
//

#include "cuda_check.cuh"
#include "render.cuh"
#include "utils.cuh"


__device__ bool intersect(const float pos[3], const float dir[3], float &t_in, float &t_out) {
    float dir_frac_x = (dir[0] == 0.0) ? 1.0e32 : 1.0f / dir[0];
    float dir_frac_y = (dir[1] == 0.0) ? 1.0e32 : 1.0f / dir[1];
    float dir_frac_z = (dir[2] == 0.0) ? 1.0e32 : 1.0f / dir[2];

    const float tx1 = (0 - pos[0]) * dir_frac_x;
    const float tx2 = (1 - pos[0]) * dir_frac_x;
    const float ty1 = (0 - pos[1]) * dir_frac_y;
    const float ty2 = (1 - pos[1]) * dir_frac_y;
    const float tz1 = (0 - pos[2]) * dir_frac_z;
    const float tz2 = (1 - pos[2]) * dir_frac_z;

    t_in  = max(max(min(tx1, tx2), min(ty1, ty2)), min(tz1, tz2));
    t_out = min(min(max(tx1, tx2), max(ty1, ty2)), max(tz1, tz2));

    /* When t_out < 0 and the ray is intersecting with AABB, the whole AABB is
     * behind us */
    if (t_out < 0) {
        return false;
    }

    return t_out >= t_in;
}


/* Held in constant memory rather than as a per-thread local array: indexing it
 * by a runtime fluid id would otherwise push all 21 floats into local memory and
 * make every thread store them again at launch. */
__constant__ float fluid_colors[7][3] = ALL_COLORS;

__device__ bool inside_volume(const float pos[3]) { return pos[0] >= 0 && pos[0] <= 1 && pos[1] >= 0 && pos[1] <= 1 && pos[2] >= 0 && pos[2] <= 1; }

__global__ void density_renderer(const float view[3][3], const float origin[3], const float focal, float **field, const unsigned active_fluids, float *output) {
    const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < WINDOW_WIDTH && y < WINDOW_HEIGHT) {
        // Reset frame buffer
        output[3 * (y * WINDOW_WIDTH + x) + 0] = 0;
        output[3 * (y * WINDOW_WIDTH + x) + 1] = 0;
        output[3 * (y * WINDOW_WIDTH + x) + 2] = 0;

        // Local coordinate (x, y, -focal) to world coordinate
        float dir[3];
        dir[0] = view[0][0] * (x - WINDOW_WIDTH / 2.0f) + view[1][0] * (y - WINDOW_HEIGHT / 2.0f) + view[2][0] * (-focal);
        dir[1] = view[0][1] * (x - WINDOW_WIDTH / 2.0f) + view[1][1] * (y - WINDOW_HEIGHT / 2.0f) + view[2][1] * (-focal);
        dir[2] = view[0][2] * (x - WINDOW_WIDTH / 2.0f) + view[1][2] * (y - WINDOW_HEIGHT / 2.0f) + view[2][2] * (-focal);

        const float norm = sqrtf(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
        dir[0] /= norm;
        dir[1] /= norm;
        dir[2] /= norm;

        if (float t_in, t_out; intersect(origin, dir, t_in, t_out)) {
            const int SampleNum = 1000;

            // The camera may sit inside the volume, in which case t_in is
            // negative and the march would start behind the eye.
            t_in = fmaxf(t_in, 0.0f);

            const float step = (t_out - t_in) / SampleNum;

            float accumlatedOpacity = 0;
            float color[3]          = {0, 0, 0};

            for (int i = 0; i < SampleNum; i++) {
                // Nothing behind an all but opaque column can still change this pixel.
                if (accumlatedOpacity > 0.995f) {
                    break;
                }

                const float t        = t_in + i * step;
                const float point[3] = {origin[0] + t * dir[0], origin[1] + t * dir[1], origin[2] + t * dir[2]};

                // Independent of the fluid, so it does not belong in the loop below.
                if (!inside_volume(point)) {
                    continue;
                }

                const float3 sample = {point[0] * (CELLS_X - 1), point[1] * (CELLS_Y - 1), point[2] * (CELLS_Z - 1)};

                for (int fluidId = 0; fluidId < NUM_FLUIDS; fluidId++) {
                    // active_fluids is uniform across the warp, so this costs no
                    // divergence and skips the eight trilinear fetches a fluid
                    // that was never seeded would contribute nothing from.
                    if (!(active_fluids & 1u << fluidId)) {
                        continue;
                    }

                    const float d = lin_interp(sample, field[fluidId]);
                    if (d > 0) {
                        const float opacity = 1 - __expf(-d * step * ALPHA_OPTION);
                        const float weight  = opacity * (1 - accumlatedOpacity);

                        color[0] += weight * fluid_colors[fluidId][0];
                        color[1] += weight * fluid_colors[fluidId][1];
                        color[2] += weight * fluid_colors[fluidId][2];
                        accumlatedOpacity += weight;
                    }
                }
            }

            output[3 * (y * WINDOW_WIDTH + x) + 0] = color[0];
            output[3 * (y * WINDOW_WIDTH + x) + 1] = color[1];
            output[3 * (y * WINDOW_WIDTH + x) + 2] = color[2];
        }
    }
}


__host__ void render_density(const float view[3][3], const float pos[3], const float focal, float **field, const unsigned active_fluids, float *output) {
    constexpr unsigned BLOCK = 16;

    const dim3 block_size(BLOCK, BLOCK);
    const dim3 grid_size((WINDOW_WIDTH + BLOCK - 1) / BLOCK, (WINDOW_HEIGHT + BLOCK - 1) / BLOCK);

    density_renderer<<<grid_size, block_size>>>(view, pos, focal, field, active_fluids, output);
    CUDA_CHECK_KERNEL();
}
