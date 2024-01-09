//
// Created by condo on 2024/1/8.
//

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


__device__ float density(const float *field, const float pos[3]) {
    if (pos[0] < 0 || pos[0] > 1 || pos[1] < 0 || pos[1] > 1 || pos[2] < 0 || pos[2] > 1) {
        return 0;
    }

    return lin_interp({pos[0] * (CELLS_X - 2) + 1, pos[1] * (CELLS_Y - 2) + 1, pos[2] * (CELLS_Z - 2) + 1}, field);
}

__global__ void density_renderer(const float view[3][3], const float origin[3], const float focal, float **field, float *output) {
    const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;

    const float colors[7][3] = ALL_COLORS;

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

            float step = (t_out - t_in) / SampleNum;

            float accumlatedOpacity = 0;
            float color[3]          = {0, 0, 0};

            for (int i = 0; i < SampleNum; i++) {
                float t = t_in + i * step;
                float point[3] = {origin[0] + t * dir[0], origin[1] + t * dir[1], origin[2] + t * dir[2]};
                for (int fluidId = 0; fluidId < NUM_FLUIDS; fluidId++) {
                    float d = density(field[fluidId], point);
                    if (d > 0) {
                        float opacity = 1 - exp(-d * step * ALPHA_OPTION);
                        float weight  = opacity * (1 - accumlatedOpacity);

                        color[0] += weight * colors[fluidId][0];
                        color[1] += weight * colors[fluidId][1];
                        color[2] += weight * colors[fluidId][2];
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


__host__ void render_density(const float view[3][3], const float pos[3], const float focal, float **field, float *output) {
    const dim3 block_size(16, 16);
    const dim3 grid_size(WINDOW_WIDTH / block_size.x + 1, WINDOW_HEIGHT / block_size.y + 1);

    density_renderer<<<grid_size, block_size>>>(view, pos, focal, field, output);
    cudaDeviceSynchronize();
}
