#include "fluid.cuh"
#include "fmt/format.h"
#include "kernels/render.cuh"
#include "kernels/solver.cuh"

void FluidCUDA::swap_grids(void) {
    float *temp;
    temp = U0_z;
    U0_z = U1_z;
    U1_z = temp;
    temp = U0_y;
    U0_y = U1_y;
    U1_y = temp;
    temp = U0_x;
    U0_x = U1_x;
    U1_x = temp;
    for (int i = 0; i < NUM_FLUIDS; i++) {
        S1[i] = S0[i];
    }
}

void FluidCUDA::init(void) {
    // Unified memory
    cudaMallocManaged(&U0_z, num_cells * sizeof(float));
    cudaMallocManaged(&U0_y, num_cells * sizeof(float));
    cudaMallocManaged(&U0_x, num_cells * sizeof(float));
    cudaMallocManaged(&U1_z, num_cells * sizeof(float));
    cudaMallocManaged(&U1_y, num_cells * sizeof(float));
    cudaMallocManaged(&U1_x, num_cells * sizeof(float));
    cudaMemset(U0_z, 0, num_cells * sizeof(float));
    cudaMemset(U0_y, 0, num_cells * sizeof(float));
    cudaMemset(U0_x, 0, num_cells * sizeof(float));
    cudaMemset(U1_z, 0, num_cells * sizeof(float));
    cudaMemset(U1_y, 0, num_cells * sizeof(float));
    cudaMemset(U1_x, 0, num_cells * sizeof(float));

    cudaMallocManaged(&S0, NUM_FLUIDS * sizeof(float *));
    cudaMallocManaged(&S1, NUM_FLUIDS * sizeof(float *));

    for (int i = 0; i < NUM_FLUIDS; i++) {
        cudaMallocManaged(&S0[i], num_cells * sizeof(float));
        cudaMallocManaged(&S1[i], num_cells * sizeof(float));
        cudaMemset(S0[i], 0, num_cells * sizeof(float));
        cudaMemset(S1[i], 0, num_cells * sizeof(float));
    }

    cudaMallocManaged(&render_buffer, 3 * WINDOW_WIDTH * WINDOW_HEIGHT * sizeof(float));

    cudaMallocManaged(&pos, 3 * sizeof(float));
    pos[0] = 0.5F;
    pos[1] = 0.5F;
    pos[2] = 2.5F;
    cudaMallocManaged(reinterpret_cast<float **>(&rot), 3 * 3 * sizeof(float));
    rot[0][0] = 1.0F;
    rot[0][1] = 0.0F;
    rot[0][2] = 0.0F;
    rot[1][0] = 0.0F;
    rot[1][1] = 1.0F;
    rot[1][2] = 0.0F;
    rot[2][0] = 0.0F;
    rot[2][1] = 0.0F;
    rot[2][2] = 1.0F;

    theta = 3.1415926F / 2;
    phi = 0.0F;
    radius = 2.0F;

    focal_length = 600.0F;
}

void FluidCUDA::step() {
    cuda_solver::v_step(U1_z, U1_y, U1_x, U0_z, U0_y, U0_x);
    auto lastErr = cudaGetLastError();
    if (lastErr != cudaSuccess) {
        fmt::print(stderr, "Error: {}\n", cudaGetErrorString(lastErr));
    }

    for (int i = 0; i < NUM_FLUIDS; i++) {
        cudaDeviceSynchronize();
        cuda_solver::s_step(S1[i], S0[i], U0_z, U0_y, U0_x);
    }
    cudaDeviceSynchronize();
    swap_grids();
}

void FluidCUDA::render(float *dest) {
    if (dest == nullptr) {
        dest = render_buffer;
    }

    auto lastErr = cudaGetLastError();
    if (lastErr != cudaSuccess) {
        fmt::print(stderr, "Error: {}\n", cudaGetErrorString(lastErr));
    }

    render_density(rot, pos, focal_length, S0, dest);

    lastErr = cudaGetLastError();
    if (lastErr != cudaSuccess) {
        fmt::print(stderr, "Error: {}\n", cudaGetErrorString(lastErr));
    }

    lastErr = cudaDeviceSynchronize();
    if (lastErr != cudaSuccess) {
        fmt::print(stderr, "Error: {}\n", cudaGetErrorString(lastErr));
    }
}

void FluidCUDA::cleanup(void) {
    cudaFree(U0_z);
    cudaFree(U0_y);
    cudaFree(U0_x);
    cudaFree(U1_z);
    cudaFree(U1_y);
    cudaFree(U1_x);
    for (int i = 0; i < NUM_FLUIDS; i++) {
        cudaFree(S0[i]);
        cudaFree(S1[i]);
    }
    cudaFree(S0);
    cudaFree(S1);
    cudaFree(render_buffer);
    cudaFree(pos);
    cudaFree(rot);
}

void FluidCUDA::add_U_z_force_at(int z, int y, int x, float force) {
    if (z > 0 && z < CELLS_Z - 1 && y > 0 && y < CELLS_Y - 1 && x > 0 && x < CELLS_X - 1) {
        U1_z[idx3d(z, y, x)] += force;
    }
}

void FluidCUDA::add_U_y_force_at(int z, int y, int x, float force) {
    if (z > 0 && z < CELLS_Z - 1 && y > 0 && y < CELLS_Y - 1 && x > 0 && x < CELLS_X - 1) {
        U1_y[idx3d(z, y, x)] += force;
    }
}

void FluidCUDA::add_U_x_force_at(int z, int y, int x, float force) {
    if (z > 0 && z < CELLS_Z - 1 && y > 0 && y < CELLS_Y - 1 && x > 0 && x < CELLS_X - 1) {
        U1_x[idx3d(z, y, x)] += force;
    }
}

void FluidCUDA::add_source_at(int z, int y, int x, int i, float source) {
    if (z > 0 && z < CELLS_Z - 1 && y > 0 && y < CELLS_Y - 1 && x > 0 && x < CELLS_X - 1) {
        S1[i][idx3d(z, y, x)] += source;
    }
}

void FluidCUDA::rot_left(float angle) {
    // Rotate the camera
    theta -= angle;
    theta = fmodf(theta, 2 * 3.1415926F);

    pos[0] = 0.5F + radius * cosf(phi) * cosf(theta);
    pos[1] = 0.5F + radius * sinf(phi);
    pos[2] = 0.5F + radius * cosf(phi) * sinf(theta);

    // Target position
    float target[3] = {0.5f, 0.5f, 0.5f};

    // Direction vector (from pos to target)
    float dir[3] = {
        pos[0] - target[0],
        pos[1] - target[1],
        pos[2] - target[2]
    };
    // Normalize the direction vector
    float norm = sqrtf(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
    dir[0] /= norm;
    dir[1] /= norm;
    dir[2] /= norm;

    // Up vector (assuming Y-up)
    float up[3] = {0.0f, 1.0f, 0.0f};

    // Right vector = up x dir
    float right[3] = {
        up[1] * dir[2] - up[2] * dir[1],
        up[2] * dir[0] - up[0] * dir[2],
        up[0] * dir[1] - up[1] * dir[0]
    };

    norm = sqrtf(right[0] * right[0] + right[1] * right[1] + right[2] * right[2]);
    right[0] /= norm;
    right[1] /= norm;
    right[2] /= norm;

    // Recompute up vector = dir x right
    float new_up[3] = {
        dir[1] * right[2] - dir[2] * right[1],
        dir[2] * right[0] - dir[0] * right[2],
        dir[0] * right[1] - dir[1] * right[0]
    };
    norm = sqrtf(new_up[0] * new_up[0] + new_up[1] * new_up[1] + new_up[2] * new_up[2]);
    new_up[0] /= norm;
    new_up[1] /= norm;
    new_up[2] /= norm;

    // Construct the rotation matrix
    rot[0][0] = right[0];
    rot[0][1] = right[1];
    rot[0][2] = right[2];
    rot[1][0] = new_up[0];
    rot[1][1] = new_up[1];
    rot[1][2] = new_up[2];
    rot[2][0] = dir[0];
    rot[2][1] = dir[1];
    rot[2][2] = dir[2];
}

void FluidCUDA::rot_up(float angle) {
    // Rotate the camera
    phi += angle;
    phi = max(min(phi, 3.1415926F / 2), -3.1415926F / 2);

    pos[0] = 0.5F + radius * cosf(phi) * cosf(theta);
    pos[1] = 0.5F + radius * sinf(phi);
    pos[2] = 0.5F + radius * cosf(phi) * sinf(theta);

    // Target position
    float target[3] = {0.5f, 0.5f, 0.5f};

    // Direction vector (from pos to target)
    float dir[3] = {
        pos[0] - target[0],
        pos[1] - target[1],
        pos[2] - target[2]
    };
    // Normalize the direction vector
    float norm = sqrtf(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
    dir[0] /= norm;
    dir[1] /= norm;
    dir[2] /= norm;

    // Up vector (assuming Y-up)
    float up[3] = {0.0f, 1.0f, 0.0f};

    // Right vector = up x dir
    float right[3] = {
        up[1] * dir[2] - up[2] * dir[1],
        up[2] * dir[0] - up[0] * dir[2],
        up[0] * dir[1] - up[1] * dir[0]
    };

    norm = sqrtf(right[0] * right[0] + right[1] * right[1] + right[2] * right[2]);
    right[0] /= norm;
    right[1] /= norm;
    right[2] /= norm;

    // Recompute up vector = dir x right
    float new_up[3] = {
        dir[1] * right[2] - dir[2] * right[1],
        dir[2] * right[0] - dir[0] * right[2],
        dir[0] * right[1] - dir[1] * right[0]
    };
    norm = sqrtf(new_up[0] * new_up[0] + new_up[1] * new_up[1] + new_up[2] * new_up[2]);
    new_up[0] /= norm;
    new_up[1] /= norm;
    new_up[2] /= norm;

    // Construct the rotation matrix
    rot[0][0] = right[0];
    rot[0][1] = right[1];
    rot[0][2] = right[2];
    rot[1][0] = new_up[0];
    rot[1][1] = new_up[1];
    rot[1][2] = new_up[2];
    rot[2][0] = dir[0];
    rot[2][1] = dir[1];
    rot[2][2] = dir[2];
}

void FluidCUDA::zoom_in(float dist) {
    radius -= dist;
    radius = max(radius, 1.0F);

    pos[0] = 0.5F + radius * cosf(phi) * cosf(theta);
    pos[1] = 0.5F + radius * sinf(phi);
    pos[2] = 0.5F + radius * cosf(phi) * sinf(theta);

    fmt::print("pos: {}, {}, {}\n", pos[0], pos[1], pos[2]);

    // Target position
    float target[3] = {0.5f, 0.5f, 0.5f};

    // Direction vector (from pos to target)
    float dir[3] = {
        pos[0] - target[0],
        pos[1] - target[1],
        pos[2] - target[2]
    };
    // Normalize the direction vector
    float norm = sqrtf(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
    dir[0] /= norm;
    dir[1] /= norm;
    dir[2] /= norm;

    // Up vector (assuming Y-up)
    float up[3] = {0.0f, 1.0f, 0.0f};

    // Right vector = up x dir
    float right[3] = {
        up[1] * dir[2] - up[2] * dir[1],
        up[2] * dir[0] - up[0] * dir[2],
        up[0] * dir[1] - up[1] * dir[0]
    };

    norm = sqrtf(right[0] * right[0] + right[1] * right[1] + right[2] * right[2]);
    right[0] /= norm;
    right[1] /= norm;
    right[2] /= norm;

    // Recompute up vector = dir x right
    float new_up[3] = {
        dir[1] * right[2] - dir[2] * right[1],
        dir[2] * right[0] - dir[0] * right[2],
        dir[0] * right[1] - dir[1] * right[0]
    };
    norm = sqrtf(new_up[0] * new_up[0] + new_up[1] * new_up[1] + new_up[2] * new_up[2]);
    new_up[0] /= norm;
    new_up[1] /= norm;
    new_up[2] /= norm;

    // Construct the rotation matrix
    rot[0][0] = right[0];
    rot[0][1] = right[1];
    rot[0][2] = right[2];
    rot[1][0] = new_up[0];
    rot[1][1] = new_up[1];
    rot[1][2] = new_up[2];
    rot[2][0] = dir[0];
    rot[2][1] = dir[1];
    rot[2][2] = dir[2];
}


float FluidCUDA::Uz_at(int z, int y, int x) { return U1_z[idx3d(z, y, x)]; }

float FluidCUDA::Uy_at(int z, int y, int x) { return U1_y[idx3d(z, y, x)]; }

float FluidCUDA::Ux_at(int z, int y, int x) { return U1_x[idx3d(z, y, x)]; }

float FluidCUDA::S_at(int z, int y, int x, int i) { return S1[i][idx3d(z, y, x)]; }

float *FluidCUDA::get_render_buffer() { return render_buffer; }
