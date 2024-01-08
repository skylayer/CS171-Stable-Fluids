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
    pos[2] = 1.0F;
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

    focal_length = 300.0F;
}

void FluidCUDA::step(void) {
    cuda_solver::v_step(U1_z, U1_y, U1_x, U0_z, U0_y, U0_x);
    auto lastErr = cudaGetLastError();
    if (lastErr != cudaSuccess) {
        fmt::print(stderr, "Error: {}\n", cudaGetErrorString(lastErr));
    }
    cudaDeviceSynchronize();
    for (int i = 0; i < NUM_FLUIDS; i++) {
        cuda_solver::s_step(S1[i], S0[i], U0_z, U0_y, U0_x);
    }
    cudaDeviceSynchronize();
    swap_grids();
}

void FluidCUDA::render() {

    auto lastErr = cudaGetLastError();
    if (lastErr != cudaSuccess) {
        fmt::print(stderr, "Error: {}\n", cudaGetErrorString(lastErr));
    }

    render_density(rot, pos, focal_length, S0, render_buffer);

    lastErr = cudaGetLastError();
    if (lastErr != cudaSuccess) {
        fmt::print(stderr, "Error: {}\n", cudaGetErrorString(lastErr));
    }

    cudaDeviceSynchronize();
}

void FluidCUDA::cleanup(void) {
    delete[] U0_z;
    delete[] U0_y;
    delete[] U0_x;
    delete[] U1_z;
    delete[] U1_y;
    delete[] U1_x;

    for (int i = 0; i < NUM_FLUIDS; i++) {
        delete[] S0[i];
        delete[] S1[i];
    }
    delete[] S0;
    delete[] S1;
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

float FluidCUDA::Uz_at(int z, int y, int x) {
    return U1_z[idx3d(z, y, x)];
}

float FluidCUDA::Uy_at(int z, int y, int x) {
    return U1_y[idx3d(z, y, x)];
}

float FluidCUDA::Ux_at(int z, int y, int x) {
    return U1_x[idx3d(z, y, x)];
}

float FluidCUDA::S_at(int z, int y, int x, int i) { return S1[i][idx3d(z, y, x)]; }

float *FluidCUDA::get_render_buffer() { return render_buffer; }
