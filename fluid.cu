#include "Fluid.cuh"

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

    S0 = new float *[NUM_FLUIDS]();
    S1 = new float *[NUM_FLUIDS]();
    for (int i = 0; i < NUM_FLUIDS; i++) {
        cudaMallocManaged(&S0[i], num_cells * sizeof(float));
        cudaMallocManaged(&S1[i], num_cells * sizeof(float));
        cudaMemset(S0[i], 0, num_cells * sizeof(float));
        cudaMemset(S1[i], 0, num_cells * sizeof(float));
    }
}

void FluidCUDA::step(void) {
    cuda_solver::v_step(U1_z, U1_y, U1_x, U0_z, U0_y, U0_x);
    auto lastErr = cudaGetLastError();
    if (lastErr != cudaSuccess) {
        fmt::print(stderr, "Error: {}\n", cudaGetErrorString(lastErr));
    }
    // for (int i = 0; i < NUM_FLUIDS; i++) {
    //     solver::s_step(S1[i], S0[i], U0_z, U0_y, U0_x);
    // }
    cudaDeviceSynchronize();
    swap_grids();
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

float FluidCUDA::S_at(int z, int y, int x, int i) {
    return S1[i][idx3d(z, y, x)];
}
