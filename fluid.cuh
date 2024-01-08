//
// Created by condo on 2024/1/5.
//

#ifndef FLUID_CUH
#define FLUID_CUH

class FluidCUDA {
private:
    float  *U0_z, *U0_y, *U0_x, *U1_z, *U1_y, *U1_x; // velocity grids
    float  *render_buffer;
    float **S0, **S1; // scalar grids

    // Position and rotation of the camera
    float *pos;
    float  (*rot)[3];
    float  focal_length = 400.0F;

    void swap_grids(void);

public:
    void init(void);
    void step(void);
    void render(void);
    void cleanup(void);

    // setters, essentially
    void add_U_z_force_at(int z, int y, int x, float force);
    void add_U_y_force_at(int z, int y, int x, float force);
    void add_U_x_force_at(int z, int y, int x, float force);
    void add_source_at(int z, int y, int x, int i, float source);

    // getters
    float  Uz_at(int z, int y, int x);
    float  Uy_at(int z, int y, int x);
    float  Ux_at(int z, int y, int x);
    float  S_at(int z, int y, int x, int i);
    float *get_render_buffer(void);
};

#endif // FLUID_CUH
