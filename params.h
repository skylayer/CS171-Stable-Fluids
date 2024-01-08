#ifndef PARAMS_H
#define PARAMS_H

#include <cassert>
#include "Eigen/Core"

// relevant systemwide parameters should go here

/* GUI parameters */
#define WINDOW_HEIGHT         600
#define WINDOW_WIDTH          600
#define WINDOW_Y              100
#define WINDOW_X              400
#define DISPLAY_KEY             0
#define ADD_AMT_INIT         0.5f
#define FORCE_SCALE          1.0f
#define ALPHA_OPTION            3
#define COLOR_SCALE            20
#define RAINBOW_HOLD_NSTEPS    20

/* Colors */
#define RED         Eigen::Vector3f{1.0f, 0.0f, 0.0f}
#define GREEN       Eigen::Vector3f{0.0f, 1.0f, 0.0f}
#define BLUE        Eigen::Vector3f{0.0f, 0.0f, 1.0f}
#define YELLOW      Eigen::Vector3f{0.5f, 0.5f, 0.0f}
#define CYAN        Eigen::Vector3f{0.0f, 0.5f, 0.5f}
#define MAGENTA     Eigen::Vector3f{0.5f, 0.0f, 0.5f}
#define WHITE       Eigen::Vector3f{0.33f, 0.33f, 0.33f}
#define ALL_COLORS  {RED, GREEN, BLUE, YELLOW, CYAN, MAGENTA, WHITE}

/* Grid parameters */
#define NDIM         3
#define CELLS_Z      200
#define CELLS_Y      200
#define CELLS_X      200
#define NUM_FLUIDS   1

/* Fluid parameters */
#define DISSIPATION 0.01F
#define VISCOSITY   0.0001F

/* Simulation parameters */
#define NUM_ITER     5
#define DT           0.001F
#define CLEANUP      false

/* Computed */
#define num_cells (CELLS_Z * CELLS_Y * CELLS_X)

/* Functions */
// inline int idx3d(const int z, const int y, const int x) {
//     return z * CELLS_Y * CELLS_X + y * CELLS_X + x;
// }

#define idx3d(z, y, x) ((z) * CELLS_Y * CELLS_X + (y) * CELLS_X + (x))

#endif
