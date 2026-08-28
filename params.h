#ifndef PARAMS_H
#define PARAMS_H

#include <cassert>

// relevant systemwide parameters should go here

/* GUI parameters */
#define WINDOW_HEIGHT       600
#define WINDOW_WIDTH        600
#define WINDOW_Y            100
#define WINDOW_X            400
#define DISPLAY_KEY         0
#define ADD_AMT_INIT        0.5f
#define FORCE_SCALE         5.0f
#define ALPHA_OPTION        0.4f
#define COLOR_SCALE         20
#define RAINBOW_HOLD_NSTEPS 20

/* Colors */
// clang-format off
#define RED         {1.0f, 0.0f, 0.0f}
#define GREEN       {0.0f, 1.0f, 0.0f}
#define BLUE        {0.0f, 0.0f, 1.0f}
#define YELLOW      {0.5f, 0.5f, 0.0f}
#define CYAN        {0.0f, 0.5f, 0.5f}
#define MAGENTA     {0.5f, 0.0f, 0.5f}
#define WHITE       {0.33f, 0.33f, 0.33f}
#define ALL_COLORS  {RED, GREEN, BLUE, YELLOW, CYAN, MAGENTA, WHITE}
// clang-format on

/* Grid parameters */
#define NDIM       3
#define CELLS_Z    200
#define CELLS_Y    200
#define CELLS_X    200
#define NUM_FLUIDS 5

/* Fluid parameters */
#define DISSIPATION     0.01F
#define VISCOSITY       1e-9F
#define BUOYANCY        2e-4F
#define AMBIENT_DENSITY 0.1F

/* Simulation parameters */
#define NUM_ITER 20
#define DT       0.01F
#define CLEANUP  false

/* Linear solver
 *
 * The Poisson (pressure) and Helmholtz (diffusion) systems are relaxed with
 * red-black -- i.e. checkerboard -- Gauss-Seidel: one kernel launch per colour,
 * so the six neighbours a cell reads are never written by the same launch.
 * That colouring is also what makes SOR possible here at all.
 *
 * SOR_OMEGA is the over-relaxation factor for the pressure solve. The
 * asymptotically optimal value for this system is 2 / (1 + sin(pi / N)) = 1.969
 * at N = 200, and measurement agrees: relative distance to the converged
 * solution from a cold start on the 200^3 grid, lower is better --
 *
 *   sweeps | omega=1.00  1.90  1.95  1.97
 *   -------+------------------------------
 *        5 |      0.998  0.984 0.981 0.980
 *       20 |      0.991  0.882 0.824 0.786
 *       40 |      0.982  0.756 0.612 0.500
 *      160 |      0.932  0.319 0.120 0.049
 *
 * Plain Gauss-Seidel barely moves this system at any sweep count a real-time
 * solver can afford, and 1.97 beats it at every count measured with no
 * overshoot, so there is no low-iteration regime where omega = 1 is preferable.
 *
 * NUM_ITER is a quality/cost dial, not a convergence one: the curve above is
 * still far from converged at 160 sweeps, and closing that gap wants a
 * multigrid or preconditioned-CG solver rather than more relaxation. 20 sweeps
 * costs roughly 4x what the previous 5 did in the dominant part of the frame;
 * halve it if the frame time matters more than the residual.
 *
 * The diffusion solve keeps omega = 1: it is strongly diagonally dominant
 * (b = 1 + 6a with a ~ 4e-7), so there is nothing to over-relax.
 */
#define SOR_OMEGA 1.97F

/* Computed */
#define num_cells (CELLS_Z * CELLS_Y * CELLS_X)

/* Functions */
// inline int idx3d(const int z, const int y, const int x) {
//     return z * CELLS_Y * CELLS_X + y * CELLS_X + x;
// }

#define idx3d(z, y, x) ((z) * CELLS_Y * CELLS_X + (y) * CELLS_X + (x))

#endif
