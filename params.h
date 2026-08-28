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
 * On the linear system alone, over-relaxation is a large win. Relative distance
 * to the converged solution, 200^3, cold start:
 *
 *   sweeps | omega=1.00  1.90  1.95  1.97
 *   -------+------------------------------
 *        5 |      0.998  0.984 0.981 0.980
 *       20 |      0.991  0.882 0.824 0.786
 *      160 |      0.932  0.319 0.120 0.049
 *
 * Inside the closed simulation loop it is not. Measured on a 64^3 run of the
 * real pipeline, divergence of the projected field relative to its velocity
 * scale after 64 steps:
 *
 *   omega   |  1.00     1.50     1.80     1.90     1.95     1.97
 *   --------+-------------------------------------------------------
 *   rel_div | 0.01223  0.01208  0.01208  0.01216   blows up  blows up
 *
 * Everything from 1.0 to 1.9 is the same flow to within 1.4%, because each
 * frame's solve only has to correct a field the previous frame already left
 * nearly divergence free -- convergence rate is not what limits the result.
 * Past ~1.93 the solver's transient outgrows what one frame can absorb and the
 * run diverges. 1.8 keeps the faster linear solve for anyone who raises
 * NUM_ITER or changes the grid, with margin to the cliff; 1.0 is the setting
 * with no cliff at all if that margin is not worth carrying.
 *
 * NUM_ITER is a quality/cost dial. Same run, omega 1.8:
 *
 *   NUM_ITER |    5        10       20       40
 *   ---------+-----------------------------------
 *   rel_div  | 0.01553  0.01253  0.01208  0.01206
 *
 * 20 is the knee; 40 buys 0.2% for twice the sweeps, and 10 gets 96% of the
 * benefit if frame time matters more. Closing the rest of the gap wants a
 * multigrid or preconditioned-CG solver, not more relaxation.
 *
 * The diffusion solve keeps omega = 1: it is strongly diagonally dominant
 * (b = 1 + 6a with a ~ 4e-7), so there is nothing to over-relax.
 */
#define SOR_OMEGA 1.8F

/* Computed */
#define num_cells (CELLS_Z * CELLS_Y * CELLS_X)

/* Functions */
// inline int idx3d(const int z, const int y, const int x) {
//     return z * CELLS_Y * CELLS_X + y * CELLS_X + x;
// }

#define idx3d(z, y, x) ((z) * CELLS_Y * CELLS_X + (y) * CELLS_X + (x))

#endif
