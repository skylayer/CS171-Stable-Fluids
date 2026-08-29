#ifndef PARAMS_H
#define PARAMS_H

#include <cassert>

// relevant systemwide parameters should go here

/* GUI parameters */
#define WINDOW_HEIGHT 600
#define WINDOW_WIDTH  600
#define WINDOW_Y      100
#define WINDOW_X      400
#define DISPLAY_KEY   0
#define ADD_AMT_INIT  0.5f
#define FORCE_SCALE   5.0f
#define ALPHA_OPTION  32.0f /* extinction per unit density per unit length */

/* Volume rendering
 *
 * Smoke is a scattering medium, not an emitting one. What reaches the eye is
 * light that came in from the light source, was attenuated on the way to the
 * scattering point, scattered towards the camera, and attenuated again on the
 * way out. The previous renderer modelled it as emission-absorption: every
 * voxel glowed with a fixed colour regardless of where the light was, so there
 * was no lit side, no shadowed side and no self-shadowing -- which is most of
 * why it read as coloured fog rather than smoke.
 *
 * LIGHT_INTENSITY is large because the Henyey-Greenstein phase function is
 * normalised over the sphere, so it carries a 1/4pi that the intensity has to
 * undo; at these angles the phase value is about 0.05.
 *
 * LIGHT_DIR is the direction the light travels. AMBIENT_LIGHT stands in for the
 * multiple scattering this single-scattering model does not compute; without it
 * the shadowed side goes pure black, which real smoke never does. HG_G is the
 * Henyey-Greenstein asymmetry -- positive means forward scattering, which is
 * what puts a bright rim on a backlit plume.
 */
#define LIGHT_DIR                                                                                                                                              \
    { -0.35f, -1.0f, -0.25f }
#define LIGHT_COLOR                                                                                                                                            \
    { 1.0f, 0.97f, 0.92f }
#define LIGHT_INTENSITY 45.0f
#define AMBIENT_LIGHT   0.08f
#define SCATTER_ALBEDO  0.9f
#define HG_G            0.3f
#define RENDER_SAMPLES  1000

/* Display transform. The render buffer holds linear radiance; a display expects
 * values already through its inverse transfer function. Handing it linear data
 * makes the shown luminance L^2.2 instead of L -- a mid grey of 0.24 arrives at
 * 0.04. Both the GL shader and the PNG writer in main.cpp apply this. */
#define EXPOSURE            1.5f
#define DISPLAY_GAMMA       2.2f
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

/* Fluid parameters
 *
 * Buoyancy is Boussinesq, following Fedkiw et al. 2001:
 *
 *     f_y = -ALPHA_SMOKE * s + BETA_TEMP * (T - T_AMBIENT)
 *
 * Two terms, not one. The smoke's own mass pulls down; the temperature excess
 * pushes up. A model with only the up term (which is what this used to have)
 * cannot produce negative buoyancy anywhere, so nothing ever overturns and no
 * Rayleigh-Taylor instability can grow -- the plume just rises as a column.
 *
 * The force is linear in both quantities. The old code raised density to the
 * 1.3 power, which has no physical basis, and multiplied by CELLS_Y, which made
 * a body force scale with resolution: the same constant meant something 3.1x
 * stronger at 200^3 than at 64^3, so it could not be tuned meaningfully.
 *
 * COOLING has to be slow enough that the plume stays buoyant while it crosses
 * the box. At 0.9 the heat was gone in about one time unit while the smoke it
 * carried decayed six times slower, so the net force turned downwards and the
 * plume sank and spread along the floor instead of rising.
 */
#define ALPHA_SMOKE 0.6F
#define BETA_TEMP   2.4F
#define T_AMBIENT   0.0F
#define COOLING     0.25F /* rate at which T relaxes back to T_AMBIENT */

#define VISCOSITY   1e-9F
#define DISSIPATION 0.15F /* smoke density decay rate, per unit time */

/* Vorticity confinement (Fedkiw et al. 2001)
 *
 *     f = VORT_EPS * h * (N x omega),   N = grad|omega| / |grad|omega||
 *
 * Feeds energy back into the small scales that the advection scheme dissipates.
 * The h factor makes it vanish under refinement, so it is a model for the
 * missing sub-grid detail rather than a permanent forcing. 0 disables it.
 *
 * It has to stay small next to the buoyancy or it simply scrambles the flow.
 * Measured on a 64^3 plume, height the smoke front reaches after 80 steps and
 * its mean rise speed:
 *
 *   VORT_EPS |    0        1        2        4
 *   ---------+--------------------------------
 *   y_tip    | 0.906    0.875    0.891    0.891
 *   rise     | 0.475    0.402    0.375    0.308
 *
 * At 18 -- roughly eps*h*|omega| ~ 5x the buoyancy -- the plume stopped rising
 * altogether and spread along the floor. 2 keeps the detail and costs about a
 * fifth of the rise speed.
 */
#define VORT_EPS 2.0F

/* Advected values are sampled with a clamped monotonic cubic rather than
 * trilinearly. Costs 64 gathers per sample instead of 8; buys a large drop in
 * numerical viscosity and in how much that viscosity depends on where the CFL
 * happens to land. Set to 0 for the cheaper trilinear path. */
#define CUBIC_ADVECTION 1

/* Source injection */
#define SOURCE_DENSITY 1.0F
#define SOURCE_TEMP    1.0F
#define SOURCE_STEPS   1000

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
