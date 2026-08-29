//
// Created by condo on 2024/1/8.
//

#ifndef RENDER_CUH
#define RENDER_CUH

#include "params.h"

/* Single-scattering volume renderer. `field` is NUM_FLUIDS density grids;
 * `active_fluids` is a bit per grid that has ever been seeded. The output is
 * linear radiance -- the display transform belongs at display time, not here. */
void render_density(const float view[3][3], const float pos[3], float focal, float **field, unsigned active_fluids, float *output);

/* Frees the light-transmittance volume the renderer keeps between frames. */
void release_render_scratch(void);

#endif // RENDER_CUH
