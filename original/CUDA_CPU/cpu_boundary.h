#ifndef __CPU_BOUNDARY
#define __CPU_BOUNDARY

#ifdef __cplusplus
extern "C" {
#endif

// CPU function to process boundary Z-layers
void CPU_ProcessBoundaries(const int sx, const int sy, const int sz, const int bord,
                           const float dx, const float dy, const float dz, const float dt,
                           float * restrict ch1dxx, float * restrict ch1dyy, float * restrict ch1dzz,
                           float * restrict ch1dxy, float * restrict ch1dyz, float * restrict ch1dxz,
                           float * restrict v2px, float * restrict v2pz, float * restrict v2sz,
                           float * restrict v2pn, float * restrict pp, float * restrict pc,
                           float * restrict qp, float * restrict qc);

// Get the number of boundary layers processed on CPU
int CPU_GetBoundaryLayers();

#ifdef __cplusplus
}
#endif

#endif
