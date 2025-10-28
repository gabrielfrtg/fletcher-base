#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "kokkos_driver_wrapper.h"  // Use wrapper instead of direct driver.h
#include "../sample.h"
#include "kokkos_stuff.h"
#include "kokkos_propagate.h"
#include "kokkos_insertsource.h"

#define MODEL_GLOBALVARS
#define MODEL_INITIALIZE

extern "C" void DRIVER_Initialize(const int sx, const int sy, const int sz, const int bord,
                                  float dx, float dy, float dz, float dt,
                                  float * restrict vpz, float * restrict vsv, float * restrict epsilon, float * restrict delta,
                                  float * restrict phi, float * restrict theta,
                                  float * restrict pp, float * restrict pc, float * restrict qp, float * restrict qc)
{
#include "../precomp.h"

    KOKKOS_Initialize(sx, sy, sz, bord,
                     dx, dy, dz, dt,
                     ch1dxx, ch1dyy, ch1dzz,
                     ch1dxy, ch1dyz, ch1dxz,
                     v2px, v2pz, v2sz, v2pn,
                     vpz, vsv, epsilon, delta,
                     phi, theta,
                     pp, pc, qp, qc);
}

extern "C" void DRIVER_Finalize()
{
    KOKKOS_Finalize();
}

extern "C" void DRIVER_Update_pointers(const int sx, const int sy, const int sz, float *pc)
{
    KOKKOS_Update_pointers(sx, sy, sz, pc);
}

extern "C" void DRIVER_Propagate(const int sx, const int sy, const int sz, const int bord,
                                 const float dx, const float dy, const float dz, const float dt, const int it,
                                 float * pp, float * pc, float * qp, float * qc)
{
    // KOKKOS_Propagate also does TimeForward
    KOKKOS_Propagate(sx, sy, sz, bord,
                    dx, dy, dz, dt, it,
                    pp, pc, qp, qc);
}

extern "C" void DRIVER_InsertSource(float dt, int it, int iSource, float *p, float*q, float src)
{
    KOKKOS_InsertSource(src, iSource, p, q);
}
