#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../driver.h"
#include "../sample.h"
#include "opencl_runtime.h"

#define MODEL_GLOBALVARS
#define MODEL_INITIALIZE

void DRIVER_Initialize(const int sx, const int sy, const int sz, const int bord,
                       float dx, float dy, float dz, float dt,
                       float * restrict vpz, float * restrict vsv, float * restrict epsilon, float * restrict delta,
                       float * restrict phi, float * restrict theta,
                       float * restrict pp, float * restrict pc, float * restrict qp, float * restrict qc)
{
#include "../precomp.h"

    OPENCL_Initialize(sx, sy, sz, bord,
                      dx, dy, dz, dt,
                      ch1dxx, ch1dyy, ch1dzz,
                      ch1dxy, ch1dyz, ch1dxz,
                      v2px, v2pz, v2sz, v2pn,
                      vpz, vsv, epsilon, delta,
                      phi, theta,
                      pp, pc, qp, qc);
}

void DRIVER_Finalize()
{
    OPENCL_Finalize();
}

void DRIVER_Update_pointers(const int sx, const int sy, const int sz, float *pc)
{
    OPENCL_Update_pointers(sx, sy, sz, pc);
}

void DRIVER_Propagate(const int sx, const int sy, const int sz, const int bord,
                      const float dx, const float dy, const float dz, const float dt, const int it,
                      float * restrict pp, float * restrict pc, float * restrict qp, float * restrict qc)
{
    OPENCL_Propagate(sx, sy, sz, bord,
                     dx, dy, dz, dt, it,
                     pp, pc, qp, qc);
}

void DRIVER_InsertSource(float dt, int it, int iSource, float *p, float *q, float src)
{
    OPENCL_InsertSource(src, iSource, p, q);
}
