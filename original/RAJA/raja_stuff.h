#ifndef _RAJA_STUFF
#define _RAJA_STUFF

#ifdef __cplusplus
extern "C" {
#endif

void RAJA_Initialize(const int sx, const int sy, const int sz, const int bord,
               float dx, float dy, float dz, float dt,
               float * restrict ch1dxx, float * restrict ch1dyy, float * restrict ch1dzz,
               float * restrict ch1dxy, float * restrict ch1dyz, float * restrict ch1dxz,
               float * restrict v2px, float * restrict v2pz, float * restrict v2sz, float * restrict v2pn,
               float * restrict vpz, float * restrict vsv, float * restrict epsilon, float * restrict delta,
               float * restrict phi, float * restrict theta,
               float * restrict pp, float * restrict pc, float * restrict qp, float * restrict qc);

void RAJA_Finalize();

void RAJA_Update_pointers(const int sx, const int sy, const int sz, float *pc);

#ifdef __cplusplus
}
#endif
#endif

