#ifndef _OMP_TARGET_PROPAGATE
#define _OMP_TARGET_PROPAGATE

#ifdef __cplusplus
extern "C" {
#endif

void OMP_TARGET_Propagate(const int sx, const int sy, const int sz, const int bord,
		    const float dx, const float dy, const float dz, const float dt, const int it,
		    float * restrict pp, float * restrict pc, float * restrict qp, float * restrict qc);

void OMP_TARGET_SwapArrays(float **pp, float **pc, float **qp, float **qc);

#ifdef __cplusplus
}
#endif
#endif
