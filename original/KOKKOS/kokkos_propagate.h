#ifndef __KOKKOS_PROPAGATE
#define __KOKKOS_PROPAGATE

#ifdef __cplusplus
extern "C" {
#ifndef restrict
#define restrict __restrict__
#endif
#endif

void KOKKOS_Propagate(const int sx, const int sy, const int sz, const int bord,
                      const float dx, const float dy, const float dz, const float dt, const int it,
                      float * restrict pp, float * restrict pc, float * restrict qp, float * restrict qc);

#ifdef __cplusplus
}
#endif

#endif
