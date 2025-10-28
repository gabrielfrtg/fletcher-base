#ifndef __KOKKOS_INSERTSOURCE
#define __KOKKOS_INSERTSOURCE

#ifdef __cplusplus
extern "C" {
#ifndef restrict
#define restrict __restrict__
#endif
#endif

void KOKKOS_InsertSource(const float val, const int iSource, float *p, float *q);

#ifdef __cplusplus
}
#endif

#endif
