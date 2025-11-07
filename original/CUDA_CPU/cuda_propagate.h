#ifndef __CUDA_PROPAGATE
#define __CUDA_PROPAGATE

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif


// Propagate: using Fletcher's equations, propagate waves one dt,
//            either forward or backward in time
void CUDA_Propagate(const int sx, const int sy, const int sz, const int bord,
	       const float dx, const float dy, const float dz, const float dt, const int it,
	       float * restrict pp, float * restrict pc, float * restrict qp, float * restrict qc,
	       cudaStream_t stream, const int z_start, const int z_end);

void CUDA_SwapArrays(float **pp, float **pc, float **qp, float **qc);

#ifdef __cplusplus
}
#endif

#endif
