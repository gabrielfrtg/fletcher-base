#ifndef __CUDA_SOURCE
#define __CUDA_SOURCE

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

void CUDA_InsertSource(const float val, const int iSource, float *p, float *q, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif
