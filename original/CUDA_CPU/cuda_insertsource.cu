#include "cuda_defines.h"
#include "cuda_insertsource.h"

__global__ void kernel_InsertSource(const float val, const int iSource,
	                            float * restrict qp, float * restrict qc)
{
  // Use only a single thread for this simple operation
  if (threadIdx.x == 0 && blockIdx.x == 0)
  {
    qp[iSource]+=val;
    qc[iSource]+=val;
  }
}


void CUDA_InsertSource(const float val, const int iSource, float *p, float *q, cudaStream_t stream)
{

  extern float* dev_pp;
  extern float* dev_pc;
  extern float* dev_qp;
  extern float* dev_qc;


  if ((dev_pp) && (dev_qp))
  {
     // Launch with just 1 thread in 1 block - much more efficient than 32 threads
     dim3 threadsPerBlock(1, 1);
     dim3 numBlocks(1, 1);

     kernel_InsertSource<<<numBlocks, threadsPerBlock, 0, stream>>> (val, iSource, dev_pc, dev_qc);
     CUDA_CALL(cudaGetLastError());
     // Note: No sync here - let stream handle it
  }
}
