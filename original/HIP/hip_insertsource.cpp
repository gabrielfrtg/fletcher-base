#include <hip/hip_runtime.h>
#include "hip_defines.h"
#include "hip_insertsource.h"

__global__ void kernel_InsertSource(const float val, const int iSource,
	                            float * restrict qp, float * restrict qc)
{
  const int ix=blockIdx.x * blockDim.x + threadIdx.x;
  if (ix==0)
  {
    qp[iSource]+=val;
    qc[iSource]+=val;
  }
}


void HIP_InsertSource(const float val, const int iSource, float *p, float *q)
{

  extern float* dev_pp;
  extern float* dev_pc;
  extern float* dev_qp;
  extern float* dev_qc;


  if ((dev_pp) && (dev_qp))
  {
     dim3 threadsPerBlock(BSIZE_X, 1);
     dim3 numBlocks(1,1);

     hipLaunchKernelGGL(kernel_InsertSource, numBlocks, threadsPerBlock, 0, 0,
                        val, iSource, dev_pc, dev_qc);
     HIP_CALL(hipGetLastError());
     HIP_CALL(hipDeviceSynchronize());
  }
}
