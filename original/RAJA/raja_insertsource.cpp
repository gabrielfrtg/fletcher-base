#include "raja_defines.h"
#include "raja_insertsource.h"

extern "C" {

void RAJA_InsertSource(const float val, const int iSource, float *p, float *q)
{

  extern float* dev_pp;
  extern float* dev_pc;
  extern float* dev_qp;
  extern float* dev_qc;

  if ((dev_pp) && (dev_qp))
  {
    // Simple single-element update using RAJA
    float* pc_ptr = dev_pc;
    float* qc_ptr = dev_qc;

    RAJA::forall<EXEC_POL>(RAJA::RangeSegment(0, 1),
      [=] RAJA_HOST_DEVICE (int idx) {
        pc_ptr[iSource] += val;
        qc_ptr[iSource] += val;
      }
    );

#if defined(RAJA_ENABLE_CUDA)
    cudaDeviceSynchronize();
#elif defined(RAJA_ENABLE_HIP)
    hipDeviceSynchronize();
#endif
  }
}

} // extern "C"
