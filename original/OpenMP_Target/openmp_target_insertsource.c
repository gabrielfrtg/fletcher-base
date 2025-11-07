#include <omp.h>
#include "openmp_target_insertsource.h"

void OMP_TARGET_InsertSource(const float val, const int iSource, float *p, float *q)
{

  extern float* dev_pp;
  extern float* dev_pc;
  extern float* dev_qp;
  extern float* dev_qc;

  if ((dev_pp) && (dev_qp))
  {
     // Insert source using OpenMP target directive - arrays already mapped
     #pragma omp target
     {
        dev_pc[iSource] += val;
        dev_qc[iSource] += val;
     }
  }
}
