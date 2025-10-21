#ifndef __RAJA_DEFINES
#define __RAJA_DEFINES

#define restrict __restrict__
#define BSIZE_X 32
#define BSIZE_Y 16
#define NPOP 4
#define TOTAL_X (BSIZE_X+2*NPOP)
#define TOTAL_Y (BSIZE_Y+2*NPOP)

#include <stdio.h>
#include "RAJA/RAJA.hpp"

// Check which backend RAJA was built with
#if defined(RAJA_ENABLE_CUDA)

// CUDA backend policies
constexpr int CUDA_BLOCK_SIZE = BSIZE_X * BSIZE_Y;
using EXEC_POL = RAJA::cuda_exec<CUDA_BLOCK_SIZE>;
using REDUCE_POL = RAJA::cuda_reduce;

// Nested policy for 2D kernel execution
using NESTED_POL = RAJA::KernelPolicy<
  RAJA::statement::CudaKernel<
    RAJA::statement::For<1, RAJA::cuda_block_y_loop,
      RAJA::statement::For<0, RAJA::cuda_thread_x_loop,
        RAJA::statement::Lambda<0>
      >
    >
  >
>;

#elif defined(RAJA_ENABLE_HIP)

// HIP backend policies
constexpr int HIP_BLOCK_SIZE = BSIZE_X * BSIZE_Y;
using EXEC_POL = RAJA::hip_exec<HIP_BLOCK_SIZE>;
using REDUCE_POL = RAJA::hip_reduce;

using NESTED_POL = RAJA::KernelPolicy<
  RAJA::statement::HipKernel<
    RAJA::statement::For<1, RAJA::hip_block_y_loop,
      RAJA::statement::For<0, RAJA::hip_thread_x_loop,
        RAJA::statement::Lambda<0>
      >
    >
  >
>;

#else

// OpenMP CPU backend (default)
using EXEC_POL = RAJA::omp_parallel_for_exec;
using REDUCE_POL = RAJA::omp_reduce;

using NESTED_POL = RAJA::KernelPolicy<
  RAJA::statement::For<1, RAJA::omp_parallel_for_exec,
    RAJA::statement::For<0, RAJA::seq_exec,
      RAJA::statement::Lambda<0>
    >
  >
>;

#endif

// Error checking macro
#define RAJA_CALL(call) do{      \
   call;                         \
   }while(0)

#endif
