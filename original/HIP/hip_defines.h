#ifndef __HIP_DEFINES
#define __HIP_DEFINES

#define restrict __restrict__
#define BSIZE_X 32
#define BSIZE_Y 16
#define NPOP 4
#define TOTAL_X (BSIZE_X+2*NPOP)
#define TOTAL_Y (BSIZE_Y+2*NPOP)


#include <stdio.h>

#define HIP_CALL(call) do{      \
   const hipError_t err=call;         \
   if (err != hipSuccess)       \
   {                             \
     fprintf(stderr, "HIP ERROR: %s on %s:%d\n", hipGetErrorString(err), __FILE__, __LINE__);\
     exit(1);                    \
   }}while(0)

#endif
