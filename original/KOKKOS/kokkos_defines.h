#ifndef __KOKKOS_DEFINES
#define __KOKKOS_DEFINES

#define restrict __restrict__
#define BSIZE_X 32
#define BSIZE_Y 16
#define NPOP 4
#define TOTAL_X (BSIZE_X+2*NPOP)
#define TOTAL_Y (BSIZE_Y+2*NPOP)

#include <stdio.h>
#include <Kokkos_Core.hpp>

// Error checking macro for Kokkos
#define KOKKOS_CHECK(msg) do {                   \
    if (!Kokkos::is_initialized()) {             \
      fprintf(stderr, "KOKKOS ERROR: %s on %s:%d\n", msg, __FILE__, __LINE__); \
      exit(1);                                   \
    }} while(0)

#endif
