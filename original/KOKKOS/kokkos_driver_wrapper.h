#ifndef __KOKKOS_DRIVER_WRAPPER_H__
#define __KOKKOS_DRIVER_WRAPPER_H__

// Wrapper to include driver.h with C++ compatibility
// This avoids modifying the base driver.h file

#ifdef __cplusplus
extern "C" {
// Define restrict for C++ before including driver.h
#ifndef restrict
#define restrict __restrict__
#endif
#endif

// Now include the actual driver header
#include "../driver.h"

#ifdef __cplusplus
}
#endif

#endif
