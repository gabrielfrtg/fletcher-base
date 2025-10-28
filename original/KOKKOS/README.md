# Kokkos Implementation of Fletcher RTM

This directory contains a Kokkos implementation of the Fletcher Reverse Time Migration (RTM) application, ported from the CUDA implementation.

## Overview

The Kokkos version provides a performance-portable implementation that can run on:
- CPUs (OpenMP, Pthreads, Serial)
- NVIDIA GPUs (CUDA)
- AMD GPUs (HIP/ROCm)
- Intel GPUs (SYCL)

## Files

- `kokkos_defines.h` - Common definitions and macros for Kokkos
- `kokkos_stuff.h/cpp` - Memory management, initialization, and finalization
- `kokkos_propagate.h/cpp` - Wave propagation kernel
- `kokkos_insertsource.h/cpp` - Source insertion kernel
- `kokkos_driver.cpp` - Driver interface matching the original API
- `flags.mk` - Compilation flags
- `Makefile` - Build configuration

## Prerequisites

1. **Kokkos 4.3.0 or later** must be installed on your system
2. A C++17 compatible compiler
3. For GPU execution: CUDA toolkit (NVIDIA) or ROCm (AMD)

## Building

### Step 1: Set Kokkos Path

Edit `flags.mk` and set `KOKKOS_PATH` to point to your Kokkos installation:

```bash
KOKKOS_PATH ?= /path/to/your/kokkos/install
```

### Step 2: Build the Kokkos Backend

From the `original` directory, run:

```bash
# Set the backend to KOKKOS
make backend=KOKKOS clean all
```

Or modify the Makefile to set:
```makefile
backend=KOKKOS
```

### Step 3: Run the Application

```bash
cd run
../ModelagemFletcher.exe
```

## Configuration

The Kokkos execution space is determined at compile time based on how Kokkos was built:

- **CUDA backend**: If Kokkos was built with CUDA support
- **OpenMP backend**: If Kokkos was built with OpenMP support
- **Serial backend**: Default fallback

## Key Differences from CUDA Implementation

1. **Execution Model**: Uses Kokkos parallel patterns instead of explicit kernel launches
2. **Memory Management**: Uses Kokkos Views instead of raw CUDA device pointers
3. **Portability**: Same source code can run on CPUs and GPUs
4. **Initialization**: Kokkos runtime is automatically initialized/finalized

## Performance Tuning

You can adjust the following parameters in `kokkos_defines.h`:

- `BSIZE_X`: Thread block size in X dimension (default: 32)
- `BSIZE_Y`: Thread block size in Y dimension (default: 16)

## Troubleshooting

### Compilation Errors

1. **Kokkos headers not found**: Check that `KOKKOS_PATH` is correctly set in `flags.mk`
2. **Linking errors**: Ensure Kokkos libraries are in your library path
3. **C++ version errors**: Kokkos 4.x requires C++17 or later

### Runtime Issues

1. **Initialization errors**: Check that your Kokkos installation matches the backend you're trying to use
2. **Memory errors**: Ensure sufficient GPU memory is available
3. **Performance issues**: Try adjusting block sizes in `kokkos_defines.h`

## Notes

- The implementation maintains the same algorithm and numerical methods as the CUDA version
- The extra plane offset used for boundary conditions is preserved
- Array swapping is done on the host side for consistency with the original implementation

## Testing

To verify correctness, compare results with the CUDA implementation:

```bash
# Run CUDA version
make backend=CUDA clean all
cd run && ../ModelagemFletcher.exe && cd ..

# Run Kokkos version
make backend=KOKKOS clean all
cd run && ../ModelagemFletcher.exe && cd ..

# Compare outputs
./compare.exe
```
