# Fletcher RTM - RAJA Backend

This directory contains the RAJA-based implementation of the Fletcher Reverse Time Migration (RTM) simulator. RAJA is a performance portability layer that enables the same source code to run on multiple architectures (CUDA, HIP, OpenMP) with minimal changes.

## Overview

The RAJA backend provides:
- **Portability**: Single source code for multiple GPU backends (CUDA, HIP) and CPU (OpenMP)
- **Performance**: Direct mapping to GPU execution similar to native CUDA
- **Flexibility**: Easy switching between backends via compile-time flags

## Prerequisites

1. **RAJA Library**: Install RAJA from https://github.com/LLNL/RAJA
   ```bash
   git clone --recursive https://github.com/LLNL/RAJA.git
   cd RAJA
   mkdir build && cd build
   cmake -DCMAKE_INSTALL_PREFIX=/path/to/install \
         -DENABLE_CUDA=On \  # or ENABLE_HIP=On for AMD GPUs
         -DCUDA_ARCH=sm_70 \  # adjust for your GPU
         ..
   make -j install
   ```

2. **Set RAJA_DIR environment variable**:
   ```bash
   export RAJA_DIR=/path/to/raja/install
   ```

3. **Backend-specific requirements**:
   - **CUDA**: NVIDIA GPU with CUDA Toolkit (nvcc)
   - **HIP**: AMD GPU with ROCm (hipcc)
   - **OpenMP**: C++ compiler with OpenMP support (gcc, clang)

## Building

### Using CUDA Backend (NVIDIA GPUs)

```bash
cd /home/gfreytag/github/fletcher-base/original
export RAJA_ENABLE_CUDA=1
export CUDA_GPU_SM=sm_70  # Adjust for your GPU: sm_60, sm_70, sm_80, etc.
make backend=RAJA
```

### Using HIP Backend (AMD GPUs)

```bash
cd /home/gfreytag/github/fletcher-base/original
export RAJA_ENABLE_HIP=1
make backend=RAJA
```

### Using OpenMP Backend (CPU)

```bash
cd /home/gfreytag/github/fletcher-base/original
make backend=RAJA
```

## File Structure

- `raja_defines.h`: RAJA policy definitions and backend selection
- `raja_stuff.{h,cpp}`: GPU/CPU memory management and initialization
- `raja_propagate.{h,cpp}`: Wave propagation kernel (main computation)
- `raja_insertsource.{h,cpp}`: Source insertion kernel
- `raja_driver.c`: Driver interface matching other backends
- `Makefile`: Build rules
- `flags.mk`: Compiler flags and backend selection

## Key Differences from CUDA Version

1. **Execution Policies**: Uses RAJA policies instead of CUDA kernel launch syntax
   - CUDA: `kernel<<<blocks, threads>>>(args)`
   - RAJA: `RAJA::kernel<NESTED_POL>(ranges, lambda)`

2. **Memory Management**: Abstracted through macros
   - Supports CUDA, HIP, or simple malloc/memcpy for CPU

3. **Device Functions**: Lambda functions with `RAJA_HOST_DEVICE` qualifier

4. **Portability**: Same source compiles for CUDA, HIP, or OpenMP

## Performance Tuning

The block sizes are defined in `raja_defines.h`:
- `BSIZE_X = 32`: Thread block size in X dimension
- `BSIZE_Y = 16`: Thread block size in Y dimension

Adjust these for your specific GPU architecture for optimal performance.

## Troubleshooting

### RAJA not found
- Ensure `RAJA_DIR` is set correctly
- Check that RAJA was compiled with the same backend you're trying to use

### Compilation errors with lambda functions
- Ensure you're using C++14 or later
- For CUDA: Add `--expt-extended-lambda` flag (already in flags.mk)

### Runtime errors
- Verify grid dimensions are multiples of block sizes
- Check that GPU memory allocation succeeded

## Testing

Run the same test cases as the CUDA version:
```bash
./ModelagemFletcher.exe
```

The output should match the CUDA version within numerical precision.

## Performance Comparison

The RAJA version should have similar performance to native CUDA:
- ~5-10% overhead from abstraction layer (negligible)
- Benefits from performance portability across vendors
- Easy to optimize for new architectures

## References

- RAJA Documentation: https://raja.readthedocs.io/
- RAJA GitHub: https://github.com/LLNL/RAJA
- Fletcher RTM: Original CUDA implementation in ../CUDA/
