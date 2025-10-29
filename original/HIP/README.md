# HIP Implementation for AMD GPUs

This directory contains the HIP (Heterogeneous-Compute Interface for Portability) implementation of the Fletcher wave propagation model for AMD GPUs.

## Overview

The HIP implementation is a port of the CUDA implementation designed to run on AMD GPUs using the ROCm platform. HIP provides a portable programming model that is very similar to CUDA, making it easy to port GPU code between NVIDIA and AMD platforms.

## Files

- `hip_defines.h` - Common definitions and error checking macros
- `hip_stuff.h/cpp` - GPU initialization, memory management, and cleanup
- `hip_propagate.h/cpp` - Wave propagation kernel and driver
- `hip_insertsource.h/cpp` - Source insertion kernel and driver
- `hip_driver.c` - Main driver interface
- `flags.mk` - Compiler flags and library settings
- `Makefile` - Build configuration

## Prerequisites

1. **ROCm Platform**: Install AMD ROCm (5.0 or later recommended)
   - Follow installation instructions at: https://rocm.docs.amd.com/

2. **HIP Compiler**: The `hipcc` compiler should be available in your PATH
   ```bash
   which hipcc
   ```

3. **AMD GPU**: Compatible AMD GPU (e.g., MI50, MI100, MI200 series, or Radeon RX 6000/7000 series)

## Building

### 1. Configure GPU Architecture

Edit `config.mk` in the parent directory to set your GPU architecture:

```makefile
HIP_GPU_ARCH=gfx908  # For MI100
# HIP_GPU_ARCH=gfx90a  # For MI200 series
# HIP_GPU_ARCH=gfx1030 # For RX 6000 series
```

To find your GPU architecture:
```bash
rocminfo | grep gfx
```

### 2. Set ROCm Path (if needed)

If ROCm is not installed in the default location `/opt/rocm`, set the path:

```makefile
ROCM_PATH=/your/rocm/path
```

### 3. Build with HIP backend

From the main project directory:

```bash
make backend=HIP arch=HIP
```

Or modify the Makefile to set HIP as the default backend:

```makefile
backend=HIP
arch=$(backend)
```

Then simply run:
```bash
make
```

## Key Differences from CUDA

### API Changes
- `cuda*` → `hip*` (e.g., `cudaMalloc` → `hipMalloc`)
- `cudaError_t` → `hipError_t`
- `cudaDeviceProp` → `hipDeviceProp_t`
- Kernel launches use `hipLaunchKernelGGL()` macro

### Kernel Launch Syntax
CUDA:
```cpp
kernel<<<numBlocks, threadsPerBlock>>>(args...);
```

HIP:
```cpp
hipLaunchKernelGGL(kernel, numBlocks, threadsPerBlock, 0, 0, args...);
```

### Architecture Specification
- CUDA uses compute capability (e.g., `sm_70`)
- HIP uses GCN/CDNA/RDNA architecture names (e.g., `gfx908`)

## Common AMD GPU Architectures

| GPU Series | Architecture | Example GPUs |
|------------|--------------|--------------|
| MI50 | gfx906 | AMD Instinct MI50 |
| MI100 | gfx908 | AMD Instinct MI100 |
| MI200 | gfx90a | AMD Instinct MI210, MI250 |
| MI300 | gfx942 | AMD Instinct MI300A, MI300X |
| RX 6000 | gfx1030 | Radeon RX 6800, 6900 XT |
| RX 7000 | gfx1100 | Radeon RX 7900 XTX |

## Performance Tuning

The block sizes are defined in `hip_defines.h`:
```c
#define BSIZE_X 32
#define BSIZE_Y 16
```

These values may need adjustment based on your specific GPU architecture for optimal performance.

## Troubleshooting

### HIP compiler not found
Ensure ROCm is properly installed and `hipcc` is in your PATH:
```bash
export PATH=/opt/rocm/bin:$PATH
```

### Wrong architecture
If you get runtime errors, verify your GPU architecture matches the compiled code:
```bash
rocminfo | grep "Name:" | grep "gfx"
```

### Memory errors
Ensure your grid dimensions (sx, sy) are multiples of the block sizes (BSIZE_X, BSIZE_Y).

## Verification

After building, you can verify the HIP implementation detects your GPU correctly by checking the output:
```
HIP source using device(0) <GPU_NAME> with compute capability X.Y.
GPU memory usage = XXX MiB
```

## References

- ROCm Documentation: https://rocm.docs.amd.com/
- HIP Programming Guide: https://rocm.docs.amd.com/projects/HIP/
- HIP Porting Guide: https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_porting_guide.html
