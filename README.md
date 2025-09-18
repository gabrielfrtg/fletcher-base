# Fletcher Base: A Foundational Implementation

This repository contains a clean and minimal base implementation of the Fletcher project, designed to be a stable starting point for developing new, specialized versions.

## About This Project

Fletcher Base was created by carefully refactoring the original [`fletcher-io` repository](https://github.com/gabrielfrtg/fletcher-io). It contains only the core algorithm and essential scripts, with all specific hardware optimizations (like advanced I/O or parallelism) removed.

The primary goal is to provide a clean slate for developers and researchers who wish to build and test their own Fletcher implementations targeting different architectures like OpenCL, CUDA, or FPGAs.

## Available backends

The reference implementation located in `original/` can be compiled for different accelerator backends by selecting the `backend` make variable:

```bash
# Build the CUDA version (default)
make -C original backend=CUDA

# Build the OpenCL GPU version
make -C original backend=OpenCL

# Build the CPU/OpenMP variants
make -C original backend=OpenMP
```

When using the OpenCL backend remember to source `env.sh` or adjust `LD_LIBRARY_PATH` so the OpenCL runtime can be discovered at execution time.
