# Flags for OpenMP Target Offload
#
# TROUBLESHOOTING: If you get "could not find accel/nvptx-none/mkoffload" error,
# see TROUBLESHOOTING.md for solutions.
#
# Quick fix options:
# 1. Install: sudo apt-get install gcc-12-offload-nvptx (if you have sudo)
# 2. Use NVIDIA HPC SDK: Uncomment the "NVIDIA HPC SDK" section below
# 3. CPU fallback: Uncomment the "CPU Fallback" section below
# 4. Use standard OpenMP: cd .. && make backend=OpenMP

# ==============================================================================
# OPTION 1: GCC with NVIDIA GPU offload (DEFAULT)
# ==============================================================================
# Requires: gcc-XX-offload-nvptx package installed
# Common architectures: sm_70 (V100), sm_80 (A100), sm_86 (RTX 3090), sm_89 (RTX 4090), sm_90 (H100)
# Use this with openmp-target-gcc.sif container

# CC=$(GCC)
# CFLAGS=-O3 -std=gnu11 -fopenmp -foffload=nvptx-none -foffload="-O3 -lm"
# LIBS=$(GCC_LIBS)

# ==============================================================================
# OPTION 2: NVIDIA HPC SDK (nvc/nvc++) - RECOMMENDED IF AVAILABLE
# ==============================================================================
# Uncomment these lines and comment out OPTION 1 above
# Adjust -gpu=ccXX to match your GPU: cc70 (V100), cc80 (A100), cc86 (RTX 30xx), cc89 (RTX 40xx), cc90 (H100)

# CC=nvc
# CFLAGS=-O3 -mp=gpu -gpu=cc80 -Minfo=accel
# LIBS=$(GCC_LIBS)

# ==============================================================================
# OPTION 3: Clang with NVIDIA GPU
# ==============================================================================
# Uncomment these lines and comment out OPTION 1 above
# Adjust -march=sm_XX to match your GPU

# CC=clang
# CFLAGS=-O3 -std=gnu11 -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_80
# LIBS=$(GCC_LIBS)

# ==============================================================================
# OPTION 4: AMD AOMP for AMD GPUs
# ==============================================================================
# Uncomment these lines and comment out OPTION 1 above
# Adjust -march=gfxXXX: gfx906 (MI50), gfx908 (MI100), gfx90a (MI200), gfx940 (MI300)

CC=/opt/rocm/llvm/bin/clang
CFLAGS=-O3 -std=gnu11 -fopenmp -fopenmp-version=51 -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx1100
LIBS=$(GCC_LIBS)

# ==============================================================================
# OPTION 5: CPU Fallback (NO GPU, for testing only)
# ==============================================================================
# Uncomment these lines and comment out OPTION 1 above
# This will compile but run on CPU only (useful for testing compilation)

# CC=$(GCC)
# CFLAGS=-O3 -fopenmp
# LIBS=$(GCC_LIBS)
