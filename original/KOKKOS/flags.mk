# Kokkos build flags
# You may need to adjust KOKKOS_PATH to point to your Kokkos installation
KOKKOS_PATH ?= /opt/kokkos/4.3.00/

# Use GCC for compiling C files
CC=$(GCC)
CFLAGS=-O3 -fPIC
# Define CXX to trigger C++ linking in base Makefile
CXX=g++

# Auto-detect Kokkos backend by checking the Kokkos configuration header
KOKKOS_HAS_CUDA := $(shell grep -q "^\#define KOKKOS_ENABLE_CUDA$$" $(KOKKOS_PATH)/include/KokkosCore_config.h 2>/dev/null && echo 1 || echo 0)
KOKKOS_HAS_HIP := $(shell grep -q "^\#define KOKKOS_ENABLE_HIP$$" $(KOKKOS_PATH)/include/KokkosCore_config.h 2>/dev/null && echo 1 || echo 0)

# Set compiler and flags based on detected backend
ifeq ($(KOKKOS_HAS_CUDA),1)
    # CUDA backend detected
    GPUCC=$(KOKKOS_PATH)/bin/nvcc_wrapper
    GPUCFLAGS=-O3 -std=c++17 -fopenmp
    GPU_LIBS=-L/usr/local/cuda/lib64 -lcudart -lcuda
    $(info Kokkos CUDA backend detected)
else ifeq ($(KOKKOS_HAS_HIP),1)
    # HIP backend detected - use hipcc or the compiler Kokkos was built with
    GPUCC=$(shell which hipcc 2>/dev/null || echo g++)
    GPUCFLAGS=-O3 -std=c++17 -fopenmp
    # ROCm libraries are usually found automatically by hipcc or in standard locations
    GPU_LIBS=-L/opt/rocm/lib -lamdhip64
    # Override CXX to use hipcc for linking when HIP is enabled
    CXX=$(GPUCC)
    $(info Kokkos HIP backend detected, using $(GPUCC))
else
    # Serial/OpenMP backend - no GPU libraries needed
    GPUCC=g++
    GPUCFLAGS=-O3 -std=c++17 -fopenmp
    GPU_LIBS=
    $(info Kokkos Serial/OpenMP backend detected)
endif

# Include Kokkos headers and libraries
KOKKOS_CXXFLAGS=-I$(KOKKOS_PATH)/include
KOKKOS_LDFLAGS=-L$(KOKKOS_PATH)/lib64 -L$(KOKKOS_PATH)/lib -lkokkoscore -lkokkoscontainers
KOKKOS_LIBS=$(KOKKOS_LDFLAGS) $(GPU_LIBS) -ldl -lpthread -lstdc++ -fopenmp

# Combined flags for the main compilation
LIBS=$(KOKKOS_LIBS) $(GCC_LIBS)
