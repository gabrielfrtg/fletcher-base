# Kokkos build flags
# You may need to adjust KOKKOS_PATH to point to your Kokkos installation
KOKKOS_PATH ?= /opt/kokkos/4.3.00/

# Use GCC for compiling C files
CC=$(GCC)
CFLAGS=-O3
# Define CXX to trigger C++ linking in base Makefile
CXX=g++
GPUCC=$(KOKKOS_PATH)/bin/nvcc_wrapper
GPUCFLAGS=-O3 -std=c++17 -fopenmp

# Include Kokkos headers and libraries
KOKKOS_CXXFLAGS=-I$(KOKKOS_PATH)/include
KOKKOS_LDFLAGS=-L$(KOKKOS_PATH)/lib64 -L$(KOKKOS_PATH)/lib -lkokkoscore -lkokkoscontainers
# Add CUDA runtime and driver libraries since Kokkos was built with CUDA support
CUDA_LIBS=-L/usr/local/cuda/lib64 -lcudart -lcuda
KOKKOS_LIBS=$(KOKKOS_LDFLAGS) $(CUDA_LIBS) -ldl -lpthread -lstdc++ -fopenmp

# Combined flags for the main compilation
LIBS=$(KOKKOS_LIBS) $(GCC_LIBS)
