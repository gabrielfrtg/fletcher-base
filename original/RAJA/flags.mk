# RAJA backend flags
# Assumes RAJA is installed and RAJA_DIR is set (e.g., export RAJA_DIR=/path/to/raja/install)
# Backend is auto-detected from RAJA's config.hpp based on how RAJA was built
# Set RAJA_ENABLE_CUDA=1 or RAJA_ENABLE_HIP=1 to select appropriate compiler/linker

CC=$(GCC)
CXX=$(GCC)
CFLAGS=-O3
CXXFLAGS=-O3 -std=c++17

# RAJA include and library paths
# Adjust these paths based on your RAJA installation
ifdef RAJA_DIR
  RAJA_INCLUDE=-I$(RAJA_DIR)/include
  RAJA_LIB=-L$(RAJA_DIR)/lib -lRAJA
else
  $(warning RAJA_DIR not set. Assuming RAJA is in system path)
  RAJA_INCLUDE=
  RAJA_LIB=-lRAJA
endif

# Backend selection
# RAJA auto-detects backend from config.hpp, so we don't define RAJA_ENABLE_*
# Just set the appropriate compiler and libraries
ifdef RAJA_ENABLE_CUDA
  # CUDA backend - use nvcc and link CUDA runtime
  BACKEND_FLAGS=
  BACKEND_LIBS=-L/usr/local/cuda/lib64 -lcudart
  CXX=$(NVCC)
  CXXFLAGS=-O3 -std=c++17 --expt-extended-lambda --expt-relaxed-constexpr -Xcompiler -fopenmp
else ifdef RAJA_ENABLE_HIP
  # HIP backend - use hipcc and link HIP runtime
  BACKEND_FLAGS=
  BACKEND_LIBS=-L/opt/rocm/lib -lamdhip64
  CXX=$(HIPCC)
  CXXFLAGS=-O3 -std=c++17
else
  # OpenMP CPU backend (default)
  BACKEND_FLAGS=-fopenmp
  BACKEND_LIBS=-fopenmp
endif

CXXFLAGS += $(RAJA_INCLUDE) $(BACKEND_FLAGS)
LIBS=$(RAJA_LIB) $(BACKEND_LIBS) -lstdc++ $(GCC_LIBS)
