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
  RAJA_LIB=-L$(RAJA_DIR)/lib -lRAJA -lcamp
else
  $(warning RAJA_DIR not set. Assuming RAJA is in system path)
  RAJA_INCLUDE=
  RAJA_LIB=-lRAJA -lcamp
endif

# Backend selection
# Define RAJA_ENABLE_* macro to match the backend RAJA was built with
ifdef RAJA_ENABLE_CUDA
  # CUDA backend - use nvcc and link CUDA runtime
  # Use -x cu to treat .cpp files as CUDA source files
  # -arch=sm_89 for RTX 4090 (compute capability 8.9)
  # --allow-unsupported-compiler for newer GCC versions
  # -Xcompiler -fpermissive to work around GCC 12 C++ standard library issues
  BACKEND_FLAGS=-DRAJA_ENABLE_CUDA -x cu -arch=sm_89 -allow-unsupported-compiler
  BACKEND_LIBS=-L/usr/local/cuda/lib64 -lcudart -fopenmp
  CXX=$(NVCC)
  CXXFLAGS=-O3 -std=c++17 --expt-extended-lambda --expt-relaxed-constexpr -Xcompiler -fopenmp -Xcompiler -fpermissive
else ifdef RAJA_ENABLE_HIP
  # HIP backend - use hipcc and link HIP runtime
  BACKEND_FLAGS=-DRAJA_ENABLE_HIP
  BACKEND_LIBS=-L/opt/rocm/lib -lamdhip64 -fopenmp
  CXX=$(HIPCC)
  CXXFLAGS=-O3 -std=c++17 -fPIE -fopenmp -Wno-unused-result
else
  # OpenMP CPU backend (default)
  BACKEND_FLAGS=-fopenmp
  BACKEND_LIBS=-fopenmp
endif

CXXFLAGS += $(RAJA_INCLUDE) $(BACKEND_FLAGS)
LIBS=$(RAJA_LIB) $(BACKEND_LIBS) -lstdc++ $(GCC_LIBS)
