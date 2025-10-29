CC=$(GCC)
CFLAGS=-O3
GPUCC=$(HIPCC)
GPUCFLAGS=-O3 -fPIC --offload-arch=$(HIP_GPU_ARCH)
LIBS=-L$(ROCM_PATH)/lib -lamdhip64 -lstdc++ $(GCC_LIBS)
