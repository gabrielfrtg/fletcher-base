CC=$(GCC)
CFLAGS=-O3 -fopenmp -I/usr/local/cuda/include
GPUCC=$(NVCC)
GPUCFLAGS=-Xptxas="-v" --maxrregcount 127 --gpu-architecture $(CUDA_GPU_SM)
LIBS=-L/usr/local/cuda/lib64 -lcudart -lstdc++ -lgomp $(GCC_LIBS)
