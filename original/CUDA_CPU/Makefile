include ../config.mk
include flags.mk

all:
	$(CC) $(CFLAGS) $(COMMON_FLAGS) -c cuda_driver.c
	$(CC) $(CFLAGS) $(COMMON_FLAGS) -c cpu_boundary.c
	$(GPUCC) $(GPUCFLAGS) $(COMMON_FLAGS) -c cuda_stuff.cu
	$(GPUCC) $(GPUCFLAGS) $(COMMON_FLAGS) -c cuda_propagate.cu
	$(GPUCC) $(GPUCFLAGS) $(COMMON_FLAGS) -c cuda_insertsource.cu

clean:
	rm -f *.o *.a

