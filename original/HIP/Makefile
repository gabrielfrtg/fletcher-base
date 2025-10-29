include ../config.mk
include flags.mk

all:
	$(CC) $(CFLAGS) $(COMMON_FLAGS) -c hip_driver.c
	$(GPUCC) $(GPUCFLAGS) $(COMMON_FLAGS) -c hip_stuff.cpp
	$(GPUCC) $(GPUCFLAGS) $(COMMON_FLAGS) -c hip_propagate.cpp
	$(GPUCC) $(GPUCFLAGS) $(COMMON_FLAGS) -c hip_insertsource.cpp

clean:
	rm -f *.o *.a

