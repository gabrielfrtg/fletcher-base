include ../config.mk
include flags.mk

# Object files to build
OBJS = kokkos_driver.o kokkos_stuff.o kokkos_propagate.o kokkos_insertsource.o

all: $(OBJS)

kokkos_driver.o: kokkos_driver.cpp kokkos_stuff.h kokkos_propagate.h kokkos_insertsource.h
	$(GPUCC) $(GPUCFLAGS) $(KOKKOS_CXXFLAGS) $(COMMON_FLAGS) -c kokkos_driver.cpp

kokkos_stuff.o: kokkos_stuff.cpp kokkos_stuff.h kokkos_defines.h
	$(GPUCC) $(GPUCFLAGS) $(KOKKOS_CXXFLAGS) $(COMMON_FLAGS) -c kokkos_stuff.cpp

kokkos_propagate.o: kokkos_propagate.cpp kokkos_propagate.h kokkos_defines.h ../derivatives.h ../map.h
	$(GPUCC) $(GPUCFLAGS) $(KOKKOS_CXXFLAGS) $(COMMON_FLAGS) -c kokkos_propagate.cpp

kokkos_insertsource.o: kokkos_insertsource.cpp kokkos_insertsource.h kokkos_defines.h
	$(GPUCC) $(GPUCFLAGS) $(KOKKOS_CXXFLAGS) $(COMMON_FLAGS) -c kokkos_insertsource.cpp

clean:
	rm -f *.o *.a
