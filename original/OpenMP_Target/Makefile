include ../config.mk
include flags.mk

all:
	$(CC) $(CFLAGS) $(COMMON_FLAGS) -c openmp_target_driver.c
	$(CC) $(CFLAGS) $(COMMON_FLAGS) -c openmp_target_stuff.c
	$(CC) $(CFLAGS) $(COMMON_FLAGS) -c openmp_target_propagate.c
	$(CC) $(CFLAGS) $(COMMON_FLAGS) -c openmp_target_insertsource.c

clean:
	rm -f *.o *.a
