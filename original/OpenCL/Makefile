include ../config.mk
include flags.mk

SRCS=\
        opencl_driver.c \
        opencl_runtime.c

OBJS=$(SRCS:.c=.o)

all: $(OBJS)

%.o: %.c
	$(CC) $(CFLAGS) $(COMMON_FLAGS) -c $< -o $@

clean:
	rm -f *.o *.a
