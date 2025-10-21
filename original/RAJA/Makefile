include ../config.mk
include flags.mk

all:
	$(CC) $(CFLAGS) $(COMMON_FLAGS) -c raja_driver.c
	$(CXX) $(CXXFLAGS) $(COMMON_FLAGS) -c raja_stuff.cpp
	$(CXX) $(CXXFLAGS) $(COMMON_FLAGS) -c raja_propagate.cpp
	$(CXX) $(CXXFLAGS) $(COMMON_FLAGS) -c raja_insertsource.cpp

clean:
	rm -f *.o *.a

