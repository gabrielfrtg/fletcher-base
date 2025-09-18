#include <sys/time.h>
#include <stddef.h>

double wtime()
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + (1.0e-6*tv.tv_usec);
}
