#ifndef OPENCL_CL_WRAPPER_H
#define OPENCL_CL_WRAPPER_H

#if defined(__has_include)
#  if __has_include(<CL/cl.h>)
#    include <CL/cl.h>
#  elif __has_include(<OpenCL/cl.h>)
#    include <OpenCL/cl.h>
#  else
#    include "opencl_stubs.h"
#  endif
#else
#  include <CL/cl.h>
#endif

#endif
