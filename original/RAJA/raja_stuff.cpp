#include "raja_defines.h"
#include "raja_stuff.h"

#if defined(RAJA_ENABLE_CUDA)
#include <cuda_runtime.h>
#define GPU_MALLOC(ptr, size) cudaMalloc(&ptr, size)
#define GPU_FREE(ptr) cudaFree(ptr)
#define GPU_MEMCPY_H2D(dst, src, size) cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice)
#define GPU_MEMCPY_D2H(dst, src, size) cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost)
#define GPU_MEMSET(ptr, val, size) cudaMemset(ptr, val, size)
#define GPU_SYNCHRONIZE() cudaDeviceSynchronize()
#define GPU_GET_DEVICE_COUNT(count) cudaGetDeviceCount(&count)
#define GPU_SET_DEVICE(device) cudaSetDevice(device)
#define GPU_GET_DEVICE_PROPERTIES(prop, device) cudaGetDeviceProperties(&prop, device)
typedef cudaDeviceProp GPU_DeviceProp;
#elif defined(RAJA_ENABLE_HIP)
#include <hip/hip_runtime.h>
#define GPU_MALLOC(ptr, size) hipMalloc(&ptr, size)
#define GPU_FREE(ptr) hipFree(ptr)
#define GPU_MEMCPY_H2D(dst, src, size) hipMemcpy(dst, src, size, hipMemcpyHostToDevice)
#define GPU_MEMCPY_D2H(dst, src, size) hipMemcpy(dst, src, size, hipMemcpyDeviceToHost)
#define GPU_MEMSET(ptr, val, size) hipMemset(ptr, val, size)
#define GPU_SYNCHRONIZE() hipDeviceSynchronize()
#define GPU_GET_DEVICE_COUNT(count) hipGetDeviceCount(&count)
#define GPU_SET_DEVICE(device) hipSetDevice(device)
#define GPU_GET_DEVICE_PROPERTIES(prop, device) hipGetDeviceProperties(&prop, device)
typedef hipDeviceProp_t GPU_DeviceProp;
#else
// CPU fallback - no GPU operations needed
#define GPU_MALLOC(ptr, size) (ptr = (float*)malloc(size), (ptr != NULL) ? 0 : -1)
#define GPU_FREE(ptr) free(ptr)
#define GPU_MEMCPY_H2D(dst, src, size) memcpy(dst, src, size)
#define GPU_MEMCPY_D2H(dst, src, size) memcpy(dst, src, size)
#define GPU_MEMSET(ptr, val, size) memset(ptr, val, size)
#define GPU_SYNCHRONIZE()
#endif

static size_t sxsy=0;

extern "C" {

void RAJA_Initialize(const int sx, const int sy, const int sz, const int bord,
	       float dx, float dy, float dz, float dt,
	       float * restrict ch1dxx, float * restrict ch1dyy, float * restrict ch1dzz,
	       float * restrict ch1dxy, float * restrict ch1dyz, float * restrict ch1dxz,
	       float * restrict v2px, float * restrict v2pz, float * restrict v2sz, float * restrict v2pn,
	       float * restrict vpz, float * restrict vsv, float * restrict epsilon, float * restrict delta,
	       float * restrict phi, float * restrict theta,
	       float * restrict pp, float * restrict pc, float * restrict qp, float * restrict qc)
{

   extern float* dev_ch1dxx;
   extern float* dev_ch1dyy;
   extern float* dev_ch1dzz;
   extern float* dev_ch1dxy;
   extern float* dev_ch1dyz;
   extern float* dev_ch1dxz;
   extern float* dev_v2px;
   extern float* dev_v2pz;
   extern float* dev_v2sz;
   extern float* dev_v2pn;
   extern float* dev_pp;
   extern float* dev_pc;
   extern float* dev_qp;
   extern float* dev_qc;

#if defined(RAJA_ENABLE_CUDA) || defined(RAJA_ENABLE_HIP)
  int deviceCount;
  GPU_GET_DEVICE_COUNT(deviceCount);
  const int device=0;
  GPU_DeviceProp deviceProp;
  GPU_GET_DEVICE_PROPERTIES(deviceProp, device);
  printf("RAJA source using device(%d) %s with compute capability %d.%d.\n", device, deviceProp.name, deviceProp.major, deviceProp.minor);
  GPU_SET_DEVICE(device);
#else
  printf("RAJA source using CPU (OpenMP) backend.\n");
#endif

  // Check sx,sy values
  if (sx%BSIZE_X != 0)
  {
     printf("sx(%d) must be multiple of BSIZE_X(%d)\n", sx, (int)BSIZE_X);
     exit(1);
  }
  if (sy%BSIZE_Y != 0)
  {
     printf("sy(%d) must be multiple of BSIZE_Y(%d)\n", sy, (int)BSIZE_Y);
     exit(1);
  }

   sxsy=sx*sy; // one plan
   const size_t sxsysz=sxsy*sz;
   const size_t msize_vol=sxsysz*sizeof(float);
   const size_t msize_vol_extra=msize_vol+2*sxsy*sizeof(float); // 2 extra plans for wave fields

   GPU_MALLOC(dev_ch1dxx, msize_vol);
   GPU_MEMCPY_H2D(dev_ch1dxx, ch1dxx, msize_vol);
   GPU_MALLOC(dev_ch1dyy, msize_vol);
   GPU_MEMCPY_H2D(dev_ch1dyy, ch1dyy, msize_vol);
   GPU_MALLOC(dev_ch1dzz, msize_vol);
   GPU_MEMCPY_H2D(dev_ch1dzz, ch1dzz, msize_vol);
   GPU_MALLOC(dev_ch1dxy, msize_vol);
   GPU_MEMCPY_H2D(dev_ch1dxy, ch1dxy, msize_vol);
   GPU_MALLOC(dev_ch1dyz, msize_vol);
   GPU_MEMCPY_H2D(dev_ch1dyz, ch1dyz, msize_vol);
   GPU_MALLOC(dev_ch1dxz, msize_vol);
   GPU_MEMCPY_H2D(dev_ch1dxz, ch1dxz, msize_vol);
   GPU_MALLOC(dev_v2px, msize_vol);
   GPU_MEMCPY_H2D(dev_v2px, v2px, msize_vol);
   GPU_MALLOC(dev_v2pz, msize_vol);
   GPU_MEMCPY_H2D(dev_v2pz, v2pz, msize_vol);
   GPU_MALLOC(dev_v2sz, msize_vol);
   GPU_MEMCPY_H2D(dev_v2sz, v2sz, msize_vol);
   GPU_MALLOC(dev_v2pn, msize_vol);
   GPU_MEMCPY_H2D(dev_v2pn, v2pn, msize_vol);

   // Wave field arrays with an extra plan
   GPU_MALLOC(dev_pp, msize_vol_extra);
   GPU_MEMSET(dev_pp, 0, msize_vol_extra);
   GPU_MALLOC(dev_pc, msize_vol_extra);
   GPU_MEMSET(dev_pc, 0, msize_vol_extra);
   GPU_MALLOC(dev_qp, msize_vol_extra);
   GPU_MEMSET(dev_qp, 0, msize_vol_extra);
   GPU_MALLOC(dev_qc, msize_vol_extra);
   GPU_MEMSET(dev_qc, 0, msize_vol_extra);
   dev_pp+=sxsy;
   dev_pc+=sxsy;
   dev_qp+=sxsy;
   dev_qc+=sxsy;

  GPU_SYNCHRONIZE();
  printf("GPU memory usage = %ld MiB\n", 15*msize_vol/1024/1024);
}


void RAJA_Finalize()
{

   extern float* dev_ch1dxx;
   extern float* dev_ch1dyy;
   extern float* dev_ch1dzz;
   extern float* dev_ch1dxy;
   extern float* dev_ch1dyz;
   extern float* dev_ch1dxz;
   extern float* dev_v2px;
   extern float* dev_v2pz;
   extern float* dev_v2sz;
   extern float* dev_v2pn;
   extern float* dev_pp;
   extern float* dev_pc;
   extern float* dev_qp;
   extern float* dev_qc;

   dev_pp-=sxsy;
   dev_pc-=sxsy;
   dev_qp-=sxsy;
   dev_qc-=sxsy;

   GPU_FREE(dev_ch1dxx);
   GPU_FREE(dev_ch1dyy);
   GPU_FREE(dev_ch1dzz);
   GPU_FREE(dev_ch1dxy);
   GPU_FREE(dev_ch1dyz);
   GPU_FREE(dev_ch1dxz);
   GPU_FREE(dev_v2px);
   GPU_FREE(dev_v2pz);
   GPU_FREE(dev_v2sz);
   GPU_FREE(dev_v2pn);
   GPU_FREE(dev_pp);
   GPU_FREE(dev_pc);
   GPU_FREE(dev_qp);
   GPU_FREE(dev_qc);

   printf("RAJA_Finalize: SUCCESS\n");
}



void RAJA_Update_pointers(const int sx, const int sy, const int sz, float *pc)
{
   extern float* dev_pc;
   const size_t sxsysz=((size_t)sx*sy)*sz;
   const size_t msize_vol=sxsysz*sizeof(float);
   if (pc) GPU_MEMCPY_D2H(pc, dev_pc, msize_vol);
}

} // extern "C"
