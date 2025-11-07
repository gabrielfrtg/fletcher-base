#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "openmp_target_stuff.h"

static size_t sxsy=0;
static size_t sxsysz=0;

void OMP_TARGET_Initialize(const int sx, const int sy, const int sz, const int bord,
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


  int num_devices = omp_get_num_devices();
  int device = omp_get_default_device();
  printf("OpenMP target using device(%d) of %d available devices.\n", device, num_devices);

  sxsy=sx*sy; // one plane
  sxsysz=sxsy*sz;
  const size_t msize_vol=sxsysz;
  const size_t msize_vol_extra=msize_vol+2*sxsy; // 2 extra planes for wave fields

  // Allocate device memory for coefficient arrays
  dev_ch1dxx = (float*)malloc(msize_vol * sizeof(float));
  dev_ch1dyy = (float*)malloc(msize_vol * sizeof(float));
  dev_ch1dzz = (float*)malloc(msize_vol * sizeof(float));
  dev_ch1dxy = (float*)malloc(msize_vol * sizeof(float));
  dev_ch1dyz = (float*)malloc(msize_vol * sizeof(float));
  dev_ch1dxz = (float*)malloc(msize_vol * sizeof(float));
  dev_v2px = (float*)malloc(msize_vol * sizeof(float));
  dev_v2pz = (float*)malloc(msize_vol * sizeof(float));
  dev_v2sz = (float*)malloc(msize_vol * sizeof(float));
  dev_v2pn = (float*)malloc(msize_vol * sizeof(float));

  // Copy coefficient data to device arrays (these will be mapped to device)
  memcpy(dev_ch1dxx, ch1dxx, msize_vol * sizeof(float));
  memcpy(dev_ch1dyy, ch1dyy, msize_vol * sizeof(float));
  memcpy(dev_ch1dzz, ch1dzz, msize_vol * sizeof(float));
  memcpy(dev_ch1dxy, ch1dxy, msize_vol * sizeof(float));
  memcpy(dev_ch1dyz, ch1dyz, msize_vol * sizeof(float));
  memcpy(dev_ch1dxz, ch1dxz, msize_vol * sizeof(float));
  memcpy(dev_v2px, v2px, msize_vol * sizeof(float));
  memcpy(dev_v2pz, v2pz, msize_vol * sizeof(float));
  memcpy(dev_v2sz, v2sz, msize_vol * sizeof(float));
  memcpy(dev_v2pn, v2pn, msize_vol * sizeof(float));

  // Wave field arrays with extra planes
  dev_pp = (float*)calloc(msize_vol_extra, sizeof(float));
  dev_pc = (float*)calloc(msize_vol_extra, sizeof(float));
  dev_qp = (float*)calloc(msize_vol_extra, sizeof(float));
  dev_qc = (float*)calloc(msize_vol_extra, sizeof(float));

  // Skip first plane (offset by sxsy)
  dev_pp += sxsy;
  dev_pc += sxsy;
  dev_qp += sxsy;
  dev_qc += sxsy;

  // Map coefficient arrays to device (read-only)
  #pragma omp target enter data map(to: dev_ch1dxx[0:msize_vol])
  #pragma omp target enter data map(to: dev_ch1dyy[0:msize_vol])
  #pragma omp target enter data map(to: dev_ch1dzz[0:msize_vol])
  #pragma omp target enter data map(to: dev_ch1dxy[0:msize_vol])
  #pragma omp target enter data map(to: dev_ch1dyz[0:msize_vol])
  #pragma omp target enter data map(to: dev_ch1dxz[0:msize_vol])
  #pragma omp target enter data map(to: dev_v2px[0:msize_vol])
  #pragma omp target enter data map(to: dev_v2pz[0:msize_vol])
  #pragma omp target enter data map(to: dev_v2sz[0:msize_vol])
  #pragma omp target enter data map(to: dev_v2pn[0:msize_vol])

  // Map wave field arrays to device (read-write, using base pointer before offset)
  float *pp_base = dev_pp - sxsy;
  float *pc_base = dev_pc - sxsy;
  float *qp_base = dev_qp - sxsy;
  float *qc_base = dev_qc - sxsy;
  #pragma omp target enter data map(to: pp_base[0:msize_vol_extra])
  #pragma omp target enter data map(to: pc_base[0:msize_vol_extra])
  #pragma omp target enter data map(to: qp_base[0:msize_vol_extra])
  #pragma omp target enter data map(to: qc_base[0:msize_vol_extra])

  printf("GPU memory usage = %ld MiB\n", 15*msize_vol*sizeof(float)/1024/1024);

}


void OMP_TARGET_Finalize()
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

   const size_t msize_vol = sxsysz;
   const size_t msize_vol_extra = msize_vol + 2*sxsy;

   // Get base pointers
   float *pp_base = dev_pp - sxsy;
   float *pc_base = dev_pc - sxsy;
   float *qp_base = dev_qp - sxsy;
   float *qc_base = dev_qc - sxsy;

   // Unmap all arrays from device
   #pragma omp target exit data map(delete: dev_ch1dxx[0:msize_vol])
   #pragma omp target exit data map(delete: dev_ch1dyy[0:msize_vol])
   #pragma omp target exit data map(delete: dev_ch1dzz[0:msize_vol])
   #pragma omp target exit data map(delete: dev_ch1dxy[0:msize_vol])
   #pragma omp target exit data map(delete: dev_ch1dyz[0:msize_vol])
   #pragma omp target exit data map(delete: dev_ch1dxz[0:msize_vol])
   #pragma omp target exit data map(delete: dev_v2px[0:msize_vol])
   #pragma omp target exit data map(delete: dev_v2pz[0:msize_vol])
   #pragma omp target exit data map(delete: dev_v2sz[0:msize_vol])
   #pragma omp target exit data map(delete: dev_v2pn[0:msize_vol])
   #pragma omp target exit data map(delete: pp_base[0:msize_vol_extra])
   #pragma omp target exit data map(delete: pc_base[0:msize_vol_extra])
   #pragma omp target exit data map(delete: qp_base[0:msize_vol_extra])
   #pragma omp target exit data map(delete: qc_base[0:msize_vol_extra])

   // Restore pointers and free host memory
   dev_pp = pp_base;
   dev_pc = pc_base;
   dev_qp = qp_base;
   dev_qc = qc_base;

   free(dev_ch1dxx);
   free(dev_ch1dyy);
   free(dev_ch1dzz);
   free(dev_ch1dxy);
   free(dev_ch1dyz);
   free(dev_ch1dxz);
   free(dev_v2px);
   free(dev_v2pz);
   free(dev_v2sz);
   free(dev_v2pn);
   free(dev_pp);
   free(dev_pc);
   free(dev_qp);
   free(dev_qc);

   printf("OMP_TARGET_Finalize: SUCCESS\n");
}



void OMP_TARGET_Update_pointers(const int sx, const int sy, const int sz, float *pc)
{
   extern float* dev_pc;
   const size_t sxsysz=((size_t)sx*sy)*sz;

   // Copy data back from device to host
   if (pc) {
      #pragma omp target update from(dev_pc[0:sxsysz])
      memcpy(pc, dev_pc, sxsysz * sizeof(float));
   }
}
