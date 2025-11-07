#include<stdio.h>
#include<math.h>
#include<stdlib.h>
#include"../driver.h"
#include"../sample.h"
#include"cuda_stuff.h"
#include"cuda_propagate.h"
#include"cuda_insertsource.h"
#include"cpu_boundary.h"

// Global device vars
float* dev_ch1dxx=NULL;
float* dev_ch1dyy=NULL;
float* dev_ch1dzz=NULL;
float* dev_ch1dxy=NULL;
float* dev_ch1dyz=NULL;
float* dev_ch1dxz=NULL;
float* dev_v2px=NULL;
float* dev_v2pz=NULL;
float* dev_v2sz=NULL;
float* dev_v2pn=NULL;
float* dev_pp=NULL;
float* dev_pc=NULL;
float* dev_qp=NULL;
float* dev_qc=NULL;

// CUDA streams (declared in cuda_stuff.cu)
extern cudaStream_t compute_stream;
extern cudaStream_t transfer_stream;

// Host copies of coefficient arrays for CPU boundary processing
static float *host_ch1dxx=NULL, *host_ch1dyy=NULL, *host_ch1dzz=NULL;
static float *host_ch1dxy=NULL, *host_ch1dyz=NULL, *host_ch1dxz=NULL;
static float *host_v2px=NULL, *host_v2pz=NULL, *host_v2sz=NULL, *host_v2pn=NULL;


#define MODEL_GLOBALVARS
#define MODEL_INITIALIZE



void DRIVER_Initialize(const int sx, const int sy, const int sz, const int bord,
                       float dx, float dy, float dz, float dt,
                       float * restrict vpz, float * restrict vsv, float * restrict epsilon, float * restrict delta,
                       float * restrict phi, float * restrict theta,
                       float * restrict pp, float * restrict pc, float * restrict qp, float * restrict qc)
{

#include"../precomp.h"

	   CUDA_Initialize(sx,   sy,   sz,   bord,
		  dx,  dy,  dz,  dt,
	          ch1dxx,    ch1dyy,    ch1dzz,
  	          ch1dxy,    ch1dyz,    ch1dxz,
  	          v2px,    v2pz,    v2sz,    v2pn,
  	          vpz,    vsv,    epsilon,    delta,
  	          phi,    theta,
  	          pp,    pc,    qp,    qc);

	   // Store host pointers for CPU boundary processing (if enabled)
	   host_ch1dxx = ch1dxx;
	   host_ch1dyy = ch1dyy;
	   host_ch1dzz = ch1dzz;
	   host_ch1dxy = ch1dxy;
	   host_ch1dyz = ch1dyz;
	   host_ch1dxz = ch1dxz;
	   host_v2px = v2px;
	   host_v2pz = v2pz;
	   host_v2sz = v2sz;
	   host_v2pn = v2pn;

	   const int cpu_layers = CPU_GetBoundaryLayers();
	   if (cpu_layers > 0) {
	       printf("CPU+GPU Hybrid Mode: CPU will process %d boundary Z-layers\n", cpu_layers);
	       printf("WARNING: CPU boundary mode requires data transfers (not yet implemented)\n");
	   } else {
	       printf("GPU-Only Mode with optimizations: 1-thread source, CUDA streams\n");
	   }

}



void DRIVER_Finalize()
{
	CUDA_Finalize();
}


void DRIVER_Update_pointers(const int sx, const int sy, const int sz, float *pc)
{
	CUDA_Update_pointers(sx,sy,sz,pc);
}




void DRIVER_Propagate(const int sx, const int sy, const int sz, const int bord,
                      const float dx, const float dy, const float dz, const float dt, const int it,
	              float * pp, float * pc, float * qp, float * qc)
{

	// GPU processes entire domain using optimized kernel with streams
	// CPU boundary processing disabled by default (requires expensive D2H/H2D transfers)

	const int cpu_layers = CPU_GetBoundaryLayers();

	if (cpu_layers > 0) {
		// CPU+GPU Hybrid mode (WARNING: requires data transfer implementation)
		// Not recommended - data transfer overhead negates benefits

		const int gpu_z_start = bord + 1 + cpu_layers;
		const int gpu_z_end = sz - bord - 1 - cpu_layers;

		CUDA_Propagate(sx, sy, sz, bord,
		               dx, dy, dz, dt, it,
		               pp, pc, qp, qc,
		               compute_stream, gpu_z_start, gpu_z_end);

		CPU_ProcessBoundaries(sx, sy, sz, bord,
		                      dx, dy, dz, dt,
		                      host_ch1dxx, host_ch1dyy, host_ch1dzz,
		                      host_ch1dxy, host_ch1dyz, host_ch1dxz,
		                      host_v2px, host_v2pz, host_v2sz, host_v2pn,
		                      pp, pc, qp, qc);

		cudaStreamSynchronize(compute_stream);
	} else {
		// GPU-only mode with optimized kernel (RECOMMENDED)
		// Benefits: optimized 1-thread source insertion, CUDA streams
		CUDA_Propagate(sx, sy, sz, bord,
		               dx, dy, dz, dt, it,
		               pp, pc, qp, qc,
		               compute_stream, bord + 1, sz - bord - 1);

		// No explicit sync needed - handled by main loop
	}

}


void DRIVER_InsertSource(float dt, int it, int iSource, float *p, float*q, float src)
{
	// Use compute_stream for source insertion
	CUDA_InsertSource(src, iSource, p, q, compute_stream);
}

