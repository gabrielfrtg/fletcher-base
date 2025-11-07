#include <omp.h>
#include "openmp_target_propagate.h"
#include "../derivatives.h"
#include "../map.h"

// Propagate: using Fletcher's equations, propagate waves one dt,
//            either forward or backward in time
void OMP_TARGET_Propagate(const int sx, const int sy, const int sz, const int bord,
		    const float dx, const float dy, const float dz, const float dt, const int it,
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

  // Launch OpenMP target kernel - arrays already mapped via target enter data
  #pragma omp target teams distribute parallel for collapse(2)
  for (int iy=bord; iy<sy-bord; iy++) {
    for (int ix=bord; ix<sx-bord; ix++) {

      // SAMPLE_PRE_LOOP section
      const int strideX=ind(1,0,0)-ind(0,0,0);
      const int strideY=ind(0,1,0)-ind(0,0,0);
      const int strideZ=ind(0,0,1)-ind(0,0,0);

      const float dxxinv=1.0f/(dx*dx);
      const float dyyinv=1.0f/(dy*dy);
      const float dzzinv=1.0f/(dz*dz);
      const float dxyinv=1.0f/(dx*dy);
      const float dxzinv=1.0f/(dx*dz);
      const float dyzinv=1.0f/(dy*dz);

      // solve both equations in all internal grid points,
      // including absortion zone

      for (int iz=bord+1; iz<sz-bord-1; iz++) {

        // SAMPLE_LOOP section
        const int i=ind(ix,iy,iz);

        // p derivatives, H1(p) and H2(p)

        const float pxx= Der2(dev_pc, i, strideX, dxxinv);
        const float pyy= Der2(dev_pc, i, strideY, dyyinv);
        const float pzz= Der2(dev_pc, i, strideZ, dzzinv);
        const float pxy= DerCross(dev_pc, i, strideX, strideY, dxyinv);
        const float pyz= DerCross(dev_pc, i, strideY, strideZ, dyzinv);
        const float pxz= DerCross(dev_pc, i, strideX, strideZ, dxzinv);

        const float cpxx=dev_ch1dxx[i]*pxx;
        const float cpyy=dev_ch1dyy[i]*pyy;
        const float cpzz=dev_ch1dzz[i]*pzz;
        const float cpxy=dev_ch1dxy[i]*pxy;
        const float cpxz=dev_ch1dxz[i]*pxz;
        const float cpyz=dev_ch1dyz[i]*pyz;
        const float h1p=cpxx+cpyy+cpzz+cpxy+cpxz+cpyz;
        const float h2p=pxx+pyy+pzz-h1p;

        // q derivatives, H1(q) and H2(q)

        const float qxx= Der2(dev_qc, i, strideX, dxxinv);
        const float qyy= Der2(dev_qc, i, strideY, dyyinv);
        const float qzz= Der2(dev_qc, i, strideZ, dzzinv);
        const float qxy= DerCross(dev_qc, i, strideX,  strideY, dxyinv);
        const float qyz= DerCross(dev_qc, i, strideY,  strideZ, dyzinv);
        const float qxz= DerCross(dev_qc, i, strideX,  strideZ, dxzinv);

        const float cqxx=dev_ch1dxx[i]*qxx;
        const float cqyy=dev_ch1dyy[i]*qyy;
        const float cqzz=dev_ch1dzz[i]*qzz;
        const float cqxy=dev_ch1dxy[i]*qxy;
        const float cqxz=dev_ch1dxz[i]*qxz;
        const float cqyz=dev_ch1dyz[i]*qyz;
        const float h1q=cqxx+cqyy+cqzz+cqxy+cqxz+cqyz;
        const float h2q=qxx+qyy+qzz-h1q;

        // p-q derivatives, H1(p-q) and H2(p-q)

        const float h1pmq=h1p-h1q;
        const float h2pmq=h2p-h2q;

        // rhs of p and q equations

        const float rhsp=dev_v2px[i]*h2p + dev_v2pz[i]*h1q + dev_v2sz[i]*h1pmq;
        const float rhsq=dev_v2pn[i]*h2p + dev_v2pz[i]*h1q - dev_v2sz[i]*h2pmq;

        // new p and q

        dev_pp[i]=2.0f*dev_pc[i] - dev_pp[i] + rhsp*dt*dt;
        dev_qp[i]=2.0f*dev_qc[i] - dev_qp[i] + rhsq*dt*dt;

      }
    }
  }

  OMP_TARGET_SwapArrays(&dev_pp, &dev_pc, &dev_qp, &dev_qc);
}

// swap array pointers on time forward array propagation
void OMP_TARGET_SwapArrays(float **pp, float **pc, float **qp, float **qc) {
  float *tmp;

  tmp=*pp;
  *pp=*pc;
  *pc=tmp;

  tmp=*qp;
  *qp=*qc;
  *qc=tmp;
}
