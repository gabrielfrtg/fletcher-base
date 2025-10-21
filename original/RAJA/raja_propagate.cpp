#include "raja_defines.h"
#include "raja_propagate.h"
#include "../derivatives.h"
#include "../map.h"

extern "C" {

// Propagate: using Fletcher's equations, propagate waves one dt,
//            either forward or backward in time
void RAJA_Propagate(const int sx, const int sy, const int sz, const int bord,
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

  // Capture pointers for lambda
  float* ch1dxx_ptr = dev_ch1dxx;
  float* ch1dyy_ptr = dev_ch1dyy;
  float* ch1dzz_ptr = dev_ch1dzz;
  float* ch1dxy_ptr = dev_ch1dxy;
  float* ch1dyz_ptr = dev_ch1dyz;
  float* ch1dxz_ptr = dev_ch1dxz;
  float* v2px_ptr = dev_v2px;
  float* v2pz_ptr = dev_v2pz;
  float* v2sz_ptr = dev_v2sz;
  float* v2pn_ptr = dev_v2pn;
  float* pp_ptr = dev_pp;
  float* pc_ptr = dev_pc;
  float* qp_ptr = dev_qp;
  float* qc_ptr = dev_qc;

  // Use RAJA kernel policy for 2D iteration over x,y and explicit z loop
  RAJA::RangeSegment rangeX(0, sx);
  RAJA::RangeSegment rangeY(0, sy);

  RAJA::kernel<NESTED_POL>(
    RAJA::make_tuple(rangeX, rangeY),
    [=] RAJA_HOST_DEVICE (int ix, int iy) {

      // Pre-loop calculations (SAMPLE_PRE_LOOP)
      const int strideX = 1;
      const int strideY = sx;
      const int strideZ = sx * sy;

      const float dxxinv = 1.0f / (dx * dx);
      const float dyyinv = 1.0f / (dy * dy);
      const float dzzinv = 1.0f / (dz * dz);
      const float dxyinv = 1.0f / (dx * dy);
      const float dxzinv = 1.0f / (dx * dz);
      const float dyzinv = 1.0f / (dy * dz);

      // solve both equations in all internal grid points,
      // including absorption zone
      for (int iz = bord + 1; iz < sz - bord - 1; iz++) {

        // SAMPLE_LOOP
        const int i = (iz * sy + iy) * sx + ix;

        // p derivatives, H1(p) and H2(p)
        const float pxx = Der2(pc_ptr, i, strideX, dxxinv);
        const float pyy = Der2(pc_ptr, i, strideY, dyyinv);
        const float pzz = Der2(pc_ptr, i, strideZ, dzzinv);
        const float pxy = DerCross(pc_ptr, i, strideX, strideY, dxyinv);
        const float pyz = DerCross(pc_ptr, i, strideY, strideZ, dyzinv);
        const float pxz = DerCross(pc_ptr, i, strideX, strideZ, dxzinv);

        const float cpxx = ch1dxx_ptr[i] * pxx;
        const float cpyy = ch1dyy_ptr[i] * pyy;
        const float cpzz = ch1dzz_ptr[i] * pzz;
        const float cpxy = ch1dxy_ptr[i] * pxy;
        const float cpxz = ch1dxz_ptr[i] * pxz;
        const float cpyz = ch1dyz_ptr[i] * pyz;
        const float h1p = cpxx + cpyy + cpzz + cpxy + cpxz + cpyz;
        const float h2p = pxx + pyy + pzz - h1p;

        // q derivatives, H1(q) and H2(q)
        const float qxx = Der2(qc_ptr, i, strideX, dxxinv);
        const float qyy = Der2(qc_ptr, i, strideY, dyyinv);
        const float qzz = Der2(qc_ptr, i, strideZ, dzzinv);
        const float qxy = DerCross(qc_ptr, i, strideX, strideY, dxyinv);
        const float qyz = DerCross(qc_ptr, i, strideY, strideZ, dyzinv);
        const float qxz = DerCross(qc_ptr, i, strideX, strideZ, dxzinv);

        const float cqxx = ch1dxx_ptr[i] * qxx;
        const float cqyy = ch1dyy_ptr[i] * qyy;
        const float cqzz = ch1dzz_ptr[i] * qzz;
        const float cqxy = ch1dxy_ptr[i] * qxy;
        const float cqxz = ch1dxz_ptr[i] * qxz;
        const float cqyz = ch1dyz_ptr[i] * qyz;
        const float h1q = cqxx + cqyy + cqzz + cqxy + cqxz + cqyz;
        const float h2q = qxx + qyy + qzz - h1q;

        // p-q derivatives, H1(p-q) and H2(p-q)
        const float h1pmq = h1p - h1q;
        const float h2pmq = h2p - h2q;

        // rhs of p and q equations
        const float rhsp = v2px_ptr[i] * h2p + v2pz_ptr[i] * h1q + v2sz_ptr[i] * h1pmq;
        const float rhsq = v2pn_ptr[i] * h2p + v2pz_ptr[i] * h1q - v2sz_ptr[i] * h2pmq;

        // new p and q
        pp_ptr[i] = 2.0f * pc_ptr[i] - pp_ptr[i] + rhsp * dt * dt;
        qp_ptr[i] = 2.0f * qc_ptr[i] - qp_ptr[i] + rhsq * dt * dt;
      }
    }
  );

  RAJA_SwapArrays(&dev_pp, &dev_pc, &dev_qp, &dev_qc);

#if defined(RAJA_ENABLE_CUDA)
  cudaDeviceSynchronize();
#elif defined(RAJA_ENABLE_HIP)
  hipDeviceSynchronize();
#endif
}

// swap array pointers on time forward array propagation
void RAJA_SwapArrays(float **pp, float **pc, float **qp, float **qc) {
  float *tmp;

  tmp=*pp;
  *pp=*pc;
  *pc=tmp;

  tmp=*qp;
  *qp=*qc;
  *qc=tmp;
}

} // extern "C"
