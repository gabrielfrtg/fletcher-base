#include "kokkos_defines.h"
#include "kokkos_propagate.h"
#include "../derivatives.h"
#include "../map.h"

// Define Kokkos execution space
using exec_space = Kokkos::DefaultExecutionSpace;
using mem_space = exec_space::memory_space;
using ViewFloat1D = Kokkos::View<float*, mem_space>;

// External accessor functions from kokkos_stuff.cpp
extern ViewFloat1D& get_dev_ch1dxx();
extern ViewFloat1D& get_dev_ch1dyy();
extern ViewFloat1D& get_dev_ch1dzz();
extern ViewFloat1D& get_dev_ch1dxy();
extern ViewFloat1D& get_dev_ch1dyz();
extern ViewFloat1D& get_dev_ch1dxz();
extern ViewFloat1D& get_dev_v2px();
extern ViewFloat1D& get_dev_v2pz();
extern ViewFloat1D& get_dev_v2sz();
extern ViewFloat1D& get_dev_v2pn();
extern ViewFloat1D& get_dev_pp();
extern ViewFloat1D& get_dev_pc();
extern ViewFloat1D& get_dev_qp();
extern ViewFloat1D& get_dev_qc();
extern size_t get_sxsy();
extern void swap_views();

// Propagate kernel
struct PropagateFunctor {
    const int sx, sy, sz, bord;
    const float dx, dy, dz, dt;
    const int it;
    ViewFloat1D ch1dxx, ch1dyy, ch1dzz;
    ViewFloat1D ch1dxy, ch1dyz, ch1dxz;
    ViewFloat1D v2px, v2pz, v2sz, v2pn;
    ViewFloat1D pp, pc, qp, qc;
    const size_t sxsy;

    PropagateFunctor(const int sx_, const int sy_, const int sz_, const int bord_,
                     const float dx_, const float dy_, const float dz_, const float dt_, const int it_,
                     ViewFloat1D ch1dxx_, ViewFloat1D ch1dyy_, ViewFloat1D ch1dzz_,
                     ViewFloat1D ch1dxy_, ViewFloat1D ch1dyz_, ViewFloat1D ch1dxz_,
                     ViewFloat1D v2px_, ViewFloat1D v2pz_, ViewFloat1D v2sz_, ViewFloat1D v2pn_,
                     ViewFloat1D pp_, ViewFloat1D pc_, ViewFloat1D qp_, ViewFloat1D qc_,
                     size_t sxsy_)
        : sx(sx_), sy(sy_), sz(sz_), bord(bord_),
          dx(dx_), dy(dy_), dz(dz_), dt(dt_), it(it_),
          ch1dxx(ch1dxx_), ch1dyy(ch1dyy_), ch1dzz(ch1dzz_),
          ch1dxy(ch1dxy_), ch1dyz(ch1dyz_), ch1dxz(ch1dxz_),
          v2px(v2px_), v2pz(v2pz_), v2sz(v2sz_), v2pn(v2pn_),
          pp(pp_), pc(pc_), qp(qp_), qc(qc_), sxsy(sxsy_) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const int ix, const int iy) const {

        // Pre-loop setup
        const int strideX = 1;
        const int strideY = sx;
        const int strideZ = sx * sy;

        const float dxxinv = 1.0f / (dx * dx);
        const float dyyinv = 1.0f / (dy * dy);
        const float dzzinv = 1.0f / (dz * dz);
        const float dxyinv = 1.0f / (dx * dy);
        const float dxzinv = 1.0f / (dx * dz);
        const float dyzinv = 1.0f / (dy * dz);

        // Solve both equations in all internal grid points, including absorption zone
        for (int iz = bord + 1; iz < sz - bord - 1; iz++) {
            const int i = ix + iy * sx + iz * sx * sy + sxsy; // offset by sxsy for extra plane

            // p derivatives, H1(p) and H2(p)
            const float pxx = Der2(pc.data(), i, strideX, dxxinv);
            const float pyy = Der2(pc.data(), i, strideY, dyyinv);
            const float pzz = Der2(pc.data(), i, strideZ, dzzinv);
            const float pxy = DerCross(pc.data(), i, strideX, strideY, dxyinv);
            const float pyz = DerCross(pc.data(), i, strideY, strideZ, dyzinv);
            const float pxz = DerCross(pc.data(), i, strideX, strideZ, dxzinv);

            const float cpxx = ch1dxx(i - sxsy) * pxx;  // ch1dxx doesn't have offset
            const float cpyy = ch1dyy(i - sxsy) * pyy;
            const float cpzz = ch1dzz(i - sxsy) * pzz;
            const float cpxy = ch1dxy(i - sxsy) * pxy;
            const float cpxz = ch1dxz(i - sxsy) * pxz;
            const float cpyz = ch1dyz(i - sxsy) * pyz;
            const float h1p = cpxx + cpyy + cpzz + cpxy + cpxz + cpyz;
            const float h2p = pxx + pyy + pzz - h1p;

            // q derivatives, H1(q) and H2(q)
            const float qxx = Der2(qc.data(), i, strideX, dxxinv);
            const float qyy = Der2(qc.data(), i, strideY, dyyinv);
            const float qzz = Der2(qc.data(), i, strideZ, dzzinv);
            const float qxy = DerCross(qc.data(), i, strideX, strideY, dxyinv);
            const float qyz = DerCross(qc.data(), i, strideY, strideZ, dyzinv);
            const float qxz = DerCross(qc.data(), i, strideX, strideZ, dxzinv);

            const float cqxx = ch1dxx(i - sxsy) * qxx;
            const float cqyy = ch1dyy(i - sxsy) * qyy;
            const float cqzz = ch1dzz(i - sxsy) * qzz;
            const float cqxy = ch1dxy(i - sxsy) * qxy;
            const float cqxz = ch1dxz(i - sxsy) * qxz;
            const float cqyz = ch1dyz(i - sxsy) * qyz;
            const float h1q = cqxx + cqyy + cqzz + cqxy + cqxz + cqyz;
            const float h2q = qxx + qyy + qzz - h1q;

            // p-q derivatives, H1(p-q) and H2(p-q)
            const float h1pmq = h1p - h1q;
            const float h2pmq = h2p - h2q;

            // rhs of p and q equations
            const float rhsp = v2px(i - sxsy) * h2p + v2pz(i - sxsy) * h1q + v2sz(i - sxsy) * h1pmq;
            const float rhsq = v2pn(i - sxsy) * h2p + v2pz(i - sxsy) * h1q - v2sz(i - sxsy) * h2pmq;

            // new p and q
            pp(i) = 2.0f * pc(i) - pp(i) + rhsp * dt * dt;
            qp(i) = 2.0f * qc(i) - qp(i) + rhsq * dt * dt;
        }
    }
};

extern "C" void KOKKOS_Propagate(const int sx, const int sy, const int sz, const int bord,
                                 const float dx, const float dy, const float dz, const float dt, const int it,
                                 float * restrict pp, float * restrict pc, float * restrict qp, float * restrict qc)
{
    ViewFloat1D& dev_ch1dxx = get_dev_ch1dxx();
    ViewFloat1D& dev_ch1dyy = get_dev_ch1dyy();
    ViewFloat1D& dev_ch1dzz = get_dev_ch1dzz();
    ViewFloat1D& dev_ch1dxy = get_dev_ch1dxy();
    ViewFloat1D& dev_ch1dyz = get_dev_ch1dyz();
    ViewFloat1D& dev_ch1dxz = get_dev_ch1dxz();
    ViewFloat1D& dev_v2px = get_dev_v2px();
    ViewFloat1D& dev_v2pz = get_dev_v2pz();
    ViewFloat1D& dev_v2sz = get_dev_v2sz();
    ViewFloat1D& dev_v2pn = get_dev_v2pn();
    ViewFloat1D& dev_pp = get_dev_pp();
    ViewFloat1D& dev_pc = get_dev_pc();
    ViewFloat1D& dev_qp = get_dev_qp();
    ViewFloat1D& dev_qc = get_dev_qc();
    size_t sxsy = get_sxsy();

    // Create the functor
    PropagateFunctor functor(sx, sy, sz, bord, dx, dy, dz, dt, it,
                            dev_ch1dxx, dev_ch1dyy, dev_ch1dzz,
                            dev_ch1dxy, dev_ch1dyz, dev_ch1dxz,
                            dev_v2px, dev_v2pz, dev_v2sz, dev_v2pn,
                            dev_pp, dev_pc, dev_qp, dev_qc, sxsy);

    // Execute the parallel computation over 2D grid matching CUDA block structure
    // Use MDRangePolicy with tile sizes matching BSIZE_X and BSIZE_Y from CUDA
    using Policy2D = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;
    Kokkos::parallel_for("Propagate",
                        Policy2D({0, 0}, {sx, sy}, {BSIZE_X, BSIZE_Y}),
                        functor);
    Kokkos::fence();

    // Swap the internal Views (device arrays)
    swap_views();
}
