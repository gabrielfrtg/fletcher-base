#include "kokkos_defines.h"
#include "kokkos_stuff.h"

// Define Kokkos execution space
using exec_space = Kokkos::DefaultExecutionSpace;
using mem_space = exec_space::memory_space;

// Define Kokkos View types
using ViewFloat1D = Kokkos::View<float*, mem_space>;

// Global device arrays (Kokkos Views)
static ViewFloat1D dev_ch1dxx;
static ViewFloat1D dev_ch1dyy;
static ViewFloat1D dev_ch1dzz;
static ViewFloat1D dev_ch1dxy;
static ViewFloat1D dev_ch1dyz;
static ViewFloat1D dev_ch1dxz;
static ViewFloat1D dev_v2px;
static ViewFloat1D dev_v2pz;
static ViewFloat1D dev_v2sz;
static ViewFloat1D dev_v2pn;
static ViewFloat1D dev_pp;
static ViewFloat1D dev_pc;
static ViewFloat1D dev_qp;
static ViewFloat1D dev_qc;

static size_t sxsy = 0;
static size_t sxsysz = 0;

// Helper function to copy data from host to device
template<typename ViewType>
void copy_to_device(ViewType& dev_view, float* host_ptr, size_t size) {
    auto host_mirror = Kokkos::create_mirror_view(dev_view);
    for (size_t i = 0; i < size; i++) {
        host_mirror(i) = host_ptr[i];
    }
    Kokkos::deep_copy(dev_view, host_mirror);
}

// Helper function to copy data from device to host
template<typename ViewType>
void copy_to_host(float* host_ptr, ViewType& dev_view, size_t size) {
    auto host_mirror = Kokkos::create_mirror_view(dev_view);
    Kokkos::deep_copy(host_mirror, dev_view);
    for (size_t i = 0; i < size; i++) {
        host_ptr[i] = host_mirror(i);
    }
}

// Track if we initialized Kokkos (so we know if we should finalize it)
static bool we_initialized_kokkos = false;

extern "C" void KOKKOS_Initialize(const int sx, const int sy, const int sz, const int bord,
                                  float dx, float dy, float dz, float dt,
                                  float * restrict ch1dxx, float * restrict ch1dyy, float * restrict ch1dzz,
                                  float * restrict ch1dxy, float * restrict ch1dyz, float * restrict ch1dxz,
                                  float * restrict v2px, float * restrict v2pz, float * restrict v2sz, float * restrict v2pn,
                                  float * restrict vpz, float * restrict vsv, float * restrict epsilon, float * restrict delta,
                                  float * restrict phi, float * restrict theta,
                                  float * restrict pp, float * restrict pc, float * restrict qp, float * restrict qc)
{
    // Initialize Kokkos if not already initialized
    if (!Kokkos::is_initialized()) {
        Kokkos::initialize();
        we_initialized_kokkos = true;
    }

    printf("Kokkos execution space: %s\n", typeid(exec_space).name());
    printf("Kokkos memory space: %s\n", typeid(mem_space).name());

    // Check sx, sy values
    if (sx % BSIZE_X != 0) {
        printf("sx(%d) must be multiple of BSIZE_X(%d)\n", sx, (int)BSIZE_X);
        exit(1);
    }
    if (sy % BSIZE_Y != 0) {
        printf("sy(%d) must be multiple of BSIZE_Y(%d)\n", sy, (int)BSIZE_Y);
        exit(1);
    }

    sxsy = sx * sy; // one plane
    sxsysz = sxsy * sz;
    const size_t msize_vol_extra = sxsysz + 2 * sxsy; // 2 extra planes for wave fields

    // Allocate device memory
    dev_ch1dxx = ViewFloat1D("dev_ch1dxx", sxsysz);
    dev_ch1dyy = ViewFloat1D("dev_ch1dyy", sxsysz);
    dev_ch1dzz = ViewFloat1D("dev_ch1dzz", sxsysz);
    dev_ch1dxy = ViewFloat1D("dev_ch1dxy", sxsysz);
    dev_ch1dyz = ViewFloat1D("dev_ch1dyz", sxsysz);
    dev_ch1dxz = ViewFloat1D("dev_ch1dxz", sxsysz);
    dev_v2px = ViewFloat1D("dev_v2px", sxsysz);
    dev_v2pz = ViewFloat1D("dev_v2pz", sxsysz);
    dev_v2sz = ViewFloat1D("dev_v2sz", sxsysz);
    dev_v2pn = ViewFloat1D("dev_v2pn", sxsysz);

    // Allocate wave field arrays with extra plane
    dev_pp = ViewFloat1D("dev_pp", msize_vol_extra);
    dev_pc = ViewFloat1D("dev_pc", msize_vol_extra);
    dev_qp = ViewFloat1D("dev_qp", msize_vol_extra);
    dev_qc = ViewFloat1D("dev_qc", msize_vol_extra);

    // Copy data from host to device
    copy_to_device(dev_ch1dxx, ch1dxx, sxsysz);
    copy_to_device(dev_ch1dyy, ch1dyy, sxsysz);
    copy_to_device(dev_ch1dzz, ch1dzz, sxsysz);
    copy_to_device(dev_ch1dxy, ch1dxy, sxsysz);
    copy_to_device(dev_ch1dyz, ch1dyz, sxsysz);
    copy_to_device(dev_ch1dxz, ch1dxz, sxsysz);
    copy_to_device(dev_v2px, v2px, sxsysz);
    copy_to_device(dev_v2pz, v2pz, sxsysz);
    copy_to_device(dev_v2sz, v2sz, sxsysz);
    copy_to_device(dev_v2pn, v2pn, sxsysz);

    // Initialize wave fields to zero
    Kokkos::deep_copy(dev_pp, 0.0f);
    Kokkos::deep_copy(dev_pc, 0.0f);
    Kokkos::deep_copy(dev_qp, 0.0f);
    Kokkos::deep_copy(dev_qc, 0.0f);

    Kokkos::fence();

    printf("GPU memory usage = %ld MiB\n", 15 * sxsysz * sizeof(float) / 1024 / 1024);
}

extern "C" void KOKKOS_Finalize()
{
    // Views will be automatically deallocated when they go out of scope
    // We can explicitly free them by resizing to 0
    dev_ch1dxx = ViewFloat1D();
    dev_ch1dyy = ViewFloat1D();
    dev_ch1dzz = ViewFloat1D();
    dev_ch1dxy = ViewFloat1D();
    dev_ch1dyz = ViewFloat1D();
    dev_ch1dxz = ViewFloat1D();
    dev_v2px = ViewFloat1D();
    dev_v2pz = ViewFloat1D();
    dev_v2sz = ViewFloat1D();
    dev_v2pn = ViewFloat1D();
    dev_pp = ViewFloat1D();
    dev_pc = ViewFloat1D();
    dev_qp = ViewFloat1D();
    dev_qc = ViewFloat1D();

    // Only finalize Kokkos if we initialized it
    if (we_initialized_kokkos && Kokkos::is_initialized()) {
        Kokkos::finalize();
        we_initialized_kokkos = false;
    }

    printf("KOKKOS_Finalize: SUCCESS\n");
}

extern "C" void KOKKOS_Update_pointers(const int sx, const int sy, const int sz, float *pc)
{
    if (pc) {
        // Copy data from device to host, accounting for the extra plane offset
        auto host_mirror = Kokkos::create_mirror_view(dev_pc);
        Kokkos::deep_copy(host_mirror, dev_pc);
        for (size_t i = 0; i < sxsysz; i++) {
            pc[i] = host_mirror(i + sxsy);
        }
    }
}

// Export global views for use in other files
ViewFloat1D& get_dev_ch1dxx() { return dev_ch1dxx; }
ViewFloat1D& get_dev_ch1dyy() { return dev_ch1dyy; }
ViewFloat1D& get_dev_ch1dzz() { return dev_ch1dzz; }
ViewFloat1D& get_dev_ch1dxy() { return dev_ch1dxy; }
ViewFloat1D& get_dev_ch1dyz() { return dev_ch1dyz; }
ViewFloat1D& get_dev_ch1dxz() { return dev_ch1dxz; }
ViewFloat1D& get_dev_v2px() { return dev_v2px; }
ViewFloat1D& get_dev_v2pz() { return dev_v2pz; }
ViewFloat1D& get_dev_v2sz() { return dev_v2sz; }
ViewFloat1D& get_dev_v2pn() { return dev_v2pn; }
ViewFloat1D& get_dev_pp() { return dev_pp; }
ViewFloat1D& get_dev_pc() { return dev_pc; }
ViewFloat1D& get_dev_qp() { return dev_qp; }
ViewFloat1D& get_dev_qc() { return dev_qc; }
size_t get_sxsy() { return sxsy; }

// Swap Views for time stepping
void swap_views() {
    ViewFloat1D tmp;

    // Swap pp and pc
    tmp = dev_pp;
    dev_pp = dev_pc;
    dev_pc = tmp;

    // Swap qp and qc
    tmp = dev_qp;
    dev_qp = dev_qc;
    dev_qc = tmp;
}
