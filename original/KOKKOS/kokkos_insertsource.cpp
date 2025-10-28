#include "kokkos_defines.h"
#include "kokkos_insertsource.h"

// Define Kokkos execution space
using exec_space = Kokkos::DefaultExecutionSpace;
using mem_space = exec_space::memory_space;
using ViewFloat1D = Kokkos::View<float*, mem_space>;

// External accessor functions from kokkos_stuff.cpp
extern ViewFloat1D& get_dev_pp();
extern ViewFloat1D& get_dev_pc();
extern ViewFloat1D& get_dev_qp();
extern ViewFloat1D& get_dev_qc();
extern size_t get_sxsy();

// Insert source kernel functor
struct InsertSourceFunctor {
    const float val;
    const int iSource;
    ViewFloat1D qp, qc;

    InsertSourceFunctor(const float val_, const int iSource_,
                       ViewFloat1D qp_, ViewFloat1D qc_)
        : val(val_), iSource(iSource_), qp(qp_), qc(qc_) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const int idx) const {
        if (idx == 0) {
            qp(iSource) += val;
            qc(iSource) += val;
        }
    }
};

extern "C" void KOKKOS_InsertSource(const float val, const int iSource, float *p, float *q)
{
    ViewFloat1D& dev_pp = get_dev_pp();
    ViewFloat1D& dev_pc = get_dev_pc();
    ViewFloat1D& dev_qp = get_dev_qp();
    ViewFloat1D& dev_qc = get_dev_qc();
    size_t sxsy = get_sxsy();

    if (dev_pp.data() && dev_qp.data()) {
        // Adjust iSource index for the extra plane offset
        const int adjusted_iSource = iSource + sxsy;

        // Create the functor - NOTE: CUDA version inserts into dev_pc, not dev_qp!
        InsertSourceFunctor functor(val, adjusted_iSource, dev_pc, dev_qc);

        // Execute the kernel with a single thread
        Kokkos::parallel_for("InsertSource", 1, functor);
        Kokkos::fence();
    }
}
