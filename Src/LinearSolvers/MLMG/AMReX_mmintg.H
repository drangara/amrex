#ifndef AMREX_MMINTG_H_
#define AMREX_MMINTG_H_

#include <AMReX_MultiFab.H>

namespace amrex::mismatched_intg {

static constexpr int i_M_x        =  0;
static constexpr int i_M_y        =  1;
static constexpr int i_M_xy       =  2;
static constexpr int i_M_1        =  3;
static constexpr int numMmIntgs   =  4;


// Order of points implicity matter.
// The first point must lie on lo face.
struct EBFaceTriangle {
    GpuArray<Real, 3> x;
    GpuArray<Real, 3> y;

    bool sign;
};

}

#endif
