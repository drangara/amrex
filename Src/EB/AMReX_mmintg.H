#ifndef AMREX_MMINTG_H_
#define AMREX_MMINTG_H_

#include <AMReX_MultiFab.H>

namespace amrex::mismatched_intg {

void compute_mmintegral (const Array<std::unique_ptr<MultiFab>, AMREX_SPACEDIM> & mmintg, int nghost = 100);
void compute_mmintegral (const Array<std::unique_ptr<MultiFab>, AMREX_SPACEDIM> & mmintgmf, IntVect nghost);

static constexpr int i_M_x        =  0;
static constexpr int i_M_y        =  1;
static constexpr int i_M_xy       =  2;
static constexpr int i_M_1        =  3;
static constexpr int numMmIntgs   =  4;

}

#endif
