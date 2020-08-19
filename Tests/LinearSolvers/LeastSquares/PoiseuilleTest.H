#ifndef POISEUILLE_TEST_
#define POISEUILLE_TEST_

#include <AMReX_EBFabFactory.H>
#include <AMReX_MLMG.H>
#include <AMReX_Array.H>

class PoiseuilleTest
{
public:

    PoiseuilleTest ();

    void compute_gradient ();
    void writePlotfile ();
    void initData ();

private:

    void initializeEB ();
    void readParameters ();
    void initGrids ();

    int max_level = 0;
    int ref_ratio = 2;
    int n_cell = 14;
    int max_grid_size = 64;
    int max_coarsening_level = 0;
    
    amrex::Vector<amrex::Geometry> geom;
    amrex::Vector<amrex::BoxArray> grids;
    amrex::Vector<amrex::DistributionMapping> dmap;
    amrex::Vector<std::unique_ptr<amrex::EBFArrayBoxFactory> > factory;
    amrex::Vector<amrex::MultiFab> phi;
    amrex::Vector<amrex::MultiFab> phieb;
    amrex::Vector<amrex::MultiFab> rhs;
    amrex::Vector<amrex::MultiFab> acoef;
    amrex::Vector<amrex::Array<amrex::MultiFab,AMREX_SPACEDIM> > bcoef;
    amrex::Vector<amrex::MultiFab> bcoef_eb;
};

#endif
