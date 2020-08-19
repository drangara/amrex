#include "PoiseuilleTest.H"

#include <AMReX_MLEBABecLap.H>
#include <AMReX_ParmParse.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_EBMultiFabUtil.H>
#include <AMReX_EB2.H>

#include <AMReX_EB_LeastSquares_2D_K.H>

#include <cmath>

using namespace amrex;

PoiseuilleTest::PoiseuilleTest ()
{
    readParameters();

    initGrids();

    initializeEB();

    initData();
}

void
PoiseuilleTest::compute_gradient ()
{
    int ilev = 0;

    bool is_eb_dirichlet = true;
    bool is_eb_inhomog  = false;

    int ncomp = phi[0].nComp();

    for (MFIter mfi(phi[ilev]); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.fabbox();
        Array4<Real> const& phi_arr = phi[ilev].array(mfi);
        Array4<Real> const& phi_eb_arr = phieb[ilev].array(mfi);

        Array4<Real const> const& fcx   = (factory[ilev]->getFaceCent())[0]->const_array(mfi);
        Array4<Real const> const& fcy   = (factory[ilev]->getFaceCent())[1]->const_array(mfi);

        Array4<Real const> const& ccent = (factory[ilev]->getCentroid()).array(mfi);
        Array4<Real const> const& bcent = (factory[ilev]->getBndryCent()).array(mfi);
        Array4<Real const> const& apx   = (factory[ilev]->getAreaFrac())[0]->const_array(mfi);
        Array4<Real const> const& apy   = (factory[ilev]->getAreaFrac())[1]->const_array(mfi);
        Array4<Real const> const& norm  = (factory[ilev]->getBndryNormal()).array(mfi);

        const FabArray<EBCellFlagFab>* flags = &(factory[ilev]->getMultiEBCellFlagFab());
        Array4<EBCellFlag const> const& flag = flags->const_array(mfi);

        amrex::ParallelFor(bx, ncomp, 
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
            const auto dx = geom[ilev].CellSizeArray();
            amrex::Print() << "phi(" << i << "," << j << "): " << phi_arr(i,j,k) << std::endl;
            Real yloc_on_xface = fcx(i,j,k);
            Real xloc_on_yface = fcy(i,j,k);
            Real nx = norm(i,j,k,0);
            Real ny = norm(i,j,k,1);
            Real phi_x_on_x_face = 
               grad_x_of_phi_on_centroids(i, j, k, n, 
                                          phi_arr,
                                          phi_eb_arr,
                                          flag,
                                          ccent, bcent, 
                                          apx, apy, 
                                          yloc_on_xface,
                                          is_eb_dirichlet, is_eb_inhomog) / dx[0];
            Real phi_y_on_y_face = 
               grad_y_of_phi_on_centroids(i, j, k, n, 
                                          phi_arr,
                                          phi_eb_arr,
                                          flag,
                                          ccent, bcent, 
                                          apx, apy, 
                                          xloc_on_yface,
                                          is_eb_dirichlet, is_eb_inhomog) / dx[0];
            Real phi_eb = 
               grad_eb_of_phi_on_centroids(i, j, k, n, 
                                          phi_arr,
                                          phi_eb_arr,
                                          flag,
                                          ccent, bcent, 
                                          nx, ny, 
                                          is_eb_inhomog) / dx[0];
           amrex::Print() << "phi_x_on_x_face / dx (" << i << "," << j <<"): " << phi_x_on_x_face << std::endl;
           amrex::Print() << "phi_y_on_y_face / dx (" << i << "," << j <<"): " << phi_y_on_y_face << std::endl;
           amrex::Print() << "phi_eb / dx (" << i << "," << j <<"): " << phi_eb << std::endl;
           amrex::Print() << "normx (" << i << "," << j <<"): " << nx << std::endl;
           amrex::Print() << "normy (" << i << "," << j <<"): " << ny << std::endl;
           amrex::Print() << "centroid(" << i << "," << j <<"): " << ccent(i,j,k,0) << ", " << ccent(i,j,k,1) << std::endl << std::endl;
        });
    }
}

void
PoiseuilleTest::readParameters ()
{
    ParmParse pp;
}

void
PoiseuilleTest::initGrids ()
{
    int nlevels = max_level + 1;
    geom.resize(nlevels);
    grids.resize(nlevels);

    RealBox rb({AMREX_D_DECL(0.,0.,0.)}, {AMREX_D_DECL(1.4,14.0,0.)});
    std::array<int,AMREX_SPACEDIM> isperiodic{AMREX_D_DECL(0,1,0)};
    Geometry::Setup(&rb, 0, isperiodic.data());
    Box domain0(IntVect{AMREX_D_DECL(0,0,0)}, IntVect{AMREX_D_DECL(n_cell-1,n_cell-1,n_cell-1)});
    Box domain = domain0;
    for (int ilev = 0; ilev < nlevels; ++ilev)
    {
        geom[ilev].define(domain);
    }

    domain = domain0;
    for (int ilev = 0; ilev < nlevels; ++ilev)
    {
        grids[ilev].define(domain);
        grids[ilev].maxSize(max_grid_size);
    }
}

void
PoiseuilleTest::initData ()
{
    int nlevels = max_level + 1;
    dmap.resize(nlevels);
    factory.resize(nlevels);
    phi.resize(nlevels);
    phieb.resize(nlevels);
    rhs.resize(nlevels);
    acoef.resize(nlevels);
    bcoef.resize(nlevels);
    bcoef_eb.resize(nlevels);

    for (int ilev = 0; ilev < nlevels; ++ilev)
    {
        dmap[ilev].define(grids[ilev]);
        const EB2::IndexSpace& eb_is = EB2::IndexSpace::top();
        const EB2::Level& eb_level = eb_is.getLevel(geom[ilev]);
        factory[ilev].reset(new EBFArrayBoxFactory(eb_level, geom[ilev], grids[ilev], dmap[ilev],
                                                   {2,2,2}, EBSupport::full));

        phi[ilev].define(grids[ilev], dmap[ilev], 1, 1, MFInfo(), *factory[ilev]);
        phieb[ilev].define(grids[ilev], dmap[ilev], 1, 1, MFInfo(), *factory[ilev]);
        rhs[ilev].define(grids[ilev], dmap[ilev], 1, 0, MFInfo(), *factory[ilev]);
        acoef[ilev].define(grids[ilev], dmap[ilev], 1, 0, MFInfo(), *factory[ilev]);
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            bcoef[ilev][idim].define(amrex::convert(grids[ilev],IntVect::TheDimensionVector(idim)),
                                     dmap[ilev], 1, 0, MFInfo(), *factory[ilev]);
        }

        phi[ilev].setVal(0.0);
        phieb[ilev].setVal(0.0);
        rhs[ilev].setVal(0.0);
        acoef[ilev].setVal(1.0);
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            bcoef[ilev][idim].setVal(1.0);
        }

        const auto dx = geom[ilev].CellSizeArray();

        for (MFIter mfi(rhs[ilev]); mfi.isValid(); ++mfi)
        {
            const Box& bx = mfi.fabbox();
            Array4<Real> const& fab = phi[ilev].array(mfi);
            Array4<Real const> const& fcy   = (factory[ilev]->getFaceCent())[1]->const_array(mfi);
            amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                Real rx = (i+0.5 + fcy(i,j,k))*dx[0];
                fab(i,j,k) = (rx-0.225)*(1.-(rx-0.225));
            });
        }
    }
}

