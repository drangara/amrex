#include <AMReX_mmintg.H>
#include <AMReX_EB2.H>
#include <AMReX_mmintg_K.H>

namespace amrex::mismatched_intg {

void
compute_mmintegral (const Array<std::unique_ptr<MultiFab>, AMREX_SPACEDIM> & mmintg, int nghost)
{
    compute_mmintegral(mmintg, IntVect(nghost));
}

void
compute_mmintegral (const Array<std::unique_ptr<MultiFab>, AMREX_SPACEDIM> & mmintgmf, IntVect nghost)
{
#if (AMREX_SPACEDIM == 2)
    amrex::ignore_unused(mmintgmf, nghost);
    amrex::Abort("amrex::mismatched_intg::compute_mmintegrals is 3D only");
#else

    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {

        nghost.min(mmintgmf[idim]->nGrowVect());
        AMREX_ASSERT(mmintgmf[idim]->nComp() >= numMmIntgs);

        const auto& my_factory = dynamic_cast<EBFArrayBoxFactory const&>(mmintgmf[idim]->Factory());

        const MultiFab&    vfrac = my_factory.getVolFrac();
        const MultiCutFab& bcent = my_factory.getBndryCent();
        const MultiCutFab& bnorm = my_factory.getBndryNormal();
        const auto&        flags = my_factory.getMultiEBCellFlagFab();

        MFItInfo mfi_info;
        if (Gpu::notInLaunchRegion()) mfi_info.EnableTiling().SetDynamic(true);

#ifdef AMREX_USE_OMP
#pragma omp parallel if(Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(*mmintgmf[idim],mfi_info); mfi.isValid(); ++mfi)
        {
            const Box& bx = mfi.growntilebox(nghost[idim]);
            Array4<Real> const& mmintg = mmintgmf[idim]->array(mfi);

            const auto& flagfab = flags[mfi];
            auto typ = flagfab.getType(bx);

            if (typ == FabType::covered || typ == FabType::regular)
            {
                auto const& fg = flagfab.array();
                AMREX_HOST_DEVICE_FOR_4D ( bx, numMmIntgs, i, j, k, n,
                {
                   mmintg(i,j,k,n) = 0.0;
                });
            }
            else
            {
                // auto const& vf = vfrac.array(mfi);
                auto const& bc = bcent.array(mfi);
                auto const& bn = bnorm.array(mfi);
                auto const& fg = flagfab.array();

                if (Gpu::inLaunchRegion())
                {
                    amrex::ParallelFor(bx,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                    {
                        const auto ebflag = fg(i,j,k);
                        if(idim == 0) {
                            compute_mmintg_on_yz_face(i,j,k,mmintg,bc,bn,fg);
                        } else if (idim == 1) {
                            compute_mmintg_on_xz_face(i,j,k,mmintg,bc,bn,fg);
                        }
                    });
                }
                else
                {
                    const auto lo = amrex::lbound(bx);
                    const auto hi = amrex::ubound(bx);
                    for (int k = lo.z; k <= hi.z; ++k)
                    for (int j = lo.y; j <= hi.y; ++j)
                    for (int i = lo.x; i <= hi.x; ++i)
                    {
                        const auto ebflag = fg(i,j,k);
                        if(idim == 0) {
                            compute_mmintg_on_yz_face(i,j,k,mmintg,bc,bn,fg);
                        } else if (idim == 1) {
                            compute_mmintg_on_xz_face(i,j,k,mmintg,bc,bn,fg);
                        } else if (idim == 2) {
                            compute_mmintg_on_xy_face(i,j,k,mmintg,bc,bn,fg);
                        }

                    }
                }
            }
        }
    }
#endif
}

}

