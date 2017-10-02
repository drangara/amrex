
#include <AMReX_EBTower.H>
#include <AMReX_EBISLevel.H>
#include <AMReX_EBLevelGrid.H>

#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace amrex {

EBTower* EBTower::m_instance = nullptr;

void
EBTower::Build ()
{
    if (!m_instance) {
        m_instance = new EBTower();
    }
}

void
EBTower::Destroy ()
{
    delete m_instance;
}

EBTower::EBTower ()
{
    BL_PROFILE("EBTower::EBTower()");

    const EBIndexSpace* ebis = AMReX_EBIS::instance();

    m_domains = ebis->getDomains();

    const int nlevels = m_domains.size();

    m_irregular_ba.resize(nlevels);
    m_covered_ba.resize(nlevels);

    for (int lev = 0; lev < nlevels; ++lev)
    {
        const auto& ebisl = ebis->getEBISLevel(lev);
        const auto& graph = ebisl.getGraph();

        Array<Box> covb;
        Array<Box> irrb;

        for (MFIter mfi(graph); mfi.isValid(); ++mfi)
        {
            const auto& g = graph[mfi];
            if (g.hasIrregular()) {
                irrb.push_back(mfi.validbox());
            } else if (g.isAllCovered()) {
                covb.push_back(mfi.validbox());
            }
        }

        amrex::AllGatherBoxes(covb);
        amrex::AllGatherBoxes(irrb);

        if (covb.size() > 0) {
            m_covered_ba[lev].define(BoxList{std::move(covb)});
        }

        if (irrb.size() > 0)
        {
            m_irregular_ba[lev].define(BoxList{std::move(irrb)});
        }
    }

    m_cellflags.resize(nlevels);
    m_volfrac.resize(nlevels);
    m_bndrycent.resize(nlevels);
    m_areafrac.resize(nlevels);
    m_facecent.resize(nlevels);

    for (int lev = 0; lev < nlevels; ++lev)
    {
        if (!m_irregular_ba[lev].empty()) 
        {
            const BoxArray& ba = m_irregular_ba[lev];
            DistributionMapping dm {ba};
            EBLevelGrid eblg(ba, dm, m_domains[lev], 1);

            m_cellflags[lev].define(ba, dm, 1, 0);
            initCellFlags(lev, eblg);

            m_volfrac[lev].define(ba, dm, 1, 0);
            initVolFrac(lev, eblg);

            m_bndrycent[lev].define(ba, dm, 3, 0, m_cellflags[lev]);
            initBndryCent(lev, eblg);
            
            for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
                const BoxArray& faceba = amrex::convert(ba, IntVect::TheDimensionVector(idim));
                m_areafrac[lev][idim].define(faceba, dm, 1, 0, m_cellflags[lev]);
                m_facecent[lev][idim].define(faceba, dm, 3, 0, m_cellflags[lev]);
            }
            initFaceGeometry(lev, eblg);
        }
    }
}

EBTower::~EBTower ()
{
}

void
EBTower::initCellFlags (int lev, const EBLevelGrid& eblg)
{
    FabArray<EBCellFlagFab>& ebcf = m_cellflags[lev];
    const auto& ebisl = eblg.getEBISL();

#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter mfi(ebcf); mfi.isValid(); ++mfi)
    {
        auto& fab = ebcf[mfi];
        const EBISBox& ebis = ebisl[mfi];
        fab.copy(ebis.getEBGraph().getEBCellFlagFab());
        fab.setType(ebis.getEBGraph().getEBCellFlagFab().getType());
        if (fab.getType() == FabType::multivalued) {
            amrex::Abort("EBTower::initCellFlags: Multi-valued cells not supported");
        }
    }
}

void
EBTower::initVolFrac (int lev, const EBLevelGrid& eblg)
{
    MultiFab& volfrac = m_volfrac[lev];
    const auto& ebisl = eblg.getEBISL();

#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter mfi(volfrac,true); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.tilebox();
        auto& fab = volfrac[mfi];
        fab.setVal(1.0, bx);

        const EBISBox& ebisbox = ebisl[mfi];
        
        for (BoxIterator bi(bx); bi.ok(); ++bi)
        {
            const IntVect& iv = bi();
            const auto& vofs = ebisbox.getVoFs(iv);
            Real vtot = 0.0;
            for (const auto& vi : vofs)
            {
                vtot += ebisbox.volFrac(vi);
            }
            fab(iv) = vtot;
        }
    }
}

void
EBTower::initBndryCent (int lev, const EBLevelGrid& eblg)
{
    auto& bndrycent = m_bndrycent[lev];
    const auto& ebisl = eblg.getEBISL();

#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter mfi(bndrycent.data(),true); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.tilebox();
        auto& fab = bndrycent[mfi];
        fab.setVal(-1.0, bx);

        const EBISBox& ebisbox = ebisl[mfi];

        for (BoxIterator bi(bx); bi.ok(); ++bi)
        {
            const IntVect& iv = bi();
            const auto& vofs = ebisbox.getVoFs(iv);
            for (const auto& vi : vofs)
            {
                const auto& bcent = ebisbox.bndryCentroid(vi);
                for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
                    fab(iv,idim) = bcent[idim];
                }
            }
        }
    }
}

void
EBTower::initFaceGeometry (int lev, const EBLevelGrid& eblg)
{
    auto& areafrac = m_areafrac[lev];
    auto& facecent = m_facecent[lev];

    const auto& ebisl = eblg.getEBISL();

#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter mfi(m_cellflags[lev]); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.validbox();

        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            areafrac[idim][mfi].setVal(1.0);
            facecent[idim][mfi].setVal(0.0);
        }

        const auto& ebisbox = ebisl[mfi];

        for (BoxIterator bi(bx); bi.ok(); ++bi)
        {
            const IntVect& iv = bi();
            for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
            {
                {
                    const auto& lo_faces = ebisbox.getAllFaces(iv, idim, Side::Lo);
                    if (lo_faces.size() == 0) {
                        areafrac[idim][mfi](iv) = 0.0;
                    } else if (lo_faces.size() == 1) {
                        areafrac[idim][mfi](iv) = ebisbox.areaFrac(lo_faces[0]);
                        const RealVect& rv = ebisbox.centroid(lo_faces[0]);
                        int icomp = 0;
                        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
                            if (dir != idim) {
                                facecent[idim][mfi](iv,icomp) = rv[dir];
                                ++icomp;
                            }
                        }
                    } else {
                        amrex::Abort("EBTower: multi-valued face not supported");
                    }
                }
                
                if (iv[idim] == bx.bigEnd(idim))
                {
                    const IntVect& ivhi = iv + IntVect::TheDimensionVector(idim);
                    const auto& hi_faces = ebisbox.getAllFaces(iv, idim, Side::Hi);
                    if (hi_faces.size() == 0) {
                        areafrac[idim][mfi](ivhi) = 0.0;
                    } else if (hi_faces.size() == 1) {
                        areafrac[idim][mfi](ivhi) = ebisbox.areaFrac(hi_faces[0]);
                        const RealVect& rv = ebisbox.centroid(hi_faces[0]);
                        int icomp = 0;
                        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
                            if (dir != idim) {
                                facecent[idim][mfi](ivhi,icomp) = rv[dir];
                                ++icomp;
                            }
                        }
                    } else {
                        amrex::Abort("EBTower: multi-valued face not supported");
                    }                        
                }
            }
        }
    }
}

int
EBTower::getIndex (const Box& domain) const
{
    auto bx_it = std::find(m_domains.begin(), m_domains.end(), domain);
    AMREX_ALWAYS_ASSERT(bx_it != m_domains.end());
    return std::distance(m_domains.begin(), bx_it);
}

void
EBTower::fillEBCellFlag (FabArray<EBCellFlagFab>& a_flag, const Geometry& a_geom)
{
    BL_PROFILE("EBTower::fillEBCellFlag()");

    const Box& domain = a_geom.Domain();

    int lev = m_instance->getIndex(domain);

    const auto& src_flag = m_instance->m_cellflags[lev];
    a_flag.ParallelCopy(src_flag, 0, 0, 1, 0, a_flag.nGrow());

    const BoxArray& cov_ba = m_instance->m_covered_ba[lev];
    auto cov_val = EBCellFlag::TheCoveredCell();
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        std::vector<std::pair<int,Box> > isects;
        for (MFIter mfi(a_flag); mfi.isValid(); ++mfi)
        {
            auto& fab = a_flag[mfi];
            const Box& bx = fab.box() & domain;

            // covered cells
            cov_ba.intersections(bx, isects);
            for (const auto& is : isects) {
                fab.setVal(cov_val, is.second);
            }

            // fix type and region for each fab
            fab.setRegion(bx);
            fab.setType(FabType::undefined);
            auto typ = fab.getType(bx);
            fab.setType(typ);
        }
    }
}

void
EBTower::fillVolFrac (MultiFab& a_volfrac, const Geometry& a_geom)
{
    BL_PROFILE("EBTower::fillVolFrac()");
    
    const Box& domain = a_geom.Domain();

    int lev = m_instance->getIndex(domain);

    const auto& src_volfrac = m_instance->m_volfrac[lev];

    a_volfrac.setVal(1.0);
    a_volfrac.ParallelCopy(src_volfrac, 0, 0, 1, 0, a_volfrac.nGrow());

    const BoxArray& cov_ba = m_instance->m_covered_ba[lev];
    Real cov_val = 0.0;
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        std::vector<std::pair<int,Box> > isects;
        for (MFIter mfi(a_volfrac); mfi.isValid(); ++mfi)
        {
            auto& fab = a_volfrac[mfi];
            const Box& bx = fab.box() & domain;

            // covered cells
            cov_ba.intersections(bx, isects);
            for (const auto& is : isects) {
                fab.setVal(cov_val, is.second);
            }
        }
    }
}

void
EBTower::fillBndryCent (MultiCutFab& a_bndrycent, const Geometry& a_geom)
{
    BL_PROFILE("EBTower::fillBndryCent()");

    const Box& domain = a_geom.Domain();

    int lev = m_instance->getIndex(domain);

    const auto& src_bndrycent = m_instance->m_bndrycent[lev];

    a_bndrycent.setVal(-1.0);

    a_bndrycent.ParallelCopy(src_bndrycent, 0, 0, a_bndrycent.nComp(), 0, a_bndrycent.nGrow());
}

void
EBTower::fillFaceGeometry (std::array<MultiCutFab*,AMREX_SPACEDIM>& a_areafrac,
                           std::array<MultiCutFab*,AMREX_SPACEDIM>& a_facecent,
                           const Geometry& a_geom)
{
    BL_PROFILE("EBTower::fillFaceGeometry()");

    const Box& domain = a_geom.Domain();

    int lev = m_instance->getIndex(domain);

    const auto& src_areafrac = m_instance->m_areafrac[lev];
    const auto& src_facecent = m_instance->m_facecent[lev];

    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        a_areafrac[idim]->setVal(1.0);
        a_facecent[idim]->setVal(0.0);
        a_areafrac[idim]->ParallelCopy(src_areafrac[idim], 0, 0, a_areafrac[idim]->nComp(),
                                       0, a_areafrac[idim]->nGrow());
        a_facecent[idim]->ParallelCopy(src_facecent[idim], 0, 0, a_facecent[idim]->nComp(),
                                       0, a_facecent[idim]->nGrow());
    }

    // fix area fraction for covered cells.  As for face centroid, we don't care.
    const BoxArray& cov_ba = m_instance->m_covered_ba[lev];
    Real cov_val = 0.0;
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        std::vector<std::pair<int,Box> > isects;
        for (MFIter mfi(a_areafrac[0]->data()); mfi.isValid(); ++mfi)
        {
            if (a_areafrac[0]->ok(mfi))
            {
                const Box& ccbx = amrex::enclosedCells((*a_areafrac[0])[mfi].box()) & domain;
                
                // covered cells
                cov_ba.intersections(ccbx, isects);
                for (const auto& is : isects) {
                    for (int idim=0; idim<AMREX_SPACEDIM; ++idim) {
                        (*a_areafrac[idim])[mfi].setVal(cov_val,
                                                        amrex::surroundingNodes(is.second,idim));
                    }
                }
            }
        }
    }    
}

}
