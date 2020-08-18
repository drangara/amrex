
#include <AMReX.H>
#include "PoiseuilleTest.H"

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);

    {
        BL_PROFILE("main");
        PoiseuilleTest mytest;

        mytest.compute_gradient();
    }

    amrex::Finalize();
}
