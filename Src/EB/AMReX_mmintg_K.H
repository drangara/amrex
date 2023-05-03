#ifndef AMREX_MMINTG_K_H_
#define AMREX_MMINTG_K_H_

#include <AMReX_mmintg.H>

namespace amrex::mismatched_intg {

struct EBTriangle {
    GpuArray<Real, 3> x;
    GpuArray<Real, 3> y;

    bool sign;
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void order_triangle (EBTriangle& t, Real nx, Real ny) noexcept
{
    bool rev_order = false;
    bool sign = true;

    Real x1 = t.x[0];
    Real y1 = t.y[0];

    Real x2 = t.x[1];
    Real y2 = t.y[1];

    Real x3 = t.x[2];
    Real y3 = t.y[2];


    if (x1 == x2 && x1 == -0.5_rt) {
        if (y1 < y2) {
            rev_order = true;
            if (ny > 0._rt) {
                sign = false;
            }
        } else {
            if (ny < 0._rt) {
                sign = false;
            }
        }
    } else if (x1 == x2 && x1 == 0.5_rt) {
        if (y1 > y2) {
            rev_order = true;
            if (ny < 0._rt) {
                sign = false;
            }
        } else {
            if (ny > 0._rt) {
                sign = false;
            }
        }
    } else if (y1 == y2 && y1 == -0.5_rt) {
        if (x1 > x2) {
            rev_order = true;
            if (nx < 0._rt) {
                sign = false;
            }
        } else {
            if (nx > 0._rt) {
                sign = false;
            }
        }

    } else if (y1 == y2 && y1 == 0.5_rt) {
        if (x1 < x2) {
            rev_order = true;
            if (nx > 0._rt) {
                sign = false;
            }
        } else {
            if (nx < 0._rt) {
                sign = false;
            }
        }
    }

    if (rev_order) {
        std::swap(t.x[0],t.x[1]);
        std::swap(t.y[0],t.y[1]);
    }

    t.sign = sign;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void add_to_mmintg (int i, int j, int k,
                    Array4<Real> const& mmintg,
                    const EBTriangle& t) noexcept
{
    Real x1 = t.x[0];
    Real x2 = t.x[1];
    Real x3 = t.x[2];

    Real y1 = t.y[0];
    Real y2 = t.y[1];
    Real y3 = t.y[2];

    Real sign = t.sign ? 1._rt : -1._rt;

    Real det = (x2 - x1)*(y3 - y1) - (x3 - x1)*(y2 - y1);
    mmintg(i,j,k, i_M_1)  += sign* 0.5_rt*det;
    mmintg(i,j,k, i_M_x)  += sign*det*(x1 + x2 + x3)/6.0_rt;
    mmintg(i,j,k, i_M_y)  += sign*det*(y1 + y2 + y3)/6.0_rt;
    mmintg(i,j,k, i_M_xy) += sign*det*((x1*y1)/2.0_rt
                            + (x1*(y2 - y1) + y1*(x2 - x1))/6.0_rt
                            + (x1*(y3 - y1) + y1*(x3 - x1))/6.0_rt
                            + ((x2 - x1)*(y3 - y1) + (x3 - x1)*(y2 - y1))/24.0_rt
                            + ((x2 - x1)*(y2 - y1))/12.0_rt
                            + ((x3 - x1)*(y3 - y1))/12.0_rt);

}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void extract_triangles_on_face (GpuArray<EBTriangle, 4>& triangles, int& nt,
                                GpuArray<Array <Real,2>, 4> const& lo_ipts, int lo_np,
                                GpuArray<Array <Real,2>, 4> const& hi_ipts, int hi_np,
                                Real lo_nx, Real lo_ny) noexcept
{
    AMREX_ASSERT(lo_np == 2 && hi_np == 2);

    Real x1 = lo_ipts[0][0];
    Real x2 = lo_ipts[1][0];
    Real y1 = lo_ipts[0][1];
    Real y2 = lo_ipts[1][1];

    Real x3 = hi_ipts[0][0];
    Real x4 = hi_ipts[1][0];
    Real y3 = hi_ipts[0][1];
    Real y4 = hi_ipts[1][1];

    if (x2 == x1) { return; }
    if (x4 == x3) { return; }

    Real alo = (y2 - y1)/(x2 - x1);
    Real blo = y1 - alo*x1;

    Real ahi = (y4 - y3)/(x4 - x3);
    Real bhi = y3 - ahi*x3;

    if (alo == ahi) { return; }

    Real xi = -(blo - bhi)/(alo - ahi);
    Real yi = alo*xi + blo;

    if (xi <= 0.5_rt && xi >= -0.5_rt && yi <= 0.5_rt && yi >= -0.5_rt) {

        bool p1 = false;
        bool p2 = false;
        bool p3 = false;
        bool p4 = false;

        if (    ((x1 == x3) && (std::abs(x1) == 0.5_rt))
             || ((y1 == y3) && (std::abs(y1) == 0.5_rt))) {
            EBTriangle t;
            t.x = {x1, x3, xi};
            t.y = {y1, y3, yi};
            order_triangle(t, lo_nx, lo_ny);
            triangles[nt++] = t;
            p1 = true;
            p3 = true;
        }

        if (   ((x2 == x4) && (std::abs(x2) == 0.5_rt))
            || ((y2 == y4) && (std::abs(y2) == 0.5_rt))) {
            EBTriangle t;
            t.x = {x2, x4, xi};
            t.y = {y2, y4, yi};
            order_triangle(t, lo_nx, lo_ny);
            triangles[nt++] = t;
            p2 = true;
            p4 = true;
        }

        if (   ((x2 == x3) && (std::abs(x2) == 0.5_rt))
             ||((y2 == y3) && (std::abs(y2) == 0.5_rt))) {
            EBTriangle t;
            t.x = {x2, x3, xi};
            t.y = {y2, y3, yi};
            order_triangle(t, lo_nx, lo_ny);
            triangles[nt++] = t;
            p2 = true;
            p3 = true;
        }

        if (   ((x1 == x4) && (std::abs(x1) == 0.5_rt))
            || ((y1 == y4) && (std::abs(y1) == 0.5_rt))) {
            EBTriangle t;
            t.x = {x1, x4, xi};
            t.y = {y1, y4, yi};
            order_triangle(t, lo_nx, lo_ny);
            triangles[nt++] = t;
            p1 = true;
            p4 = true;
        }

        if (y2 == -0.5_rt && x3 == -0.5_rt && !p2 && !p3) {
            EBTriangle t1;
            t1.x = {x2, -0.5_rt, xi};
            t1.y = {y2, -0.5_rt, yi};
            order_triangle(t1, lo_nx, lo_ny);
            triangles[nt++] = t1;

            EBTriangle t2;
            t2.x = {x3, -0.5_rt, xi};
            t2.y = {y3, -0.5_rt, yi};
            order_triangle(t2, lo_nx, lo_ny);
            t2.sign = t1.sign;
            triangles[nt++] = t2;

            p2 = true;
            p3 = true;
        }

        if (y2 == 0.5_rt && x3 == -0.5_rt && !p2 && !p3) {
            EBTriangle t1;
            t1.x = {x2, -0.5_rt, xi};
            t1.y = {y2,  0.5_rt, yi};
            order_triangle(t1, lo_nx, lo_ny);
            triangles[nt++] = t1;

            EBTriangle t2;
            t2.x = {x3, -0.5_rt, xi};
            t2.y = {y3,  0.5_rt, yi};
            order_triangle(t2, lo_nx, lo_ny);
            t2.sign = t1.sign;
            triangles[nt++] = t2;

            p2 = true;
            p3 = true;
        }

        if (y2 == -0.5_rt && x4 == 0.5_rt && !p2 && !p4) {
            EBTriangle t1;
            t1.x = {x2,  0.5_rt, xi};
            t1.y = {y2, -0.5_rt, yi};
            order_triangle(t1, lo_nx, lo_ny);
            triangles[nt++] = t1;

            EBTriangle t2;
            t2.x = {x4,  0.5_rt, xi};
            t2.y = {y4, -0.5_rt, yi};
            order_triangle(t2, lo_nx, lo_ny);
            t2.sign = t1.sign;
            triangles[nt++] = t2;

            p2 = true;
            p4 = true;
        }

        if (x2 ==  0.5_rt && y4 == -0.5_rt && !p2 && !p4) {
            EBTriangle t1;
            t1.x = {x2,  0.5_rt, xi};
            t1.y = {y2, -0.5_rt, yi};
            order_triangle(t1, lo_nx, lo_ny);
            triangles[nt++] = t1;

            EBTriangle t2;
            t2.x = {x4,  0.5_rt, xi};
            t2.y = {y4, -0.5_rt, yi};
            order_triangle(t2, lo_nx, lo_ny);
            t2.sign = t1.sign;
            triangles[nt++] = t2;

            p2 = true;
            p4 = true;
        }

        if (x1 ==  -0.5_rt && y4 == -0.5_rt && !p1 && !p4) {
            EBTriangle t1;
            t1.x = {x1, -0.5_rt, xi};
            t1.y = {y1, -0.5_rt, yi};
            order_triangle(t1, lo_nx, lo_ny);
            triangles[nt++] = t1;

            EBTriangle t2;
            t2.x = {x4, -0.5_rt, xi};
            t2.y = {y4, -0.5_rt, yi};
            order_triangle(t2, lo_nx, lo_ny);
            t2.sign = t1.sign;
            triangles[nt++] = t2;

            p1 = true;
            p4 = true;
        }

        if (x1 ==  -0.5_rt && y4 == 0.5_rt && !p1 && !p4) {
            EBTriangle t1;
            t1.x = {x1, -0.5_rt, xi};
            t1.y = {y1,  0.5_rt, yi};
            order_triangle(t1, lo_nx, lo_ny);
            triangles[nt++] = t1;

            EBTriangle t2;
            t2.x = {x4, -0.5_rt, xi};
            t2.y = {y4,  0.5_rt, yi};
            order_triangle(t2, lo_nx, lo_ny);
            t2.sign = t1.sign;
            triangles[nt++] = t2;

            p1 = true;
            p4 = true;
        }

        if (y2 == 0.5_rt && x4 == 0.5_rt && !p2 && !p4) {
            EBTriangle t1;
            t1.x = {x2, 0.5_rt, xi};
            t1.y = {y2, 0.5_rt, yi};
            order_triangle(t1, lo_nx, lo_ny);
            triangles[nt++] = t1;

            EBTriangle t2;
            t2.x = {x4, 0.5_rt, xi};
            t2.y = {y4, 0.5_rt, yi};
            order_triangle(t2, lo_nx, lo_ny);
            t2.sign = t1.sign;
            triangles[nt++] = t2;

            p2 = true;
            p4 = true;
        }

        if (x2 ==  0.5_rt && y4 == 0.5_rt && !p2 && !p4) {
            EBTriangle t1;
            t1.x = {x2, 0.5_rt, xi};
            t1.y = {y2, 0.5_rt, yi};
            order_triangle(t1, lo_nx, lo_ny);
            triangles[nt++] = t1;

            EBTriangle t2;
            t2.x = {x4, 0.5_rt, xi};
            t2.y = {y4, 0.5_rt, yi};
            order_triangle(t2, lo_nx, lo_ny);
            t2.sign = t1.sign;
            triangles[nt++] = t2;

            p2 = true;
            p4 = true;
        }
    }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void compute_mmintg_on_yz_face(int i, int j, int k,
                               Array4<Real> const& mmintg_x,
                               Array4<Real const> const& bcent,
                               Array4<Real const> const& bnorm,
                               Array4<EBCellFlag const> const& flag) noexcept
{
    mmintg_x(i,j,k, i_M_1)  = 0._rt;
    mmintg_x(i,j,k, i_M_x)  = 0._rt;
    mmintg_x(i,j,k, i_M_y)  = 0._rt;
    mmintg_x(i,j,k, i_M_xy) = 0._rt;

    if (flag(i-1,j,k).isRegular() || flag(i-1,j,k).isCovered()) {
        return;
    }

    // lo side x =  0.5
    // hi side x = -0.5
    //
    Real lo_Cx = bcent(i-1,j,k,0);
    Real lo_Cy = bcent(i-1,j,k,1);
    Real lo_Cz = bcent(i-1,j,k,2);
    Real lo_nx = bnorm(i-1,j,k,0);
    Real lo_ny = bnorm(i-1,j,k,1);
    Real lo_nz = bnorm(i-1,j,k,2);

    Real hi_Cx = bcent(i,j,k,0);
    Real hi_Cy = bcent(i,j,k,1);
    Real hi_Cz = bcent(i,j,k,2);
    Real hi_nx = bnorm(i,j,k,0);
    Real hi_ny = bnorm(i,j,k,1);
    Real hi_nz = bnorm(i,j,k,2);

    GpuArray<Array<Real,2>, 4> lo_ipts;
    GpuArray<Array<Real,2>, 4> hi_ipts;
    int lo_np = 0;
    int hi_np = 0;

    // y = -0.5
    {
        Real lo_z = lo_Cz -(( 0.5_rt-lo_Cx)*lo_nx + (-0.5_rt-lo_Cy)*lo_ny)/lo_nz;
        Real hi_z = hi_Cz -((-0.5_rt-hi_Cx)*hi_nx + (-0.5_rt-hi_Cy)*hi_ny)/hi_nz;

        if (lo_z <= 0.5_rt && lo_z >= -0.5_rt) {lo_ipts[lo_np++] = {-0.5_rt,lo_z};}
        if (hi_z <= 0.5_rt && hi_z >= -0.5_rt) {hi_ipts[hi_np++] = {-0.5_rt,hi_z};}
    }

    // y = 0.5
    {
        Real lo_z = lo_Cz -(( 0.5_rt-lo_Cx)*lo_nx + (0.5_rt-lo_Cy)*lo_ny)/lo_nz;
        Real hi_z = hi_Cz -((-0.5_rt-hi_Cx)*hi_nx + (0.5_rt-hi_Cy)*hi_ny)/hi_nz;

        if (lo_z <= 0.5_rt && lo_z >= -0.5_rt) {lo_ipts[lo_np++] = {0.5_rt,lo_z};}
        if (hi_z <= 0.5_rt && hi_z >= -0.5_rt) {hi_ipts[hi_np++] = {0.5_rt,hi_z};}
    }

    // z = -0.5
    {
        Real lo_y = lo_Cy -(( 0.5_rt-lo_Cx)*lo_nx + (-0.5_rt-lo_Cz)*lo_nz)/lo_ny;
        Real hi_y = hi_Cy -((-0.5_rt-hi_Cx)*hi_nx + (-0.5_rt-hi_Cz)*hi_nz)/hi_ny;

        if (lo_y <= 0.5_rt && lo_y >= -0.5_rt) {lo_ipts[lo_np++] = {lo_y,-0.5_rt};}
        if (hi_y <= 0.5_rt && hi_y >= -0.5_rt) {hi_ipts[hi_np++] = {hi_y,-0.5_rt};}
    }

    // z = 0.5
    {
        Real lo_y = lo_Cy -(( 0.5_rt-lo_Cx)*lo_nx + (0.5_rt-lo_Cz)*lo_nz)/lo_ny;
        Real hi_y = hi_Cy -((-0.5_rt-hi_Cx)*hi_nx + (0.5_rt-hi_Cz)*hi_nz)/hi_ny;

        if (lo_y <= 0.5_rt && lo_y >= -0.5_rt) {lo_ipts[lo_np++] = {lo_y,0.5_rt};}
        if (hi_y <= 0.5_rt && hi_y >= -0.5_rt) {hi_ipts[hi_np++] = {hi_y,0.5_rt};}
    }

    GpuArray<EBTriangle, 4> triangles;
    int nt = 0;

    if (lo_np == 2 && hi_np == 2) {
        extract_triangles_on_face(triangles, nt, lo_ipts, lo_np, hi_ipts, hi_np,
                                  lo_ny, lo_nz);
    }

    for (int n = 0; n < nt; ++n) {
        add_to_mmintg(i,j,k,mmintg_x,triangles[n]);
    }
}

}

#endif
