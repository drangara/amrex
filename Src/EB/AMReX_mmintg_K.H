#ifndef AMREX_MMINTG_K_H_
#define AMREX_MMINTG_K_H_

#include <AMReX_mmintg.H>

namespace amrex::mismatched_intg {

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void swap_vars (Real& a, Real& b) noexcept {
    Real tmp = a;
    a = b;
    b = tmp;
}

// Order of points implicity matter.
// The first point must lie on lo face.
struct EBFaceTriangle {
    GpuArray<Real, 3> x;
    GpuArray<Real, 3> y;

    bool sign;
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void sanitize_ipts (GpuArray<Array <Real,2>, 4>& ipts, int& np) noexcept {
    Real eps = 100*std::numeric_limits<Real>::epsilon();

    for (int n = 0; n < np; ++n) {
        for (int c = 0; c < 2; ++c) {
            if (std::abs(ipts[n][c] - (-0.5_rt)) <= eps) {
                ipts[n][c] = -0.5_rt;
            } else if (std::abs(ipts[n][c] - 0.5_rt) <= eps) {
                ipts[n][c] = 0.5_rt;
            }
        }
    }

    // Remove duplicates
    if (np == 3) {
        // More relaxed tolerance
        Real epsd = 1.e-12;

        if (   std::abs(ipts[0][0] - ipts[1][0]) <= epsd
            && std::abs(ipts[0][1] - ipts[1][1]) <= epsd) {
            ipts[1] = {ipts[2][0], ipts[2][1]};
            --np;
        } else if (   std::abs(ipts[1][0] - ipts[2][0]) <= epsd
                   && std::abs(ipts[1][1] - ipts[2][1]) <= epsd) {
            --np;
        } else if (   std::abs(ipts[0][0] - ipts[2][0]) <= epsd
                   && std::abs(ipts[0][1] - ipts[2][1]) <= epsd) {
            --np;
        }
    }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void order_triangle (EBFaceTriangle& t, Real nx, Real ny) noexcept
{
    bool rev_order = false;
    bool sign = true;

    Real x1 = t.x[0];
    Real y1 = t.y[0];

    Real x2 = t.x[1];
    Real y2 = t.y[1];

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
        swap_vars(t.x[0],t.x[1]);
        swap_vars(t.y[0],t.y[1]);
    }

    t.sign = sign;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void add_to_mmintg (int i, int j, int k,
                    Array4<Real> const& mmintg,
                    const EBFaceTriangle& t) noexcept
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

//TODO: try to consolidate into "extract_triangles_on_face"
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void extract_triangle_on_face_hi (GpuArray<EBFaceTriangle, 4>& triangles, int& nt,
                                  GpuArray<Array <Real,2>, 4> const& ipts, int np,
                                  bool sign) noexcept
{
    AMREX_ALWAYS_ASSERT(np == 2);

    Real x1 = ipts[0][0];
    Real x2 = ipts[1][0];
    Real y1 = ipts[0][1];
    Real y2 = ipts[1][1];

    EBFaceTriangle t;
    if (x1 == -0.5_rt && y2 == 0.5_rt) {
        t.x = {x1, x2, -0.5_rt};
        t.y = {y1, y2,  0.5_rt};
    } else if (x1 == 0.5_rt && y2 == 0.5_rt) {
        t.x = {x2, x1, 0.5_rt};
        t.y = {y2, y1, 0.5_rt};
    } else if (x1 == -0.5_rt && y2 == -0.5_rt) {
        t.x = {x1, -0.5_rt, x2};
        t.y = {y1, -0.5_rt, y2};
    } else if (x1 == 0.5_rt && y2 == -0.5_rt) {
        t.x = {x1, x2,  0.5_rt};
        t.y = {y1, y2, -0.5_rt};
    }

    t.sign = sign;
    triangles[nt++] = t;
}

//TODO: try to consolidate into "extract_triangles_on_face"
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void extract_triangle_on_face_lo (GpuArray<EBFaceTriangle, 4>& triangles, int& nt,
                                  GpuArray<Array <Real,2>, 4> const& ipts, int np,
                                  Real nx, Real ny) noexcept
{
    AMREX_ALWAYS_ASSERT(np == 2);
    ignore_unused(ny);

    Real x1 = ipts[0][0];
    Real x2 = ipts[1][0];
    Real y1 = ipts[0][1];
    Real y2 = ipts[1][1];

    bool sign = true;

    EBFaceTriangle t;
    if (x1 == -0.5_rt && y2 == 0.5_rt) {
        t.x = {x1, x2, -0.5_rt};
        t.y = {y1, y2,  0.5_rt};
        if (nx < 0._rt) {
            sign = false;
        }
    } else if (x1 == 0.5_rt && y2 == 0.5_rt) {
        t.x = {x2, x1, 0.5_rt};
        t.y = {y2, y1, 0.5_rt};
        if (nx > 0._rt) {
            sign = false;
        }
    } else if (x1 == -0.5_rt && y2 == -0.5_rt) {
        t.x = {x1, -0.5_rt, x2};
        t.y = {y1, -0.5_rt, y2};
        if (nx < 0._rt) {
            sign = false;
        }
    } else if (x1 == 0.5_rt && y2 == -0.5_rt) {
        t.x = {x1, x2,  0.5_rt};
        t.y = {y1, y2, -0.5_rt};
        if (nx > 0._rt) {
            sign = false;
        }
    }

    t.sign = sign;
    triangles[nt++] = t;
}

//TODO: Improve this routine. Too many special cases!!
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void extract_triangles_on_face (GpuArray<EBFaceTriangle, 4>& triangles, int& nt,
                                GpuArray<Array <Real,2>, 4> const& lo_ipts, int lo_np,
                                GpuArray<Array <Real,2>, 4> const& hi_ipts, int hi_np,
                                Real lo_nx, Real lo_ny, Real hi_nx, Real hi_ny) noexcept
{
    AMREX_ALWAYS_ASSERT(lo_np == 2 && hi_np == 2);

    Real x1 = lo_ipts[0][0];
    Real x2 = lo_ipts[1][0];
    Real y1 = lo_ipts[0][1];
    Real y2 = lo_ipts[1][1];

    Real x3 = hi_ipts[0][0];
    Real x4 = hi_ipts[1][0];
    Real y3 = hi_ipts[0][1];
    Real y4 = hi_ipts[1][1];

    Real eps = 100*std::numeric_limits<Real>::epsilon();
    if ( (std::abs(x1 - x3) <= eps) &&
         (std::abs(y1 - y3) <= eps) &&
         (std::abs(x2 - x4) <= eps) &&
         (std::abs(y2 - y4) <= eps)) {
        return;
    }
    if ( (std::abs(x1 - x4) <= eps) &&
         (std::abs(y1 - y4) <= eps) &&
         (std::abs(x2 - x3) <= eps) &&
         (std::abs(y2 - y3) <= eps)) {
        return;
    }

    Real xi = 1.0_rt;
    Real yi = 1.0_rt;

    if ((std::abs(x1 - x2) <= eps) && (std::abs(x3 - x4) > eps)) {
        Real ahi = (y4 - y3)/(x4 - x3);
        Real bhi = y3 - ahi*x3;

        xi = x1;
        yi = ahi*xi + bhi;

    } else if ((std::abs(x1 - x2) > eps) && (std::abs(x3 - x4) <= eps)) {
        Real alo = (y2 - y1)/(x2 - x1);
        Real blo = y1 - alo*x1;

        xi = x3;
        yi = alo*xi + blo;

    } else if ((std::abs(x1 - x2) > eps) && (std::abs(x3 - x4) > eps)) {
        Real alo = (y2 - y1)/(x2 - x1);
        Real blo = y1 - alo*x1;

        Real ahi = (y4 - y3)/(x4 - x3);
        Real bhi = y3 - ahi*x3;

        if (alo != ahi) {
            xi = -(blo - bhi)/(alo - ahi);
        }

        yi = alo*xi + blo;

    } else {
        return;
    }

    if (xi <= 0.5_rt && xi >= -0.5_rt && yi <= 0.5_rt && yi >= -0.5_rt) {

        bool p1 = false;
        bool p2 = false;
        bool p3 = false;
        bool p4 = false;

        // Triangles
        if (    ((x1 == x3) && (std::abs(x1) == 0.5_rt) &&
                 (lo_ny*hi_ny >= 0._rt))
             || ((y1 == y3) && (std::abs(y1) == 0.5_rt) &&
                 (lo_nx*hi_nx >= 0._rt))) {
            EBFaceTriangle t;
            t.x = {x1, x3, xi};
            t.y = {y1, y3, yi};
            order_triangle(t, lo_nx, lo_ny);
            triangles[nt++] = t;
            p1 = true;
            p3 = true;
        }

        if (   ((x2 == x4) && (std::abs(x2) == 0.5_rt) &&
                (lo_ny*hi_ny >= 0._rt))
            || ((y2 == y4) && (std::abs(y2) == 0.5_rt) &&
                (lo_nx*hi_nx >= 0._rt))) {
            EBFaceTriangle t;
            t.x = {x2, x4, xi};
            t.y = {y2, y4, yi};
            order_triangle(t, lo_nx, lo_ny);
            triangles[nt++] = t;
            p2 = true;
            p4 = true;
        }

        if (   ((x2 == x3) && (std::abs(x2) == 0.5_rt) &&
                (lo_ny*hi_ny >= 0._rt))
            || ((y2 == y3) && (std::abs(y2) == 0.5_rt) &&
                (lo_nx*hi_nx >= 0._rt))) {
            EBFaceTriangle t;
            t.x = {x2, x3, xi};
            t.y = {y2, y3, yi};
            order_triangle(t, lo_nx, lo_ny);
            triangles[nt++] = t;
            p2 = true;
            p3 = true;
        }

        if (   ((x1 == x4) && (std::abs(x1) == 0.5_rt) &&
                (lo_ny*hi_ny >= 0._rt))
            || ((y1 == y4) && (std::abs(y1) == 0.5_rt) &&
                (lo_nx*hi_nx >= 0._rt))) {
            EBFaceTriangle t;
            t.x = {x1, x4, xi};
            t.y = {y1, y4, yi};
            order_triangle(t, lo_nx, lo_ny);
            triangles[nt++] = t;
            p1 = true;
            p4 = true;
        }

        // Squares
        if ((y2 == -0.5_rt && x3 == -0.5_rt) &&
            (lo_nx*hi_ny <= 0._rt) &&
            (!p2 && !p3)) {
            EBFaceTriangle t1;
            t1.x = {x2, -0.5_rt, xi};
            t1.y = {y2, -0.5_rt, yi};
            order_triangle(t1, lo_nx, lo_ny);
            triangles[nt++] = t1;

            EBFaceTriangle t2;
            t2.x = {x3, -0.5_rt, xi};
            t2.y = {y3, -0.5_rt, yi};
            order_triangle(t2, lo_nx, lo_ny);
            t2.sign = t1.sign;
            triangles[nt++] = t2;

            p2 = true;
            p3 = true;
        }

        if ((y2 == -0.5_rt && x3 == 0.5_rt) &&
            (lo_nx*hi_ny >= 0._rt) &&
            (!p2 && !p3)) {
            EBFaceTriangle t1;
            t1.x = {x2,  0.5_rt, xi};
            t1.y = {y2, -0.5_rt, yi};
            order_triangle(t1, lo_nx, lo_ny);
            triangles[nt++] = t1;

            EBFaceTriangle t2;
            t2.x = {x3,  0.5_rt, xi};
            t2.y = {y3, -0.5_rt, yi};
            order_triangle(t2, lo_nx, lo_ny);
            t2.sign = t1.sign;
            triangles[nt++] = t2;

            p2 = true;
            p3 = true;
        }

        if ((y2 == 0.5_rt && x3 == -0.5_rt) &&
            (lo_nx*hi_ny >= 0._rt) &&
            (!p2 && !p3)) {
            EBFaceTriangle t1;
            t1.x = {x2, -0.5_rt, xi};
            t1.y = {y2,  0.5_rt, yi};
            order_triangle(t1, lo_nx, lo_ny);
            triangles[nt++] = t1;

            EBFaceTriangle t2;
            t2.x = {x3, -0.5_rt, xi};
            t2.y = {y3,  0.5_rt, yi};
            order_triangle(t2, lo_nx, lo_ny);
            t2.sign = t1.sign;
            triangles[nt++] = t2;

            p2 = true;
            p3 = true;
        }

        if ((y2 == 0.5_rt && x3 == 0.5_rt) &&
            (lo_nx*hi_ny <= 0._rt) &&
            (!p2 && !p3)) {
            EBFaceTriangle t1;
            t1.x = {x2, 0.5_rt, xi};
            t1.y = {y2, 0.5_rt, yi};
            order_triangle(t1, lo_nx, lo_ny);
            triangles[nt++] = t1;

            EBFaceTriangle t2;
            t2.x = {x3, 0.5_rt, xi};
            t2.y = {y3, 0.5_rt, yi};
            order_triangle(t2, lo_nx, lo_ny);
            t2.sign = t1.sign;
            triangles[nt++] = t2;

            p2 = true;
            p3 = true;
        }

        if ((x2 == 0.5_rt && y3 == -0.5_rt) &&
            (lo_ny*hi_nx >= 0._rt) &&
            (!p2 && !p3)) {
            EBFaceTriangle t1;
            t1.x = {x2,  0.5_rt, xi};
            t1.y = {y2, -0.5_rt, yi};
            order_triangle(t1, lo_nx, lo_ny);
            triangles[nt++] = t1;

            EBFaceTriangle t2;
            t2.x = {x3,  0.5_rt, xi};
            t2.y = {y3, -0.5_rt, yi};
            order_triangle(t2, lo_nx, lo_ny);
            t2.sign = t1.sign;
            triangles[nt++] = t2;

            p2 = true;
            p3 = true;
        }

        if ((y1 == -0.5_rt && x3 == -0.5_rt) &&
            (lo_nx*hi_ny <= 0._rt) &&
            (!p1 && !p3)) {
            EBFaceTriangle t1;
            t1.x = {x1, -0.5_rt, xi};
            t1.y = {y1, -0.5_rt, yi};
            order_triangle(t1, lo_nx, lo_ny);
            triangles[nt++] = t1;

            EBFaceTriangle t2;
            t2.x = {x3, -0.5_rt, xi};
            t2.y = {y3, -0.5_rt, yi};
            order_triangle(t2, lo_nx, lo_ny);
            t2.sign = t1.sign;
            triangles[nt++] = t2;

            p1 = true;
            p3 = true;
        }

        if ((y1 == -0.5_rt && x3 == 0.5_rt) &&
            (lo_nx*hi_ny >= 0._rt) &&
            (!p1 && !p3)) {
            EBFaceTriangle t1;
            t1.x = {x1,  0.5_rt, xi};
            t1.y = {y1, -0.5_rt, yi};
            order_triangle(t1, lo_nx, lo_ny);
            triangles[nt++] = t1;

            EBFaceTriangle t2;
            t2.x = {x3,  0.5_rt, xi};
            t2.y = {y3, -0.5_rt, yi};
            order_triangle(t2, lo_nx, lo_ny);
            t2.sign = t1.sign;
            triangles[nt++] = t2;

            p1 = true;
            p3 = true;
        }

        if ((x1 == -0.5_rt && y3 == -0.5_rt) &&
            (lo_ny*hi_nx <= 0._rt) &&
            (!p1 && !p3)) {
            EBFaceTriangle t1;
            t1.x = {x1, -0.5_rt, xi};
            t1.y = {y1, -0.5_rt, yi};
            order_triangle(t1, lo_nx, lo_ny);
            triangles[nt++] = t1;

            EBFaceTriangle t2;
            t2.x = {x3, -0.5_rt, xi};
            t2.y = {y3, -0.5_rt, yi};
            order_triangle(t2, lo_nx, lo_ny);
            t2.sign = t1.sign;
            triangles[nt++] = t2;

            p1 = true;
            p3 = true;
        }

        if ((x1 == 0.5_rt && y3 == -0.5_rt) &&
            (lo_ny*hi_nx >= 0._rt) &&
            (!p1 && !p3)) {
            EBFaceTriangle t1;
            t1.x = {x1,  0.5_rt, xi};
            t1.y = {y1, -0.5_rt, yi};
            order_triangle(t1, lo_nx, lo_ny);
            triangles[nt++] = t1;

            EBFaceTriangle t2;
            t2.x = {x3,  0.5_rt, xi};
            t2.y = {y3, -0.5_rt, yi};
            order_triangle(t2, lo_nx, lo_ny);
            t2.sign = t1.sign;
            triangles[nt++] = t2;

            p1 = true;
            p3 = true;
        }

        if ((y2 == -0.5_rt && x4 == 0.5_rt) &&
            (lo_nx*hi_ny >= 0._rt) &&
            (!p2 && !p4)) {
            EBFaceTriangle t1;
            t1.x = {x2,  0.5_rt, xi};
            t1.y = {y2, -0.5_rt, yi};
            order_triangle(t1, lo_nx, lo_ny);
            triangles[nt++] = t1;

            EBFaceTriangle t2;
            t2.x = {x4,  0.5_rt, xi};
            t2.y = {y4, -0.5_rt, yi};
            order_triangle(t2, lo_nx, lo_ny);
            t2.sign = t1.sign;
            triangles[nt++] = t2;

            p2 = true;
            p4 = true;
        }

        if ((y2 == 0.5_rt && x4 == 0.5_rt) &&
            (lo_nx*hi_ny <= 0._rt) &&
            (!p2 && !p4)) {
            EBFaceTriangle t1;
            t1.x = {x2, 0.5_rt, xi};
            t1.y = {y2, 0.5_rt, yi};
            order_triangle(t1, lo_nx, lo_ny);
            triangles[nt++] = t1;

            EBFaceTriangle t2;
            t2.x = {x4, 0.5_rt, xi};
            t2.y = {y4, 0.5_rt, yi};
            order_triangle(t2, lo_nx, lo_ny);
            t2.sign = t1.sign;
            triangles[nt++] = t2;

            p2 = true;
            p4 = true;
        }

        if ((x2 == 0.5_rt && y4 == -0.5_rt) &&
            (lo_ny*hi_nx >= 0._rt) &&
            (!p2 && !p4)) {
            EBFaceTriangle t1;
            t1.x = {x2,  0.5_rt, xi};
            t1.y = {y2, -0.5_rt, yi};
            order_triangle(t1, lo_nx, lo_ny);
            triangles[nt++] = t1;

            EBFaceTriangle t2;
            t2.x = {x4,  0.5_rt, xi};
            t2.y = {y4, -0.5_rt, yi};
            order_triangle(t2, lo_nx, lo_ny);
            t2.sign = t1.sign;
            triangles[nt++] = t2;

            p2 = true;
            p4 = true;
        }

        if ((x2 == 0.5_rt && y4 == 0.5_rt) &&
            (lo_ny*hi_nx <= 0._rt) &&
            (!p2 && !p4)) {
            EBFaceTriangle t1;
            t1.x = {x2, 0.5_rt, xi};
            t1.y = {y2, 0.5_rt, yi};
            order_triangle(t1, lo_nx, lo_ny);
            triangles[nt++] = t1;

            EBFaceTriangle t2;
            t2.x = {x4, 0.5_rt, xi};
            t2.y = {y4, 0.5_rt, yi};
            order_triangle(t2, lo_nx, lo_ny);
            t2.sign = t1.sign;
            triangles[nt++] = t2;

            p2 = true;
            p4 = true;
        }

        if ((y1 == -0.5_rt && x4 == 0.5_rt) &&
            (lo_nx*hi_ny >= 0._rt) &&
            (!p1 && !p4)) {
            EBFaceTriangle t1;
            t1.x = {x1,  0.5_rt, xi};
            t1.y = {y1, -0.5_rt, yi};
            order_triangle(t1, lo_nx, lo_ny);
            triangles[nt++] = t1;

            EBFaceTriangle t2;
            t2.x = {x4,  0.5_rt, xi};
            t2.y = {y4, -0.5_rt, yi};
            order_triangle(t2, lo_nx, lo_ny);
            t2.sign = t1.sign;
            triangles[nt++] = t2;

            p1 = true;
            p4 = true;
        }

        if ((x1 == -0.5_rt && y4 == -0.5_rt) &&
            (lo_ny*hi_nx <= 0._rt) &&
            (!p1 && !p4)) {
            EBFaceTriangle t1;
            t1.x = {x1, -0.5_rt, xi};
            t1.y = {y1, -0.5_rt, yi};
            order_triangle(t1, lo_nx, lo_ny);
            triangles[nt++] = t1;

            EBFaceTriangle t2;
            t2.x = {x4, -0.5_rt, xi};
            t2.y = {y4, -0.5_rt, yi};
            order_triangle(t2, lo_nx, lo_ny);
            t2.sign = t1.sign;
            triangles[nt++] = t2;

            p1 = true;
            p4 = true;
        }

        if ((x1 == 0.5_rt && y4 == -0.5_rt) &&
            (lo_ny*hi_nx >= 0._rt) &&
            (!p1 && !p4)) {
            EBFaceTriangle t1;
            t1.x = {x1,  0.5_rt, xi};
            t1.y = {y1, -0.5_rt, yi};
            order_triangle(t1, lo_nx, lo_ny);
            triangles[nt++] = t1;

            EBFaceTriangle t2;
            t2.x = {x4,  0.5_rt, xi};
            t2.y = {y4, -0.5_rt, yi};
            order_triangle(t2, lo_nx, lo_ny);
            t2.sign = t1.sign;
            triangles[nt++] = t2;

            p1 = true;
            p4 = true;
        }

        if ((x1 == -0.5_rt && y4 == 0.5_rt) &&
            (lo_ny*hi_nx >= 0._rt) &&
            (!p1 && !p4)) {
            EBFaceTriangle t1;
            t1.x = {x1, -0.5_rt, xi};
            t1.y = {y1,  0.5_rt, yi};
            order_triangle(t1, lo_nx, lo_ny);
            triangles[nt++] = t1;

            EBFaceTriangle t2;
            t2.x = {x4, -0.5_rt, xi};
            t2.y = {y4,  0.5_rt, yi};
            order_triangle(t2, lo_nx, lo_ny);
            t2.sign = t1.sign;
            triangles[nt++] = t2;

            p1 = true;
            p4 = true;
        }

        if ((x1 == 0.5_rt && y4 == 0.5_rt) &&
            (lo_ny*hi_nx <= 0._rt) &&
            (!p1 && !p4)) {
            EBFaceTriangle t1;
            t1.x = {x1, 0.5_rt, xi};
            t1.y = {y1, 0.5_rt, yi};
            order_triangle(t1, lo_nx, lo_ny);
            triangles[nt++] = t1;

            EBFaceTriangle t2;
            t2.x = {x4, 0.5_rt, xi};
            t2.y = {y4, 0.5_rt, yi};
            order_triangle(t2, lo_nx, lo_ny);
            t2.sign = t1.sign;
            triangles[nt++] = t2;

            p1 = true;
            p4 = true;
        }

        // Sanity check
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE((p1 && p2 && p3 && p4), "Something went wrong when extracting triangles");

    } else {

        // Trapezoids
        if (x1 == 0.5_rt && y2 == 0.5_rt && x3 == 0.5_rt && y4 == 0.5_rt &&
            (lo_ny*hi_ny >= 0._rt) && (lo_nx*hi_nx >= 0._rt)) {
            EBFaceTriangle t1;
            t1.x = {x1, x3, x2};
            t1.y = {y1, y3, y2};
            order_triangle(t1, lo_nx, lo_ny);
            t1.sign = true;

            if (y1 > y3) {
                if (lo_ny < 0._rt) {
                    t1.sign = false;
                }
            } else {
                if (lo_ny > 0._rt) {
                    t1.sign = false;
                }
            }
            triangles[nt++] = t1;

            EBFaceTriangle t2;
            t2.x = {x2, x4, x3};
            t2.y = {y2, y4, y3};
            order_triangle(t2, lo_nx, lo_ny);
            t2.sign = t1.sign;
            triangles[nt++] = t2;

        } else if (x1 == -0.5_rt && y2 == -0.5_rt && x3 == -0.5_rt && y4 == -0.5_rt &&
                   (lo_ny*hi_ny >= 0._rt) && (lo_nx*hi_nx >= 0._rt)) {
            EBFaceTriangle t1;
            t1.x = {x1, x3, x2};
            t1.y = {y1, y3, y2};
            order_triangle(t1, lo_nx, lo_ny);
            t1.sign = true;

            if (y1 < y3) {
                if (lo_ny > 0._rt) {
                    t1.sign = false;
                }
            } else {
                if (lo_ny < 0._rt) {
                    t1.sign = false;
                }
            }
            triangles[nt++] = t1;

            EBFaceTriangle t2;
            t2.x = {x2, x4, x3};
            t2.y = {y2, y4, y3};
            order_triangle(t2, lo_nx, lo_ny);
            t2.sign = t1.sign;
            triangles[nt++] = t2;

        } else if (x1 == -0.5_rt && y2 == 0.5_rt && x3 == -0.5_rt && y4 == 0.5_rt &&
                   (lo_ny*hi_ny >= 0._rt) && (lo_nx*hi_nx >= 0._rt)) {
            EBFaceTriangle t1;
            t1.x = {x1, x3, x2};
            t1.y = {y1, y3, y2};
            order_triangle(t1, lo_nx, lo_ny);
            t1.sign = true;

            if (y1 < y3) {
                if (lo_ny > 0._rt) {
                    t1.sign = false;
                }
            } else {
                if (lo_ny < 0._rt) {
                    t1.sign = false;
                }
            }
            triangles[nt++] = t1;

            EBFaceTriangle t2;
            t2.x = {x2, x4, x3};
            t2.y = {y2, y4, y3};
            order_triangle(t2, lo_nx, lo_ny);
            t2.sign = t1.sign;
            triangles[nt++] = t2;

        } else if (x1 == 0.5_rt && y2 == -0.5_rt && x3 == 0.5_rt && y4 == -0.5_rt &&
                   (lo_ny*hi_ny >= 0._rt) && (lo_nx*hi_nx >= 0._rt)) {
            EBFaceTriangle t1;
            t1.x = {x1, x3, x2};
            t1.y = {y1, y3, y2};
            order_triangle(t1, lo_nx, lo_ny);
            t1.sign = true;

            if (y1 > y3) {
                if (lo_ny < 0._rt) {
                    t1.sign = false;
                }
            } else {
                if (lo_ny > 0._rt) {
                    t1.sign = false;
                }
            }
            triangles[nt++] = t1;

            EBFaceTriangle t2;
            t2.x = {x2, x4, x3};
            t2.y = {y2, y4, y3};
            order_triangle(t2, lo_nx, lo_ny);
            t2.sign = t1.sign;
            triangles[nt++] = t2;

        } else if (x1 == -0.5_rt && x2 == 0.5_rt && x3 == -0.5_rt && x4 == 0.5_rt &&
                   (lo_ny*hi_ny >= 0._rt)) {
            EBFaceTriangle t1;
            t1.x = {x1, x3, x2};
            t1.y = {y1, y3, y2};
            order_triangle(t1, lo_nx, lo_ny);
            t1.sign = true;

            if (y1 > y3) {
                if (lo_ny < 0._rt) {
                    t1.sign = false;
                }
            } else {
                if (lo_ny > 0._rt) {
                    t1.sign = false;
                }
            }
            triangles[nt++] = t1;

            EBFaceTriangle t2;
            t2.x = {x2, x4, x3};
            t2.y = {y2, y4, y3};
            order_triangle(t2, lo_nx, lo_ny);
            t2.sign = t1.sign;
            triangles[nt++] = t2;

        } else if (y1 == -0.5_rt && y2 == 0.5_rt && y3 == -0.5_rt && y4 == 0.5_rt &&
                   (lo_nx*hi_nx >= 0._rt)) {
            EBFaceTriangle t1;
            t1.x = {x1, x3, x2};
            t1.y = {y1, y3, y2};
            order_triangle(t1, lo_nx, lo_ny);
            t1.sign = true;

            if (x1 > x3) {
                if (lo_nx < 0._rt) {
                    t1.sign = false;
                }
            } else {
                if (lo_nx > 0._rt) {
                    t1.sign = false;
                }
            }
            triangles[nt++] = t1;

            EBFaceTriangle t2;
            t2.x = {x2, x4, x3};
            t2.y = {y2, y4, y3};
            order_triangle(t2, lo_nx, lo_ny);
            t2.sign = t1.sign;
            triangles[nt++] = t2;

        // Pentagons
        } else if (x1 == -0.5_rt && y2 == 0.5_rt && y3 == -0.5_rt && y4 == 0.5_rt &&
                   (lo_nx*hi_nx >= 0._rt)) {
            EBFaceTriangle t1;
            t1.x = {x1, -0.5_rt, x2};
            t1.y = {y1, -0.5_rt, y2};
            order_triangle(t1, lo_nx, lo_ny);
            t1.sign = true;

            if (lo_nx > 0._rt) {
                t1.sign = false;
            }
            triangles[nt++] = t1;

            EBFaceTriangle t2;
            t2.x = {x2, x4, -0.5_rt};
            t2.y = {y2, y4, -0.5_rt};
            order_triangle(t2, lo_nx, lo_ny);
            t2.sign = t1.sign;
            triangles[nt++] = t2;

            EBFaceTriangle t3;
            t3.x = {x3, -0.5_rt, x4};
            t3.y = {y3, -0.5_rt, y4};
            order_triangle(t3, lo_nx, lo_ny);
            t3.sign = t1.sign;
            triangles[nt++] = t3;

        } else if (x1 == -0.5_rt && y2 == 0.5_rt && x3 == -0.5_rt && x4 == 0.5_rt &&
                   (lo_ny*hi_ny >= 0._rt)) {
            EBFaceTriangle t1;
            t1.x = {x1, 0.5_rt, x2};
            t1.y = {y1, 0.5_rt, y2};
            order_triangle(t1, lo_nx, lo_ny);
            t1.sign = true;

            if (lo_ny < 0._rt) {
                t1.sign = false;
            }
            triangles[nt++] = t1;

            EBFaceTriangle t2;
            t2.x = {x1, x3, 0.5_rt};
            t2.y = {y1, y3, 0.5_rt};
            order_triangle(t2, lo_nx, lo_ny);
            t2.sign = t1.sign;
            triangles[nt++] = t2;

            EBFaceTriangle t3;
            t3.x = {0.5_rt, x4, x3};
            t3.y = {0.5_rt, y4, y3};
            order_triangle(t3, lo_nx, lo_ny);
            t3.sign = t1.sign;
            triangles[nt++] = t3;

        } else if (x1 == 0.5_rt && y2 == 0.5_rt && x3 == -0.5_rt && x4 == 0.5_rt &&
                   (lo_ny*hi_ny >= 0._rt)) {
            EBFaceTriangle t1;
            t1.x = {x2, -0.5_rt, x1};
            t1.y = {y2,  0.5_rt, y1};
            order_triangle(t1, lo_nx, lo_ny);
            t1.sign = true;

            if (lo_ny < 0._rt) {
                t1.sign = false;
            }
            triangles[nt++] = t1;

            EBFaceTriangle t2;
            t2.x = {x4, -0.5_rt, x1};
            t2.y = {y4,  0.5_rt, x1};
            order_triangle(t2, lo_nx, lo_ny);
            t2.sign = t1.sign;
            triangles[nt++] = t2;

            EBFaceTriangle t3;
            t3.x = {x3, -0.5_rt, x4};
            t3.y = {y3,  0.5_rt, x4};
            order_triangle(t3, lo_nx, lo_ny);
            t3.sign = t1.sign;
            triangles[nt++] = t3;

        } else if (x1 == -0.5_rt && x2 == 0.5_rt && x3 == -0.5_rt && y4 == -0.5_rt &&
                   (lo_ny*hi_ny >= 0._rt)) {
            EBFaceTriangle t1;
            t1.x = {x2,  0.5_rt, x1};
            t1.y = {y2, -0.5_rt, y1};
            order_triangle(t1, lo_nx, lo_ny);
            t1.sign = true;

            if (lo_ny < 0._rt) {
                t1.sign = false;
            }
            triangles[nt++] = t1;

            EBFaceTriangle t2;
            t2.x = {x1, x3,  0.5_rt};
            t2.y = {y1, y3, -0.5_rt};
            order_triangle(t2, lo_nx, lo_ny);
            t2.sign = t1.sign;
            triangles[nt++] = t2;

            EBFaceTriangle t3;
            t3.x = {x4,  0.5_rt, x3};
            t3.y = {y4, -0.5_rt, y3};
            order_triangle(t3, lo_nx, lo_ny);
            t3.sign = t1.sign;
            triangles[nt++] = t3;

        } else if (y1 == -0.5_rt && y2 == 0.5_rt && x3 == -0.5_rt && y4 == -0.5_rt &&
                   (lo_nx*hi_nx >= 0._rt)) {
            EBFaceTriangle t1;
            t1.x = {x2, -0.5_rt, x1};
            t1.y = {y2,  0.5_rt, y1};
            order_triangle(t1, lo_nx, lo_ny);
            t1.sign = true;

            if (lo_nx < 0._rt) {
                t1.sign = false;
            }
            triangles[nt++] = t1;

            EBFaceTriangle t2;
            t2.x = {x1, x4, -0.5_rt};
            t2.y = {y1, y4,  0.5_rt};
            order_triangle(t2, lo_nx, lo_ny);
            t2.sign = t1.sign;
            triangles[nt++] = t2;

            EBFaceTriangle t3;
            t3.x = {x3, -0.5_rt, x4};
            t3.y = {y3,  0.5_rt, y4};
            order_triangle(t3, lo_nx, lo_ny);
            t3.sign = t1.sign;
            triangles[nt++] = t3;

        } else if (y1 == -0.5_rt && y2 == 0.5_rt && x3 == -0.5_rt && y4 == 0.5_rt &&
                   (lo_nx*hi_nx >= 0._rt)) {
            EBFaceTriangle t1;
            t1.x = {x1, -0.5_rt, x2};
            t1.y = {y1, -0.5_rt, y2};
            order_triangle(t1, lo_nx, lo_ny);
            t1.sign = true;

            if (lo_nx < 0._rt) {
                t1.sign = false;
            }
            triangles[nt++] = t1;

            EBFaceTriangle t2;
            t2.x = {x2, x4, -0.5_rt};
            t2.y = {y2, y4, -0.5_rt};
            order_triangle(t2, lo_nx, lo_ny);
            t2.sign = t1.sign;
            triangles[nt++] = t2;

            EBFaceTriangle t3;
            t3.x = {x3, -0.5_rt, x4};
            t3.y = {y3, -0.5_rt, y4};
            order_triangle(t3, lo_nx, lo_ny);
            t3.sign = t1.sign;
            triangles[nt++] = t3;

        // 2 disjoined triangles
        } else if (x1 == 0.5_rt && y2 == -0.5_rt && x3 == 0.5_rt && y4 == 0.5_rt) {
            EBFaceTriangle t1;
            t1.x = {x1,  0.5_rt, x2};
            t1.y = {y1, -0.5_rt, y2};
            order_triangle(t1, lo_nx, lo_ny);
            triangles[nt++] = t1;

            EBFaceTriangle t2;
            t2.x = {x3, 0.5_rt, x4};
            t2.y = {y3, 0.5_rt, y4};
            order_triangle(t2, lo_nx, lo_ny);
            t2.sign = !t1.sign;
            triangles[nt++] = t2;

        } else if (x1 == 0.5_rt && y2 == 0.5_rt && x3 == 0.5_rt && y4 == -0.5_rt) {
            EBFaceTriangle t1;
            t1.x = {x1, 0.5_rt, x2};
            t1.y = {y1, 0.5_rt, y2};
            order_triangle(t1, lo_nx, lo_ny);
            triangles[nt++] = t1;

            EBFaceTriangle t2;
            t2.x = {x3,  0.5_rt, x4};
            t2.y = {y3, -0.5_rt, y4};
            order_triangle(t2, lo_nx, lo_ny);
            t2.sign = !t1.sign;
            triangles[nt++] = t2;

        } else if (x1 == 0.5_rt && y2 == 0.5_rt && x3 == -0.5_rt && y4 == 0.5_rt) {
            EBFaceTriangle t1;
            t1.x = {x1, 0.5_rt, x2};
            t1.y = {y1, 0.5_rt, y2};
            order_triangle(t1, lo_nx, lo_ny);
            triangles[nt++] = t1;

            EBFaceTriangle t2;
            t2.x = {x3, -0.5_rt, x4};
            t2.y = {y3,  0.5_rt, y4};
            order_triangle(t2, lo_nx, lo_ny);
            t2.sign = !t1.sign;
            triangles[nt++] = t2;

        } else if (x1 == -0.5_rt && y2 == -0.5_rt && x3 == -0.5_rt && y4 == 0.5_rt) {
            EBFaceTriangle t1;
            t1.x = {x1, -0.5_rt, x2};
            t1.y = {y1, -0.5_rt, y2};
            order_triangle(t1, lo_nx, lo_ny);
            triangles[nt++] = t1;

            EBFaceTriangle t2;
            t2.x = {x3, -0.5_rt, x4};
            t2.y = {y3,  0.5_rt, y4};
            order_triangle(t2, lo_nx, lo_ny);
            t2.sign = !t1.sign;
            triangles[nt++] = t2;

        } else if (x1 == -0.5_rt && y2 == 0.5_rt && x3 == -0.5_rt && y4 == -0.5_rt) {
            EBFaceTriangle t1;
            t1.x = {x1, -0.5_rt, x2};
            t1.y = {y1,  0.5_rt, y2};
            order_triangle(t1, lo_nx, lo_ny);
            triangles[nt++] = t1;

            EBFaceTriangle t2;
            t2.x = {x3, -0.5_rt, x4};
            t2.y = {y3, -0.5_rt, y4};
            order_triangle(t2, lo_nx, lo_ny);
            t2.sign = !t1.sign;
            triangles[nt++] = t2;

        } else if (x1 == 0.5_rt && y2 == -0.5_rt && x3 == -0.5_rt && y4 == -0.5_rt) {
            EBFaceTriangle t1;
            t1.x = {x1,  0.5_rt, x2};
            t1.y = {y1, -0.5_rt, y2};
            order_triangle(t1, lo_nx, lo_ny);
            triangles[nt++] = t1;

            EBFaceTriangle t2;
            t2.x = {x3, -0.5_rt, x4};
            t2.y = {y3, -0.5_rt, y4};
            order_triangle(t2, lo_nx, lo_ny);
            t2.sign = !t1.sign;
            triangles[nt++] = t2;

        } else if (x1 == -0.5_rt && y2 == 0.5_rt && x3 == 0.5_rt && y4 == 0.5_rt) {
            EBFaceTriangle t1;
            t1.x = {x1, -0.5_rt, x2};
            t1.y = {y1,  0.5_rt, y2};
            order_triangle(t1, lo_nx, lo_ny);
            triangles[nt++] = t1;

            EBFaceTriangle t2;
            t2.x = {x3, 0.5_rt, x4};
            t2.y = {y3, 0.5_rt, y4};
            order_triangle(t2, lo_nx, lo_ny);
            t2.sign = !t1.sign;
            triangles[nt++] = t2;

        } else if (x1 == -0.5_rt && y2 == -0.5_rt && x3 == 0.5_rt && y4 == -0.5_rt) {
            EBFaceTriangle t1;
            t1.x = {x1, -0.5_rt, x2};
            t1.y = {y1, -0.5_rt, y2};
            order_triangle(t1, lo_nx, lo_ny);
            triangles[nt++] = t1;

            EBFaceTriangle t2;
            t2.x = {x3,  0.5_rt, x4};
            t2.y = {y3, -0.5_rt, y4};
            order_triangle(t2, lo_nx, lo_ny);
            t2.sign = !t1.sign;
            triangles[nt++] = t2;

        } else {
            Abort("Something went wrong when extracting triangles");
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

    GpuArray<Array<Real,2>, 4> lo_ipts;
    GpuArray<Array<Real,2>, 4> hi_ipts;
    int lo_np = 0;
    int hi_np = 0;

    if (!(flag(i-1,j,k).isSingleValued() || flag(i,j,k).isSingleValued())) {
        return;
    }

    if (flag(i-1,j,k).isSingleValued()) {
        // lo side x =  0.5
        Real lo_Cx = bcent(i-1,j,k,0);
        Real lo_Cy = bcent(i-1,j,k,1);
        Real lo_Cz = bcent(i-1,j,k,2);
        Real lo_nx = bnorm(i-1,j,k,0);
        Real lo_ny = bnorm(i-1,j,k,1);
        Real lo_nz = bnorm(i-1,j,k,2);

        // y = -0.5
        if (lo_nz != 0._rt) {
            Real lo_z = lo_Cz -(( 0.5_rt-lo_Cx)*lo_nx + (-0.5_rt-lo_Cy)*lo_ny)/lo_nz;
            if (lo_z <= 0.5_rt && lo_z >= -0.5_rt) {lo_ipts[lo_np++] = {-0.5_rt,lo_z};}
        }

        // y = 0.5
        if (lo_nz != 0._rt) {
            Real lo_z = lo_Cz -(( 0.5_rt-lo_Cx)*lo_nx + (0.5_rt-lo_Cy)*lo_ny)/lo_nz;
            if (lo_z <= 0.5_rt && lo_z >= -0.5_rt) {lo_ipts[lo_np++] = {0.5_rt,lo_z};}
        }

        // z = -0.5
        if (lo_ny != 0._rt) {
            Real lo_y = lo_Cy -(( 0.5_rt-lo_Cx)*lo_nx + (-0.5_rt-lo_Cz)*lo_nz)/lo_ny;
            if (lo_y <= 0.5_rt && lo_y >= -0.5_rt) {lo_ipts[lo_np++] = {lo_y,-0.5_rt};}
        }

        // z = 0.5
        if (lo_ny != 0._rt) {
            Real lo_y = lo_Cy -(( 0.5_rt-lo_Cx)*lo_nx + (0.5_rt-lo_Cz)*lo_nz)/lo_ny;
            if (lo_y <= 0.5_rt && lo_y >= -0.5_rt) {lo_ipts[lo_np++] = {lo_y,0.5_rt};}
        }
    }

    if (flag(i,j,k).isSingleValued()) {
        // hi side x = -0.5
        Real hi_Cx = bcent(i,j,k,0);
        Real hi_Cy = bcent(i,j,k,1);
        Real hi_Cz = bcent(i,j,k,2);
        Real hi_nx = bnorm(i,j,k,0);
        Real hi_ny = bnorm(i,j,k,1);
        Real hi_nz = bnorm(i,j,k,2);


        // y = -0.5
        if (hi_nz != 0_rt) {
            Real hi_z = hi_Cz -((-0.5_rt-hi_Cx)*hi_nx + (-0.5_rt-hi_Cy)*hi_ny)/hi_nz;
            if (hi_z <= 0.5_rt && hi_z >= -0.5_rt) {hi_ipts[hi_np++] = {-0.5_rt,hi_z};}
        }

        // y = 0.5
        if (hi_nz != 0._rt) {
            Real hi_z = hi_Cz -((-0.5_rt-hi_Cx)*hi_nx + (0.5_rt-hi_Cy)*hi_ny)/hi_nz;
            if (hi_z <= 0.5_rt && hi_z >= -0.5_rt) {hi_ipts[hi_np++] = {0.5_rt,hi_z};}
        }

        // z = -0.5
        if (hi_ny != 0._rt) {
            Real hi_y = hi_Cy -((-0.5_rt-hi_Cx)*hi_nx + (-0.5_rt-hi_Cz)*hi_nz)/hi_ny;
            if (hi_y <= 0.5_rt && hi_y >= -0.5_rt) {hi_ipts[hi_np++] = {hi_y,-0.5_rt};}
        }

        // z = 0.5
        if (hi_ny != 0._rt) {
            Real hi_y = hi_Cy -((-0.5_rt-hi_Cx)*hi_nx + (0.5_rt-hi_Cz)*hi_nz)/hi_ny;
            if (hi_y <= 0.5_rt && hi_y >= -0.5_rt) {hi_ipts[hi_np++] = {hi_y,0.5_rt};}
        }
    }

    sanitize_ipts(lo_ipts, lo_np);
    sanitize_ipts(hi_ipts, hi_np);

    GpuArray<EBFaceTriangle, 4> triangles;
    int nt = 0;

    if (lo_np == 0 && hi_np == 2) {
        bool sign = true;
        if (flag(i-1,j,k).isSingleValued()) {
            if (bnorm(i-1,j,k,0) > 0._rt) {
                sign = false;
            }
        } else {
            sign = flag(i-1,j,k).isRegular();
        }
        extract_triangle_on_face_hi(triangles, nt, hi_ipts, hi_np, sign);
    } else if (lo_np == 2 && hi_np ==0) {
        extract_triangle_on_face_lo(triangles, nt, lo_ipts, lo_np, bnorm(i-1,j,k,1), bnorm(i-1,j,k,2));
    } else if (lo_np == 2 && hi_np == 2) {
        extract_triangles_on_face(triangles, nt, lo_ipts, lo_np, hi_ipts, hi_np,
                                  bnorm(i-1,j,k,1), bnorm(i-1,j,k,2),
                                  bnorm(i,j,k,1), bnorm(i,j,k,2));
    }

    for (int n = 0; n < nt; ++n) {
        add_to_mmintg(i,j,k,mmintg_x,triangles[n]);
    }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void compute_mmintg_on_xz_face(int i, int j, int k,
                               Array4<Real> const& mmintg_y,
                               Array4<Real const> const& bcent,
                               Array4<Real const> const& bnorm,
                               Array4<EBCellFlag const> const& flag) noexcept
{
    mmintg_y(i,j,k, i_M_1)  = 0._rt;
    mmintg_y(i,j,k, i_M_x)  = 0._rt;
    mmintg_y(i,j,k, i_M_y)  = 0._rt;
    mmintg_y(i,j,k, i_M_xy) = 0._rt;

    GpuArray<Array<Real,2>, 4> lo_ipts;
    GpuArray<Array<Real,2>, 4> hi_ipts;
    int lo_np = 0;
    int hi_np = 0;

    if (!(flag(i,j-1,k).isSingleValued() || flag(i,j,k).isSingleValued())) {
        return;
    }

    if (flag(i,j-1,k).isSingleValued()) {
        // lo side y =  0.5
        Real lo_Cx = bcent(i,j-1,k,0);
        Real lo_Cy = bcent(i,j-1,k,1);
        Real lo_Cz = bcent(i,j-1,k,2);
        Real lo_nx = bnorm(i,j-1,k,0);
        Real lo_ny = bnorm(i,j-1,k,1);
        Real lo_nz = bnorm(i,j-1,k,2);

        // x = -0.5
        if (lo_nz != 0._rt) {
            Real lo_z = lo_Cz -((-0.5_rt-lo_Cx)*lo_nx + (0.5_rt-lo_Cy)*lo_ny)/lo_nz;
            if (lo_z <= 0.5_rt && lo_z >= -0.5_rt) {lo_ipts[lo_np++] = {-0.5_rt,lo_z};}
        }

        // x = 0.5
        if (lo_nz != 0._rt) {
            Real lo_z = lo_Cz -(( 0.5_rt-lo_Cx)*lo_nx + (0.5_rt-lo_Cy)*lo_ny)/lo_nz;
            if (lo_z <= 0.5_rt && lo_z >= -0.5_rt) {lo_ipts[lo_np++] = {0.5_rt,lo_z};}
        }

        // z = -0.5
        if (lo_nx != 0._rt) {
            Real lo_x = lo_Cx -((0.5_rt-lo_Cy)*lo_ny + (-0.5_rt-lo_Cz)*lo_nz)/lo_nx;
            if (lo_x <= 0.5_rt && lo_x >= -0.5_rt) {lo_ipts[lo_np++] = {lo_x,-0.5_rt};}
        }

        // z = 0.5
        if (lo_nx != 0._rt) {
            Real lo_x = lo_Cx -((0.5_rt-lo_Cy)*lo_ny + (0.5_rt-lo_Cz)*lo_nz)/lo_nx;
            if (lo_x <= 0.5_rt && lo_x >= -0.5_rt) {lo_ipts[lo_np++] = {lo_x,0.5_rt};}
        }
    }

    if (flag(i,j,k).isSingleValued()) {
        // hi side y = -0.5
        Real hi_Cx = bcent(i,j,k,0);
        Real hi_Cy = bcent(i,j,k,1);
        Real hi_Cz = bcent(i,j,k,2);
        Real hi_nx = bnorm(i,j,k,0);
        Real hi_ny = bnorm(i,j,k,1);
        Real hi_nz = bnorm(i,j,k,2);


        // x = -0.5
        if (hi_nz != 0_rt) {
            Real hi_z = hi_Cz -((-0.5_rt-hi_Cx)*hi_nx + (-0.5_rt-hi_Cy)*hi_ny)/hi_nz;
            if (hi_z <= 0.5_rt && hi_z >= -0.5_rt) {hi_ipts[hi_np++] = {-0.5_rt,hi_z};}
        }

        // x = 0.5
        if (hi_nz != 0._rt) {
            Real hi_z = hi_Cz -(( 0.5_rt-hi_Cx)*hi_nx + (-0.5_rt-hi_Cy)*hi_ny)/hi_nz;
            if (hi_z <= 0.5_rt && hi_z >= -0.5_rt) {hi_ipts[hi_np++] = {0.5_rt,hi_z};}
        }

        // z = -0.5
        if (hi_nx != 0._rt) {
            Real hi_x = hi_Cx -((-0.5_rt-hi_Cy)*hi_ny + (-0.5_rt-hi_Cz)*hi_nz)/hi_nx;
            if (hi_x <= 0.5_rt && hi_x >= -0.5_rt) {hi_ipts[hi_np++] = {hi_x,-0.5_rt};}
        }

        // z = 0.5
        if (hi_nx != 0._rt) {
            Real hi_x = hi_Cx -((-0.5_rt-hi_Cy)*hi_ny + (0.5_rt-hi_Cz)*hi_nz)/hi_nx;
            if (hi_x <= 0.5_rt && hi_x >= -0.5_rt) {hi_ipts[hi_np++] = {hi_x,0.5_rt};}
        }
    }

    sanitize_ipts(lo_ipts, lo_np);
    sanitize_ipts(hi_ipts, hi_np);

    GpuArray<EBFaceTriangle, 4> triangles;
    int nt = 0;

    if (lo_np == 0 && hi_np == 2) {
        bool sign = true;
        if (flag(i,j-1,k).isSingleValued()) {
            if (bnorm(i,j-1,k,1) > 0._rt) {
                sign = false;
            }
        } else {
            sign = flag(i,j-1,k).isRegular();
        }
        extract_triangle_on_face_hi(triangles, nt, hi_ipts, hi_np, sign);
    } else if (lo_np == 2 && hi_np ==0) {
        extract_triangle_on_face_lo(triangles, nt, lo_ipts, lo_np, bnorm(i,j-1,k,0), bnorm(i,j-1,k,2));
    } else if (lo_np == 2 && hi_np == 2) {
        extract_triangles_on_face(triangles, nt, lo_ipts, lo_np, hi_ipts, hi_np,
                                  bnorm(i,j-1,k,0), bnorm(i,j-1,k,2),
                                  bnorm(i,j,k,0), bnorm(i,j,k,2));
    }

    for (int n = 0; n < nt; ++n) {
        add_to_mmintg(i,j,k,mmintg_y,triangles[n]);
    }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void compute_mmintg_on_xy_face(int i, int j, int k,
                               Array4<Real> const& mmintg_z,
                               Array4<Real const> const& bcent,
                               Array4<Real const> const& bnorm,
                               Array4<EBCellFlag const> const& flag) noexcept
{
    mmintg_z(i,j,k, i_M_1)  = 0._rt;
    mmintg_z(i,j,k, i_M_x)  = 0._rt;
    mmintg_z(i,j,k, i_M_y)  = 0._rt;
    mmintg_z(i,j,k, i_M_xy) = 0._rt;

    GpuArray<Array<Real,2>, 4> lo_ipts;
    GpuArray<Array<Real,2>, 4> hi_ipts;
    int lo_np = 0;
    int hi_np = 0;

    if (!(flag(i,j,k-1).isSingleValued() || flag(i,j,k).isSingleValued())) {
        return;
    }

    if (flag(i,j,k-1).isSingleValued()) {
        // lo side z =  0.5
        Real lo_Cx = bcent(i,j,k-1,0);
        Real lo_Cy = bcent(i,j,k-1,1);
        Real lo_Cz = bcent(i,j,k-1,2);
        Real lo_nx = bnorm(i,j,k-1,0);
        Real lo_ny = bnorm(i,j,k-1,1);
        Real lo_nz = bnorm(i,j,k-1,2);

        // x = -0.5
        if (lo_ny != 0._rt) {
            Real lo_y = lo_Cy -((-0.5_rt-lo_Cx)*lo_nx + (0.5_rt-lo_Cz)*lo_nz)/lo_ny;
            if (lo_y <= 0.5_rt && lo_y >= -0.5_rt) {lo_ipts[lo_np++] = {-0.5_rt,lo_y};}
        }

        // x = 0.5
        if (lo_ny != 0._rt) {
            Real lo_y = lo_Cy -(( 0.5_rt-lo_Cx)*lo_nx + (0.5_rt-lo_Cz)*lo_nz)/lo_ny;
            if (lo_y <= 0.5_rt && lo_y >= -0.5_rt) {lo_ipts[lo_np++] = {0.5_rt,lo_y};}
        }

        // y = -0.5
        if (lo_nx != 0._rt) {
            Real lo_x = lo_Cx -((-0.5_rt-lo_Cy)*lo_ny + (0.5_rt-lo_Cz)*lo_nz)/lo_nx;
            if (lo_x <= 0.5_rt && lo_x >= -0.5_rt) {lo_ipts[lo_np++] = {lo_x,-0.5_rt};}
        }

        // y = 0.5
        if (lo_nx != 0._rt) {
            Real lo_x = lo_Cx -(( 0.5_rt-lo_Cy)*lo_ny + (0.5_rt-lo_Cz)*lo_nz)/lo_nx;
            if (lo_x <= 0.5_rt && lo_x >= -0.5_rt) {lo_ipts[lo_np++] = {lo_x,0.5_rt};}
        }
    }

    if (flag(i,j,k).isSingleValued()) {
        // hi side z = -0.5
        Real hi_Cx = bcent(i,j,k,0);
        Real hi_Cy = bcent(i,j,k,1);
        Real hi_Cz = bcent(i,j,k,2);
        Real hi_nx = bnorm(i,j,k,0);
        Real hi_ny = bnorm(i,j,k,1);
        Real hi_nz = bnorm(i,j,k,2);


        // x = -0.5
        if (hi_ny != 0_rt) {
            Real hi_y = hi_Cy -((-0.5_rt-hi_Cx)*hi_nx + (-0.5_rt-hi_Cz)*hi_nz)/hi_ny;
            if (hi_y <= 0.5_rt && hi_y >= -0.5_rt) {hi_ipts[hi_np++] = {-0.5_rt,hi_y};}
        }

        // x = 0.5
        if (hi_ny != 0._rt) {
            Real hi_y = hi_Cy -(( 0.5_rt-hi_Cx)*hi_nx + (-0.5_rt-hi_Cz)*hi_nz)/hi_ny;
            if (hi_y <= 0.5_rt && hi_y >= -0.5_rt) {hi_ipts[hi_np++] = {0.5_rt,hi_y};}
        }

        // y = -0.5
        if (hi_nx != 0._rt) {
            Real hi_x = hi_Cx -((-0.5_rt-hi_Cy)*hi_ny + (-0.5_rt-hi_Cz)*hi_nz)/hi_nx;
            if (hi_x <= 0.5_rt && hi_x >= -0.5_rt) {hi_ipts[hi_np++] = {hi_x,-0.5_rt};}
        }

        // y = 0.5
        if (hi_nx != 0._rt) {
            Real hi_x = hi_Cx -(( 0.5_rt-hi_Cy)*hi_ny + (-0.5_rt-hi_Cz)*hi_nz)/hi_nx;
            if (hi_x <= 0.5_rt && hi_x >= -0.5_rt) {hi_ipts[hi_np++] = {hi_x,0.5_rt};}
        }
    }

    sanitize_ipts(lo_ipts, lo_np);
    sanitize_ipts(hi_ipts, hi_np);

    GpuArray<EBFaceTriangle, 4> triangles;
    int nt = 0;

    if (lo_np == 0 && hi_np == 2) {
        bool sign = true;
        if (flag(i,j,k-1).isSingleValued()) {
            if (bnorm(i,j,k-1,2) > 0._rt) {
                sign = false;
            }
        } else {
            sign = flag(i,j,k-1).isRegular();
        }
        extract_triangle_on_face_hi(triangles, nt, hi_ipts, hi_np, sign);
    } else if (lo_np == 2 && hi_np ==0) {
        extract_triangle_on_face_lo(triangles, nt, lo_ipts, lo_np, bnorm(i,j,k-1,0), bnorm(i,j,k-1,1));
    } else if (lo_np == 2 && hi_np == 2) {
        extract_triangles_on_face(triangles, nt, lo_ipts, lo_np, hi_ipts, hi_np,
                                  bnorm(i,j,k-1,0), bnorm(i,j,k-1,1),
                                  bnorm(i,j,k,0), bnorm(i,j,k,1));
    }

    for (int n = 0; n < nt; ++n) {
        add_to_mmintg(i,j,k,mmintg_z,triangles[n]);
    }
}

}

#endif
