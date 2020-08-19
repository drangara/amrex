#ifndef AMREX_MLEB_LEASTSQUARES_K_H_
#define AMREX_MLEB_LEASTSQUARES_K_H_

#include <AMReX_EBCellFlag.H>

namespace amrex {

// This is the 2D version (i.e. for 6x6 (A^T A) matrix)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void decomp_chol_np6(Array2D<Real,0,5,0,5>& aa)
{
    int neq = 6;

    Real p[neq];
    Real sum1;
    int ising;

    for (int ii = 0; ii < neq; ii++)
    {
        ising = 0;

        for (int jj = ii; jj < neq; jj++)
        {
            sum1 = aa(ii,jj);

            for (int kk = ii-1; kk >= 0; kk--)
            {
                sum1 = sum1 - aa(ii,kk)*aa(jj,kk);
            }
 
            if (ii == jj)
            {
                 if (sum1 <= 0.) 
                 {
                     p[ii] = 0.0;
                     ising = 1;
                 } else {
                     p[ii] = std::sqrt(sum1);
                 }
            } else {
                if (ising == 0) 
                   aa(jj,ii) = sum1 / p[ii];
                else
                   aa(jj,ii) = 0.0;
            }
        }
    }

    for (int ii = 0; ii < neq; ii++)
    {
        for (int jj = ii+1; jj < neq; jj++)
        {
           aa(ii,jj) = 0.0;       // Zero upper triangle
           aa(ii,jj) = aa(jj,ii);  // Zero upper triangle
        }
          
        aa(ii,ii) = p[ii];
    }
}

// This is the 2D version (i.e. for 6x6 (A^T A) matrix)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void  cholsol_np6(Array2D<Real,0,11,0,5>& Amatrix, Array1D<Real,0,5>& b)
{
    int neq = 6;

    Array2D<Real,0,5,0,5> AtA;

    for (int irow = 0; irow < neq; irow++)
        for (int icol = 0; icol < neq; icol++)
            AtA(irow,icol) =  0.0;

    for (int irow = 0; irow < 12; irow++)
    {
         AtA(0,0) += Amatrix(irow,0)*Amatrix(irow,0); // e^T e
         AtA(0,1) += Amatrix(irow,0)*Amatrix(irow,1); // e^T x
         AtA(0,2) += Amatrix(irow,0)*Amatrix(irow,2); // e^T y
         AtA(0,3) += Amatrix(irow,0)*Amatrix(irow,3); // e^T x^2
         AtA(0,4) += Amatrix(irow,0)*Amatrix(irow,4); // e^T x*y
         AtA(0,5) += Amatrix(irow,0)*Amatrix(irow,5); // e^T y^2
         AtA(1,1) += Amatrix(irow,1)*Amatrix(irow,1); // x^T x
         AtA(1,2) += Amatrix(irow,1)*Amatrix(irow,2); // x^T y
         AtA(1,3) += Amatrix(irow,1)*Amatrix(irow,3); // x^T (x^2)
         AtA(1,4) += Amatrix(irow,1)*Amatrix(irow,4); // x^T (xy)
         AtA(1,5) += Amatrix(irow,1)*Amatrix(irow,5); // x^T (y^2)
         AtA(2,2) += Amatrix(irow,2)*Amatrix(irow,2); // y^T y
         AtA(2,3) += Amatrix(irow,2)*Amatrix(irow,3); // y^T (x^2)
         AtA(2,4) += Amatrix(irow,2)*Amatrix(irow,4); // y^T (xy)
         AtA(2,5) += Amatrix(irow,2)*Amatrix(irow,5); // y^T (y^2)
         AtA(3,3) += Amatrix(irow,3)*Amatrix(irow,3); // (x^2)^T (x^2)
         AtA(3,4) += Amatrix(irow,3)*Amatrix(irow,4); // (x^2)^T (x*y)
         AtA(3,5) += Amatrix(irow,3)*Amatrix(irow,5); // (x^2)^T (y^2)
         AtA(4,4) += Amatrix(irow,4)*Amatrix(irow,4); // (x*y)^T (x*y)
         AtA(4,5) += Amatrix(irow,4)*Amatrix(irow,5); // (x*y)^T (y^2)
         AtA(5,5) += Amatrix(irow,5)*Amatrix(irow,5); // (y^2)^T (y^2) 
    }

    for (int irow = 0; irow < neq-1; irow++)
        for (int icol = irow+1; icol < neq; icol++)
           AtA(icol,irow) = AtA(irow,icol);

    decomp_chol_np6(AtA);

    if (AtA(0,0) > 0.)
       b(0) = b(0) / AtA(0,0);
    else
       b(0) = 0.;

    for (int ii = 1; ii < neq; ii++)
    {
       if (AtA(ii,ii) > 0.)
       {
          for (int jj = 0; jj < ii; jj++)
            b(ii) = b(ii) - AtA(ii,jj)*b(jj);

          b(ii) = b(ii) / AtA(ii,ii);
       }
       else
       {
          b(ii) = 0.0;
       }
    }

    if (AtA(neq-1,neq-1) > 0.)
       b(neq-1) = b(neq-1) / AtA(neq-1,neq-1);
    else
       b(neq-1) = 0.0;

    for (int ii = neq-2; ii >= 0; ii--)
    {
       if (AtA(ii,ii) > 0.)
       {
          for (int jj = ii+1; jj < neq; jj++)
            b(ii) = b(ii) - AtA(ii,jj)*b(jj);

          b(ii) = b(ii) / AtA(ii,ii);
       }
       else
       {
          b(ii) = 0.0;
       }
    }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void  cholsol_for_eb(Array2D<Real,0,17,0,5>& Amatrix, Array1D<Real,0,5>& b)
{
    int neq = 6;

    Array2D<Real,0,5,0,5> AtA;

    for (int irow = 0; irow < neq; irow++)
        for (int icol = 0; icol < neq; icol++)
            AtA(irow,icol) =  0.0;

    for (int irow = 0; irow < 18; irow++)
    {
         AtA(0,0) += Amatrix(irow,0)*Amatrix(irow,0); // e^T e
         AtA(0,1) += Amatrix(irow,0)*Amatrix(irow,1); // e^T x
         AtA(0,2) += Amatrix(irow,0)*Amatrix(irow,2); // e^T y
         AtA(0,3) += Amatrix(irow,0)*Amatrix(irow,3); // e^T x^2
         AtA(0,4) += Amatrix(irow,0)*Amatrix(irow,4); // e^T x*y
         AtA(0,5) += Amatrix(irow,0)*Amatrix(irow,5); // e^T y^2
         AtA(1,1) += Amatrix(irow,1)*Amatrix(irow,1); // x^T x
         AtA(1,2) += Amatrix(irow,1)*Amatrix(irow,2); // x^T y
         AtA(1,3) += Amatrix(irow,1)*Amatrix(irow,3); // x^T (x^2)
         AtA(1,4) += Amatrix(irow,1)*Amatrix(irow,4); // x^T (xy)
         AtA(1,5) += Amatrix(irow,1)*Amatrix(irow,5); // x^T (y^2)
         AtA(2,2) += Amatrix(irow,2)*Amatrix(irow,2); // y^T y
         AtA(2,3) += Amatrix(irow,2)*Amatrix(irow,3); // y^T (x^2)
         AtA(2,4) += Amatrix(irow,2)*Amatrix(irow,4); // y^T (xy)
         AtA(2,5) += Amatrix(irow,2)*Amatrix(irow,5); // y^T (y^2)
         AtA(3,3) += Amatrix(irow,3)*Amatrix(irow,3); // (x^2)^T (x^2)
         AtA(3,4) += Amatrix(irow,3)*Amatrix(irow,4); // (x^2)^T (x*y)
         AtA(3,5) += Amatrix(irow,3)*Amatrix(irow,5); // (x^2)^T (y^2)
         AtA(4,4) += Amatrix(irow,4)*Amatrix(irow,4); // (x*y)^T (x*y)
         AtA(4,5) += Amatrix(irow,4)*Amatrix(irow,5); // (x*y)^T (y^2)
         AtA(5,5) += Amatrix(irow,5)*Amatrix(irow,5); // (y^2)^T (y^2) 
    }

    for (int irow = 0; irow < neq-1; irow++)
        for (int icol = irow+1; icol < neq; icol++)
           AtA(icol,irow) = AtA(irow,icol);

    decomp_chol_np6(AtA);

    if (AtA(0,0) > 0.)
       b(0) = b(0) / AtA(0,0);
    else
       b(0) = 0.;

    for (int ii = 1; ii < neq; ii++)
    {
       if (AtA(ii,ii) > 0.)
       {
          for (int jj = 0; jj < ii; jj++)
            b(ii) = b(ii) - AtA(ii,jj)*b(jj);

          b(ii) = b(ii) / AtA(ii,ii);
       }
       else
       {
          b(ii) = 0.0;
       }
    }

    if (AtA(neq-1,neq-1) > 0.)
       b(neq-1) = b(neq-1) / AtA(neq-1,neq-1);
    else
       b(neq-1) = 0.0;

    for (int ii = neq-2; ii >= 0; ii--)
    {
       if (AtA(ii,ii) > 0.)
       {
          for (int jj = ii+1; jj < neq; jj++)
            b(ii) = b(ii) - AtA(ii,jj)*b(jj);

          b(ii) = b(ii) / AtA(ii,ii);
       }
       else
       {
          b(ii) = 0.0;
       }
    }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real grad_x_of_phi_on_centroids(int i,int j,int k,int n,
                                Array4<Real const> const& phi, 
                                Array4<Real const> const& phieb, 
                                Array4<EBCellFlag const> const& flag,
                                Array4<Real const> const& ccent,
                                Array4<Real const> const& bcent, 
                                Array4<Real const> const& apx, 
                                Array4<Real const> const& apy, 
                                Real& yloc_on_xface, 
                                bool is_eb_dirichlet, bool is_eb_inhomog)
{
    Array2D<Real,0,11,0,5> Amatrix;
    Array1D<Real,0, 5    > rhs;

    // Order of column -- first  six are cell centroids: (i,j-1) (i,j) (i,j+1) (i-1,j-1) (i-1,j) (i-1,j+1)
    // Order of column -- second six are   EB centroids: (i,j-1) (i,j) (i,j+1) (i-1,j-1) (i-1,j) (i-1,j+1)

    for (int irow = 0; irow < 12; irow++)
        for (int icol = 0; icol < 6; icol++)
            Amatrix(irow,icol) =  0.0;

    // Columns: [e x y x*x x*y y*y]
    for (int ii = i-1; ii <= i; ii++) // Normal to face
        for (int jj = j-1; jj <= j+1; jj++)  // Tangential to face
        {
            if (!flag(ii,jj,k).isCovered())
            {
                int a_ind  = (jj-(j-1)) + 3*(ii-(i-1));
    
                Real x_off = static_cast<Real>(ii-i) + 0.5;
                Real y_off = static_cast<Real>(jj-j); 
    
                Amatrix(a_ind,0) = 1.0;
                Amatrix(a_ind,1) = x_off + ccent(ii,jj,k,0);
                Amatrix(a_ind,2) = y_off + ccent(ii,jj,k,1) - yloc_on_xface;
                Amatrix(a_ind,3) = Amatrix(a_ind,1) * Amatrix(a_ind,1);
                Amatrix(a_ind,4) = Amatrix(a_ind,1) * Amatrix(a_ind,2);
                Amatrix(a_ind,5) = Amatrix(a_ind,2) * Amatrix(a_ind,2);

                if (!flag(ii,jj,k).isRegular())
                {
                    int a_ind_eb  = a_ind + 6;
                    if (is_eb_dirichlet) {
                        Amatrix(a_ind_eb,0) = 1.0;
                        Amatrix(a_ind_eb,1) = x_off + bcent(ii,jj,k,0);
                        Amatrix(a_ind_eb,2) = y_off + bcent(ii,jj,k,1) - yloc_on_xface; 
                        Amatrix(a_ind_eb,3) = Amatrix(a_ind_eb,1) * Amatrix(a_ind_eb,1);
                        Amatrix(a_ind_eb,4) = Amatrix(a_ind_eb,1) * Amatrix(a_ind_eb,2);
                        Amatrix(a_ind_eb,5) = Amatrix(a_ind_eb,2) * Amatrix(a_ind_eb,2);
                    } else { // Neumann bc's on EB face
                        Real dapx = apx(ii,jj,k)-apx(ii+1,jj,k);
                        Real dapy = apy(ii,jj,k)-apy(ii,jj+1,k);;
                        Real anorm = std::hypot(dapx,dapy);
                        Real norm_x = dapx / anorm;
                        Real norm_y = dapy / anorm;
                        Amatrix(a_ind_eb,0) = 0.0;
                        Amatrix(a_ind_eb,1) = norm_x;
                        Amatrix(a_ind_eb,2) = norm_y;
                        Amatrix(a_ind_eb,3) = 2.0*norm_x*(x_off + bcent(ii,jj,k,1));
                        Amatrix(a_ind_eb,4) = norm_x*(y_off + bcent(ii,jj,k,1) - yloc_on_xface)
                                             +norm_y*(x_off + bcent(ii,jj,k,0));
                        Amatrix(a_ind_eb,5) = 2.0*norm_y*(y_off + bcent(ii,jj,k,1) - yloc_on_xface);
                    }
                }
            }
        }

    // Make the RHS = A^T v
    for (int irow = 0; irow < 6; irow++)
    {
        rhs(irow) = 0.; // Only non-zero when inhomogeneous Dirichlet

        for (int ii = i-1; ii <= i; ii++)  // Normal to face
            for (int jj = j-1; jj <= j+1; jj++)  // Tangential to face
            {
                if (!flag(ii,jj,k).isCovered())
                {
                    int a_ind  = (jj-(j-1)) + 3*(ii-(i-1));
                    rhs(irow) += Amatrix(a_ind  ,irow)*  phi(ii,jj,k,n);

                    if (flag(ii,jj,k).isSingleValued() && 
                        is_eb_dirichlet && is_eb_inhomog)
                            rhs(irow) += Amatrix(a_ind+6,irow)*phieb(ii,jj,k,n);
                }
            }
    }

    cholsol_np6(Amatrix, rhs);

    return rhs(1);     
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real grad_y_of_phi_on_centroids(int i,int j,int k,int n,
                                Array4<Real const> const& phi, 
                                Array4<Real const> const& phieb, 
                                Array4<EBCellFlag const> const& flag,
                                Array4<Real const> const& ccent,
                                Array4<Real const> const& bcent, 
                                Array4<Real const> const& apx, 
                                Array4<Real const> const& apy, 
                                Real& xloc_on_yface,
                                bool is_eb_dirichlet, bool is_eb_inhomog)
{
    Array2D<Real,0,11,0,5> Amatrix;
    Array1D<Real,0, 5    > rhs;

    // Order of column -- first  six are cell centroids: (i-1,j) (i,j) (i+1,j) (i-1,j-1) (i,j-1) (i+1,j-1)
    // Order of column -- second six are   EB centroids: (i-1,j) (i,j) (i+1,j) (i-1,j-1) (i,j-1) (i+1,j-1)

    for (int irow = 0; irow < 12; irow++)
        for (int icol = 0; icol < 6; icol++)
            Amatrix(irow,icol) =  0.0;

    // Columns: [e x y x*x x*y y*y]
    for (int jj = j-1; jj <= j; jj++) // Normal to face
        for (int ii = i-1; ii <= i+1; ii++)  // Tangential to face
        {
            if (!flag(ii,jj,k).isCovered())
            {
                int a_ind  = (ii-(i-1)) + 3*(jj-(j-1));

                Real x_off = static_cast<Real>(ii-i); 
                Real y_off = static_cast<Real>(jj-j) + 0.5;
    
                Amatrix(a_ind,0) = 1.0;
                Amatrix(a_ind,1) = x_off + ccent(ii,jj,k,0) - xloc_on_yface;
                Amatrix(a_ind,2) = y_off + ccent(ii,jj,k,1);
                Amatrix(a_ind,3) = Amatrix(a_ind,1) * Amatrix(a_ind,1);
                Amatrix(a_ind,4) = Amatrix(a_ind,1) * Amatrix(a_ind,2);
                Amatrix(a_ind,5) = Amatrix(a_ind,2) * Amatrix(a_ind,2);

                if (!flag(ii,jj,k).isRegular())
                {
                    int a_ind_eb  = a_ind + 6;
                    if (is_eb_dirichlet) {
                        Amatrix(a_ind_eb,0) = 1.0;
                        Amatrix(a_ind_eb,1) = x_off + bcent(ii,jj,k,0) - xloc_on_yface;
                        Amatrix(a_ind_eb,2) = y_off + bcent(ii,jj,k,1);
                        Amatrix(a_ind_eb,3) = Amatrix(a_ind_eb,1) * Amatrix(a_ind_eb,1);
                        Amatrix(a_ind_eb,4) = Amatrix(a_ind_eb,1) * Amatrix(a_ind_eb,2);
                        Amatrix(a_ind_eb,5) = Amatrix(a_ind_eb,2) * Amatrix(a_ind_eb,2);
                    } else { // Neumann bc's on EB face
                        Real dapx = apx(ii,jj,k)-apx(ii+1,jj,k);
                        Real dapy = apy(ii,jj,k)-apy(ii,jj+1,k);;
                        Real anorm = std::hypot(dapx,dapy);
                        Real norm_x = dapx / anorm;
                        Real norm_y = dapy / anorm;
                        Amatrix(a_ind_eb,0) = 0.0;
                        Amatrix(a_ind_eb,1) = norm_x;
                        Amatrix(a_ind_eb,2) = norm_y;
                        Amatrix(a_ind_eb,3) = 2.0*norm_x*(x_off + bcent(ii,jj,k,0) - xloc_on_yface);
                        Amatrix(a_ind_eb,4) = norm_x*(y_off + bcent(ii,jj,k,1))
                                             +norm_y*(x_off + bcent(ii,jj,k,0) - xloc_on_yface);
                        Amatrix(a_ind_eb,5) = 2.0*norm_y*(y_off + bcent(ii,jj,k,1));
                    }
                }
            }
        }

    // Make the RHS = A^T v
    for (int irow = 0; irow < 6; irow++)
    {
        rhs(irow) = 0.; // Only non-zero when inhomogeneous Dirichlet

        for (int jj = j-1; jj <= j; jj++) // Normal to face
            for (int ii = i-1; ii <= i+1; ii++) // Tangential to face
                if (!flag(ii,jj,k).isCovered())
                {
                    int a_ind  = (ii-(i-1)) + 3*(jj-(j-1));
                    rhs(irow) += Amatrix(a_ind  ,irow)*  phi(ii,jj,k,n);

                    if (flag(ii,jj,k).isSingleValued() && 
                        is_eb_dirichlet && is_eb_inhomog)
                            rhs(irow) += Amatrix(a_ind+6,irow)*phieb(ii,jj,k,n);
                }
    }

    cholsol_np6(Amatrix, rhs);

    return rhs(2);     
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real grad_eb_of_phi_on_centroids(int i,int j,int k,int n,
                                Array4<Real const> const& phi, 
                                Array4<Real const> const& phieb, 
                                Array4<EBCellFlag const> const& flag,
                                Array4<Real const> const& ccent,
                                Array4<Real const> const& bcent,
                                Real& nrmx, Real& nrmy,
                                bool is_eb_inhomog)
{
    Array2D<Real,0,17,0,5> Amatrix;
    Array1D<Real,0, 5    > rhs;

    // Order of column -- first 9 are cell centroids, next 9 are EB centroids

    for (int irow = 0; irow < 18; irow++)
        for (int icol = 0; icol < 6; icol++)
            Amatrix(irow,icol) =  0.0;

    //  Column 0-2: [e x y]
    for (int ii = i-1; ii <= i+1; ii++)
        for (int jj = j-1; jj <= j+1; jj++)
        {
            if (!flag(ii,jj,k).isCovered())
            {
                int a_ind  = (jj-(j-1)) + 3*(ii-(i-1));
    
                Real x_off = static_cast<Real>(ii-i);
                Real y_off = static_cast<Real>(jj-j);
    
                if (flag(i,j,k).isConnected((jj-j),(ii-i),0))
                {
                   Amatrix(a_ind,0) = 1.0;
                   Amatrix(a_ind,1) = x_off + ccent(ii,jj,k,0) - bcent(i,j,k,0);
                   Amatrix(a_ind,2) = y_off + ccent(ii,jj,k,1) - bcent(i,j,k,1);
                }
#if 0
                if (i == ii and j == jj)
                {
                   Amatrix(a_ind,0) *= 2.0;
                   Amatrix(a_ind,1) *= 2.0;
                   Amatrix(a_ind,2) *= 2.0;
                }
#endif

                if (flag(i,j,k).isConnected((jj-j),(ii-i),0) and !flag(ii,jj,k).isRegular())
                {
                    Amatrix(a_ind+9,0) = 1.0;
                    Amatrix(a_ind+9,1) = x_off + bcent(ii,jj,k,0) - bcent(i,j,k,0);
                    Amatrix(a_ind+9,2) = y_off + bcent(ii,jj,k,1) - bcent(i,j,k,1);

#if 0
                    if (i == ii and j == jj)
                        Amatrix(a_ind+9,0) *= 2.0;
#endif
                }
            }
        }

    // Columns 3 : [x*x  x*y  y*y]

    for (int irow = 0; irow < 18; irow++)
    {
       Amatrix(irow,3) =  Amatrix(irow,1) * Amatrix(irow,1);
       Amatrix(irow,4) =  Amatrix(irow,1) * Amatrix(irow,2);
       Amatrix(irow,5) =  Amatrix(irow,2) * Amatrix(irow,2);
    }

    // Make the RHS = A^T v
    for (int irow = 0; irow < 6; irow++)
    {
        rhs(irow) = 0.;

        for (int ii = i-1; ii <= i+1; ii++)
            for (int jj = j-1; jj <= j+1; jj++)
                if (!flag(ii,jj,k).isCovered())
                {
                    int a_ind  = (jj-(j-1)) + 3*(ii-(i-1));
                    rhs(irow) += Amatrix(a_ind,irow) * phi(ii,jj,k,n);
                    if (flag(ii,jj,k).isSingleValued() and is_eb_inhomog)
                        rhs(irow) += Amatrix(a_ind+9,irow)*phieb(ii,jj,k,n);
                }
    }

    cholsol_for_eb(Amatrix, rhs);

    Real dphidn = rhs(1)*nrmx + rhs(2)*nrmy;
    return dphidn;
}

}
#endif
