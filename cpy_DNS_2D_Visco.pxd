# /* -------------------------------------------------------------------------- *
#  *									        *
#  *  cpy_DNS_2D_Visco.pxd						        *
#  *                                                                            *
#  *  Time stepping DNS simulation cython header                                *
#  *                                                                            *
#  *                                                                            *
#  * -------------------------------------------------------------------------- */

# Last modified: Sun  4 Mar 11:11:40 2018

cimport numpy as np
# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef np.double_t DTYPE_DOUB_t
ctypedef np.complex128_t DTYPE_CMPLX_t

cdef extern from "include/DNS_2D_Visco.h":

    ctypedef struct flow_params:
        int N;
        int M;
        int dealiasing;
        int oscillatory_flow;
        int Nf;
        int Mf;
        double kx;
        double U0;
        double Re;
        double Wi;
        double beta;
        double Omega;
        double De;
        double P;
        double dt;
        int stepsPerFrame;
        int numTimeSteps;
        double initTime;

cdef extern from "include/fields_2D.h":

    ctypedef struct flow_scratch:
        DTYPE_CMPLX_t *scratch;
        DTYPE_CMPLX_t *scratch2;
        DTYPE_CMPLX_t *scratch3;
        DTYPE_CMPLX_t *scratch4;
        DTYPE_CMPLX_t *scratch5;
        DTYPE_CMPLX_t *U0;
        DTYPE_CMPLX_t *u;
        DTYPE_CMPLX_t *v;
        DTYPE_CMPLX_t *udxlplpsi;
        DTYPE_CMPLX_t *vdylplpsi;
        DTYPE_CMPLX_t *biharmpsi;
        DTYPE_CMPLX_t *lplpsi;
        DTYPE_CMPLX_t *dyyypsi;
        DTYPE_CMPLX_t *dypsi;
        DTYPE_CMPLX_t *vdyypsi;
        DTYPE_CMPLX_t *d4ypsi;
        DTYPE_CMPLX_t *d4xpsi;
        DTYPE_CMPLX_t *d2xd2ypsi;
        DTYPE_CMPLX_t *dxu;
        DTYPE_CMPLX_t *dyu;
        DTYPE_CMPLX_t *dxv;
        DTYPE_CMPLX_t *dyv;

        DTYPE_CMPLX_t *d2ycxy;
        DTYPE_CMPLX_t *d2xcxy;
        DTYPE_CMPLX_t *dxycyy_cxx;
        DTYPE_CMPLX_t *dycxy;
        DTYPE_CMPLX_t *d2ycxyN;
        DTYPE_CMPLX_t *d2xcxyN;
        DTYPE_CMPLX_t *dxycyy_cxxN;
        DTYPE_CMPLX_t *dycxyN;

        DTYPE_CMPLX_t *cxxdxu;
        DTYPE_CMPLX_t *cxydyu;
        DTYPE_CMPLX_t *vgradcxx;
        DTYPE_CMPLX_t *cxydxv;
        DTYPE_CMPLX_t *cyydyv;
        DTYPE_CMPLX_t *vgradcyy;
        DTYPE_CMPLX_t *cxxdxv;
        DTYPE_CMPLX_t *cyydyu;
        DTYPE_CMPLX_t *vgradcxy;

        DTYPE_CMPLX_t *scratchin;
        DTYPE_CMPLX_t *scratchout;

        DTYPE_CMPLX_t *RHSvec;

        DTYPE_CMPLX_t *opsList;
        DTYPE_CMPLX_t *hopsList;
        DTYPE_CMPLX_t *tmpop;

        DTYPE_DOUB_t *scratchp1;
        DTYPE_DOUB_t *scratchp2;
        DTYPE_DOUB_t *scratchp3;

        #fftw_plan *phys_plan, *spec_plan;

        #fftw_plan act_phys_plan, act_spec_plan;


cdef extern from "include/DNS_2D_Visco.h":
    int DNS_2D_Visco(double complex *psi, double complex *cij, double complex
                     *forcing, double complex *psi_lam, double complex *opsList,
                     double complex *hopsList, flow_params params);
