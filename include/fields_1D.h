#ifndef FIELDS_1D_C_H
#define FIELDS_1D_C_H
/* -------------------------------------------------------------------------- *
 *									      *
 *  fields_1D.h								      *
 *                                                                            *
 *  Header File for Chebyshev 1D fields					      *
 *                                                                            *
 *                                                                            *
 * -------------------------------------------------------------------------- */

// Last modified: Wed 30 Sep 12:15:13 2015

#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<math.h>
#include<complex.h>
#include"hdf5.h"
#include"fftw3.h"

// Macros

#define complex_D _Complex double
#define ind(i, j) (M*(i) + (j))
//#define ind(i, j) ((2*M-2)*(i) + (j))
#define indfft(i, j) ((2*Mf-2)*(i) + (j))

// Prototypes

typedef struct flow_params flow_params;
typedef struct complex_hdf complex_hdf;
typedef struct flow_scratch flow_scratch;

void single_dx(fftw_complex *arrin, fftw_complex *arrout,  flow_params cnsts);

void single_d2x(fftw_complex *arrin, fftw_complex *arrout,  flow_params cnsts);

void single_d4x(fftw_complex *arrin, fftw_complex *arrout,  flow_params cnsts);

void single_dy(fftw_complex *arrin, fftw_complex *arrout,  flow_params cnsts);

void to_cheby_physical(complex *arrin, double *arrout, flow_scratch scr,
	flow_params cnsts);

void to_cheby_spectral(double *arrin, complex *arrout,
	flow_scratch scr,  flow_params cnsts);

void fft_cheby_convolve(complex *arr1, complex *arr2, complex *arrout,
		    flow_scratch scr, flow_params cnsts);

double calc_KE_mode(fftw_complex *u, fftw_complex *v, int n, flow_params cnsts);

int trC_tensor(complex *cij, complex *trC, flow_scratch scr, flow_params cnsts);

void diagonalised_C(complex *cij, complex *ecij, double *rcij,
	flow_scratch scr, flow_params cnsts);

double calc_EE_mode(complex *trC, int n, flow_params cnsts);

struct flow_params {
    int N;
    int M;
    int dealiasing;
    int Nf;
    int Mf;
    double kx;
    double U0;
    double Re;
    double Wi;
    double beta;
    double Omega;
};

struct flow_scratch {
    complex *scratch, *scratch2, *scratch3, *scratch4, *scratch5;
    complex *U0, *u, *v, *udxlplpsi, *vdylplpsi, *biharmpsi, *lplpsi;
    complex *d2ypsi;
    complex *dyyyPSI0, *dypsi, *vdyypsi;
    complex *d4ypsi, *d4xpsi, *d2xd2ypsi;
    complex *dxu, *dyu, *dxv, *dyv;

    complex *d2ycxy, *d2xcxy, *dxycyy_cxx, *dycxy;
    complex *d2ycxyN, *d2xcxyN, *dxycyy_cxxN, *dycxyN;

    complex *cxxdxu, *cxydyu, *vgradcxx, *cxydxv, *cyydyv;
    complex *vgradcyy, *cxxdxv, *cyydyu, *vgradcxy;

    fftw_complex *scratchin, *scratchout;

    double *scratchp1, *scratchp2;

    fftw_complex *RHSvec;
    
    fftw_complex *opsList, *hopsList, *tmpop;

    fftw_plan *phys_plan, *spec_plan;
};


struct complex_hdf {
    double r;
    double i;
};

#endif // FIELDS_1D_C_H
