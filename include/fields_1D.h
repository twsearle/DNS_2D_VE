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

// Last modified: Tue  6 Oct 17:00:38 2015

#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<math.h>
#include<complex.h>
#include"hdf5.h"
#include"fftw3.h"
#include"fields_IO.h"

// Macros

#define complex_d _Complex double
#define ind(i, j) (M*(i) + (j))

// Prototypes

typedef struct lin_flow_scratch lin_flow_scratch;

void single_dx(fftw_complex *arrin, fftw_complex *arrout, int fou, flow_params cnsts);

void single_d2x(fftw_complex *arrin, fftw_complex *arrout, int fou, flow_params cnsts);

void single_d4x(fftw_complex *arrin, fftw_complex *arrout, int fou, flow_params cnsts);

void single_dy(fftw_complex *arrin, fftw_complex *arrout,  flow_params cnsts);

void to_cheby_physical(complex_d *arrin, complex_d *arrout, lin_flow_scratch scr,
	flow_params cnsts);

void to_cheby_spectral(complex_d *arrin, complex_d *arrout, lin_flow_scratch scr,  flow_params cnsts);

void fft_cheby_convolve(complex_d *arr1, complex_d *arr2, complex_d *arrout, lin_flow_scratch scr, flow_params cnsts);

double calc_cheby_KE_mode(fftw_complex *u, fftw_complex *v, int n, flow_params cnsts);

struct lin_flow_scratch {
    complex_d *scratch, *scratch2, *scratch3, *scratch4, *scratch5;
    complex_d *U0, *u, *v, *udxlplpsi, *vdylplpsi, *biharmpsi, *lplpsi;

    complex_d *d2yPSI0, *d3yPSI0;

    complex_d *d2ypsi, *d3ypsi;
    complex_d *d4ypsi, *d4xpsi, *d2xd2ypsi;
    complex_d *dxu, *dyu, *dxv, *dyv;

    complex_d *d2ycxy, *d2xcxy, *dxycyy_cxx, *dycxy, *dycxy0;
    complex_d *d2ycxyN, *d2xcxyN, *dxycyy_cxxN, *dycxyN, *dycxy0N;

    complex_d *cxxdxu, *cxydyu, *vgradcxx, *cxydxv, *cyydyv;
    complex_d *cxy0dyU0, *cyy0dyU0;
    complex_d *vgradcyy, *cxxdxv, *cyydyu, *vgradcxy;

    fftw_complex *scratchin, *scratchout;

    complex_d *scratchp1, *scratchp2;

    fftw_complex *RHSvec;
    
    fftw_plan *phys_plan, *spec_plan;
};

#endif // FIELDS_1D_C_H
