#ifndef FIELDS_2D_C_H
#define FIELDS_2D_C_H
/* -------------------------------------------------------------------------- *
 *									      *
 *  fields_2D_c.h							      *
 *                                                                            *
 *  Header File for 2D fields						      *
 *                                                                            *
 *                                                                            *
 * -------------------------------------------------------------------------- */

// Last modified: Wed 28 Sep 17:37:01 2016

#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<math.h>
#include<complex.h>
#include"hdf5.h"
#include"fftw3.h"
#include"fields_IO.h"

// Macros

#define ind(i, j) (M*(i) + (j))
//#define ind(i, j) ((2*M-2)*(i) + (j))
#define indfft(i, j) ((2*Mf-2)*(i) + (j))

// Prototypes

typedef struct flow_scratch flow_scratch;

void dx(fftw_complex *arrin, fftw_complex *arrout,  flow_params cnsts);

void d2x(fftw_complex *arrin, fftw_complex *arrout,  flow_params cnsts);

void d4x(fftw_complex *arrin, fftw_complex *arrout,  flow_params cnsts);

void dy(fftw_complex *arrin, fftw_complex *arrout,  flow_params cnsts);

void matdy(fftw_complex *matarr, flow_params cnsts);

void to_physical_1(fftw_complex *arrin, fftw_complex *arrout,
		 fftw_complex *scratchFin, fftw_complex *scratchFout,
		 fftw_complex *scratchCin, fftw_complex *scratchCout,
		 fftw_plan *phys_fou_plan, fftw_plan *phys_cheb_plan,
		 flow_params cnsts);

void to_physical_r(complex_d *arrin, double *arrout, flow_scratch scr,
	flow_params cnsts);

void to_spectral_1(fftw_complex *arrin, fftw_complex *arrout, fftw_complex *scratch,
		 fftw_complex *scratchFin, fftw_complex *scratchFout,
		 fftw_complex *scratchCin, fftw_complex *scratchCout,
		 fftw_plan *spec_fou_plan, fftw_plan *spec_cheb_plan,
		 flow_params cnsts);

void to_spectral_r(double *arrin, complex_d *arrout,
	flow_scratch scr,  flow_params cnsts);

void fft_convolve_r(complex_d *arr1, complex_d *arr2, complex_d *arrout,
		    flow_scratch scr, flow_params cnsts);

double calc_KE_mode(fftw_complex *u, fftw_complex *v, int n, flow_params cnsts);

int trC_tensor(complex_d *cij, complex_d *trC, flow_scratch scr, flow_params cnsts);

void diagonalised_C(complex_d *cij, complex_d *ecij, double *rcij,
	flow_scratch scr, flow_params cnsts);

double calc_EE_mode(complex_d *trC, int n, flow_params cnsts);

struct flow_scratch {
    complex_d *scratch, *scratch2, *scratch3, *scratch4, *scratch5;
    complex_d *U0, *u, *v, *udxlplpsi, *vdylplpsi, *biharmpsi, *lplpsi;
    complex_d *dyyypsi, *dypsi, *vdyypsi;
    complex_d *d4ypsi, *d4xpsi, *d2xd2ypsi;
    complex_d *dxu, *dyu, *dxv, *dyv;

    complex_d *d2ycxy, *d2xcxy, *dxycyy_cxx, *dycxy;
    complex_d *d2ycxyN, *d2xcxyN, *dxycyy_cxxN, *dycxyN;

    complex_d *cxxdxu, *cxydyu, *vgradcxx, *cxydxv, *cyydyv;
    complex_d *vgradcyy, *cxxdxv, *cyydyu, *vgradcxy;

    fftw_complex *scratchin, *scratchout;

    double *scratchp1, *scratchp2, *scratchp3;

    fftw_complex *RHSvec;
    
    fftw_complex *opsList, *hopsList, *tmpop;

    fftw_plan *phys_plan, *spec_plan;

    // A stupid fix. I began by only carrying a pointer to the plan but have
    // decided to switch to carrying the full plan
    fftw_plan act_phys_plan, act_spec_plan;
};

#endif // FIELDS_2D_C_H
