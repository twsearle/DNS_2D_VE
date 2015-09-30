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

// Last modified: Wed 30 Sep 12:55:08 2015

#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<math.h>
#include<complex.h>
#include"hdf5.h"
#include"fftw3.h"

// Macros

#define complex_d complex double
#define ind(i, j) (M*(i) + (j))
//#define ind(i, j) ((2*M-2)*(i) + (j))
#define indfft(i, j) ((2*Mf-2)*(i) + (j))

// Prototypes

typedef struct flow_params flow_params;
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
    complex_d *scratch, *scratch2, *scratch3, *scratch4, *scratch5;
    complex_d *U0, *u, *v, *udxlplpsi, *vdylplpsi, *biharmpsi, *lplpsi;
    complex_d *d2ypsi;
    complex_d *dyyypsi, *dypsi, *vdyypsi;
    complex_d *d4ypsi, *d4xpsi, *d2xd2ypsi;
    complex_d *dxu, *dyu, *dxv, *dyv;

    complex_d *d2ycxy, *d2xcxy, *dxycyy_cxx, *dycxy;
    complex_d *d2ycxyN, *d2xcxyN, *dxycyy_cxxN, *dycxyN;

    complex_d *cxxdxu, *cxydyu, *vgradcxx, *cxydxv, *cyydyv;
    complex_d *vgradcyy, *cxxdxv, *cyydyu, *vgradcxy;

    fftw_complex *scratchin, *scratchout;

    double *scratchp1, *scratchp2;

    fftw_complex *RHSvec;
    
    fftw_complex *opsList, *hopsList, *tmpop;

    fftw_plan *phys_plan, *spec_plan;
};

#endif // FIELDS_2D_C_H
