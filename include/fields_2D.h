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

// Last modified: Mon 15 Jun 22:11:12 2015

#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<math.h>
#include<complex.h>
#include"hdf5.h"
#include"fftw3.h"

// Macros

#define ind(i, j) (M*(i) + (j))
//#define ind(i, j) ((2*M-2)*(i) + (j))
#define indfft(i, j) ((2*Mf-2)*(i) + (j))

// Prototypes

typedef struct flow_params flow_params;
typedef struct complex_hdf complex_hdf;
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

void to_physical_r(complex *arrin, double *arrout, flow_scratch scr,
	flow_params cnsts);

void to_spectral_1(fftw_complex *arrin, fftw_complex *arrout, fftw_complex *scratch,
		 fftw_complex *scratchFin, fftw_complex *scratchFout,
		 fftw_complex *scratchCin, fftw_complex *scratchCout,
		 fftw_plan *spec_fou_plan, fftw_plan *spec_cheb_plan,
		 flow_params cnsts);

void to_spectral_r(double *arrin, complex *arrout,
	flow_scratch scr,  flow_params cnsts);

void fft_convolve_r(complex *arr1, complex *arr2, complex *arrout,
		    flow_scratch scr, flow_params cnsts);

void save_hdf5_state(char *filename, fftw_complex *arr, flow_params cnsts);

void save_hdf5_arr(char *filename, fftw_complex *arr, int size);

void save_hdf5_snapshot(hid_t *file_id, hid_t *filetype_id, hid_t *datatype_id,
	fftw_complex *arr, double time, flow_params cnsts);

void save_hdf5_snapshot_visco(hid_t *file_id, hid_t *filetype_id, hid_t *datatype_id,
	complex *psi, complex *cxx, complex *cyy, complex *cxy, double time, flow_params cnsts);

void save_state(FILE *fp, fftw_complex *arr, flow_params cnsts);

void save_hdf5_state_visco(hid_t *file_id, hid_t *filetype_id, hid_t *datatype_id,
	complex *psi, complex *cxx, complex *cyy, complex *cxy, flow_params cnsts);

void load_hdf5_state(char *filename, fftw_complex *arr, flow_params cnsts);

void load_hdf5_state_visco(char *filename, complex *psi, complex *cxx, complex *cyy, complex *cxy, flow_params cnsts);

void load_state(FILE *fp, fftw_complex *arr, flow_params cnsts);

void load_operator(FILE *fp, fftw_complex *arr, flow_params cnsts);

void load_hdf5_operator(char *filename, fftw_complex *arr, flow_params cnsts);

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
    complex *u, *v, *udxlplpsi, *vdylplpsi, *biharmpsi, *lplpsi;
    complex *d2ypsi;
    complex *dyyypsi, *dypsi, *vdyypsi;
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

#endif // FIELDS_2D_C_H
