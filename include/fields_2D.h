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

// Last modified: Fri 20 Mar 11:17:38 2015

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

void to_physical_r(complex *arrin, double *arrout,
	fftw_complex *scratchin, fftw_complex *scratchout,
	fftw_plan *phys_plan,  flow_params cnsts);

void to_physical(fftw_complex *arrin, fftw_complex *arrout,
		 fftw_complex *scratchin, fftw_complex *scratchout,
		 fftw_plan *phys_plan,  flow_params cnsts);

void to_spectral_1(fftw_complex *arrin, fftw_complex *arrout, fftw_complex *scratch,
		 fftw_complex *scratchFin, fftw_complex *scratchFout,
		 fftw_complex *scratchCin, fftw_complex *scratchCout,
		 fftw_plan *spec_fou_plan, fftw_plan *spec_cheb_plan,
		 flow_params cnsts);

void to_spectral_r(double *arrin, complex *arrout,
	fftw_complex *scratchin, fftw_complex *scratchout,
	fftw_plan *spec_plan,  flow_params cnsts);

void to_spectral(fftw_complex *arrin, fftw_complex *arrout,
		 fftw_complex *scratchin, fftw_complex *scratchout,
		 fftw_plan *spec_plan,  flow_params cnsts);

void fft_convolve(fftw_complex *arr1, fftw_complex *arr2, fftw_complex *arrout,
	fftw_complex *scratchp1, fftw_complex *scratchp2, fftw_complex
	*scratchin, fftw_complex *scratchout, fftw_plan *physplan, fftw_plan
	*spec_plan, flow_params cnsts);

void fft_convolve_r(complex *arr1, complex *arr2, complex *arrout,
	double *scratchp1, double *scratchp2, fftw_complex
	*scratchin, fftw_complex *scratchout, fftw_plan *phys_plan, fftw_plan
	*spec_plan, flow_params cnsts);

void save_hdf5_state(char *filename, fftw_complex *arr, flow_params cnsts);

void save_hdf5_arr(char *filename, fftw_complex *arr, int size);

void save_hdf5_snapshot(hid_t *file_id, hid_t *filetype_id, hid_t *datatype_id,
	fftw_complex *arr, double time, flow_params cnsts);

void save_hdf5_snapshot_visco(hid_t *file_id, hid_t *filetype_id, hid_t *datatype_id,
	complex *psi, complex *cxx, complex *cyy, complex *cxy, double time, flow_params cnsts);

void save_state(FILE *fp, fftw_complex *arr, flow_params cnsts);

void load_hdf5_state(char *filename, fftw_complex *arr, flow_params cnsts);

void load_hdf5_state_visco(char *filename, complex *psi, complex *cxx, complex *cyy, complex *cxy, flow_params cnsts);

void load_state(FILE *fp, fftw_complex *arr, flow_params cnsts);

void load_operator(FILE *fp, fftw_complex *arr, flow_params cnsts);

void load_hdf5_operator(char *filename, fftw_complex *arr, flow_params cnsts);

double calc_KE0(fftw_complex *usq, fftw_complex *vsq, flow_params cnsts);

double calc_KE1(fftw_complex *usq, fftw_complex *vsq, flow_params cnsts);

double calc_KE2(fftw_complex *usq, fftw_complex *vsq, flow_params cnsts);

double calc_KE(fftw_complex *usq, fftw_complex *vsq, flow_params cnsts);

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
};

struct complex_hdf {
    double r;
    double i;
};

#endif // FIELDS_2D_C_H
