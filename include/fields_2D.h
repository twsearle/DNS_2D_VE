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

// Last modified: Fri  6 Feb 17:57:52 2015

#include<stdio.h>
#include<stdlib.h>
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

void dy(fftw_complex *arrin, fftw_complex *arrout,  flow_params cnsts);

void to_physical(fftw_complex *arrin, fftw_complex *arrout,
		 fftw_complex *scratchin, fftw_complex *scratchout,
		 fftw_plan *phys_plan,  flow_params cnsts);

void to_spectral(fftw_complex *arrin, fftw_complex *arrout,
		 fftw_complex *scratchin, fftw_complex *scratchout,
		 fftw_plan *spec_plan,  flow_params cnsts);

void save_hdf5_state(char *filename, fftw_complex *arr, flow_params cnsts);

void save_hdf5_arr(char *filename, fftw_complex *arr, int size);

void save_state(FILE *fp, fftw_complex *arr, flow_params cnsts);

void load_hdf5_state(char *filename, fftw_complex *arr, flow_params cnsts);

void load_state(FILE *fp, fftw_complex *arr, flow_params cnsts);

void load_operator(FILE *fp, fftw_complex *arr, flow_params cnsts);

void load_hdf5_operator(char *filename, fftw_complex *arr, flow_params cnsts);

double calc_KE0(fftw_complex *arrin, flow_params cnsts);

struct flow_params {
    int N;
    int M;
    int dealiasing;
    int Nf;
    int Mf;
    double kx;
    double Ly;
    double Re;
    double Wi;
    double beta;
};

struct complex_hdf {
    double r;
    double i;
};

#endif // FIELDS_2D_C_H
