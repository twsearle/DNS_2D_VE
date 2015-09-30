#ifndef FIELDS_IO_C_H
#define FIELDS_IO_C_H
/* -------------------------------------------------------------------------- *
 *									      *
 *  fields_IO.h								      *
 *                                                                            *
 *  Header File for flow field input and output				      *
 *                                                                            *
 *                                                                            *
 * -------------------------------------------------------------------------- */

// Last modified: Wed 30 Sep 15:46:43 2015

#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<math.h>
#include<complex.h>
#include"hdf5.h"
#include"fftw3.h"
#include"fields_2D.h"

// Prototypes

typedef struct complex_hdf complex_hdf;

void save_hdf5_state(char *filename, fftw_complex *arr, flow_params cnsts);

void save_hdf5_real_arr(char *filename, double *arr, int size);

void save_hdf5_arr(char *filename, fftw_complex *arr, int size);

void save_hdf5_snapshot(hid_t *file_id, hid_t *filetype_id, hid_t *datatype_id,
	fftw_complex *arr, double time, flow_params cnsts);

void save_hdf5_snapshot_visco(hid_t *file_id, hid_t *filetype_id, hid_t *datatype_id,
	complex_d *psi, complex_d *cxx, complex_d *cyy, complex_d *cxy, double time, flow_params cnsts);

void save_state(FILE *fp, fftw_complex *arr, flow_params cnsts);

void save_hdf5_state_visco(hid_t *file_id, hid_t *filetype_id, hid_t *datatype_id,
	complex_d *psi, complex_d *cxx, complex_d *cyy, complex_d *cxy, flow_params cnsts);

void load_hdf5_state(char *filename, fftw_complex *arr, flow_params cnsts);

void load_hdf5_state_visco(char *filename, complex_d *psi, complex_d *cxx, complex_d *cyy, complex_d *cxy, flow_params cnsts);

void load_state(FILE *fp, fftw_complex *arr, flow_params cnsts);

void load_operator(FILE *fp, fftw_complex *arr, flow_params cnsts);

void load_hdf5_operator(char *filename, fftw_complex *arr, flow_params cnsts);

struct complex_hdf {
    double r;
    double i;
};

#endif // FIELDS_IO_C_H
