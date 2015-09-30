#ifndef TIME_STEPPERS_H
#define TIME_STEPPERS_H
/* -------------------------------------------------------------------------- *
 *									      *
 *  time_steppers.h							      *
 *                                                                            *
 *  Header File for the time stepping routines				      *
 *                                                                            *
 *                                                                            *
 * -------------------------------------------------------------------------- */

// Last modified: Wed 30 Sep 12:57:03 2015

#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<math.h>
#include<complex.h>
#include"fftw3.h"

//prototypes

void step_sf_SI_Crank_Nicolson(
	complex_d *psi, complex_d *psi2, double dt, int timeStep, complex_d
	*forcing, complex_d *opsList, flow_scratch scr, flow_params params);

void step_sf_SI_Crank_Nicolson_visco(
	complex_d *psiOld, complex_d *psi, complex_d *cijOld, complex_d *cij, complex_d
	*psiNL, complex_d *forcing, complex_d *forcingN, double dt, int timeStep,
	complex_d *opsList, flow_scratch scr, flow_params params);

void step_conformation_Crank_Nicolson(
	 complex_d *cijOld, complex_d *cij, complex_d *psi, complex_d *cijNL, double
	 dt, flow_scratch scr, flow_params params);

void stress_time_derivative(
	complex_d *psi, complex_d *cxx, complex_d *cyy, complex_d *cxy, complex_d *fxx,
	complex_d *fyy, complex_d *fxy, complex_d *txx, complex_d *tyy, complex_d *txy,
	flow_scratch scr, flow_params params
	);

void equilibriate_stress(
	complex_d *psiOld, complex_d *psi_lam, complex_d *cijOld, complex_d *cij,
	complex_d *cijNL, double dt,flow_scratch scr, flow_params params, hid_t
	*file_id, hid_t *filetype_id, hid_t *datatype_id
	);

#endif // TIME_STEPPERS_H

