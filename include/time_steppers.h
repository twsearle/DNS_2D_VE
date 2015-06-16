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

// Last modified: Mon 18 May 12:19:24 2015

#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<math.h>
#include<complex.h>
#include"fftw3.h"

//prototypes

void step_sf_SI_Crank_Nicolson(
	complex *psi, complex *psi2, double dt, int timeStep, complex
	*forcing, complex *opsList, flow_scratch scr, flow_params params);

void step_sf_SI_Crank_Nicolson_visco(
	complex *psiOld, complex *psi, complex *cijOld, complex *cij, complex
	*psiNL, complex *forcing, complex *forcingN, double dt, int timeStep,
	complex *opsList, flow_scratch scr, flow_params params);

void step_conformation_Crank_Nicolson(
	 complex *cijOld, complex *cij, complex *psi, complex *cijNL, double
	 dt, flow_scratch scr, flow_params params);

void stress_time_derivative(
	complex *psi, complex *cxx, complex *cyy, complex *cxy, complex *fxx,
	complex *fyy, complex *fxy, complex *txx, complex *tyy, complex *txy,
	flow_scratch scr, flow_params params
	);

void equilibriate_stress(
	complex *psiOld, complex *psi_lam, complex *cijOld, complex *cij,
	complex *cijNL, double dt,flow_scratch scr, flow_params params, hid_t
	*file_id, hid_t *filetype_id, hid_t *datatype_id
	);

#endif // FIELDS_2D_C_H

