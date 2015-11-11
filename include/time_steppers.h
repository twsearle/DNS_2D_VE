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

// Last modified: Wed 11 Nov 11:06:59 2015

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

void step_sf_SI_oscil_visco(
	complex_d *psiOld, complex_d *psi, complex_d *cijOld, complex_d *cij, complex_d
	*psiNL, complex_d *forcing, complex_d *forcingN, double dt, int timeStep,
	complex_d *opsList, flow_scratch scr, flow_params params);

void step_conformation_oscil(
	 complex_d *cijOld, complex_d *cij, complex_d *psi, complex_d *cijNL, double
	 dt, flow_scratch scr, flow_params params);

void equilibriate_stress(
	complex_d *psiOld, complex_d *psi_lam, complex_d *cijOld, complex_d *cij,
	complex_d *cijNL, double dt,flow_scratch scr, flow_params params, hid_t
	*file_id, hid_t *filetype_id, hid_t *datatype_id
	);

#endif // TIME_STEPPERS_H

