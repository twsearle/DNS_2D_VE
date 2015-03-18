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

// Last modified: Wed 18 Mar 16:07:09 2015

#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<math.h>
#include<complex.h>
#include"fftw3.h"

//prototypes

void step_sf_SI_Crank_Nicolson(
	complex *psi, complex *psi2, double dt, int timeStep, double oneOverRe, flow_params params, complex
	*scratch, complex *scratch2, complex *u, complex *v, complex *lplpsi,
	complex *biharmpsi, complex *d2ypsi, complex *dyyypsi, complex *d4ypsi,
	complex *d2xd2ypsi, complex *d4xpsi, complex *udxlplpsi, complex
	*vdylplpsi, complex *vdyypsi, complex *RHSvec, complex *opsList,
	fftw_plan *phys_plan, fftw_plan *spec_plan, complex *scratchin, complex
	*scratchout, double *scratchp1, double *scratchp2 
	);

#endif // FIELDS_2D_C_H

