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

// Last modified: Wed 30 Sep 12:03:57 2015

#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<math.h>
#include<complex.h>
#include"fftw3.h"

//prototypes

void step_sf_linear_SI_Crank_Nicolson(
	complex *psi, complex *psi2, double dt, int timeStep, complex
	*forcing, complex *opsList, flow_scratch scr, flow_params params);

#endif // FIELDS_2D_C_H