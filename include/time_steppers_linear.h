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

// Last modified: Fri  2 Oct 11:03:21 2015

#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<math.h>
#include<complex.h>
#include"fftw3.h"

//prototypes

void step_sf_linear_SI_Crank_Nicolson(
	complex_d *psi, complex_d *psi2, double dt, int timeStep, complex_d
	*forcing, complex_d *opsList, lin_flow_scratch scr, flow_params params);

#endif // FIELDS_2D_C_H
