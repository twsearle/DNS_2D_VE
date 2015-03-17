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

// Last modified: Tue 10 Mar 14:51:26 2015

#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<math.h>
#include<complex.h>
#include"fftw3.h"

//prototypes

void step_sf_SI_Crank_Nicolson_visco(
	complex *psi, complex *cxx, complex *cyy, complex *cxy, double dt, int
	timeStep, flow_params params, complex *scratch, complex *scratch2,
	complex *u, complex *v, complex *lplpsi, complex *biharmpsi, complex
	*d2ypsi, complex *dyyypsi, complex *d4ypsi, complex *d2xd2ypsi, complex
	*d4xpsi, complex *udxlplpsi, complex *vdylplpsi, complex *vdyypsi,
	complex *txx, complex *tyy, complex *txy, complex *d2ytxy, complex
	*d2xtxy, complex *dxytyy_txx, complex *dytxy, complex *RHSvec, complex
	*opsList, fftw_plan *phys_plan, fftw_plan *spec_plan, complex
	*scratchin, complex *scratchout, double *scratchp1, double *scratchp2 
	);

void step_sf_SI_Crank_Nicolson(
	complex *psi, double dt, int timeStep, double oneOverRe, flow_params params, complex
	*scratch, complex *scratch2, complex *u, complex *v, complex *lplpsi,
	complex *biharmpsi, complex *d2ypsi, complex *dyyypsi, complex *d4ypsi,
	complex *d2xd2ypsi, complex *d4xpsi, complex *udxlplpsi, complex
	*vdylplpsi, complex *vdyypsi, complex *RHSvec, complex *opsList,
	fftw_plan *phys_plan, fftw_plan *spec_plan, complex *scratchin, complex
	*scratchout, double *scratchp1, double *scratchp2 
	);

void stress_time_derivative(
	complex *psi, complex *cxx, complex *cyy, complex *cxy, complex *fxx,
	complex *fyy, complex *fxy, double oneOverWi, flow_params params,
	complex *u, complex *v, complex *dxu, complex *dyu, complex *dxv,
	complex *dyv, complex *txx, complex *tyy, complex *txy, complex
	*cxxdxu, complex *cxydyu, complex *vgradcxx, complex *cxydxv, complex
	*cyydyv, complex *vgradcyy, complex *cxxdxv, complex *cyydyu, complex
	*vgradcxy, complex *scratch, complex *scratch2, fftw_plan *phys_plan,
	fftw_plan *spec_plan, complex *scratchin, complex *scratchout, double
	*scratchp1, double *scratchp2 
	);

void step_stresses_RK4();

void step_stresses_ABM();

#endif // FIELDS_2D_C_H

