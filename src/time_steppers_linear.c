
#ifndef TIME_STEPPERS_H
#define TIME_STEPPERS_H
/* -------------------------------------------------------------------------- *
 *									      *
 *  time_steppers_linear.c						      *
 *                                                                            *
 *  linearised time stepping routines					      *
 *                                                                            *
 *                                                                            *
 * -------------------------------------------------------------------------- */

// Last modified: Wed 30 Sep 12:01:53 2015

#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<math.h>
#include<complex.h>
#include"fftw3.h"

//prototypes

void step_sf_linear_SI_Crank_Nicolson(
	complex *psi, complex *psi2, double dt, int timeStep, complex
	*forcing, complex *opsList, flow_scratch scr, flow_params params)
{
    int i, j, l;
    int N = params.N;
    int M = params.M;
    double oneOverRe = 1. / params.Re;

    // First of all calculate the linearised nonlinear terms on psi2 (includeds
    // PSI0_2 and dpsi2), then calculate linear terms on psi then calculate RHS
    // for each mode, then solve for the new streamfunction, psi, at this
    // timestep.

    // -----------Nonlinear Terms --------------
    //
    // u
    single_dy(&psi2[ind(0,0)], scr.U0, params);
    single_dy(&psi2[ind(1,0)], scr.u, params);

    // v = -dydpsi
    single_dx(&psi2[ind(1,0)], scr.v, params);
    for(j=0; j<M; j++)
    {
	scr.v[j] = -scr.v[j];
    }


    // lpldpsi = dyy(dpsi) + dxx(dpsi)
    single_d2x(&psi2[ind(1,0)], scr.scratch, params);
    single_dy(scr.u, scr.lplpsi, params);
    for(i=0; i<N+1; i++)
    {
	for(j=0; j<M; j++)
	{
	    scr.lplpsi[ind(i,j)] = scr.lplpsi[ind(i,j)] + scr.scratch[ind(i,j)];
	}
    }

    // lplPSI0 = dyy(PSI0)
    single_dy(scr.U0, scr.dyyPSI0, params);

    // udxlplpsi = U0dxlpldpsi 
    single_dx(scr.lplpsi, scr.udxlplpsi, params);
    #ifdef MYDEBUG
    if(timeStep==0)
    {
	save_hdf5_arr("./output/dxlplpsi.h5", &scr.udxlplpsi[0], M);
    }
    #endif

    fft_cheb_convolve_r(scr.udxlplpsi, scr.U0, scr.udxlplpsi, 
	    scr, params);

    // vdylplpsi = vdylplPSI0 = vdyyyPSI0
    single_dy(scr.dyyPSI0, scr.vdylplpsi, params);

    #ifdef MYDEBUG
    if(timeStep==0)
    {
	save_hdf5_arr("./output/dylplpsi.h5", &scr.vdylplpsi[0], M);
    }
    #endif


    fft_cheb_convolve_r(scr.vdylplpsi, scr.v, scr.vdylplpsi, 
	    scr, params);

    //vdyypsi = vdyu = vdyyPSI0

    fft_cheb_convolve_r(scr.dyyPSI0, scr.v, scr.vdyypsi, 
	    scr, params);

#ifdef MYDEBUG
    if(timeStep==0)
    {
	save_hdf5_arr("./output/vdyypsi.h5", &scr.vdyypsi[0], M);
    }
#endif

    // ----------- linear Terms --------------
    
    // lplpsi dyy(psi) + dxx(psi)

    single_d2x(&psi[ind(1,0)], scr.scratch, params);
    single_dy(&psi[ind(1,0)], scr.u, params);
    single_dy(scr.u, scr.lplpsi, params);

    for(j=0; j<M; j++)
    {
	scr.lplpsi[j] = scr.lplpsi[j] + scr.scratch[j];
    }

    // biharmpsi (dyy + dxx)lplpsi

    single_dy(scr.u, scr.dyu, params);
    single_dy(scr.dyu, scr.dyyypsi, params);
    single_dy(scr.dyyypsi, scr.d4ypsi, params);

    single_d2x(&psi[ind(1,0)], scr.scratch, params);

#ifdef MYDEBUG
    if(timeStep==0)
    {
	save_hdf5_arr("./output/d2xpsi.h5", &scr.scratch[0], M);
    }
#endif

    single_dy(scr.scratch, scr.scratch2, params);
    single_dy(scr.scratch2, scr.d2xd2ypsi, params);


    single_d4x(psi, scr.d4xpsi, params);

    for(j=0; j<M; j++)
    {
	scr.biharmpsi[j] = scr.d4xpsi[j] + 2.*scr.d2xd2ypsi[j];
	scr.biharmpsi[j] = scr.biharmpsi[j] + scr.d4ypsi[j];
    }

    // RHSVec = dt*0.5*oneOverRe*BIHARMPSI 
    // 	+ LPLPSI 
    // 	- dt*UDXLPLPSI 
    // 	- dt*VDYLPLPSI 
    
#ifdef MYDEBUG
    if(timeStep==0)
    {
	save_hdf5_arr("./output/u.h5",  &scr.u[0], M);
	save_hdf5_arr("./output/v.h5", &scr.v[0], M);
	save_hdf5_arr("./output/lplpsi.h5", &scr.lplpsi[0], M);
	save_hdf5_arr("./output/d2ypsi.h5", &scr.dyu[0], M);
	save_hdf5_arr("./output/d3ypsi.h5", &scr.dyyypsi[0], M);
	save_hdf5_arr("./output/d4ypsi.h5", &scr.d4ypsi[0], M);
	save_hdf5_arr("./output/d2xd2ypsi.h5", &scr.d2xd2ypsi[0], M);
	save_hdf5_arr("./output/d4xpsi.h5", &scr.d4xpsi[0], M);
	save_hdf5_arr("./output/biharmpsi.h5", &scr.biharmpsi[0], M);
	save_hdf5_arr("./output/udxlplpsi.h5", &scr.udxlplpsi[0], M);
	save_hdf5_arr("./output/vdylplpsi.h5", &scr.vdylplpsi[0], M);
    }
#endif

    for (j=0; j<M; j++)
    {

	scr.RHSvec[j] = 0.5*dt*oneOverRe*scr.biharmpsi[j];
	scr.RHSvec[j] += + scr.lplpsi[j];
	scr.RHSvec[j] += - dt*scr.udxlplpsi[j];
	scr.RHSvec[j] += - dt*scr.vdylplpsi[j];
	scr.RHSvec[j] += dt*forcing[j]; 

    }

    //impose BCs

    scr.RHSvec[M-2] = 0;
    scr.RHSvec[M-1] = 0;

#ifdef MYDEBUG
    if(timeStep==0)
    {
	char fn[30];
	fn = "./output/RHSVec1.h5";
	printf("writing %s\n", fn);
	save_hdf5_arr(fn, &scr.RHSvec[0], M);
    }
#endif

    // perform dot product to calculate new streamfunction.
    for (j=M-1; j>=0; j=j-1)
    {
	psi[ind(1,j)] = 0;

	for (l=M-1; l>=0; l=l-1)
	{
	    psi[ind(1,j)] += opsList[(M + j)*M + l] * scr.RHSvec[l];
	}
    }


    // Zeroth mode
    //
    // RHSVec[N*M:(N+1)*M] = 0
    // RHSVec[N*M:(N+1)*M] = dt*0.5*oneOverRe*dot(MDYYY, PSI)[N*M:(N+1)*M] 
    // 	+ dot(MDY, PSI)[N*M:(N+1)*M] 
    // 	- dt*dot(dot(MMV, MDYY), PSI)[N*M:(N+1)*M]
    // RHSVec[N*M] += dt*2*oneOverRe

    for (j=0; j<M; j++)
    {
	scr.RHSvec[j] = dt*0.5*oneOverRe*scr.dyyyPSI0[j];
	scr.RHSvec[j] += scr.U0[ind(0,j)]; 
	scr.RHSvec[j] += dt*forcing[ind(0,j)]; 
    }

    // apply BCs
    // # dyPsi0(+-1) = 0
    // RHSVec[N*M + M-3] = 0
    // RHSVec[N*M + M-2] = 0
    // # Psi0(-1) = 0
    // RHSVec[N*M + M-1] = 0

    scr.RHSvec[M-3] = params.U0; 
    scr.RHSvec[M-2] = -params.U0; 
    scr.RHSvec[M-1] = 0; 

#ifdef MYDEBUG
    if(timeStep==0)
    {
	char fn[30];
	sprintf(fn, "./output/RHSVec%d.h5", 0);
	save_hdf5_arr(fn, &scr.RHSvec[0], M);
    }
#endif


    // step the zeroth mode

    //for (j=M-1; j>=0; j=j-1)
    for (j=0; j<M; j++)
    {
	psi[ind(0,j)] = 0;
	//for (l=M-1; l>=0; l=l-1)
	for (l=0; l<M; l++)
	{
	    psi[ind(0,j)] += creal(opsList[j*M + l] * scr.RHSvec[l]);

	}
    }

}

#endif // FIELDS_2D_C_H
