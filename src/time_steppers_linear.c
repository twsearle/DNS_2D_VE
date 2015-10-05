
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

// Last modified: Fri  2 Oct 17:09:25 2015

#include"fields_1D.h"
#include"fields_IO.h"

//prototypes

void step_sf_linear_SI_Crank_Nicolson(
	complex_d *psi, complex_d *psi2, double dt, int timeStep, complex_d
	*forcing, complex_d *opsList, lin_flow_scratch scr, flow_params params)
{
    int j, l;
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
    single_dx(&psi2[ind(1,0)], scr.v, 1, params);
    for(j=0; j<M; j++)
    {
	scr.v[j] = -scr.v[j];
    }


    // lpldpsi = dyy(dpsi) + dxx(dpsi)
    single_d2x(&psi2[ind(1,0)], scr.scratch, 1, params);
    single_dy(scr.u, scr.lplpsi, params);

    for(j=0; j<M; j++)
    {
	scr.lplpsi[j] = scr.lplpsi[j] + scr.scratch[j];
    }

    // lplPSI0 = dyy(PSI0)
    single_dy(scr.U0, scr.d2yPSI0, params);

    // udxlplpsi = U0dxlpldpsi 
    single_dx(scr.lplpsi, scr.udxlplpsi, 1,  params);
    #ifdef MYDEBUG
    if(timeStep==0)
    {
	save_hdf5_arr("./output/dxlplpsi.h5", &scr.udxlplpsi[0], M);
    }
    #endif

    fft_cheby_convolve(scr.udxlplpsi, scr.U0, scr.udxlplpsi, 
	    scr, params);

    // vdylplpsi = vdylplPSI0 = vdyyyPSI0
    single_dy(scr.d2yPSI0, scr.d3yPSI0, params);

    for (j=0; j<M; j++)
    {
        scr.vdylplpsi[j] = scr.d3yPSI0[j];
    }

    #ifdef MYDEBUG
    if(timeStep==0)
    {
	save_hdf5_arr("./output/dylplpsi.h5", &scr.vdylplpsi[0], M);
    }
    #endif


    fft_cheby_convolve(scr.vdylplpsi, scr.v, scr.vdylplpsi, 
	    scr, params);

    // ----------- linear Terms --------------
    
    // lplpsi dyy(psi) + dxx(psi)

    single_d2x(&psi[ind(1,0)], scr.scratch, 1, params);
    single_dy(&psi[ind(1,0)], scr.u, params);
    single_dy(scr.u, scr.lplpsi, params);

    for(j=0; j<M; j++)
    {
	scr.lplpsi[j] = scr.lplpsi[j] + scr.scratch[j];
    }

    // d3yPSI0
    
    single_dy(&psi[ind(0,0)], scr.U0, params);
    single_dy(scr.U0, scr.d2yPSI0, params);
    single_dy(scr.d2yPSI0, scr.d3yPSI0, params);

    // biharmpsi (dyy + dxx)lplpsi

    single_dy(scr.u, scr.d2ypsi, params);
    single_dy(scr.d2ypsi, scr.d3ypsi, params);
    single_dy(scr.d3ypsi, scr.d4ypsi, params);

    single_d2x(&psi[ind(1,0)], scr.scratch, 1, params);


#ifdef MYDEBUG
    if(timeStep==0)
    {
	save_hdf5_arr("./output/d2xpsi.h5", &scr.scratch[0], M);
    }
#endif

    single_dy(scr.scratch, scr.scratch2, params);
    single_dy(scr.scratch2, scr.d2xd2ypsi, params);


    single_d4x(&psi[ind(1,0)], scr.d4xpsi, 1, params);

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
	save_hdf5_arr("./output/U0.h5",  &scr.U0[0], M);
	save_hdf5_arr("./output/u.h5",  &scr.u[0], M);
	save_hdf5_arr("./output/v.h5", &scr.v[0], M);
	save_hdf5_arr("./output/lplpsi.h5", &scr.lplpsi[0], M);
	save_hdf5_arr("./output/d2yPSI0.h5", &scr.d2yPSI0[0], M);
	save_hdf5_arr("./output/d3yPSI0.h5", &scr.d3yPSI0[0], M);
	save_hdf5_arr("./output/d2ypsi.h5", &scr.d2ypsi[0], M);
	save_hdf5_arr("./output/d3ypsi.h5", &scr.d3ypsi[0], M);
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
    }

    //impose BCs

    scr.RHSvec[M-2] = 0;
    scr.RHSvec[M-1] = 0;

#ifdef MYDEBUG
    if(timeStep==0)
    {
	char fn[30];
	sprintf(fn, "./output/RHSVec%d.h5", 1);
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


    // Zeroth mode linearised
    //
    // RHSVec[N*M:(N+1)*M] = 0
    // RHSVec[N*M:(N+1)*M] = dt*0.5*oneOverRe*dot(MDYYY, PSI)[N*M:(N+1)*M] 
    // 	+ dot(MDY, PSI)[N*M:(N+1)*M] 
    // RHSVec[N*M] += dt*2*oneOverRe

    for (j=0; j<M; j++)
    {
	scr.RHSvec[j] = dt*0.5*oneOverRe*scr.d3yPSI0[j];
	scr.RHSvec[j] += scr.U0[j]; 
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
