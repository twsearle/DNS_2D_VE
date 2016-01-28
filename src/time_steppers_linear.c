
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

// Last modified: Thu 28 Jan 14:59:28 2016

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

void step_conformation_linear_Crank_Nicolson(
	 complex_d *cijOld, complex_d *cij, complex_d *psi, complex_d *cijNL, double
	 dt, lin_flow_scratch scr, flow_params params)
{
    int N = params.N;
    int M = params.M;
    int i,j;
    double oneOverWi = 1./params.Wi;
    

    // Nonlinear terms ----------------------------------------------------
    // dxU, dyU, dxV, dyV
    
    single_dy(&psi[ind(0,0)], scr.U0, params);
    single_dy(&psi[ind(1,0)], scr.u, params);
    single_dx(&psi[ind(1,0)], scr.v, 1, params);

    for (i =0; i<M; i++)
    {
	scr.v[i] = -scr.v[i];
    }

    single_dx(scr.u, scr.dxu, 1, params);
    single_dy(scr.u, scr.dyu, params);
    single_dy(scr.U0, scr.d2yPSI0, params);
    single_dx(scr.v, scr.dxv, 1, params);
    single_dy(scr.v, scr.dyv, params);
    
    // Cxx*dxu 
    fft_cheby_convolve(&cijNL[ind(0,0)], scr.dxu, scr.cxxdxu, scr, params);

    // Cxy*dyu = Cxy*dyu + cxy*dyU
    fft_cheby_convolve(&cijNL[2*(N+1)*M + ind(0,0)], scr.dyu, scr.cxydyu, scr, params);
    fft_cheby_convolve(&cijNL[2*(N+1)*M + ind(1,0)], scr.d2yPSI0, scr.scratch, scr, params);

    for (i =0; i<M; i++)
    {
	scr.cxydyu[i] += scr.scratch[i];
    }

    // Cxy0*dyU0 
    fft_cheby_convolve(&cijNL[2*(N+1)*M + ind(0,0)], scr.d2yPSI0, scr.cxy0dyU0, scr, params);

    // VGrad*Cxx
    single_dx(&cijNL[0 + ind(1,0)], scr.scratch, 1, params);
    fft_cheby_convolve(scr.U0, scr.scratch, scr.scratch, scr, params);

    single_dy(&cijNL[0 + ind(0,0)], scr.scratch2, params);
    fft_cheby_convolve(scr.v, scr.scratch2, scr.scratch2, scr, params);

    for (i=0; i<M; i++)
    {
	scr.vgradcxx[i] = scr.scratch[i] + scr.scratch2[i];
    }
    
    // Cxy*dxV
    fft_cheby_convolve(&cijNL[2*(N+1)*M + ind(0,0)], scr.dxv, scr.cxydxv, scr, params);

    // Cyy*dyV
    fft_cheby_convolve(&cijNL[(N+1)*M + ind(0,0)], scr.dyv, scr.cyydyv, scr, params);

    // vgrad*Cyy
    single_dx(&cijNL[(N+1)*M + ind(1,0)], scr.scratch, 1,  params);
    fft_cheby_convolve(scr.U0, scr.scratch, scr.scratch, scr, params);

    single_dy(&cijNL[(N+1)*M + ind(0,0)], scr.scratch2, params);
    fft_cheby_convolve(scr.v, scr.scratch2, scr.scratch2, scr, params);

    for (i=0; i<M; i++)
    {
	scr.vgradcyy[i] = scr.scratch[i] + scr.scratch2[i];
    }
    
    // Cxx*dxV
    fft_cheby_convolve(&cijNL[0 + ind(0,0)], scr.dxv, scr.cxxdxv, scr, params);

    // CyydyU = Cyydyu + cyydyU
    fft_cheby_convolve(&cijNL[(N+1)*M + ind(0,0)], scr.dyu, scr.cyydyu, scr, params);
    fft_cheby_convolve(&cijNL[(N+1)*M + ind(1,0)], scr.d2yPSI0, scr.scratch, scr, params);
    for (i =0; i<M; i++)
    {
	scr.cyydyu[i] += scr.scratch[i];
    }

    fft_cheby_convolve(&cijNL[(N+1)*M + ind(0,0)], scr.d2yPSI0, scr.cyy0dyU0, scr, params);
    

    // Vgrad*Cxy
    single_dx(&cijNL[2*(N+1)*M + ind(1,0)], scr.scratch, 1, params);
    fft_cheby_convolve(scr.U0, scr.scratch, scr.scratch, scr, params);

    single_dy(&cijNL[2*(N+1)*M + ind(0,0)], scr.scratch2, params);
    fft_cheby_convolve(scr.v, scr.scratch2, scr.scratch2, scr, params);

    for (i=0; i<M; i++)
    {
	scr.vgradcxy[i] = scr.scratch[i] + scr.scratch2[i];
    }

    
    // Calculate polymeric stress components
     
    // dCxxdt 
    // SVNew[:vecLen] = 2*dot(MMCXX, dot(MDX, U)) + 2*dot(MMCXY, dot(MDY, U))
    // 		 - dot(VGRAD, CxxOld) - TxxOld 
    // dCyydt
    // SVNew[vecLen:2*vecLen] = + 2*dot(MMCXY, dot(MDX, V)) 
    // 			 + 2*dot(MMCYY, dot(MDY, V)) 
    // 			 - dot(VGRAD, CyyOld) - TyyOld
    // dCxydt
    // SVNew[2*vecLen:3*vecLen] = dot(MMCXX, dot(MDX, V)) 
    // 			  + dot(MMCYY, dot(MDY, U))
    // 			  - dot(VGRAD, CxyOld) - TxyOld
    //
    
	// First Mode
    for (j=0; j<M; j++)
    {
	cij[ind(1,j)] = cijOld[ind(1,j)];
	cij[ind(1,j)] += - 0.5*dt*oneOverWi*cijOld[ind(1,j)];
	cij[ind(1,j)] += dt*2.*scr.cxydyu[j];
	cij[ind(1,j)] += dt*2.*scr.cxxdxu[j];
	cij[ind(1,j)] += - dt*scr.vgradcxx[j];

	cij[ind(1,j)] *= params.Wi/(params.Wi+dt*0.5);
	

	cij[(N+1)*M + ind(1,j)] = cijOld[(N+1)*M + ind(1,j)];
	cij[(N+1)*M + ind(1,j)] += - 0.5*dt*oneOverWi*cijOld[(N+1)*M + ind(1,j)];
	cij[(N+1)*M + ind(1,j)] += dt*2.*scr.cxydxv[j]; 
	cij[(N+1)*M + ind(1,j)] += dt*2.*scr.cyydyv[j];
	cij[(N+1)*M + ind(1,j)] += - dt*scr.vgradcyy[j];

	cij[(N+1)*M + ind(1,j)] *= params.Wi/(params.Wi+dt*0.5);
	

	cij[2*(N+1)*M + ind(1,j)] = cijOld[2*(N+1)*M + ind(1,j)];
	cij[2*(N+1)*M + ind(1,j)] += - 0.5*dt*oneOverWi*cijOld[2*(N+1)*M + ind(1,j)];
	cij[2*(N+1)*M + ind(1,j)] += dt*scr.cxxdxv[j];
	cij[2*(N+1)*M + ind(1,j)] += dt*scr.cyydyu[j];
	cij[2*(N+1)*M + ind(1,j)] += - dt*scr.vgradcxy[j];

	cij[2*(N+1)*M + ind(1,j)] *= params.Wi/(params.Wi+dt*0.5);
    }

	// zeroth Mode
    for (j=0; j<M; j++)
    {
	cij[ind(0,j)] = cijOld[ind(0,j)];
	cij[ind(0,j)] += - 0.5*dt*oneOverWi*cijOld[ind(0,j)];
	cij[ind(0,j)] += dt*2.*scr.cxy0dyU0[j];

	cij[ind(0,j)] *= params.Wi/(params.Wi+dt*0.5);
	

	cij[(N+1)*M + ind(0,j)] = cijOld[(N+1)*M + ind(0,j)];
	cij[(N+1)*M + ind(0,j)] += - 0.5*dt*oneOverWi*cijOld[(N+1)*M + ind(0,j)];

	cij[(N+1)*M + ind(0,j)] *= params.Wi/(params.Wi+dt*0.5);
	

	cij[2*(N+1)*M + ind(0,j)] = cijOld[2*(N+1)*M + ind(0,j)];
	cij[2*(N+1)*M + ind(0,j)] += - 0.5*dt*oneOverWi*cijOld[2*(N+1)*M + ind(0,j)];
	cij[2*(N+1)*M + ind(0,j)] += dt*scr.cyy0dyU0[j];

	cij[2*(N+1)*M + ind(0,j)] *= params.Wi/(params.Wi+dt*0.5);

    }

    cij[0] += dt/(params.Wi+dt*0.5);
    cij[(N+1)*M] += dt/(params.Wi+dt*0.5);
    
    // Zero off the imaginary part of the zeroth mode of the stresses

    for (j=0; j<M; j++)
    {
        cij[ind(0,j)] = creal(cij[ind(0,j)]);
        cij[(N+1)*M + ind(0,j)] = creal(cij[(N+1)*M + ind(0,j)]);
        cij[2*(N+1)*M + ind(0,j)] = creal(cij[2*(N+1)*M + ind(0,j)]);
    }
    
    #ifdef MYDEBUG
    save_hdf5_arr("./output/cyy0dyu0.h5", &scr.cyy0dyU0[0], M);
    save_hdf5_arr("./output/cxy0dyu0.h5", &scr.cxy0dyU0[0], M);
    #endif

}

void step_sf_linear_SI_Crank_Nicolson_visco(
	complex_d *psiOld, complex_d *psi, complex_d *cijOld, complex_d *cij, complex_d
	*psiNL, complex_d *forcing, complex_d *forcingN, double dt, int timeStep,
	complex_d *opsList, lin_flow_scratch scr, flow_params params)
{
    int j, l;
    int N = params.N;
    int M = params.M;
    double oneOverRe = 1./params.Re;
    double oneOverWi = 1./params.Wi;
    double beta = params.beta;

    // -----------Nonlinear Terms --------------
    //
    // u
    single_dy(&psi[ind(0,0)], scr.U0, params);
    single_dy(&psi[ind(1,0)], scr.u, params);

    // v = -dydpsi
    single_dx(&psi[ind(1,0)], scr.v, 1, params);
    for(j=0; j<M; j++)
    {
	scr.v[j] = -scr.v[j];
    }


    // lpldpsi = dyy(dpsi) + dxx(dpsi)
    single_d2x(&psi[ind(1,0)], scr.scratch, 1, params);
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

    // stresses

    // AT FUTURE STEP
    
    //dycxy
    single_dy(&cij[2*(N+1)*M + ind(1,0)], scr.dycxyN, params);
    single_dy(&cij[2*(N+1)*M + ind(0,0)], scr.dycxy0N, params);
    
    // d2x cxy
    single_d2x(&cij[2*(N+1)*M + ind(1,0)], scr.d2xcxyN, 1, params);
    
    // d2y cxy
    single_dy(scr.dycxyN, scr.d2ycxyN, params);

    // dxy cyy-cxx
    for (j=0; j<M; j++)
    {
	scr.scratch[j] = cij[(N+1)*M + ind(1,j)] - cij[ind(1,j)];
    }

    single_dx(scr.scratch, scr.scratch2, 1, params);
    single_dy(scr.scratch2, scr.dxycyy_cxxN, params);

    // AT PREVIOUS STEP
    
    //dycxy
    single_dy(&cijOld[2*(N+1)*M + ind(1,0)], scr.dycxy, params);
    single_dy(&cijOld[2*(N+1)*M + ind(0,0)], scr.dycxy0, params);
    
    // d2x cxy
    single_d2x(&cijOld[2*(N+1)*M + ind(1,0)], scr.d2xcxy, 1, params);
    
    // d2y cxy
    single_dy(scr.dycxy, scr.d2ycxy, params);

    // dxy cyy-cxx
    for (j=0; j<M; j++)
    {
	scr.scratch[j] = cijOld[(N+1)*M + ind(1,j)] - cijOld[ind(1,j)];
    }

    single_dx(scr.scratch, scr.scratch2, 1, params);
    single_dy(scr.scratch2, scr.dxycyy_cxx, params);
    
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
	save_hdf5_arr("./output/dycxy0.h5", &scr.dycxy0[0], M);
	save_hdf5_arr("./output/d2ycxy.h5", &scr.d2ycxy[0], M);
	save_hdf5_arr("./output/d2xcxy.h5", &scr.d2xcxy[0], M);
	save_hdf5_arr("./output/dxycyy_cxx.h5", &scr.dxycyy_cxx[0], M);
	save_hdf5_arr("./output/dycxy0N.h5", &scr.dycxy0N[0], M);
	save_hdf5_arr("./output/d2ycxyN.h5", &scr.d2ycxyN[0], M);
	save_hdf5_arr("./output/d2xcxyN.h5", &scr.d2xcxyN[0], M);
	save_hdf5_arr("./output/dxycyy_cxxN.h5", &scr.dxycyy_cxxN[0], M);
    }
    #endif
    
    // Streamfunction equation:

    // RHSVec = dt*0.5*oneOverRe*BIHARMPSI 
    // 	+ LPLPSI 
    // 	- dt*UDXLPLPSI 
    // 	- dt*VDYLPLPSI 
    // + _dt*(1.-beta)*oneOverRe(dot(MDXX, Txy) 
    //                + dot(MDXY,(Tyy - Txx)) 
    //                - dot(MDYY, txy) )

    for (j=0; j<M; j++)
    {

	scr.RHSvec[j] = 0.5*dt*oneOverRe*beta*scr.biharmpsi[j];
	scr.RHSvec[j] += 0.5*dt*(1.0-beta)*oneOverRe*oneOverWi*( 
				     scr.d2ycxy[j]
				   - scr.d2xcxy[j] 
				   - scr.dxycyy_cxx[j]);
	scr.RHSvec[j] += 0.5*dt*(1.0-beta)*oneOverRe*oneOverWi*( 
				     scr.d2ycxyN[j]
				   - scr.d2xcxyN[j] 
				   - scr.dxycyy_cxxN[j]); 
	scr.RHSvec[j] += - dt*scr.udxlplpsi[j];
	scr.RHSvec[j] += - dt*scr.vdylplpsi[j];
	scr.RHSvec[j] += + scr.lplpsi[j];
	scr.RHSvec[j] += 0.5*dt*forcing[ind(1,j)];
	scr.RHSvec[j] += 0.5*dt*forcingN[ind(1,j)];
    }

    //impose BCs

    scr.RHSvec[M-2] = 0;
    scr.RHSvec[M-1] = 0;

    #ifdef MYDEBUG
    if(timeStep==0)
    {
	char fn[30];
	sprintf(fn, "./output/RHSVec%d.h5", 1);
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

    


    // # Zeroth mode, U equatioj
    // RHSVec[N*M:(N+1)*M] = 0
    // RHSVec[N*M:(N+1)*M] = dt*0.5*oneOverRe*dot(MDYYY, PSI)[N*M:(N+1)*M] 
    // 	+ dot(MDY, PSI)[N*M:(N+1)*M] 
    // 	- dt*dot(dot(MMV, MDYY), PSI)[N*M:(N+1)*M]
    //  + _dt*(1.-beta)*oneOverRe*dot(MDY, Txy)[N*M:(N+1)*M]
    // RHSVec[N*M] += dt*2*oneOverRe


    for (j=0; j<M; j++)
    {
	scr.RHSvec[j]  = dt*0.5*beta*oneOverRe*creal(scr.d3yPSI0[j]);
	scr.RHSvec[j] += dt*0.5*(1.-beta)*oneOverRe*oneOverWi*creal(scr.dycxy0[j]); 
	scr.RHSvec[j] += dt*0.5*(1.-beta)*oneOverRe*oneOverWi*creal(scr.dycxy0N[j]); 
	scr.RHSvec[j] += dt*0.5*creal(forcing[ind(0,j)]);
	scr.RHSvec[j] += dt*0.5*creal(forcingN[ind(0,j)]);
	scr.RHSvec[j] += creal(scr.U0[j]); 
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
	    psi[ind(0,j)] += opsList[j*M + l] * scr.RHSvec[l];

	}
    }

}

void step_conformation_linear_oscil(
	 complex_d *cijOld, complex_d *cij, complex_d *psi, complex_d *cijNL, double
	 dt, lin_flow_scratch scr, flow_params params)
{
    int N = params.N;
    int M = params.M;
    int i,j;
    double oneOverWi = 1./params.Wi;
    

    // Nonlinear terms ----------------------------------------------------
    // dxU, dyU, dxV, dyV
    
    single_dy(&psi[ind(0,0)], scr.U0, params);
    single_dy(&psi[ind(1,0)], scr.u, params);
    single_dx(&psi[ind(1,0)], scr.v, 1, params);

    for (i =0; i<M; i++)
    {
	scr.v[i] = -scr.v[i];
    }

    single_dx(scr.u, scr.dxu, 1, params);
    single_dy(scr.u, scr.dyu, params);
    single_dy(scr.U0, scr.d2yPSI0, params);
    single_dx(scr.v, scr.dxv, 1, params);
    single_dy(scr.v, scr.dyv, params);
    
    // Cxx*dxU 
    fft_cheby_convolve(&cijNL[ind(0,0)], scr.dxu, scr.cxxdxu, scr, params);

    // Cxy*dyu = Cxy*dyu + cxy*dyU
    fft_cheby_convolve(&cijNL[2*(N+1)*M + ind(0,0)], scr.dyu, scr.cxydyu, scr, params);
    fft_cheby_convolve(&cijNL[2*(N+1)*M + ind(1,0)], scr.d2yPSI0, scr.scratch, scr, params);

    for (i =0; i<M; i++)
    {
	scr.cxydyu[i] += scr.scratch[i];
    }

    // Cxy0*dyU0 
    fft_cheby_convolve(&cijNL[2*(N+1)*M + ind(0,0)], scr.d2yPSI0, scr.cxy0dyU0, scr, params);

    //save_hdf5_arr("./output/cxy0dyU0.h5", scr.cxy0dyU0, M);
    //exit(1);

    // VGrad*Cxx = U0dxcxx + vdyCxx
    single_dx(&cijNL[0 + ind(1,0)], scr.scratch, 1, params);
    fft_cheby_convolve(scr.U0, scr.scratch, scr.scratch, scr, params);

    single_dy(&cijNL[0 + ind(0,0)], scr.scratch2, params);
    fft_cheby_convolve(scr.v, scr.scratch2, scr.scratch2, scr, params);

    for (i=0; i<M; i++)
    {
	scr.vgradcxx[i] = scr.scratch[i] + scr.scratch2[i];
    }
    
    // Cxy*dxV
    fft_cheby_convolve(&cijNL[2*(N+1)*M + ind(0,0)], scr.dxv, scr.cxydxv, scr, params);

    // Cyy*dyV
    fft_cheby_convolve(&cijNL[(N+1)*M + ind(0,0)], scr.dyv, scr.cyydyv, scr, params);

    // vgrad*Cyy = Udxcyy + vdyCyy
    single_dx(&cijNL[(N+1)*M + ind(1,0)], scr.scratch, 1,  params);
    fft_cheby_convolve(scr.U0, scr.scratch, scr.scratch, scr, params);

    single_dy(&cijNL[(N+1)*M + ind(0,0)], scr.scratch2, params);
    fft_cheby_convolve(scr.v, scr.scratch2, scr.scratch2, scr, params);

    for (i=0; i<M; i++)
    {
	scr.vgradcyy[i] = scr.scratch[i] + scr.scratch2[i];
    }
    
    // Cxx*dxV
    fft_cheby_convolve(&cijNL[0 + ind(0,0)], scr.dxv, scr.cxxdxv, scr, params);

    // CyydyU = Cyydyu + cyydyU
    fft_cheby_convolve(&cijNL[(N+1)*M + ind(0,0)], scr.dyu, scr.cyydyu, scr, params);
    fft_cheby_convolve(&cijNL[(N+1)*M + ind(1,0)], scr.d2yPSI0, scr.scratch, scr, params);
    for (i =0; i<M; i++)
    {
	scr.cyydyu[i] += scr.scratch[i];
    }

    fft_cheby_convolve(&cijNL[(N+1)*M + ind(0,0)], scr.d2yPSI0, scr.cyy0dyU0, scr, params);
    

    // Vgrad*Cxy
    single_dx(&cijNL[2*(N+1)*M + ind(1,0)], scr.scratch, 1, params);
    fft_cheby_convolve(scr.U0, scr.scratch, scr.scratch, scr, params);

    single_dy(&cijNL[2*(N+1)*M + ind(0,0)], scr.scratch2, params);
    fft_cheby_convolve(scr.v, scr.scratch2, scr.scratch2, scr, params);

    for (i=0; i<M; i++)
    {
	scr.vgradcxy[i] = scr.scratch[i] + scr.scratch2[i];
    }

    
    // Calculate polymeric stress components
     
    // dCxxdt 
    // SVNew[:vecLen] = 2*dot(MMCXX, dot(MDX, U)) + 2*dot(MMCXY, dot(MDY, U))
    // 		 - dot(VGRAD, CxxOld) - TxxOld 
    // dCyydt
    // SVNew[vecLen:2*vecLen] = + 2*dot(MMCXY, dot(MDX, V)) 
    // 			 + 2*dot(MMCYY, dot(MDY, V)) 
    // 			 - dot(VGRAD, CyyOld) - TyyOld
    // dCxydt
    // SVNew[2*vecLen:3*vecLen] = dot(MMCXX, dot(MDX, V)) 
    // 			  + dot(MMCYY, dot(MDY, U))
    // 			  - dot(VGRAD, CxyOld) - TxyOld
    //
    
    double old_fac = (M_PI / (2.*params.Wi)) * (params.De - 0.5*dt);	
    double new_fac = 1.0 / ( (M_PI / (2.*params.Wi)) * (params.De + 0.5*dt) );	

    // First Mode
    for (j=0; j<M; j++)
    {
	cij[ind(1,j)] = old_fac*cijOld[ind(1,j)];
	cij[ind(1,j)] += dt*2.*scr.cxydyu[j];
	cij[ind(1,j)] += dt*2.*scr.cxxdxu[j];
	cij[ind(1,j)] += - dt*scr.vgradcxx[j];

	cij[ind(1,j)] *= new_fac;
	

	cij[(N+1)*M + ind(1,j)] = old_fac*cijOld[(N+1)*M + ind(1,j)];
	cij[(N+1)*M + ind(1,j)] += dt*2.*scr.cxydxv[j]; 
	cij[(N+1)*M + ind(1,j)] += dt*2.*scr.cyydyv[j];
	//cij[(N+1)*M + ind(1,j)] += - dt*scr.vgradcyy[j];

	cij[(N+1)*M + ind(1,j)] *= new_fac;
	

	cij[2*(N+1)*M + ind(1,j)] = old_fac*cijOld[2*(N+1)*M + ind(1,j)];
	cij[2*(N+1)*M + ind(1,j)] += dt*scr.cxxdxv[j];
	cij[2*(N+1)*M + ind(1,j)] += dt*scr.cyydyu[j];
	cij[2*(N+1)*M + ind(1,j)] += - dt*scr.vgradcxy[j];

	cij[2*(N+1)*M + ind(1,j)] *= new_fac;
    }

    // zeroth Mode
    //for (j=0; j<M; j++)
    //{
    //    cij[ind(0,j)] = old_fac*cijOld[ind(0,j)];
    //    cij[ind(0,j)] += dt*2.*scr.cxy0dyU0[j];
    //    cij[ind(0,j)] *= new_fac;
    //    

    //    cij[(N+1)*M + ind(0,j)] = old_fac*cijOld[(N+1)*M + ind(0,j)];
    //    cij[(N+1)*M + ind(0,j)] *= new_fac;
    //    

    //    cij[2*(N+1)*M + ind(0,j)] = old_fac*cijOld[2*(N+1)*M + ind(0,j)];
    //    cij[2*(N+1)*M + ind(0,j)] += dt*scr.cyy0dyU0[j];
    //    cij[2*(N+1)*M + ind(0,j)] *= new_fac;

    //}

    //cij[0] += dt/(params.De + 0.5*dt);
    //cij[(N+1)*M] += dt/(params.De + 0.5*dt);
    //
    //// Zero off the imaginary part of the zeroth mode of the stresses

    //for (j=0; j<M; j++)
    //{
    //    cij[ind(0,j)] = creal(cij[ind(0,j)]);
    //    cij[(N+1)*M + ind(0,j)] = creal(cij[(N+1)*M + ind(0,j)]);
    //    cij[2*(N+1)*M + ind(0,j)] = creal(cij[2*(N+1)*M + ind(0,j)]);
    //}

    #ifdef MYDEBUG
    save_hdf5_arr("./output/dxu.h5", &scr.dxu[0], M);
    save_hdf5_arr("./output/d2ypsi.h5", &scr.dyu[0], M);
    save_hdf5_arr("./output/cyy0dyu0.h5", &scr.cyy0dyU0[0], M);
    save_hdf5_arr("./output/cxy0dyu0.h5", &scr.cxy0dyU0[0], M);
    #endif

}

void step_sf_linear_SI_oscil_visco(
	complex_d *psiOld, complex_d *psi, complex_d *cijOld, complex_d *cij, complex_d
	*psiNL, complex_d *forcing, complex_d *forcingN, double dt, int timeStep,
	complex_d *opsList, lin_flow_scratch scr, flow_params params)
{
    int j, l;
    int N = params.N;
    int M = params.M;
    double WiFac = M_PI / (2.*params.Wi);
    double beta = params.beta;

    // -----------Nonlinear Terms --------------
    //
    // u
    single_dy(&psiNL[ind(0,0)], scr.U0, params);
    single_dy(&psiNL[ind(1,0)], scr.u, params);

    // v = -dxdpsi
    single_dx(&psiNL[ind(1,0)], scr.v, 1, params);
    for(j=0; j<M; j++)
    {
	scr.v[j] = -scr.v[j];
    }


    // lpldpsi = dyy(dpsi) + dxx(dpsi)
    single_d2x(&psiNL[ind(1,0)], scr.scratch, 1, params);
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

    // SCREW WITH EVERYTHING
    //double cumsum=0;
    //for (j=0; j<M; j++)
    //{
    //    cumsum += cimag(scr.d3yPSI0[j]);
    //    //cumsum += cimag(psi[j]);
    //}
    //printf("SUM %e \n", cumsum); 
    //cumsum = 0.0;
    //for (j=0; j<M; j++)
    //{
    //    cumsum += creal(scr.v[j]);
    //}
    //printf("SUM %e \n", cumsum); 
    //fft_cheby_convolve(scr.v, scr.v, scr.scratch, scr, params);
    //fft_cheby_convolve(scr.d3yPSI0, scr.d3yPSI0, scr.scratch2, scr, params);
    //
    ////fft_cheby_convolve(scr.v, scr.lplpsi, scr.scratch, scr, params);

    //char file1[50];
    //char file2[50];
    //char file3[50];
    //sprintf(file1, "./output/v%d.h5", timeStep);
    //sprintf(file2, "./output/d3ypsi%d.h5", timeStep);
    //sprintf(file3, "./output/vdylplpsi%d.h5", timeStep);

    //// save_hdf5_arr(file1, &scr.v[0], M);
    //// save_hdf5_arr(file2, &scr.scratch2[0], M);
    //// save_hdf5_arr(file3, &scr.vdylplpsi[0], M);

    #ifdef MYDEBUG
    if(timeStep==0)
    {
	save_hdf5_arr("./output/dylplpsi.h5", &scr.vdylplpsi[0], M);
    }
    
    #endif


    fft_cheby_convolve(scr.vdylplpsi, scr.v, scr.vdylplpsi, 
	    scr, params);

    //#ifdef MYDEBUG
    //if(timeStep==0)
    //{
    //    save_hdf5_arr("./output/vdylplpsi.h5", &scr.vdylplpsi[0], M);
    //}
    ////add on the nonlinear term in the perturbation
    ////for (j=0; j<M; j++)
    ////{
    ////    scr.vdylplpsi[j] += scr.scratch[j];
    ////}
    //#endif

    // ----------- linear Terms --------------
    
    // lplpsi dyy(psi) + dxx(psi)

    single_d2x(&psiOld[ind(1,0)], scr.scratch, 1, params);
    single_dy(&psiOld[ind(1,0)], scr.u, params);
    single_dy(scr.u, scr.lplpsi, params);

    for(j=0; j<M; j++)
    {
	scr.lplpsi[j] = scr.lplpsi[j] + scr.scratch[j];
    }

    // d3yPSI0
    
    single_dy(&psiOld[ind(0,0)], scr.U0, params);
    single_dy(scr.U0, scr.d2yPSI0, params);
    single_dy(scr.d2yPSI0, scr.d3yPSI0, params);

    // biharmpsi (dyy + dxx)lplpsi

    single_dy(scr.u, scr.d2ypsi, params);
    single_dy(scr.d2ypsi, scr.d3ypsi, params);
    single_dy(scr.d3ypsi, scr.d4ypsi, params);

    single_d2x(&psiOld[ind(1,0)], scr.scratch, 1, params);


#ifdef MYDEBUG
    if(timeStep==0)
    {
	save_hdf5_arr("./output/d2xpsi.h5", &scr.scratch[0], M);
    }
#endif

    single_dy(scr.scratch, scr.scratch2, params);
    single_dy(scr.scratch2, scr.d2xd2ypsi, params);


    single_d4x(&psiOld[ind(1,0)], scr.d4xpsi, 1, params);

    for(j=0; j<M; j++)
    {
	scr.biharmpsi[j] = scr.d4xpsi[j] + 2.*scr.d2xd2ypsi[j];
	scr.biharmpsi[j] = scr.biharmpsi[j] + scr.d4ypsi[j];
    }

    // stresses

    // AT FUTURE STEP
    
    //dycxy
    single_dy(&cij[2*(N+1)*M + ind(1,0)], scr.dycxyN, params);
    single_dy(&cij[2*(N+1)*M + ind(0,0)], scr.dycxy0N, params);
    
    // d2x cxy
    single_d2x(&cij[2*(N+1)*M + ind(1,0)], scr.d2xcxyN, 1, params);
    
    // d2y cxy
    single_dy(scr.dycxyN, scr.d2ycxyN, params);

    // dxy cyy-cxx
    for (j=0; j<M; j++)
    {
	scr.scratch[j] = cij[(N+1)*M + ind(1,j)] - cij[ind(1,j)];
    }

    single_dx(scr.scratch, scr.scratch2, 1, params);
    single_dy(scr.scratch2, scr.dxycyy_cxxN, params);

    // AT PREVIOUS STEP
    
    //dycxy
    single_dy(&cijOld[2*(N+1)*M + ind(1,0)], scr.dycxy, params);
    single_dy(&cijOld[2*(N+1)*M + ind(0,0)], scr.dycxy0, params);
    
    // d2x cxy
    single_d2x(&cijOld[2*(N+1)*M + ind(1,0)], scr.d2xcxy, 1, params);
    
    // d2y cxy
    single_dy(scr.dycxy, scr.d2ycxy, params);

    // dxy cyy-cxx
    for (j=0; j<M; j++)
    {
	scr.scratch[j] = cijOld[(N+1)*M + ind(1,j)] - cijOld[ind(1,j)];
    }

    single_dx(scr.scratch, scr.scratch2, 1, params);
    single_dy(scr.scratch2, scr.dxycyy_cxx, params);
    
    
    // Streamfunction equation:

    // RHSVec = dt*0.5*oneOverRe*BIHARMPSI 
    // 	+ LPLPSI 
    // 	- dt*UDXLPLPSI 
    // 	- dt*VDYLPLPSI 
    // + _dt*(1.-beta)*oneOverRe(dot(MDXX, Txy) 
    //                + dot(MDXY,(Tyy - Txx)) 
    //                - dot(MDYY, txy) )

    double BFac = (M_PI*params.Re*params.De) / (2.0*params.Wi);

    for (j=0; j<M; j++)
    {

	scr.RHSvec[j] = 0.5*dt*beta*scr.biharmpsi[j];
	scr.RHSvec[j] += 0.5*dt*(1.0-beta)*WiFac*( 
				     scr.d2ycxy[j]
				   - scr.d2xcxy[j] 
				   - scr.dxycyy_cxx[j]);
	scr.RHSvec[j] += 0.5*dt*(1.0-beta)*WiFac*( 
				     scr.d2ycxyN[j]
				   - scr.d2xcxyN[j] 
				   - scr.dxycyy_cxxN[j]); 
	scr.RHSvec[j] += - params.Re*dt*scr.udxlplpsi[j];
	scr.RHSvec[j] += - params.Re*dt*scr.vdylplpsi[j];
	scr.RHSvec[j] += + BFac*scr.lplpsi[j];
	//scr.RHSvec[j] += 0.5*dt*forcing[ind(1,j)];
	//scr.RHSvec[j] += 0.5*dt*forcingN[ind(1,j)];
    }

    //impose BCs

    scr.RHSvec[M-2] = 0;
    scr.RHSvec[M-1] = 0;

    #ifdef MYDEBUG
    //if(timeStep==0)
    //{
    char fn[30];
    sprintf(fn, "./output/RHSVec%d.h5", 1);
    printf("writing %s\n", fn);
    save_hdf5_arr(fn, &scr.RHSvec[0], M);
    for (j=0;j<M;j++)
    {
	scr.scratch[j] = scr.RHSvec[j];
    }
    //}
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

    


    // # Zeroth mode, U equatioj
    // RHSVec[N*M:(N+1)*M] = 0
    // RHSVec[N*M:(N+1)*M] = dt*0.5*oneOverRe*dot(MDYYY, PSI)[N*M:(N+1)*M] 
    // 	+ dot(MDY, PSI)[N*M:(N+1)*M] 
    // 	- dt*dot(dot(MMV, MDYY), PSI)[N*M:(N+1)*M]
    //  + _dt*(1.-beta)*oneOverRe*dot(MDY, Txy)[N*M:(N+1)*M]
    // RHSVec[N*M] += dt*2*oneOverRe


    // for (j=0; j<M; j++)
    // {
    //     scr.RHSvec[j]  = dt*0.5*beta*creal(scr.d3yPSI0[j]);
    //     scr.RHSvec[j] += dt*0.5*(1.-beta)*WiFac*creal(scr.dycxy0[j]); 
    //     scr.RHSvec[j] += dt*0.5*(1.-beta)*WiFac*creal(scr.dycxy0N[j]); 
    //     scr.RHSvec[j] += dt*0.5*creal(forcing[ind(0,j)]);
    //     scr.RHSvec[j] += dt*0.5*creal(forcingN[ind(0,j)]);
    //     scr.RHSvec[j] += BFac*creal(scr.U0[j]); 
    // }

    // apply BCs
    // # dyPsi0(+-1) = 0
    // RHSVec[N*M + M-3] = 0
    // RHSVec[N*M + M-2] = 0
    // # Psi0(-1) = 0
    // RHSVec[N*M + M-1] = 0

    //scr.RHSvec[M-3] = params.U0; 
    //scr.RHSvec[M-2] = -params.U0; 
    //scr.RHSvec[M-1] = 0; 

#ifdef MYDEBUG
    if(timeStep==0)
    {
	char fn[30];
	sprintf(fn, "./output/RHSVec%d.h5", 0);
	save_hdf5_arr(fn, &scr.RHSvec[0], M);
    }
#endif


    // step the zeroth mode

    //for (j=0; j<M; j++)
    //{
    //    psi[ind(0,j)] = 0;
    //    //for (l=M-1; l>=0; l=l-1)
    //    for (l=0; l<M; l++)
    //    {
    //        psi[ind(0,j)] += opsList[j*M + l] * scr.RHSvec[l];

    //    }
    //    psi[ind(0,j)] = creal(psi[ind(0,j)]);
    //}

}

void calc_base_cij(
	complex_d *cij, double time, lin_flow_scratch scr, flow_params params)
{

    if (params.oscillatory_flow == 0) 
    {
	printf("NOT OSCILLATORY FLOW!");
	exit(1);
    }

    int i;
    int N = params.N;
    int M = params.M;

    double Re = params.Re;
    double Wi = params.Wi;
    double De = params.De;
    double beta = params.beta;
    double y;

    complex_d tmp = beta + (1.-beta) / (1. + 1.*I*De);
    //printf("tmp %16.14f+%16.14fI\n", creal(tmp), cimag(tmp));

    complex_d alpha = csqrt( (I*M_PI*Re*De) / (2.*Wi*tmp) );
    //printf("alpha %16.14f+%16.14fI\n", creal(alpha), cimag(alpha));

    complex_d Chi = creal( (1.-I)*(1. - ctanh(alpha) / alpha) );
    //printf("Chi %16.14f+%16.14fI\n", creal(Chi), cimag(Chi));

    complex_d dyu_cmplx = 0;
    complex_d cxy_cmplx = 0;
    complex_d cxx_cmplx = 0;


    for(i=0; i<params.Mf; i++)
    {
        y = cos(M_PI*i/(params.Mf-1.));


        dyu_cmplx = M_PI/(2.*I*Chi) *(-alpha*csinh(alpha*y)/(ccosh(alpha)));
        cxy_cmplx = (1.0/(1.0+I*De)) * ((2*Wi/M_PI) * dyu_cmplx);

        scr.scratchp3[i] = creal( cxy_cmplx * cexp(I*time) );

        cxx_cmplx = (1.0/(1.0+2.*I*De))*(Wi/M_PI)*(cxy_cmplx*dyu_cmplx)*cexp(2.*I*time);
	cxx_cmplx +=(1.0/(1.0-2.*I*De))*(Wi/M_PI)*(conj(cxy_cmplx)*conj(dyu_cmplx))*cexp(-2.*I*time); 
        cxx_cmplx += 1. + (params.Wi/M_PI)*( cxy_cmplx*conj(dyu_cmplx) + conj(cxy_cmplx)*dyu_cmplx ); 

        scr.scratchp2[i] = creal(cxx_cmplx);
    }

    // cxx
    to_cheby_spectral(scr.scratchp2, &cij[0], scr, params);
    // cxy
    to_cheby_spectral(scr.scratchp3, &cij[2*(N+1)*M], scr, params);

    /*
    for (i=0; i<params.M; i++)
    {
	cij[ind(0,i)] = 0.0;
	cij[2*(N+1)*M + ind(0,i)] = 0.0;
    }

    // Cxy = Wi * U'
    cij[2*(N+1)*M + ind(0,1)] = Wi * -2.0 * cos(time);
    // Cxx = Wi^2 * U'^2
    cij[ind(0,0)] = (Wi*Wi * 2.0) * cos(time)*cos(time) + 1.0;
    cij[ind(0,2)] = Wi*Wi * 2.0 * cos(time)*cos(time);
    */
    
    // mean flow, so must have zero imaginary part!
    for (i=0; i<params.M; i++)
    {
	cij[ind(0,i)] = creal(cij[ind(0,i)]);
	cij[(N+1)*M + ind(0,i)] = 0.0;
	cij[2*(N+1)*M + ind(0,i)] = creal(cij[2*(N+1)*M + ind(0,i)]);
    }
    cij[(N+1)*M + ind(0,0)] = 1.0;

}

void calc_base_sf(
	complex_d *psi, double time, lin_flow_scratch scr, flow_params params)
{
    if (params.oscillatory_flow == 0) 
    {
	printf("NOT OSCILLATORY FLOW!");
	exit(1);
    }
    int i;
    int N = params.N;
    int M = params.M;

    double Re = params.Re;
    double Wi = params.Wi;
    double De = params.De;
    double beta = params.beta;
    double y;

    complex_d tmp = beta + (1.-beta) / (1. + 1.*I*De);
    //printf("tmp %16.14f+%16.14fI\n", creal(tmp), cimag(tmp));

    complex_d alpha = csqrt( (I*M_PI*Re*De) / (2.*Wi*tmp) );
    //printf("alpha %16.14f+%16.14fI\n", creal(alpha), cimag(alpha));

    complex_d Chi = creal( (1.-I)*(1. - ctanh(alpha) / alpha) );
    //printf("Chi %16.14f+%16.14fI\n", creal(Chi), cimag(Chi));

    complex_d psi_im = 0;

    for(i=0; i<params.Mf; i++)
    {
        y = cos(M_PI*i/(params.Mf-1.));

	psi_im = M_PI/(2.*I*Chi) *( y - csinh(alpha*y)/(alpha*ccosh(alpha))
                                      + csinh(alpha*-1.)/(alpha*ccosh(alpha)) );


        scr.scratchp1[i] = creal(psi_im * cexp(I*time));

	//printf("psi %f+%fI\n", creal(scr.scratchp1[i]), cimag(scr.scratchp1[i]));

    }

    // psi
    to_cheby_spectral(scr.scratchp1, &psi[0], scr, params);

    // Poiseuille flow
    /*
    for (i=0; i<params.M; i++)
    {
	psi[ind(0,i)] = 0;
    }

    psi[ind(0,0)] = 2.0/3.0 * cos(time);
               
    psi[ind(0,1)] = 3.0/4.0 * cos(time);
               
    psi[ind(0,2)] = 0.0 * cos(time);
               
    psi[ind(0,3)] = -1.0/12.0 * cos(time);
    */

    for (i=0; i<params.M; i++)
    {
	psi[ind(0,i)] = creal(psi[ind(0,i)]);
    }

}
#endif // FIELDS_2D_C_H
