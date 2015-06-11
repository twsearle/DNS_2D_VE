/* -------------------------------------------------------------------------- *
 *									      *
 *  time_steppers.c							      *
 *                                                                            *
 *  functions for time stepping 2D fields 				      *
 *                                                                            *
 *                                                                            *
 * -------------------------------------------------------------------------- */

// Last modified: Thu 11 Jun 11:36:45 2015

#include"fields_2D.h"

// Functions

void step_sf_SI_Crank_Nicolson(
	complex *psi, complex *psi2, double dt, int timeStep, complex
	*forcing, complex *opsList, flow_scratch scr, flow_params params)
{
    int i, j, l;
    int N = params.N;
    int M = params.M;
    double oneOverRe = 1. / params.Re;

    // First of all calculate the nonlinear terms on psi2, then calculate
    // linear terms on psi then calculate RHS for each mode, then solve for the
    // new streamfunction, psi, at this timestep.

    // -----------Nonlinear Terms --------------
    //
    // u
    dy(psi2, scr.u, params);

    // v
    dx(psi2, scr.v, params);
    for(i=0; i<N+1; i++)
    {
	for(j=0; j<M; j++)
	{
	    scr.v[ind(i,j)] = -scr.v[ind(i,j)];
	}
    }


    // lplpsi dyy(psi) + dxx(psi)

    d2x(psi2, scr.scratch, params);
    dy(scr.u, scr.lplpsi, params);
    for(i=0; i<N+1; i++)
    {
	for(j=0; j<M; j++)
	{
	    scr.lplpsi[ind(i,j)] = scr.lplpsi[ind(i,j)] + scr.scratch[ind(i,j)];
	}
    }


    // udxlplpsi 
    dx(scr.lplpsi, scr.udxlplpsi, params);
    #ifdef MYDEBUG
    if(timeStep==0)
    {
	save_hdf5_state("./output/dxlplpsi.h5", &scr.udxlplpsi[0], params);
    }
    #endif


    fft_convolve_r(scr.udxlplpsi, scr.u, scr.udxlplpsi, 
	    scr, params);

    // vdylplpsi 
    dy(scr.lplpsi, scr.vdylplpsi, params);

    #ifdef MYDEBUG
    if(timeStep==0)
    {
	save_hdf5_state("./output/dylplpsi.h5", &scr.vdylplpsi[0], params);
    }
    #endif


    fft_convolve_r(scr.vdylplpsi, scr.v, scr.vdylplpsi, 
	    scr, params);

    //vdyypsi = vdyu
    dy(scr.u, scr.dyu, params);

    fft_convolve_r(scr.dyu, scr.v, scr.vdyypsi, 
	    scr, params);

#ifdef MYDEBUG
    if(timeStep==0)
    {
	save_hdf5_state("./output/vdyypsi.h5", &scr.vdyypsi[0], params);
    }
#endif

    // ----------- linear Terms --------------
    
    // lplpsi dyy(psi) + dxx(psi)

    d2x(psi, scr.scratch, params);
    dy(psi, scr.u, params);
    dy(scr.u, scr.lplpsi, params);

    for(i=0; i<N+1; i++)
    {
	for(j=0; j<M; j++)
	{
	    scr.lplpsi[ind(i,j)] = scr.lplpsi[ind(i,j)] + scr.scratch[ind(i,j)];
	}
    }

    // biharmpsi (dyy + dxx)lplpsi

    dy(scr.u, scr.dyu, params);
    dy(scr.dyu, scr.dyyypsi, params);
    dy(scr.dyyypsi, scr.d4ypsi, params);

    d2x(psi, scr.scratch, params);

#ifdef MYDEBUG
    if(timeStep==0)
    {
	save_hdf5_state("./output/d2xpsi.h5", &scr.scratch[0], params);
    }
#endif

    dy(scr.scratch, scr.scratch2, params);
    dy(scr.scratch2, scr.d2xd2ypsi, params);


    d4x(psi, scr.d4xpsi, params);

    for(i=0; i<N+1; i++)
    {
	for(j=0; j<M; j++)
	{
	    // biharmpsi[ind(i,j)] = biharmpsi[ind(i,j)] + scr.scratch2[ind(i,j)];

	    scr.biharmpsi[ind(i,j)] = scr.d4xpsi[ind(i,j)] + 2.*scr.d2xd2ypsi[ind(i,j)];
	    scr.biharmpsi[ind(i,j)] = scr.biharmpsi[ind(i,j)] + scr.d4ypsi[ind(i,j)];
	}
    }

    // RHSVec = dt*0.5*oneOverRe*BIHARMPSI 
    // 	+ LPLPSI 
    // 	- dt*UDXLPLPSI 
    // 	- dt*VDYLPLPSI 
    
#ifdef MYDEBUG
    if(timeStep==0)
    {
	save_hdf5_state("./output/u.h5",  &scr.u[0], params);
	save_hdf5_state("./output/v.h5", &scr.v[0], params);
	save_hdf5_state("./output/lplpsi.h5", &scr.lplpsi[0], params);
	save_hdf5_state("./output/d2ypsi.h5", &scr.dyu[0], params);
	save_hdf5_state("./output/d3ypsi.h5", &scr.dyyypsi[0], params);
	save_hdf5_state("./output/d4ypsi.h5", &scr.d4ypsi[0], params);
	save_hdf5_state("./output/d2xd2ypsi.h5", &scr.d2xd2ypsi[0], params);
	save_hdf5_state("./output/d4xpsi.h5", &scr.d4xpsi[0], params);
	save_hdf5_state("./output/biharmpsi.h5", &scr.biharmpsi[0], params);
	save_hdf5_state("./output/udxlplpsi.h5", &scr.udxlplpsi[0], params);
	save_hdf5_state("./output/vdylplpsi.h5", &scr.vdylplpsi[0], params);
    }
#endif

    for (i=1; i<N+1; i++)
    {
	for (j=0; j<M; j++)
	{

	    scr.RHSvec[j] = 0.5*dt*oneOverRe*scr.biharmpsi[ind(i,j)];
	    scr.RHSvec[j] += + scr.lplpsi[ind(i,j)];
	    scr.RHSvec[j] += - dt*scr.udxlplpsi[ind(i,j)];
	    scr.RHSvec[j] += - dt*scr.vdylplpsi[ind(i,j)];
	    scr.RHSvec[j] += dt*forcing[ind(i,j)]; 

	}

	//impose BCs

	scr.RHSvec[M-2] = 0;
	scr.RHSvec[M-1] = 0;

#ifdef MYDEBUG
	if(timeStep==0)
	{
	    char fn[30];
	    sprintf(fn, "./output/RHSVec%d.h5", i);
	    printf("writing %s\n", fn);
	    save_hdf5_arr(fn, &scr.RHSvec[0], M);
	}
#endif

	// perform dot product to calculate new streamfunction.
	for (j=M-1; j>=0; j=j-1)
	{
	    psi[ind(i,j)] = 0;

	    for (l=M-1; l>=0; l=l-1)
	    {
		psi[ind(i,j)] += opsList[(i*M + j)*M + l] * scr.RHSvec[l];
	    }
	}

    }


    // # Zeroth mode
    // RHSVec[N*M:(N+1)*M] = 0
    // RHSVec[N*M:(N+1)*M] = dt*0.5*oneOverRe*dot(MDYYY, PSI)[N*M:(N+1)*M] 
    // 	+ dot(MDY, PSI)[N*M:(N+1)*M] 
    // 	- dt*dot(dot(MMV, MDYY), PSI)[N*M:(N+1)*M]
    // RHSVec[N*M] += dt*2*oneOverRe

    for (j=0; j<M; j++)
    {
	//RHSvec[j] = u[ind(0,j)];
	scr.RHSvec[j] = dt*0.5*oneOverRe*scr.dyyypsi[ind(0,j)] - dt*scr.vdyypsi[ind(0,j)];
	scr.RHSvec[j] += scr.u[ind(0,j)]; 
	scr.RHSvec[j] += dt*forcing[ind(0,j)]; 
    }
    //scr.RHSvec[0] += 2*dt*oneOverRe;

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


void step_conformation_Crank_Nicolson(
	 complex *cijOld, complex *cij, complex *psi, complex *cijNL, double
	 dt, flow_scratch scr, flow_params params)
{
    int N = params.N;
    int M = params.M;
    int i,j;
    double oneOverWi = 1./params.Wi;
    

    // Nonlinear terms ----------------------------------------------------
    // dxU, dyU, dxV, dyV
    
    dy(psi, scr.u, params);
    dx(psi, scr.v, params);

    for (i =0; i<M*(N+1); i++)
    {
	scr.v[i] = -scr.v[i];
    }

    dx(scr.u, scr.dxu, params);
    dy(scr.u, scr.dyu, params);
    dx(scr.v, scr.dxv, params);
    dy(scr.v, scr.dyv, params);
    
    // Cxx*dxU 
    fft_convolve_r(&cijNL[0], scr.dxu, scr.cxxdxu, scr, params);

    // Cxy*dyU 
    fft_convolve_r(&cijNL[2*(N+1)*M], scr.dyu, scr.cxydyu, scr, params);

    // VGrad*Cxx
    dx(&cijNL[0], scr.scratch, params);
    fft_convolve_r(scr.u, scr.scratch, scr.scratch, scr, params);

    dy(&cijNL[0], scr.scratch2, params);
    fft_convolve_r(scr.v, scr.scratch2, scr.scratch2, scr, params);

    for (i=0; i<N+1; i++)
    {
	for (j=0; j<M; j++)
	{
	    scr.vgradcxx[ind(i,j)] = scr.scratch[ind(i,j)] + scr.scratch2[ind(i,j)];
	}
    }
    
    // Cxy*dxV
    fft_convolve_r(&cijNL[2*(N+1)*M], scr.dxv, scr.cxydxv, scr, params);

    // Cyy*dyV
    fft_convolve_r(&cijNL[(N+1)*M], scr.dyv, scr.cyydyv, scr, params);

    // vgrad*Cyy
    dx(&cijNL[(N+1)*M], scr.scratch, params);
    fft_convolve_r(scr.u, scr.scratch, scr.scratch, scr, params);

    dy(&cijNL[(N+1)*M], scr.scratch2, params);
    fft_convolve_r(scr.v, scr.scratch2, scr.scratch2, scr, params);

    for (i=0; i<N+1; i++)
    {
	for (j=0; j<M; j++)
	{
	    scr.vgradcyy[ind(i,j)] = scr.scratch[ind(i,j)] + scr.scratch2[ind(i,j)];
	}
    }
    
    // Cxx*dxV
    fft_convolve_r(&cijNL[0], scr.dxv, scr.cxxdxv, scr, params);

    // CyydyU
    fft_convolve_r(&cijNL[(N+1)*M], scr.dyu, scr.cyydyu, scr, params);

    // Vgrad*Cxy
    dx(&cijNL[2*(N+1)*M], scr.scratch, params);
    fft_convolve_r(scr.u, scr.scratch, scr.scratch, scr, params);

    dy(&cijNL[2*(N+1)*M], scr.scratch2, params);
    fft_convolve_r(scr.v, scr.scratch2, scr.scratch2, scr, params);

    for (i=0; i<N+1; i++)
    {
	for (j=0; j<M; j++)
	{
	    scr.vgradcxy[ind(i,j)] = scr.scratch[ind(i,j)] + scr.scratch2[ind(i,j)];
	}
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
    
    for (i=0; i<N+1; i++)
    {
        for (j=0; j<M; j++)
        {
            cij[ind(i,j)] = cijOld[ind(i,j)];
            cij[ind(i,j)] += - 0.5*dt*oneOverWi*cijOld[ind(i,j)];
	    cij[ind(i,j)] += dt*2.*scr.cxydyu[ind(i,j)];
            cij[ind(i,j)] += dt*2.*scr.cxxdxu[ind(i,j)];
            cij[ind(i,j)] += - dt*scr.vgradcxx[ind(i,j)];

            cij[ind(i,j)] *= params.Wi/(params.Wi+dt*0.5);
            

            cij[(N+1)*M + ind(i,j)] = cijOld[(N+1)*M + ind(i,j)];
            cij[(N+1)*M + ind(i,j)] += - 0.5*dt*oneOverWi*cijOld[(N+1)*M + ind(i,j)];
            cij[(N+1)*M + ind(i,j)] += dt*2.*scr.cxydxv[ind(i,j)]; 
	    cij[(N+1)*M + ind(i,j)] += dt*2.*scr.cyydyv[ind(i,j)];
            cij[(N+1)*M + ind(i,j)] += - dt*scr.vgradcyy[ind(i,j)];

            cij[(N+1)*M + ind(i,j)] *= params.Wi/(params.Wi+dt*0.5);
            

            cij[2*(N+1)*M + ind(i,j)] = cijOld[2*(N+1)*M + ind(i,j)];
            cij[2*(N+1)*M + ind(i,j)] += - 0.5*dt*oneOverWi*cijOld[2*(N+1)*M + ind(i,j)];
            cij[2*(N+1)*M + ind(i,j)] += dt*scr.cxxdxv[ind(i,j)];
	    cij[2*(N+1)*M + ind(i,j)] += dt*scr.cyydyu[ind(i,j)];
            cij[2*(N+1)*M + ind(i,j)] += - dt*scr.vgradcxy[ind(i,j)];

            cij[2*(N+1)*M + ind(i,j)] *= params.Wi/(params.Wi+dt*0.5);

        }
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

}

void step_sf_SI_Crank_Nicolson_visco(
	complex *psiOld, complex *psi, complex *cijOld, complex *cij, complex
	*psiNL, complex *forcing, complex *forcingN, double dt, int timeStep,
	complex *opsList, flow_scratch scr, flow_params params)
{
    int i, j, l;
    int N = params.N;
    int M = params.M;
    double oneOverRe = 1./params.Re;
    double oneOverWi = 1./params.Wi;
    double beta = params.beta;

    // -----------Nonlinear Terms --------------
    //
    // 
    dy(psiNL, scr.u, params);

    // v
    dx(psiNL, scr.v, params);
    for(i=0; i<N+1; i++)
    {
	for(j=0; j<M; j++)
	{
	    scr.v[ind(i,j)] = -scr.v[ind(i,j)];
	}
    }


    // lplpsi dyy(psi) + dxx(psi)

    d2x(psiNL, scr.scratch, params);
    dy(scr.u, scr.lplpsi, params);
    for(i=0; i<N+1; i++)
    {
	for(j=0; j<M; j++)
	{
	    scr.lplpsi[ind(i,j)] = scr.lplpsi[ind(i,j)] + scr.scratch[ind(i,j)];
	}
    }


    // udxlplpsi 
    dx(scr.lplpsi, scr.udxlplpsi, params);
    #ifdef MYDEBUG
    if(timeStep==0)
    {
	save_hdf5_state("./output/dxlplpsi.h5", &scr.udxlplpsi[0], params);
    }
    #endif


    fft_convolve_r(scr.udxlplpsi, scr.u, scr.udxlplpsi, scr, params);

    // vdylplpsi 
    dy(scr.lplpsi, scr.vdylplpsi, params);

    #ifdef MYDEBUG
    if(timeStep==0)
    {
	save_hdf5_state("./output/dylplpsi.h5", &scr.vdylplpsi[0], params);
    }
    #endif


    fft_convolve_r(scr.vdylplpsi, scr.v, scr.vdylplpsi, scr, params);

    //vdyypsi = vdyu
    dy(scr.u, scr.dyu, params);

    fft_convolve_r(scr.dyu, scr.v, scr.vdyypsi, scr, params);

#ifdef MYDEBUG
    if(timeStep==0)
    {
	save_hdf5_state("./output/vdyypsi.h5", &scr.vdyypsi[0], params);
    }
#endif

    // ----------- linear Terms --------------
    
    
    // lplpsi dyy(psi) + dxx(psi)

    d2x(psiOld, scr.scratch, params);

    #ifdef MYDEBUG
    if(timeStep==0)
    {
	save_hdf5_state("./output/d2xpsi.h5", &scr.scratch[0], params);
    }
    #endif

    dy(psiOld, scr.u, params);
    dy(scr.u, scr.dyu, params);

    for(i=0; i<N+1; i++)
    {
	for(j=0; j<M; j++)
	{
	    scr.lplpsi[ind(i,j)] = scr.dyu[ind(i,j)] + scr.scratch[ind(i,j)];
	}
    }

    // biharmpsi (dyy + dxx)lplpsi

    dy(scr.dyu, scr.dyyypsi, params);
    dy(scr.dyyypsi, scr.d4ypsi, params);

    dy(scr.scratch, scr.scratch2, params);
    dy(scr.scratch2, scr.d2xd2ypsi, params);


    d4x(psiOld, scr.d4xpsi, params);

    for(i=0; i<N+1; i++)
    {
	for(j=0; j<M; j++)
	{
	    // biharmpsi[ind(i,j)] = biharmpsi[ind(i,j)] + scr.scratch2[ind(i,j)];

	    scr.biharmpsi[ind(i,j)] = scr.d4xpsi[ind(i,j)] + 2.*scr.d2xd2ypsi[ind(i,j)];
	    scr.biharmpsi[ind(i,j)] = scr.biharmpsi[ind(i,j)] + scr.d4ypsi[ind(i,j)];
	}
    }

    // RHSVec = dt*0.5*oneOverRe*BIHARMPSI 
    // 	+ LPLPSI 
    // 	- dt*UDXLPLPSI 
    // 	- dt*VDYLPLPSI 
    
#ifdef MYDEBUG
    if(timeStep==0)
    {
	printf("should see some output?\n");
	save_hdf5_state("./output/u.h5",  &scr.u[0], params);
	save_hdf5_state("./output/v.h5", &scr.v[0], params);
	save_hdf5_state("./output/lplpsi.h5", &scr.lplpsi[0], params);
	save_hdf5_state("./output/d2ypsi.h5", &scr.dyu[0], params);
	save_hdf5_state("./output/d3ypsi.h5", &scr.dyyypsi[0], params);
	save_hdf5_state("./output/d4ypsi.h5", &scr.d4ypsi[0], params);
	save_hdf5_state("./output/d2xd2ypsi.h5", &scr.d2xd2ypsi[0], params);
	save_hdf5_state("./output/d4xpsi.h5", &scr.d4xpsi[0], params);
	save_hdf5_state("./output/biharmpsi.h5", &scr.biharmpsi[0], params);
	save_hdf5_state("./output/udxlplpsi.h5", &scr.udxlplpsi[0], params);
	save_hdf5_state("./output/vdylplpsi.h5", &scr.vdylplpsi[0], params);
    }
#endif

    // stresses

    // AT FUTURE STEP
    
    //dycxy
    dy(&cij[2*(N+1)*M], scr.dycxyN, params);
    
    // d2x cxy
    d2x(&cij[2*(N+1)*M], scr.d2xcxyN, params);
    
    // d2y cxy
    dy(&cij[2*(N+1)*M], scr.scratch, params);
    dy(scr.scratch, scr.d2ycxyN, params);

    // dxy cyy-cxx
    for (i=0; i<N+1; i++)
    {
	for (j=0; j<M; j++)
	{
	    scr.scratch[ind(i,j)] = cij[(N+1)*M + ind(i,j)] - cij[ind(i,j)];
	}
    }
    dx(scr.scratch, scr.scratch2, params);
    dy(scr.scratch2, scr.dxycyy_cxxN, params);

    // AT PREVIOUS STEP
    
    //dycxy
    dy(&cijOld[2*(N+1)*M], scr.dycxy, params);
    
    // d2x cxy
    d2x(&cijOld[2*(N+1)*M], scr.d2xcxy, params);
    
    // d2y cxy
    dy(&cijOld[2*(N+1)*M], scr.scratch, params);
    dy(scr.scratch, scr.d2ycxy, params);

    // dxy cyy-cxx
    for (i=0; i<N+1; i++)
    {
	for (j=0; j<M; j++)
	{
	    scr.scratch[ind(i,j)] = cijOld[(N+1)*M + ind(i,j)] - cijOld[ind(i,j)];
	}
    }

    dx(scr.scratch, scr.scratch2, params);
    dy(scr.scratch2, scr.dxycyy_cxx, params);
    
    // Streamfunction equation:

    // RHSVec = dt*0.5*oneOverRe*BIHARMPSI 
    // 	+ LPLPSI 
    // 	- dt*UDXLPLPSI 
    // 	- dt*VDYLPLPSI 
    // + _dt*(1.-beta)*oneOverRe(dot(MDXX, Txy) 
    //                + dot(MDXY,(Tyy - Txx)) 
    //                - dot(MDYY, txy) )

    for (i=1; i<N+1; i++)
    {
	for (j=0; j<M; j++)
	{

	    scr.RHSvec[j] = 0.5*dt*oneOverRe*beta*scr.biharmpsi[ind(i,j)];
	    scr.RHSvec[j] += + 0.5*dt*(1.-beta)*oneOverRe*oneOverWi*( 
				         scr.d2ycxy[ind(i,j)]
			               - scr.d2xcxy[ind(i,j)] 
				       - scr.dxycyy_cxx[ind(i,j)]);
	    scr.RHSvec[j] += + 0.5*dt*(1.-beta)*oneOverRe*oneOverWi*( 
				         scr.d2ycxyN[ind(i,j)]
			               - scr.d2xcxyN[ind(i,j)] 
				       - scr.dxycyy_cxxN[ind(i,j)]); 
	    scr.RHSvec[j] += 0.5*dt*forcing[ind(i,j)];
	    scr.RHSvec[j] += 0.5*dt*forcingN[ind(i,j)];
	    scr.RHSvec[j] += - dt*scr.udxlplpsi[ind(i,j)];
	    scr.RHSvec[j] += - dt*scr.vdylplpsi[ind(i,j)];
	    scr.RHSvec[j] += + scr.lplpsi[ind(i,j)];


	}

	//impose BCs

	scr.RHSvec[M-2] = 0;
	scr.RHSvec[M-1] = 0;

	#ifdef MYDEBUG
	if(timeStep==0)
	{
	    char fn[30];
	    sprintf(fn, "./output/RHSVec%d.h5", i);
	    printf("writing %s\n", fn);
	    save_hdf5_arr(fn, &scr.RHSvec[0], M);
	}
	#endif

	// perform dot product to calculate new streamfunction.
	for (j=M-1; j>=0; j=j-1)
	{
	    psi[ind(i,j)] = 0;

	    for (l=M-1; l>=0; l=l-1)
	    {
		psi[ind(i,j)] += opsList[(i*M + j)*M + l] * scr.RHSvec[l];
	    }
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
	scr.RHSvec[j]  = dt*0.5*beta*oneOverRe*creal(scr.dyyypsi[ind(0,j)]);
	scr.RHSvec[j] += dt*0.5*(1.-beta)*oneOverRe*oneOverWi*creal(scr.dycxy[ind(0,j)]); 
	scr.RHSvec[j] += dt*0.5*(1.-beta)*oneOverRe*oneOverWi*creal(scr.dycxyN[ind(0,j)]); 
	scr.RHSvec[j] += dt*0.5*creal(forcing[ind(0,j)]);
	scr.RHSvec[j] += dt*0.5*creal(forcingN[ind(0,j)]);
	scr.RHSvec[j] += - dt*creal(scr.vdyypsi[ind(0,j)]);
	scr.RHSvec[j] += creal(scr.u[ind(0,j)]); 
    }

    // RHSvec[0] += 2*dt*oneOverRe;

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



void stress_time_derivative(
	complex *psi, complex *cxx, complex *cyy, complex *cxy, complex *fxx,
	complex *fyy, complex *fxy, complex *txx, complex *tyy, complex *txy,
	flow_scratch scr, flow_params params
	)
{
    // Calculate f({psi, cxx, cyy, cxy}, dt)
    int N = params.N;
    int M = params.M;
    int i,j;
    double oneOverWi = 1./params.Wi;
    
    // dxU, dyU, dxV, dyV
    dy(psi, scr.u, params);
    dx(psi, scr.v, params);

    for (i =0; i<M*(N+1); i++)
    {
	scr.v[i] = -scr.v[i];
    }

    dx(scr.u, scr.dxu, params);
    dy(scr.u, scr.dyu, params);
    dx(scr.v, scr.dxv, params);
    dy(scr.v, scr.dyv, params);
    
    // Cxx*dxU 
    fft_convolve_r(cxx, scr.dxu, scr.cxxdxu, scr, params);

    // Cxy*dyU 
    fft_convolve_r(cxy, scr.dyu, scr.cxydyu, scr, params);

    // VGrad*Cxx
    dx(cxx, scr.scratch, params);
    fft_convolve_r(scr.u, scr.scratch, scr.scratch, scr, params);

    dy(cxx, scr.scratch2, params);
    fft_convolve_r(scr.v, scr.scratch2, scr.scratch2, scr, params);

    for (i=0; i<N+1; i++)
    {
	for (j=0; j<M; j++)
	{
	    scr.vgradcxx[ind(i,j)] = scr.scratch[ind(i,j)] + scr.scratch2[ind(i,j)];
	}
    }
    
    // Cxy*dxV
    fft_convolve_r(cxy, scr.dxv, scr.cxydxv, scr, params);

    // Cyy*dyV
    fft_convolve_r(cyy, scr.dyv, scr.cyydyv, scr, params);

    // vgrad*Cyy
    dx(cyy, scr.scratch, params);
    fft_convolve_r(scr.u, scr.scratch, scr.scratch, scr, params);

    dy(cyy, scr.scratch2, params);
    fft_convolve_r(scr.v, scr.scratch2, scr.scratch2, scr, params);

    for (i=0; i<N+1; i++)
    {
	for (j=0; j<M; j++)
	{
	    scr.vgradcyy[ind(i,j)] = scr.scratch[ind(i,j)] + scr.scratch2[ind(i,j)];
	}
    }
    
    // Cxx*dxV
    fft_convolve_r(cxx, scr.dxv, scr.cxxdxv, scr, params);

    // CyydyU
    fft_convolve_r(cyy, scr.dyu, scr.cyydyu, scr, params);

    // Vgrad*Cxy
    dx(cxy, scr.scratch, params);
    fft_convolve_r(scr.u, scr.scratch, scr.scratch, scr, params);

    dy(cxy, scr.scratch2, params);
    fft_convolve_r(scr.v, scr.scratch2, scr.scratch2, scr, params);

    for (i=0; i<N+1; i++)
    {
	for (j=0; j<M; j++)
	{
	    scr.vgradcxy[ind(i,j)] = scr.scratch[ind(i,j)] + scr.scratch2[ind(i,j)];
	}
    }
    
    // Calculate polymeric stress components
    // TxxOld = oneOverWi*CxxOld
    // TxxOld[N*M] += -oneOverWi
    // TyyOld = oneOverWi*CyyOld
    // TyyOld[N*M] += -oneOverWi
    // TxyOld = oneOverWi*CxyOld
    
    for (i=0; i<N+1; i++)
    {
	for (j=0; j<M; j++)
	{
	txx[ind(i,j)] = oneOverWi*cxx[ind(i,j)];
	tyy[ind(i,j)] = oneOverWi*cyy[ind(i,j)];
	txy[ind(i,j)] = oneOverWi*cxy[ind(i,j)];
	}
    }
    txx[ind(0,0)] += -oneOverWi;
    tyy[ind(0,0)] += -oneOverWi;
     
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
    for (i=0; i<N+1; i++)
    {
	for (j=0; j<M; j++)
	{
	    fxx[ind(i,j)] = 2*scr.cxxdxu[ind(i,j)] + 2*scr.cxydyu[ind(i,j)];
	    fxx[ind(i,j)] += - scr.vgradcxx[ind(i,j)] - txx[ind(i,j)];

	    fyy[ind(i,j)] = 2*scr.cxydxv[ind(i,j)] + 2*scr.cyydyv[ind(i,j)];
	    fyy[ind(i,j)] += - scr.vgradcyy[ind(i,j)] - tyy[ind(i,j)];

	    fxy[ind(i,j)] = scr.cxxdxv[ind(i,j)] + scr.cyydyu[ind(i,j)];
	    fxy[ind(i,j)] += - scr.vgradcxy[ind(i,j)] - txy[ind(i,j)];
	}
    }
}

void equilibriate_stress(
	complex *psiOld, complex *psi_lam, complex *cijOld, complex *cij,
	complex *cijNL, double dt,flow_scratch scr, flow_params params, hid_t
	*file_id, hid_t *filetype_id, hid_t *datatype_id
	)

{
    int i;
    int N = params.N;
    int M = params.M;
    double Wi = params.Wi;
    double time = 0;
    int timeStep, numSteps;
#ifdef MYDEBUG 
    double EE0 = 1.0;
    double EE1 = 0.0;
    double EE2 = 0.0;
    double EE_tot = 0.0;
    double EE_xdepend = 0.0;
#endif


    // Time step for approximately 2 the elastic timescale, 2*Wi.
    // Do homotopy between laminar flow and the initial streamfunction.
    // Update the stress according to the velocity at each timestep, but update
    // the streamfunction according to the homotopy 
    // psi = 0.5*(1+cos((Pi/2Wi) * time)) * psi_lam + 
    //	     0.5*(1-cos((Pi/2Wi) * time)) * psi  

    FILE *tracefp = fopen("trace_ramp.dat", "w");

    // setup temporary psi to perform the homotopy from the laminar flow to the
    // initial state.

    complex *psi_tmp = (complex*) fftw_malloc(M*(N+1) * sizeof(complex));

    complex *trC = (complex*) fftw_malloc(M*(N+1) * sizeof(complex));

    numSteps = 2.0*Wi/dt;

    for (timeStep = 0; timeStep<numSteps; timeStep++)
    {
	// set the velocity via the homotopy

	time = timeStep * dt;
	for (i=0; i<M*(N+1); i++)
	{
	    psi_tmp[i] = 0.5*(1.-cos(time * M_PI/(2.0*Wi))) * psiOld[i];
	    psi_tmp[i] += 0.5*(1.+cos(time * M_PI/(2.0*Wi))) * psi_lam[i];
	}

	// step the stress to t star

	step_conformation_Crank_Nicolson( cijOld, cijNL, psi_tmp, cijOld, 0.5*dt, scr,
		params);

	// assume psi star = psi_tmp
	// TODO: Account for a time dependent forcing, psi star will be different?

	// step the stress to t + h

	step_conformation_Crank_Nicolson( cijOld, cij, psi_tmp, cijNL, dt, scr,
		params);

#ifdef MYDEBUG
	// output some trajectories to check everything is going ok! 
	if ((timeStep % (numSteps / 100)) == 0 )
	{
	    trC_tensor(cij, trC, scr.scratchp1, scr.scratchp2, scr.scratchin, scr.scratchout,
		    scr.phys_plan, scr.spec_plan, params);

	//    diagonalised_C(complex *cij, complex *ecij, double *rcij, double
	//	    *scr.scratchp1, double *scr.scratchp2, fftw_complex *scr.scratchin, fftw_complex
	//	    *scr.scratchout, fftw_plan *phys_plan, fftw_plan
	//	    *spec_plan, flow_params cnsts)

	    calc_EE_mode(&trC[0], 0, params);

	    EE0 = calc_EE_mode(&trC[0], 0, params);
	    EE1 = calc_EE_mode(&trC[0], 1, params);
	    EE2 = calc_EE_mode(&trC[0], 2, params);

	    EE_xdepend = EE1 + EE2; 
	    for (i=3; i<N+1; i++)
	    {
		EE_xdepend += calc_EE_mode(&trC[0], i, params);
	    }

	    EE_tot = EE0 + EE_xdepend;

	    save_hdf5_snapshot_visco(file_id, filetype_id, datatype_id,
		    psi_tmp, &cij[0], &cij[(N+1)*M], &cij[2*(N+1)*M], time, params);

	    fprintf(tracefp, "%e\t%e\t%e\t%e\t%e\t%e\n", time, EE_tot, EE0, EE1, EE2, EE_xdepend);

            printf("%e\t%e\t%e\t%e\t%e\n", time, EE_tot, EE0, EE1, EE2);

	    fflush(tracefp);
	    H5Fflush(*file_id, H5F_SCOPE_GLOBAL);
	}
#endif

    }

    fclose(tracefp);
    free(psi_tmp);
    free(trC);
}



