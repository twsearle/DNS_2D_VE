/* -------------------------------------------------------------------------- *
 *									      *
 *  time_steppers.c							      *
 *                                                                            *
 *  functions for time stepping 2D fields 				      *
 *                                                                            *
 *                                                                            *
 * -------------------------------------------------------------------------- */

// Last modified: Wed  6 May 12:08:13 2015

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
    dy(psi2, scr.u, scr, params);

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
    dy(scr.u, scr.lplpsi, scr, params);
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
    dy(scr.lplpsi, scr.vdylplpsi, scr, params);

    #ifdef MYDEBUG
    if(timeStep==0)
    {
	save_hdf5_state("./output/dylplpsi.h5", &scr.vdylplpsi[0], params);
    }
    #endif


    fft_convolve_r(scr.vdylplpsi, scr.v, scr.vdylplpsi, 
	    scr, params);

    //vdyypsi = vdyu
    dy(scr.u, scr.d2ypsi, scr, params);

    fft_convolve_r(scr.d2ypsi, scr.v, scr.vdyypsi, 
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
    dy(psi, scr.u, scr, params);
    dy(scr.u, scr.lplpsi, scr, params);

    for(i=0; i<N+1; i++)
    {
	for(j=0; j<M; j++)
	{
	    scr.lplpsi[ind(i,j)] = scr.lplpsi[ind(i,j)] + scr.scratch[ind(i,j)];
	}
    }

    // biharmpsi (dyy + dxx)lplpsi

    dy(scr.u, scr.d2ypsi, scr, params);
    dy(scr.d2ypsi, scr.dyyypsi, scr, params);
    dy(scr.dyyypsi, scr.d4ypsi, scr, params);

    d2x(psi, scr.scratch, params);

#ifdef MYDEBUG
    if(timeStep==0)
    {
	save_hdf5_state("./output/d2xpsi.h5", &scr.scratch[0], params);
    }
#endif

    dy(scr.scratch, scr.scratch2, scr, params);
    dy(scr.scratch2, scr.d2xd2ypsi, scr, params);


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
	save_hdf5_state("./output/d2ypsi.h5", &scr.d2ypsi[0], params);
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
