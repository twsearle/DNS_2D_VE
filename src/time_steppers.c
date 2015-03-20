/* -------------------------------------------------------------------------- *
 *									      *
 *  time_steppers.c							      *
 *                                                                            *
 *  functions for time stepping 2D fields 				      *
 *                                                                            *
 *                                                                            *
 * -------------------------------------------------------------------------- */

// Last modified: Thu 19 Mar 14:43:24 2015

#include"fields_2D.h"

// Functions

void step_sf_SI_Crank_Nicolson(
	complex *psi, complex *psi2, double dt, int timeStep, complex
	*forcing, double oneOverRe, flow_params params, complex *scratch,
	complex *scratch2, complex *u, complex *v, complex *lplpsi, complex
	*biharmpsi, complex *d2ypsi, complex *dyyypsi, complex *d4ypsi,
	complex *d2xd2ypsi, complex *d4xpsi, complex *udxlplpsi, complex
	*vdylplpsi, complex *vdyypsi, complex *RHSvec, complex *opsList,
	fftw_plan *phys_plan, fftw_plan *spec_plan, complex *scratchin,
	complex *scratchout, double *scratchp1, double *scratchp2 
	)
{
    int i, j, l;
    int N = params.N;
    int M = params.M;

    // First of all calculate the nonlinear terms on psi2, then calculate
    // linear terms on psi then calculate RHS for each mode, then solve for the
    // new streamfunction, psi, at this timestep.

    // -----------Nonlinear Terms --------------
    //
    // u
    dy(psi2, u, params);

    // v
    dx(psi2, v, params);
    for(i=0; i<N+1; i++)
    {
	for(j=0; j<M; j++)
	{
	    v[ind(i,j)] = -v[ind(i,j)];
	}
    }


    // lplpsi dyy(psi) + dxx(psi)

    d2x(psi2, scratch, params);
    dy(u, lplpsi, params);
    for(i=0; i<N+1; i++)
    {
	for(j=0; j<M; j++)
	{
	    lplpsi[ind(i,j)] = lplpsi[ind(i,j)] + scratch[ind(i,j)];
	}
    }


    // udxlplpsi 
    dx(lplpsi, udxlplpsi, params);
    #ifdef MYDEBUG
    if(timeStep==0)
    {
	save_hdf5_state("./output/dxlplpsi.h5", &udxlplpsi[0], params);
    }
    #endif


    fft_convolve_r(udxlplpsi, u, udxlplpsi, scratchp1, scratchp2, scratchin,
	    scratchout, phys_plan, spec_plan, params);

    // vdylplpsi 
    dy(lplpsi, vdylplpsi, params);

    #ifdef MYDEBUG
    if(timeStep==0)
    {
	save_hdf5_state("./output/dylplpsi.h5", &vdylplpsi[0], params);
    }
    #endif


    fft_convolve_r(vdylplpsi, v, vdylplpsi, scratchp1, scratchp2, scratchin,
	    scratchout, phys_plan, spec_plan, params);

    //vdyypsi = vdyu
    dy(u, d2ypsi, params);

    fft_convolve_r(d2ypsi, v, vdyypsi, scratchp1, scratchp2, scratchin,
	    scratchout, phys_plan, spec_plan, params);

#ifdef MYDEBUG
    if(timeStep==0)
    {
	save_hdf5_state("./output/vdyypsi.h5", &vdyypsi[0], params);
    }
#endif

    // ----------- linear Terms --------------
    
    // lplpsi dyy(psi) + dxx(psi)

    d2x(psi, scratch, params);
    dy(psi, u, params);
    dy(u, lplpsi, params);

    for(i=0; i<N+1; i++)
    {
	for(j=0; j<M; j++)
	{
	    lplpsi[ind(i,j)] = lplpsi[ind(i,j)] + scratch[ind(i,j)];
	}
    }

    // biharmpsi (dyy + dxx)lplpsi

    dy(u, d2ypsi, params);
    dy(d2ypsi, dyyypsi, params);
    dy(dyyypsi, d4ypsi, params);

    d2x(psi, scratch, params);

#ifdef MYDEBUG
    if(timeStep==0)
    {
	save_hdf5_state("./output/d2xpsi.h5", &scratch[0], params);
    }
#endif

    dy(scratch, scratch2, params);
    dy(scratch2, d2xd2ypsi, params);


    d4x(psi, d4xpsi, params);

    for(i=0; i<N+1; i++)
    {
	for(j=0; j<M; j++)
	{
	    // biharmpsi[ind(i,j)] = biharmpsi[ind(i,j)] + scratch2[ind(i,j)];

	    biharmpsi[ind(i,j)] = d4xpsi[ind(i,j)] + 2.*d2xd2ypsi[ind(i,j)];
	    biharmpsi[ind(i,j)] = biharmpsi[ind(i,j)] + d4ypsi[ind(i,j)];
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
	save_hdf5_state("./output/u.h5",  &u[0], params);
	save_hdf5_state("./output/v.h5", &v[0], params);
	save_hdf5_state("./output/lplpsi.h5", &lplpsi[0], params);
	save_hdf5_state("./output/d2ypsi.h5", &d2ypsi[0], params);
	save_hdf5_state("./output/d3ypsi.h5", &dyyypsi[0], params);
	save_hdf5_state("./output/d4ypsi.h5", &d4ypsi[0], params);
	save_hdf5_state("./output/d2xd2ypsi.h5", &d2xd2ypsi[0], params);
	save_hdf5_state("./output/d4xpsi.h5", &d4xpsi[0], params);
	save_hdf5_state("./output/biharmpsi.h5", &biharmpsi[0], params);
	save_hdf5_state("./output/udxlplpsi.h5", &udxlplpsi[0], params);
	save_hdf5_state("./output/vdylplpsi.h5", &vdylplpsi[0], params);
    }
#endif

    for (i=1; i<N+1; i++)
    {
	for (j=0; j<M; j++)
	{

	    RHSvec[j] = 0.5*dt*oneOverRe*biharmpsi[ind(i,j)];
	    RHSvec[j] += + lplpsi[ind(i,j)];
	    RHSvec[j] += - dt*udxlplpsi[ind(i,j)];
	    RHSvec[j] += - dt*vdylplpsi[ind(i,j)];
	    RHSvec[j] += dt*forcing[ind(i,j)]; 

	}

	//impose BCs

	RHSvec[M-2] = 0;
	RHSvec[M-1] = 0;

#ifdef MYDEBUG
	if(timeStep==0)
	{
	    char fn[30];
	    sprintf(fn, "./output/RHSVec%d.h5", i);
	    printf("writing %s\n", fn);
	    save_hdf5_arr(fn, &RHSvec[0], M);
	}
#endif

	// perform dot product to calculate new streamfunction.
	for (j=M-1; j>=0; j=j-1)
	{
	    psi[ind(i,j)] = 0;

	    for (l=M-1; l>=0; l=l-1)
	    {
		psi[ind(i,j)] += opsList[(i*M + j)*M + l] * RHSvec[l];
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
	RHSvec[j] = dt*0.5*oneOverRe*dyyypsi[ind(0,j)] - dt*vdyypsi[ind(0,j)];
	RHSvec[j] += u[ind(0,j)]; 
	RHSvec[j] += dt*forcing[ind(0,j)]; 
    }
    //RHSvec[0] += 2*dt*oneOverRe;

    // apply BCs
    // # dyPsi0(+-1) = 0
    // RHSVec[N*M + M-3] = 0
    // RHSVec[N*M + M-2] = 0
    // # Psi0(-1) = 0
    // RHSVec[N*M + M-1] = 0

    RHSvec[M-3] = params.U0; 
    RHSvec[M-2] = -params.U0; 
    RHSvec[M-1] = 0; 

#ifdef MYDEBUG
    if(timeStep==0)
    {
	char fn[30];
	sprintf(fn, "./output/RHSVec%d.h5", 0);
	save_hdf5_arr(fn, &RHSvec[0], M);
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
	    psi[ind(0,j)] += opsList[j*M + l] * RHSvec[l];

	}
    }

}
