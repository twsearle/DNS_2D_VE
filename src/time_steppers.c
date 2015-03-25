/* -------------------------------------------------------------------------- *
 *									      *
 *  time_steppers.c							      *
 *                                                                            *
 *  functions for time stepping 2D fields 				      *
 *                                                                            *
 *                                                                            *
 * -------------------------------------------------------------------------- */

// Last modified: Tue 24 Mar 11:48:13 2015

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
    // 
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

void step_conformation_Crank_Nicolson(
	complex *psi, complex *cij, complex *cijNL, double dt, flow_params params,
	complex *u, complex *v, complex *dxu, complex *dyu, complex *dxv,
	complex *dyv, complex *cxxdxu, complex *cxydyu, complex *vgradcxx,
	complex *cxydxv, complex *cyydyv, complex *vgradcyy, complex *cxxdxv,
	complex *cyydyu, complex *vgradcxy, complex *scratch, complex
	*scratch2, fftw_plan *phys_plan, fftw_plan *spec_plan, complex
	*scratchin, complex *scratchout, double *scratchp1, double *scratchp2 
	)
{
    int N = params.N;
    int M = params.M;
    int i,j;
    double oneOverWi = 1./params.Wi;
    

    // Nonlinear terms ----------------------------------------------------
    // dxU, dyU, dxV, dyV
    dy(psi, u, params);
    dx(psi, v, params);

    for (i =0; i<M*(N+1); i++)
    {
	v[i] = -v[i];
    }

    dx(u, dxu, params);
    dy(u, dyu, params);
    dx(v, dxv, params);
    dy(v, dyv, params);
    
    // Cxx*dxU 
    fft_convolve_r(&cijNL[0], dxu, cxxdxu, scratchp1, scratchp2, scratchin,
	    scratchout, phys_plan, spec_plan, params);

    // Cxy*dyU 
    fft_convolve_r(&cijNL[2*(N+1)*M], dyu, cxydyu, scratchp1, scratchp2, scratchin,
	    scratchout, phys_plan, spec_plan, params);

    // VGrad*Cxx
    dx(&cijNL[0], scratch, params);
    fft_convolve_r(u, scratch, scratch, scratchp1, scratchp2, scratchin,
	    scratchout, phys_plan, spec_plan, params);

    dy(&cijNL[0], scratch2, params);
    fft_convolve_r(v, scratch2, scratch2, scratchp1, scratchp2, scratchin,
	    scratchout, phys_plan, spec_plan, params);

    for (i=0; i<N+1; i++)
    {
	for (j=0; j<M; j++)
	{
	    vgradcxx[ind(i,j)] = scratch[ind(i,j)] + scratch2[ind(i,j)];
	}
    }
    
    // Cxy*dxV
    fft_convolve_r(&cijNL[2*(N+1)*M], dxv, cxydxv, scratchp1, scratchp2, scratchin,
	    scratchout, phys_plan, spec_plan, params);

    // Cyy*dyV
    fft_convolve_r(&cijNL[(N+1)*M], dyv, cyydyv, scratchp1, scratchp2, scratchin,
	    scratchout, phys_plan, spec_plan, params);

    // vgrad*Cyy
    dx(&cijNL[(N+1)*M], scratch, params);
    fft_convolve_r(u, scratch, scratch, scratchp1, scratchp2, scratchin,
	    scratchout, phys_plan, spec_plan, params);

    dy(&cijNL[(N+1)*M], scratch2, params);
    fft_convolve_r(v, scratch2, scratch2, scratchp1, scratchp2, scratchin,
	    scratchout, phys_plan, spec_plan, params);

    for (i=0; i<N+1; i++)
    {
	for (j=0; j<M; j++)
	{
	    vgradcyy[ind(i,j)] = scratch[ind(i,j)] + scratch2[ind(i,j)];
	}
    }
    
    // Cxx*dxV
    fft_convolve_r(&cijNL[0], dxv, cxxdxv, scratchp1, scratchp2, scratchin,
	    scratchout, phys_plan, spec_plan, params);

    // CyydyU
    fft_convolve_r(&cijNL[(N+1)*M], dyu, cyydyu, scratchp1, scratchp2, scratchin,
	    scratchout, phys_plan, spec_plan, params);

    // Vgrad*Cxy
    dx(&cijNL[2*(N+1)*M], scratch, params);
    fft_convolve_r(u, scratch, scratch, scratchp1, scratchp2, scratchin,
	    scratchout, phys_plan, spec_plan, params);

    dy(&cijNL[2*(N+1)*M], scratch2, params);
    fft_convolve_r(v, scratch2, scratch2, scratchp1, scratchp2, scratchin,
	    scratchout, phys_plan, spec_plan, params);

    for (i=0; i<N+1; i++)
    {
	for (j=0; j<M; j++)
	{
	    vgradcxy[ind(i,j)] = scratch[ind(i,j)] + scratch2[ind(i,j)];
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
	    cij[ind(i,j)] = dt*2.*cxxdxu[ind(i,j)] + dt*2.*cxydyu[ind(i,j)];
	    cij[ind(i,j)] += - dt*vgradcxx[ind(i,j)];
	    cij[ind(i,j)] += - 0.5*dt*oneOverWi*cij[ind(i,j)];
	    cij[ind(i,j)] += cij[ind(i,j)];

	    cij[ind(i,j)] *= 1./(1.+0.5*oneOverWi);

	    cij[(N+1)*M + ind(i,j)] = dt*2.*cxydxv[ind(i,j)] + dt*2.*cyydyv[ind(i,j)];
	    cij[(N+1)*M + ind(i,j)] += - dt*vgradcyy[ind(i,j)];
	    cij[(N+1)*M + ind(i,j)] += 0.5*dt*oneOverWi*cij[(N+1)*M + ind(i,j)];
	    cij[(N+1)*M + ind(i,j)] += cij[(N+1)*M + ind(i,j)];

	    cij[2*(N+1)*M + ind(i,j)] = dt*cxxdxv[ind(i,j)] + dt*cyydyu[ind(i,j)];
	    cij[2*(N+1)*M + ind(i,j)] += - dt*vgradcxy[ind(i,j)];
	    cij[2*(N+1)*M + ind(i,j)] += 0.5*dt*oneOverWi*cij[2*(N+1)*M + ind(i,j)];
	    cij[2*(N+1)*M + ind(i,j)] += cij[2*(N+1)*M + ind(i,j)];
	}
    }
    cij[0] += oneOverWi;
    cij[(N+1)*M] += oneOverWi;
}

void step_sf_SI_Crank_Nicolson_visco(
	complex *psi, complex *psiNL, complex *cij, complex *cijNL, complex
	*forcing, complex *forcing2, double dt, int timeStep, flow_params
	params, complex *scratch, complex *scratch2, complex *u, complex *v,
	complex *lplpsi, complex *biharmpsi, complex *d2ypsi, complex *dyyypsi,
	complex *d4ypsi, complex *d2xd2ypsi, complex *d4xpsi, complex
	*udxlplpsi, complex *vdylplpsi, complex *vdyypsi, complex *d2ycxy,
	complex *d2xcxy, complex *dxycyy_cxx, complex *dycxy, complex
	*d2ycxyNL, complex *d2xcxyNL, complex *dxycyy_cxxNL, complex *dycxyNL,
	complex *RHSvec, complex *opsList, fftw_plan *phys_plan, fftw_plan
	*spec_plan, complex *scratchin, complex *scratchout, double *scratchp1,
	double *scratchp2 )
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
    dy(psiNL, u, params);

    // v
    dx(psiNL, v, params);
    for(i=0; i<N+1; i++)
    {
	for(j=0; j<M; j++)
	{
	    v[ind(i,j)] = -v[ind(i,j)];
	}
    }


    // lplpsi dyy(psi) + dxx(psi)

    d2x(psiNL, scratch, params);
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

    // d2x cxy
    d2x(&cijNL[2*(N+1)*M], d2xcxyNL, params);
    
    // d2y cxy
    dy(&cijNL[2*(N+1)*M], scratch, params);
    dy(scratch, d2ycxyNL, params);

    // dxy cyy-cxx
    for (i=0; i<N+1; i++)
    {
	for (j=0; j<M; j++)
	{
	    scratch[ind(i,j)] = cijNL[(N+1)*M + ind(i,j)] - cijNL[ind(i,j)];
	}
    }

    dx(scratch, scratch2, params);
    dy(scratch2, dxycyy_cxxNL, params);
    
    //dycxy
    dy(&cijNL[2*(N+1)*M], dycxyNL, params);
    

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

    // stresses

    // d2x cxy
    d2x(&cij[2*(N+1)*M], d2xcxy, params);
    
    // d2y cxy
    dy(&cij[2*(N+1)*M], scratch, params);
    dy(scratch, d2ycxy, params);

    for (i=0; i<N+1; i++)
    {
	for (j=0; j<M; j++)
	{
	    scratch[ind(i,j)] = cij[(N+1)*M + ind(i,j)] - cij[ind(i,j)];
	}
    }

    dx(scratch, scratch2, params);
    dy(scratch2, dxycyy_cxx, params);
    
    //dycxy
    dy(&cij[2*(N+1)*M], dycxy, params);
    

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

	    RHSvec[j] = 0.5*dt*oneOverRe*biharmpsi[ind(i,j)];
	    RHSvec[j] += - dt*udxlplpsi[ind(i,j)];
	    RHSvec[j] += - dt*vdylplpsi[ind(i,j)];
	    RHSvec[j] += + 0.5*dt*(1.-beta)*oneOverRe*oneOverWi*( 
			                 d2xcxy[ind(i,j)] 
				       + dxycyy_cxx[ind(i,j)] 
				       - d2ycxy[ind(i,j)]);
	    RHSvec[j] += + 0.5*dt*(1.-beta)*oneOverRe*oneOverWi*( 
			                 d2xcxyNL[ind(i,j)] 
				       + dxycyy_cxxNL[ind(i,j)] 
				       - d2ycxyNL[ind(i,j)]);
	    RHSvec[j] += 0.5*dt*(forcing[ind(i,j)] + forcing2[ind(i,j)]);
	    RHSvec[j] += + lplpsi[ind(i,j)];


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
    //  + _dt*(1.-beta)*oneOverRe*dot(MDY, Txy)[N*M:(N+1)*M]
    // RHSVec[N*M] += dt*2*oneOverRe

    for (j=0; j<M; j++)
    {
	//RHSvec[j] = u[ind(0,j)];
	RHSvec[j] = dt*0.5*oneOverRe*dyyypsi[ind(0,j)] - dt*vdyypsi[ind(0,j)];
	RHSvec[j] += dt*0.5*(1.-beta)*oneOverRe*dycxy[ind(0,j)]; 
	RHSvec[j] += dt*0.5*(1.-beta)*oneOverRe*dycxyNL[ind(0,j)]; 
	RHSvec[j] += 0.5*dt*(forcing[ind(0,j)] + forcing2[ind(0,j)]);
	RHSvec[j] += u[ind(0,j)]; 
    }
    // RHSvec[0] += 2*dt*oneOverRe;

    // apply BCs
    // # dyPsi0(+-1) = 0
    // RHSVec[N*M + M-3] = 0
    // RHSVec[N*M + M-2] = 0
    // # Psi0(-1) = 0
    // RHSVec[N*M + M-1] = 0

    RHSvec[M-3] = 0; 
    RHSvec[M-2] = 0; 
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
	)
{
    // Calculate f({psi, cxx, cyy, cxy}, dt)
    int N = params.N;
    int M = params.M;
    int i,j;
    
    // dxU, dyU, dxV, dyV
    dy(psi, u, params);
    dx(psi, v, params);

    for (i =0; i<M*(N+1); i++)
    {
	v[i] = -v[i];
    }

    dx(u, dxu, params);
    dy(u, dyu, params);
    dx(v, dxv, params);
    dy(v, dyv, params);
    
    // Cxx*dxU 
    fft_convolve_r(cxx, dxu, cxxdxu, scratchp1, scratchp2, scratchin,
	    scratchout, phys_plan, spec_plan, params);

    // Cxy*dyU 
    fft_convolve_r(cxy, dyu, cxydyu, scratchp1, scratchp2, scratchin,
	    scratchout, phys_plan, spec_plan, params);

    // VGrad*Cxx
    dx(cxx, scratch, params);
    fft_convolve_r(u, scratch, scratch, scratchp1, scratchp2, scratchin,
	    scratchout, phys_plan, spec_plan, params);

    dy(cxx, scratch2, params);
    fft_convolve_r(v, scratch2, scratch2, scratchp1, scratchp2, scratchin,
	    scratchout, phys_plan, spec_plan, params);

    for (i=0; i<N+1; i++)
    {
	for (j=0; j<M; j++)
	{
	    vgradcxx[ind(i,j)] = scratch[ind(i,j)] + scratch2[ind(i,j)];
	}
    }
    
    // Cxy*dxV
    fft_convolve_r(cxy, dxv, cxydxv, scratchp1, scratchp2, scratchin,
	    scratchout, phys_plan, spec_plan, params);

    // Cyy*dyV
    fft_convolve_r(cyy, dyv, cyydyv, scratchp1, scratchp2, scratchin,
	    scratchout, phys_plan, spec_plan, params);

    // vgrad*Cyy
    dx(cyy, scratch, params);
    fft_convolve_r(u, scratch, scratch, scratchp1, scratchp2, scratchin,
	    scratchout, phys_plan, spec_plan, params);

    dy(cyy, scratch2, params);
    fft_convolve_r(v, scratch2, scratch2, scratchp1, scratchp2, scratchin,
	    scratchout, phys_plan, spec_plan, params);

    for (i=0; i<N+1; i++)
    {
	for (j=0; j<M; j++)
	{
	    vgradcyy[ind(i,j)] = scratch[ind(i,j)] + scratch2[ind(i,j)];
	}
    }
    
    // Cxx*dxV
    fft_convolve_r(cxx, dxv, cxxdxv, scratchp1, scratchp2, scratchin,
	    scratchout, phys_plan, spec_plan, params);

    // CyydyU
    fft_convolve_r(cyy, dyu, cyydyu, scratchp1, scratchp2, scratchin,
	    scratchout, phys_plan, spec_plan, params);

    // Vgrad*Cxy
    dx(cxy, scratch, params);
    fft_convolve_r(u, scratch, scratch, scratchp1, scratchp2, scratchin,
	    scratchout, phys_plan, spec_plan, params);

    dy(cxy, scratch2, params);
    fft_convolve_r(v, scratch2, scratch2, scratchp1, scratchp2, scratchin,
	    scratchout, phys_plan, spec_plan, params);

    for (i=0; i<N+1; i++)
    {
	for (j=0; j<M; j++)
	{
	    vgradcxy[ind(i,j)] = scratch[ind(i,j)] + scratch2[ind(i,j)];
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
	    fxx[ind(i,j)] = 2*cxxdxu[ind(i,j)] + 2*cxydyu[ind(i,j)];
	    fxx[ind(i,j)] += - vgradcxx[ind(i,j)] - txx[ind(i,j)];

	    fyy[ind(i,j)] = 2*cxydxv[ind(i,j)] + 2*cyydyv[ind(i,j)];
	    fyy[ind(i,j)] += - vgradcyy[ind(i,j)] - tyy[ind(i,j)];

	    fxy[ind(i,j)] = cxxdxv[ind(i,j)] + cyydyu[ind(i,j)];
	    fxy[ind(i,j)] += - vgradcxy[ind(i,j)] - txy[ind(i,j)];
	}
    }
}
