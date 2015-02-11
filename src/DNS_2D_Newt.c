/* -------------------------------------------------------------------------- *
 *									      *
 *  DNS_2D_Newt.c							      *
 *                                                                            *
 *  Time stepping DNS program for 2D Newtonian fluid.			      *
 *                                                                            *
 *                                                                            *
 * -------------------------------------------------------------------------- */

// Last modified: Wed 11 Feb 19:03:03 2015

/* Program Description:
 *
 * This program is written to work with a python setup program. The setup
 * program will write a series of files containing matrix operators of all
 * Fourier modes in the problem. These will then be imported and this program
 * will perform the timestepping using FFT's for the products and my own
 * functions to perform derivatives. Every so often I want to be able to save
 * the state of the fluid, this will mean outputting the current field into a
 * data file. Also I would like to output several properties of the fluid at
 * this point, such as energy etc.
 * 
 * Functions required:
 *
 * dy - returns y derivative of field
 *
 * dx - returns x derivative of a field
 *
 * to_physical - transforms from fully spectral to real space on the GL +
 *		 uniform grid
 *
 * to_spectral - transforms from physical space to fully spectral Chebyshev +
 *		 Fourier space.
 *
 * load_operator - loads a spectral operator from a text file generated in
 *		    python.
 *
 * save_state - saves the flow at the current time to a text file
 *
 * load_state - load the flow from a previous time from a text file.
 *
 * Plenty of other functions would be useful, but these are the essential ones.
 *
 * Unit Testing:
 *
 * Testing will be performed by comparing fields with those generated by python.
 *
 * TODO:
 *	- testing
 *	- profiling
 *	- faster transforms
 *	- faster products
 *	- faster derivatives
 *	- less variables
 *	- take advantage of hermitian properties
 *
 */

// Headers

#include<unistd.h>

#include"fields_2D.h"

// Main

int main(int argc, char **argv) 
{
    flow_params params;
    int stepsPerFrame = 0;
    int numTimeSteps = 0;
    int timeStep = 0;
    double dt = 0;
    double KE0 = 1.0;

    opterr = 0;
    int shortArg;

    //params.N = 5;
    //params.M = 40;
    //params.Ly = 2.;
    //params.kx = 1.31;
    //params.Re = 400;
    //params.Wi = 1e-05;
    //params.beta = 1.0;

    // Read in parameters from cline args.

    while ((shortArg = getopt (argc, argv, "N:M:L:k:R:W:b:t:s:T:d")) != -1)
	switch (shortArg)
	  {
	  case 'N':
	    params.N = atoi(optarg);
	    break;
	  case 'M':
	    params.M = atoi(optarg);
	    break;
	  case 'L':
	    params.Ly = atof(optarg);
	    break;
	  case 'k':
	    params.kx = atof(optarg);
	    break;
	  case 'R':
	    params.Re = atof(optarg);
	    break;
	  case 'W':
	    params.Wi = atof(optarg);
	    break;
	  case 'b':
	    params.beta = atof(optarg);
	    break;
	  case 't':
	    dt = atof(optarg);
	    break;
	  case 's':
	    stepsPerFrame = atoi(optarg);
	    break;
	  case 'T':
	    numTimeSteps = atoi(optarg);
	    break;
	  case 'd':
	    params.dealiasing = 1;
	    break;
	  case '?':
	    fprintf (stderr, "Option -%c requires an argument.\n", optopt);
	    if (isprint (optopt))
	      fprintf (stderr, "Unknown option `-%c'.\n", optopt);
	    else
	      fprintf (stderr,
		       "Unknown option character `\\x%x'.\n",
		       optopt);
	      return 1;
	  default:
	    abort ();
	  }

    if (params.dealiasing == 1)
    {
	params.Nf = 3*params.N/2;
	params.Mf = 3*params.M/2;
    } else
    {
	params.Nf = params.N;
	params.Mf = params.M;
    }


    printf("PARAMETERS: ");
    printf("\nN                   \t %d ", params.N);
    printf("\nM                   \t %d ", params.M);
    printf("\nLy                  \t %f ", params.Ly);
    printf("\nkx                  \t %e ", params.kx);
    printf("\nRe                  \t %e ", params.Re);
    printf("\nWi                  \t %e ", params.Wi);
    printf("\nbeta                \t %e ", params.beta);
    printf("\nTime Step           \t %e ", dt);
    printf("\nNumber of Time Steps\t %d ", numTimeSteps);
    printf("\nTime Steps per frame\t %d \n", stepsPerFrame);

    // Declare variables
    FILE *fpi = NULL;

    int i, j, l;
    int N = params.N;
    int M = params.M;
    int Nf = params.N;
    int Mf = params.M;

    // field arrays are declared as pointers and then I malloc.
    fftw_complex *scratch, *scratch2, *scratch3, *scratch4, *tmpop;
    fftw_complex *scratchin, *scratchout;
    fftw_complex *psi, *u, *v, *udxlplpsi, *vdylplpsi, *biharmpsi, *lplpsi;
    fftw_complex *dyyypsi, *dypsi, *vdyypsi;
    fftw_complex *scratchp1, *scratchp2;

    fftw_complex *RHSvec;
    double norm;
    double time;
    double oneOverRe = 1./params.Re;
    
    // Ops list is an array of arrays. Inner arrays are flattened 2D operators.
    // outer array is just so that we keep al the operators together
    fftw_complex *opsList;
    // fftw_complex **opsList;
    // opsList = malloc(N+1 * sizeof * (opsList));
    // for (i=0; i<N+1; i++) 
    // {
    //     (opsList)[i] = malloc(M*M * sizeof * (opsList)[i]);
    // }

    fftw_plan phys_plan, spec_plan;

    unsigned fftwFlag = FFTW_ESTIMATE;

    int shape[2] = { M*(2*N+1), 0 };
    int shapefft[2] = { (2*Mf-2)*(2*Nf+1), 0 };
    int shape2[2] = { M, 0 };

    int shapeOp[2] = {M*M,0};

    // dynamically malloc array of complex numbers.
    tmpop = (fftw_complex*) fftw_malloc(M*M * sizeof(fftw_complex));
    opsList = (fftw_complex*) fftw_malloc((N+1)*M*M * sizeof(fftw_complex));

    scratch = (fftw_complex*) fftw_malloc((M)*(2*N+1) * sizeof(fftw_complex));
    scratch2 = (fftw_complex*) fftw_malloc((M)*(2*N+1) * sizeof(fftw_complex));
    scratch3 = (fftw_complex*) fftw_malloc((M)*(2*N+1) * sizeof(fftw_complex));
    scratch4 = (fftw_complex*) fftw_malloc((M)*(2*N+1) * sizeof(fftw_complex));

    psi = (fftw_complex*) fftw_malloc((M)*(2*N+1) * sizeof(fftw_complex));
    u = (fftw_complex*) fftw_malloc((M)*(2*N+1) * sizeof(fftw_complex));
    v = (fftw_complex*) fftw_malloc((M)*(2*N+1) * sizeof(fftw_complex));
    udxlplpsi = (fftw_complex*) fftw_malloc((M)*(2*N+1) * sizeof(fftw_complex));
    vdylplpsi = (fftw_complex*) fftw_malloc((M)*(2*N+1) * sizeof(fftw_complex));
    lplpsi = (fftw_complex*) fftw_malloc((M)*(2*N+1) * sizeof(fftw_complex));
    biharmpsi = (fftw_complex*) fftw_malloc((M)*(2*N+1) * sizeof(fftw_complex));
    dyyypsi = (fftw_complex*) fftw_malloc((M)*(2*N+1) * sizeof(fftw_complex));
    dypsi = (fftw_complex*) fftw_malloc((M)*(2*N+1) * sizeof(fftw_complex));
    vdyypsi = (fftw_complex*) fftw_malloc((M)*(2*N+1) * sizeof(fftw_complex));

    scratchin = (fftw_complex*) fftw_malloc((2*Mf-2)*(2*Nf+1) * sizeof(fftw_complex));
    scratchout = (fftw_complex*) fftw_malloc((2*Mf-2)*(2*Nf+1) * sizeof(fftw_complex));
    scratchp1 = (fftw_complex*) fftw_malloc((2*Mf-2)*(2*Nf+1) * sizeof(fftw_complex));
    scratchp2 = (fftw_complex*) fftw_malloc((2*Mf-2)*(2*Nf+1) * sizeof(fftw_complex));

    RHSvec = (fftw_complex*) fftw_malloc(M * sizeof(fftw_complex));

    // Set up some dft plans
    printf("\n------\nSetting up fftw3 plans\n------\n");
    phys_plan = fftw_plan_dft_2d(2*Nf+1, 2*Mf-2,  scratchin, scratchout,
			 FFTW_BACKWARD, fftwFlag);
    spec_plan = fftw_plan_dft_2d(2*Nf+1, 2*Mf-2,  scratchin, scratchout,
			 FFTW_FORWARD, fftwFlag);

    printf("\n------\nLoading initial streamfunction and operators\n------\n");

    // load the initial field from scipy

    // fpi = fopen("initial.dat", "r");
    // load_state(fpi, psi, params);
    // fclose(fpi);

    load_hdf5_state("initial.h5", psi, params);

    // load the operators from scipy 
    for (i=0; i<N+1; i++) 
    {
	char fn[30];
	sprintf(fn, "./operators/op%d.h5", i);
	printf("opening: %s\n", fn);
	load_hdf5_operator(fn, tmpop, params);

	for (j=0; j<M*M; j++)
	{
	    opsList[i*M*M + j] = tmpop[j];
	    //printf(" (%f + %fj) ", creal(opsList[i][j]), cimag(opsList[i][j]) );
	}

	// fpi = fopen(fn, "r");
	// load_operator(fpi, opsList[i], params);
	// fclose(fpi);
    }

    for (i=0; i<N+1; i++) 
    {
	char fn[30];
	sprintf(fn, "./output/op%d.h5", i);
	printf("writing: %s\n", fn);
	for (j=0; j<M*M; j++)
	{
	    tmpop[j] = opsList[i*M*M + j];
	}

	save_hdf5_arr(fn, &tmpop[0], shapeOp[0]);
    }
    
    // perform the time iteration
    printf("\n------\nperforming the time iteration\n------\n");
    printf("\nTime:\t\tKE0:\n");

    time = timeStep*dt;
    dy(psi, u, params);
    KE0 = (15./ 8.) * calc_KE0(u, params);

    printf("%e\t\t%e\n", time, KE0);


    for (timeStep=0; timeStep<numTimeSteps; timeStep++)
    {
	// calculate RHS 
	// First of all calculate some useful variables then product terms,
	// then calculate RHS for each mode, then solve for the new
	// streamfunction at each time.
	
	// u
	dy(psi, u, params);

	if(timeStep==0)
	{
	    save_hdf5_state("./output/psi.h5", &psi[0], params);
	    save_hdf5_state("./output/u.h5",  &u[0], params);
	}

	//if((timeStep==1) || (timeStep==0) )
	//{
	//    printf("\n");
	//    for (j=0;j<M;j++)
	//    {
	//	printf(" %f ", creal(u[ind(0,j)]));
	//    }
	//    printf("\n");
	//}
	

      // to_physical(u, scratchp1, scratchin, scratchout, &phys_plan, params);

      // v
      dx(psi, v, params);
      for(i=0; i<2*N+1; i++)
      {
          for(j=0; j<M; j++)
          {
      	v[ind(i,j)] = -v[ind(i,j)];
          }
      }

      if(timeStep==0)
      {
          save_hdf5_state("./output/v.h5", &v[0], params);
      }

      to_physical(psi, scratchp2, scratchin, scratchout, &phys_plan, params);
      //exit(1);
      to_spectral(scratchp2, psi, scratchin, scratchout, &spec_plan, params);

	// lplpsi dyy(psi) + dxx(psi)

	dx(v, scratch, params);
	dy(u, lplpsi, params);
	for(i=0; i<2*N+1; i++)
	{
	    for(j=0; j<M; j++)
	    {
		lplpsi[ind(i,j)] = lplpsi[ind(i,j)] - scratch[ind(i,j)];
	    }
	}

	if(timeStep==0)
	{
	    save_hdf5_state("./output/lplpsi.h5", &lplpsi[0], params);
	}



	// biharmpsi (dyy + dxx)lplpsi
	
	dy(u, scratch, params);
	dy(scratch, scratch2, params);
	dy(scratch2, scratch, params);
	if(timeStep==0)
	{
	    save_hdf5_state("./output/d4ypsi.h5", &scratch[0], params);
	}

	dx(psi, scratch, params);
	dx(scratch, scratch2, params);
	dx(scratch2, scratch, params);
	dx(scratch, scratch2, params);
	if(timeStep==0)
	{
	    save_hdf5_state("./output/d4xpsi.h5", &scratch2[0], params);
	}

	dx(psi, scratch, params);
	dx(scratch, scratch2, params);
	dy(scratch2, scratch, params);
	dy(scratch, scratch2, params);
	if(timeStep==0)
	{
	    save_hdf5_state("./output/d2xd2ypsi.h5", &scratch2[0], params);
	}

	dx(lplpsi, scratch, params);
	dx(scratch, scratch2, params);

	dy(lplpsi, scratch, params);
	dy(scratch, biharmpsi, params);

	for(i=0; i<2*N+1; i++)
	{
	    for(j=0; j<M; j++)
	    {
		biharmpsi[ind(i,j)] = biharmpsi[ind(i,j)] + scratch2[ind(i,j)];
	    }
	}
	
	if(timeStep==0)
	{
	    save_hdf5_state("./output/biharmpsi.h5", &biharmpsi[0], params);
	}

	// udxlplpsi 
	dx(lplpsi, udxlplpsi, params);
	if(timeStep==0)
	{
	    save_hdf5_state("./output/dxlplpsi.h5", &udxlplpsi[0], params);
	}

	fft_convolve(udxlplpsi, u, udxlplpsi, scratchp1, scratchp2, scratchin,
		scratchout, &phys_plan, &spec_plan, params);

	if(timeStep==0)
	{
	    save_hdf5_state("./output/udxlplpsi.h5", &udxlplpsi[0], params);
	}

	// vdylplpsi 
	dy(lplpsi, vdylplpsi, params);
	if(timeStep==0)
	{
	    save_hdf5_state("./output/dylplpsi.h5", &vdylplpsi[0], params);
	}

	fft_convolve(vdylplpsi, v, vdylplpsi, scratchp1, scratchp2, scratchin,
		scratchout, &phys_plan, &spec_plan, params);

	if(timeStep==0)
	{
	    save_hdf5_state("./output/vdylplpsi.h5", &vdylplpsi[0], params);

	}
	
	//vdyypsi = vdyu
	dy(u, vdyypsi, params);

	fft_convolve(vdyypsi, v, vdyypsi, scratchp1, scratchp2, scratchin,
		scratchout, &phys_plan, &spec_plan, params);

	if(timeStep==0)
	{
	    save_hdf5_state("./output/vdyypsi.h5", &vdyypsi[0], params);
	}

	// RHSVec = dt*0.5*oneOverRe*dot(BIHARM, PSI) 
	// 	+ dot(LAPLAC, PSI) 
	// 	- dt*dot(MMU, dot(MDXLAPLAC, PSI)) 
	// 	- dt*dot(MMV, dot(MDYLAPLAC, PSI)) 

	for (i=1; i<N+1; i++)
	{
	    for (j=0; j<M; j++)
	    {

		RHSvec[j] = 0.5*dt*oneOverRe*biharmpsi[ind(i,j)] 
			    + lplpsi[ind(i,j)]
			    - dt*udxlplpsi[ind(i,j)]
			    - dt*vdylplpsi[ind(i,j)];
			    
		//printf("%p\n", &RHSvec[0]);
		//printf("%f\n", creal(RHSvec[0]));
		//printf("%p\n", &opsList[4*M*M]);
		//printf("%f\n", creal(opsList[4*M*M]));

	    }

	    //impose BCs
	    
	    RHSvec[M-2] = 0;
	    RHSvec[M-1] = 0;

	    if(timeStep==0)
	    {
		char fn[30];
		sprintf(fn, "./output/RHSVec%d.h5", i);
		printf("writing %s\n", fn);
		save_hdf5_arr(fn, &RHSvec[0], shape2[0]);
	    }



	    // perform dot product to calculate new streamfunction.
	    for (j=0; j<M; j++)
	    {
		psi[ind(i,j)] = 0;
		psi[ind(2*N+1-i,j)] = 0;
	        for (l=0; l<M; l++)
	        {
	            psi[ind(i,j)] += opsList[(i*M + j)*M + l] * conj(RHSvec[l]);
		    psi[ind(2*N+1-i,j)] += conj(psi[ind(i,j)]);
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
	    RHSvec[j] = 0.5*dt*oneOverRe*dyyypsi[ind(0,j)]
			+ u[ind(0,j)]
			- dt*vdyypsi[ind(0,j)];
	}
	RHSvec[0] += 2*dt*oneOverRe;

	// apply BCs
	// # dyPsi0(+-1) = 0
	// RHSVec[N*M + M-3] = 0
	// RHSVec[N*M + M-2] = 0
	// # Psi0(-1) = 0
	// RHSVec[N*M + M-1] = 0
	
	RHSvec[M-3] = 0; 
	RHSvec[M-2] = 0; 
	RHSvec[M-1] = 0; 

	//printf("0th element RHSVEC: %f %f\n", creal(RHSvec[0]), cimag(RHSvec[0]));
	//printf("2nd element RHSVEC: %f %f\n", creal(RHSvec[2]), cimag(RHSvec[2]));

	// step the zeroth mode
	
	for (j=0; j<M; j++)
	{
	    psi[ind(0,j)] = 0;
	    for (l=0; l<M; l++)
	    {
		psi[ind(0,j)] += opsList[j*M + l] * RHSvec[l];

		// if ((j==0) || (j==1) || (j==2) || (j==3))
		// {
		// if ((l==0)||(l==2))
		// {
		// printf("%d,%d: %f %f %f\n", j,l, creal(psi[ind(0,j)]),
		// 	creal(opsList[j*M+l]), creal(RHSvec[l]));
		// }
		// }

	    }
	}

	//printf("psi[0] %f + %fj\n", creal(psi[0]), cimag(psi[0]));
	//printf("psi[1] %f + %fj\n", creal(psi[1]), cimag(psi[1]));
	//printf("psi[2] %f + %fj\n", creal(psi[2]), cimag(psi[2]));
	//printf("psi[3] %f + %fj\n", creal(psi[3]), cimag(psi[3]));

	if(timeStep==0)
	{
	    save_hdf5_state("./output/psi2.h5", &psi[0], params);
	}

	if(timeStep==0)
	{
	    char fn[30];
	    sprintf(fn, "./output/RHSVec%d.h5", 0);
	    save_hdf5_arr(fn, &RHSvec[0], shape2[0]);
	}
	
	// output some information at every frame
	if ((timeStep % stepsPerFrame) == 0 )
	{
	  time = timeStep*dt;
	  KE0 = calc_KE0(u, params) * (15.0/ 8.0);

	  printf("%e\t\t%e\n", time, KE0);

	    char fn[30];
	    sprintf(fn, "./output/t%e.h5", time);
	    save_hdf5_state(fn, &psi[0], params);
	}

    }

    // save the final state
    save_hdf5_state("./output/final.h5", &psi[0], params);

    // garbage collection
    fftw_destroy_plan(phys_plan);
    fftw_destroy_plan(spec_plan);

    fftw_free(scratch);
    fftw_free(scratch2);
    fftw_free(scratch3);
    fftw_free(scratch4);
    fftw_free(psi);
    fftw_free(u);
    fftw_free(v);
    fftw_free(udxlplpsi);
    fftw_free(vdylplpsi);
    fftw_free(lplpsi);
    fftw_free(biharmpsi);
    fftw_free(scratchp1);
    fftw_free(scratchp2);
    fftw_free(dyyypsi);
    fftw_free(dypsi);
    fftw_free(vdyypsi);
    fftw_free(RHSvec);
    printf("quitting c program\n");

    return 0;
}
