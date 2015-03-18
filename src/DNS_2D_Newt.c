/* -------------------------------------------------------------------------- *
 *									      *
 *  DNS_2D_Newt.c							      *
 *                                                                            *
 *  Time stepping DNS program for 2D Newtonian fluid.			      *
 *                                                                            *
 *                                                                            *
 * -------------------------------------------------------------------------- */

// Last modified: Wed 18 Mar 19:24:20 2015

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

#include"fields_2D.h"
#include"time_steppers.h"

// Main

int main(int argc, char **argv) 
{
    flow_params params;
    int stepsPerFrame = 0;
    int numTimeSteps = 0;
    int timeStep = 0;
    double dt = 0;
    double KE0 = 1.0;
    double KE1 = 0.0;
    double KE2 = 0.0;
    double KE = 0.0;

    opterr = 0;
    int shortArg;

    //default parameters
    params.N = 5;
    params.M = 40;
    params.Ly = 2.;
    params.kx = 1.31;
    params.Re = 400;
    params.Wi = 1e-05;
    params.beta = 1.0;
    params.dealiasing = 0;

    // Read in parameters from cline args.

    while ((shortArg = getopt (argc, argv, "dN:M:L:k:R:W:b:t:s:T:")) != -1)
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
	    printf("Dealiasing on\n");
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
	params.Nf = (3*params.N)/2 + 1;
	params.Mf = 2*params.M; //(3*params.M)/2;
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

    FILE *tracefp, *traceU, *trace1mode;
    char *trace_fn, *traj_fn;
    int i, j;
    int N = params.N;
    int M = params.M;
    int Nf = params.Nf;
    int Mf = params.Mf;

    trace_fn = "./output/trace.dat";
    traj_fn = "./output/traj_psi.h5";

    tracefp = fopen(trace_fn, "w");
    traceU = fopen("./output/traceU.dat", "w");
    trace1mode = fopen("./output/traceMode.dat", "w");

    // Variables for HDF5 output
    hid_t hdf5fp, datatype_id, filetype_id;
    herr_t status;
    
    // create the datatype for scipy complex numbers
    datatype_id = H5Tcreate(H5T_COMPOUND, sizeof (complex_hdf));
    status = H5Tinsert(datatype_id, "r",
                HOFFSET(complex_hdf, r), H5T_NATIVE_DOUBLE);
    status = H5Tinsert(datatype_id, "i",
                HOFFSET(complex_hdf, i), H5T_NATIVE_DOUBLE);

    // create the filetype for the scipy complex numbers
    filetype_id = H5Tcreate(H5T_COMPOUND, 8 + 8);
    status = H5Tinsert(filetype_id, "r", 0, H5T_NATIVE_DOUBLE);
    status = H5Tinsert(filetype_id, "i", 8, H5T_NATIVE_DOUBLE);

    // create Hdf5 output file
    hdf5fp = H5Fcreate(traj_fn, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    // field arrays are declared as pointers and then I malloc.
    complex *scratch, *scratch2, *scratch3, *scratch4, *tmpop;
    complex *psi, *psi2, *u, *v, *udxlplpsi, *vdylplpsi, *biharmpsi, *lplpsi;
    complex *dyyypsi, *dypsi, *vdyypsi;
    complex *d2ypsi, *d4ypsi, *d4xpsi, *d2xd2ypsi;

    fftw_complex *scratchin, *scratchout;
    double *scratchp1, *scratchp2;

    fftw_complex *RHSvec;
    double time = 0;
    double oneOverRe = 1./params.Re;
    
    fftw_complex *opsList, *hopsList;

    fftw_plan phys_plan, spec_plan;

    unsigned fftwFlag;
    #ifdef MYDEBUG 
    fftwFlag = FFTW_ESTIMATE;
    #else
    fftwFlag = FFTW_MEASURE;
    #endif

    // dynamically malloc array of complex numbers.
    tmpop = (complex*) fftw_malloc(M*M * sizeof(complex));
    opsList = (complex*) fftw_malloc((N+1)*M*M * sizeof(complex));
    hopsList = (complex*) fftw_malloc((N+1)*M*M * sizeof(complex));

    scratch = (complex*) fftw_malloc(M*(N+1) * sizeof(complex));
    scratch2 = (complex*) fftw_malloc(M*(N+1) * sizeof(complex));
    scratch3 = (complex*) fftw_malloc(M*(N+1) * sizeof(complex));
    scratch4 = (complex*) fftw_malloc(M*(N+1) * sizeof(complex));

    psi = (complex*) fftw_malloc(M*(N+1) * sizeof(complex));
    psi2 = (complex*) fftw_malloc(M*(N+1) * sizeof(complex));
    u = (complex*) fftw_malloc(M*(N+1) * sizeof(complex));
    v = (complex*) fftw_malloc(M*(N+1) * sizeof(complex));
    udxlplpsi = (complex*) fftw_malloc(M*(N+1) * sizeof(complex));
    vdylplpsi = (complex*) fftw_malloc(M*(N+1) * sizeof(complex));
    lplpsi = (complex*) fftw_malloc(M*(N+1) * sizeof(complex));
    biharmpsi = (complex*) fftw_malloc(M*(N+1) * sizeof(complex));
    d2ypsi = (complex*) fftw_malloc(M*(N+1) * sizeof(complex));
    dyyypsi = (complex*) fftw_malloc(M*(N+1) * sizeof(complex));
    d4ypsi = (complex*) fftw_malloc(M*(N+1) * sizeof(complex));
    d4xpsi = (complex*) fftw_malloc(M*(N+1) * sizeof(complex));
    d2xd2ypsi = (complex*) fftw_malloc(M*(N+1) * sizeof(complex));
    dypsi = (complex*) fftw_malloc(M*(N+1) * sizeof(complex));
    vdyypsi = (complex*) fftw_malloc(M*(N+1) * sizeof(complex));

    scratchin = (fftw_complex*) fftw_malloc((2*Mf-2)*(2*Nf+1) * sizeof(fftw_complex));
    scratchout = (fftw_complex*) fftw_malloc((2*Mf-2)*(2*Nf+1) * sizeof(fftw_complex));

    scratchp1 = (double*) fftw_malloc((2*Mf-2)*(2*Nf+1) * sizeof(double));
    scratchp2 = (double*) fftw_malloc((2*Mf-2)*(2*Nf+1) * sizeof(double));

    RHSvec = (complex*) fftw_malloc(M * sizeof(complex));

    // Set up some dft plans
    printf("\n------\nSetting up fftw3 plans\n------\n");
    phys_plan = fftw_plan_dft_2d(2*Nf+1, 2*Mf-2,  scratchin, scratchout,
			 FFTW_BACKWARD, fftwFlag);
    spec_plan = fftw_plan_dft_2d(2*Nf+1, 2*Mf-2,  scratchin, scratchout,
			 FFTW_FORWARD, fftwFlag);

    printf("\n------\nLoading initial streamfunction and operators\n------\n");

    // load the initial field from scipy
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
	}

	sprintf(fn, "./operators/hop%d.h5", i);
	printf("opening: %s\n", fn);
	load_hdf5_operator(fn, tmpop, params);

	for (j=0; j<M*M; j++)
	{
	    hopsList[i*M*M + j] = tmpop[j];
	}

    }

    #ifdef MYDEBUG
    for (i=0; i<N+1; i++) 
    {
	char fn[30];
	sprintf(fn, "./output/op%d.h5", i);
	printf("writing: %s\n", fn);
	for (j=0; j<M*M; j++)
	{
	    tmpop[j] = opsList[i*M*M + j];
	}

	save_hdf5_arr(fn, &tmpop[0], M*M);

	sprintf(fn, "./output/hop%d.h5", i);
	printf("writing: %s\n", fn);
	for (j=0; j<M*M; j++)
	{
	    tmpop[j] = hopsList[i*M*M + j];
	}

	save_hdf5_arr(fn, &tmpop[0], M*M);
    }
    save_hdf5_state("./output/psi.h5", &psi[0], params);
    #endif
    
    // perform the time iteration
    printf("\n------\nperforming the time iteration\n------\n");
    printf("\nTime:\t\tKE0:\n");

    for (timeStep=0; timeStep<numTimeSteps; timeStep++)
    {

	// predictor step to calculate nonlinear terms
	
	for (i=0; i<N+1; i++)
	{
	    for(j=0; j<M; j++)
	    {
		psi2[ind(i,j)] = psi[ind(i,j)];
	    }
	}

	step_sf_SI_Crank_Nicolson(
	    psi2, psi, dt/2., timeStep, oneOverRe, params, scratch, scratch2, u, v, lplpsi,
	    biharmpsi, d2ypsi, dyyypsi, d4ypsi, d2xd2ypsi, d4xpsi, udxlplpsi,
	    vdylplpsi, vdyypsi, RHSvec, hopsList, &phys_plan, &spec_plan,
	    scratchin, scratchout, scratchp1, scratchp2);

	// 'corrector' step to calculate full step based on nonlinear terms from predictor step
	step_sf_SI_Crank_Nicolson(
	    psi, psi2, dt, timeStep, oneOverRe, params, scratch, scratch2, u, v, lplpsi,
	    biharmpsi, d2ypsi, dyyypsi, d4ypsi, d2xd2ypsi, d4xpsi, udxlplpsi,
	    vdylplpsi, vdyypsi, RHSvec, opsList, &phys_plan, &spec_plan,
	    scratchin, scratchout, scratchp1, scratchp2);

	if (timeStep==0)
	{
	    save_hdf5_state("./output/psi2.h5", &psi[0], params);
	}

	// output some information at every frame
	if ((timeStep % stepsPerFrame) == 0 )
	{
	    time = timeStep*dt;

	    double normU1 = 0;
	    double normU2 = 0;
	    double normU0 = 0;

	    for (i=0; i<N+1; i++)
	    {
		for (j=0; j<M; j++)
		{
		    scratch[ind(i,j)] = u[ind(i,j)];
		}
	    }
	    scratch[ind(0,0)] -= 0.5;
	    scratch[ind(0,2)] += 0.5;
	    fprintf(trace1mode, "%e\t%e\t%e\t%e\t%e\t%e\t%e\n", 
		    time, creal(scratch[ind(0,3)]), cimag(scratch[ind(0,3)]),
		     creal(scratch[ind(1,3)]), cimag(scratch[ind(1,3)]),
		      creal(scratch[ind(2,3)]), cimag(scratch[ind(2,3)]));

	    for (j=M-1; j>=0; j=j-1)
	    {
		normU0 += creal(scratch[ind(0,j)]*scratch[ind(0,j)]); 
		normU1 += creal(u[ind(1,j)]*conj(u[ind(1,j)])); 
		normU2 += creal(u[ind(2,j)]*conj(u[ind(2,j)])); 
	    }
	    normU0 = sqrt(normU0);//-(1./sqrt(2.));
	    normU1 = sqrt(normU1);
	    normU2 = sqrt(normU2);

	    fprintf(traceU, "%e\t%e\t%e\t%e\t\n", time, normU0, normU1, normU2);

	    fft_convolve_r(u, u, u, scratchp1, scratchp2, scratchin, scratchout,
		    &phys_plan, &spec_plan, params);
	    fft_convolve_r(v, v, v, scratchp1, scratchp2, scratchin, scratchout,
		    &phys_plan, &spec_plan, params);
	    

	    KE0 = calc_KE0(u, v, params) * (15.0/ 8.0);
	    KE1 = calc_KE1(u, v, params) * (15.0/ 8.0);
	    KE2 = calc_KE2(u, v, params) * (15.0/ 8.0);
	    KE = calc_KE(u, v, params) * (15.0/ 8.0);

	    printf("%e\t%e\t%e\t%e\n", time, KE0, KE1, KE2);

	    
	    save_hdf5_snapshot(&hdf5fp, &filetype_id, &datatype_id,
		    psi, time, params);
	    
	    fprintf(tracefp, "%e\t%e\t%e\t%e\t%e\n", time, KE0, KE1, KE2, KE);

	    fflush(traceU);
	    fflush(trace1mode);
	    fflush(tracefp);
	    H5Fflush(hdf5fp, H5F_SCOPE_GLOBAL);

	}

    }

    // save the final state
    save_hdf5_state("./output/final.h5", &psi[0], params);

    fclose(tracefp);
    fclose(traceU);
    fclose(trace1mode);

    // clean up hdf5
    status = H5Tclose(datatype_id);
    status = H5Tclose(filetype_id);
    status = H5Fclose(hdf5fp);

    // garbage collection
    fftw_destroy_plan(phys_plan);
    fftw_destroy_plan(spec_plan);

    fftw_free(tmpop);
    fftw_free(opsList);
    fftw_free(hopsList);
    fftw_free(scratch);
    fftw_free(scratch2);
    fftw_free(scratch3);
    fftw_free(scratch4);
    fftw_free(psi);
    fftw_free(psi2);
    fftw_free(u);
    fftw_free(v);
    fftw_free(udxlplpsi);
    fftw_free(vdylplpsi);
    fftw_free(lplpsi);
    fftw_free(biharmpsi);
    fftw_free(scratchin);
    fftw_free(scratchout);
    fftw_free(scratchp1);
    fftw_free(scratchp2);
    fftw_free(d2ypsi);
    fftw_free(dyyypsi);
    fftw_free(d4ypsi);
    fftw_free(d4xpsi);
    fftw_free(d2xd2ypsi);
    fftw_free(dypsi);
    fftw_free(vdyypsi);
    fftw_free(RHSvec);

    printf("quitting c program\n");

    return 0;
}
