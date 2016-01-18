/* ------------------------------------------------------------------------ *
 *									      *
 *  DNS_2D_linear_Newt.c						      *
 *                                                                            *
 *  Time stepping linear in x, DNS program for 2D Newtonian fluid.	      *
 *                                                                            *
 *                                                                            *
 * -------------------------------------------------------------------------- */

// Last modified: Mon 18 Jan 16:03:57 2016

/* Program Description:
 *
 * This program is written to work with a python setup program. The setup
 * program will write a series of files containing matrix operators of all the
 * zeroth and first fourier modes. These will then be imported and this program
 * will perform the timestepping using FFT's for the chebyshev products and my
 * own functions to perform derivatives. Every so often I want to be able to
 * save the state of the fluid, this will mean outputting the current field
 * into a data file. Also I would like to output several properties of the
 * fluid at this point, such as energy etc.
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
 */

// Headers

#include"fields_IO.h"
#include"fields_1D.h"
#include"time_steppers_linear.h"

// Main

int main(int argc, char **argv) 
{

    flow_params params;
    int stepsPerFrame = 0;
    int numTimeSteps = 0;
    double dt = 0;
    double initTime=0;

    opterr = 0;
    int shortArg;

    //default parameters
    params.N = 5;
    params.M = 40;
    params.U0 = 0;
    params.kx = 1.31;
    params.Re = 400;
    params.Wi = 1e-05;
    params.beta = 1.0;
    params.De = 1.0;
    params.P = 1.0;
    params.dealiasing = 0;
    params.oscillatory_flow = 0;

    // Read in parameters from cline args.


    while ((shortArg = getopt (argc, argv, "OdN:M:U:k:R:W:b:D:P:t:s:T:i:")) != -1)
	switch (shortArg)
	{
	    case 'N':
		params.N = atoi(optarg);
		break;
	    case 'M':
		params.M = atoi(optarg);
		break;
	    case 'U':
		params.U0 = atof(optarg);
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
	    case 'D':
		params.De = atof(optarg);
		break;
	    case 'P':
		params.P = atof(optarg);
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
	    case 'i':
		initTime = atof(optarg);
		break;
	    case 'd':
		params.dealiasing = 1;
		printf("Dealiasing on\n");
		break;
	    case 'O':
		params.oscillatory_flow = 1;
		printf("oscillatory flow\n");
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
	params.Mf = 2*params.M; //(3*params.M)/2;
    } else
    {
	params.Mf = params.M;
    }

//    DNS_2D_linear_Visco(params.oscillatory_flow, params.dealiasing, params.N,
//	    params.Nf, params.M, params.Mf, params.U0, params.kx, params.Re, params.Wi, params.beta,
//	    params.De, params.P, dt, stepsPerFrame, numTimeSteps, initTime);
//
//}

//void DNS_2D_linear_Visco(int oscilFlag, int dealiasFlag, int N, int Nf, int M, int Mf, double U0, double kx,
//	double Re, double Wi, double beta, double De, double P, double dt, int
//	stepsPerFrame, int numTimeSteps, double initTime)
//{

    //initTime = 0;
    //flow_params params;
    //params.N = N;
    //params.M = M;
    //params.Nf = Nf;
    //params.Mf = Mf;
    //params.U0 = U0;
    //params.kx = kx;
    //params.Re = Re; 
    //params.Wi = Wi;
    //params.beta =beta;
    //params.De = De;
    //params.P = P; 
    //params.dealiasing = dealiasFlag;
    //params.oscillatory_flow = oscilFlag;


    printf("PARAMETERS: ");
    printf("\nN                   \t %d ", params.N);
    printf("\nM                   \t %d ", params.M);
    printf("\nU0                  \t %f ", params.U0);
    printf("\nkx                  \t %e ", params.kx);
    printf("\nRe                  \t %e ", params.Re);
    printf("\nWi                  \t %e ", params.Wi);
    printf("\nbeta                \t %e ", params.beta);
    printf("\nDe                  \t %e ", params.De);
    printf("\nTime Step           \t %e ", dt);
    printf("\nNumber of Time Steps\t %d ", numTimeSteps);
    printf("\nTime Steps per frame\t %d ", stepsPerFrame);
    printf("\nInitial Time\t %f \n", initTime);

    FILE *tracefp, *tracePSI, *trace1mode;
    char *trace_fn, *traj_fn;
    char trPSI_fn[100]=" ";
    int i, j;
    int N = params.N;
    int M = params.M;
    int Nf = params.Nf;
    int Mf = params.Mf;
    double KE0 = 1.0;
    double KE1 = 0.0;
    double KE_tot = 0.0;
    double periods, phase;
    double time = initTime;
    int timeStep;

    int save_traj = 0;

    trace_fn = "./output/trace.dat";
    traj_fn = "./output/traj.h5";
    tracefp = fopen(trace_fn, "w");

    sprintf(trPSI_fn, "./output/tracePSI_kx%06.3f.dat", params.kx);
    tracePSI = fopen(trPSI_fn, "w");
    trace1mode = fopen("./output/traceMode.dat", "w");

    // Variables for HDF5 output
    hid_t hdf5final, hdf5fp, datatype_id, filetype_id;
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

    if (save_traj==1)
    {
	// create Hdf5 output file
	hdf5fp = H5Fcreate(traj_fn, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    }

    // field arrays are declared as pointers and then I malloc.
    complex_d *psi, *psiOld, *forcing;
    complex_d *trC, *psiNL, *forcingN;
    complex_d *cij, *cijOld, *cijNL;
    complex_d *tmpop;

    complex_d *opsList, *hopsList;

    lin_flow_scratch scr;

    fftw_plan phys_plan, spec_plan;

    unsigned fftwFlag;
#ifdef MYDEBUG 
    fftwFlag = FFTW_ESTIMATE;
#else
    fftwFlag = FFTW_MEASURE;
#endif

    // dynamically malloc array of complex numbers.
    tmpop = (complex_d*) fftw_malloc(M*M * sizeof(complex_d));
    opsList = (complex_d*) fftw_malloc((N+1)*M*M * sizeof(complex_d));
    hopsList = (complex_d*) fftw_malloc((N+1)*M*M * sizeof(complex_d));
    psi = (complex_d*) fftw_malloc(M*(N+1) * sizeof(complex_d));
    forcing = (complex_d*) fftw_malloc(M*(N+1) * sizeof(complex_d));
    forcingN = (complex_d*) fftw_malloc(M*(N+1) * sizeof(complex_d));
    psiOld = (complex_d*) fftw_malloc(M*(N+1) * sizeof(complex_d));
    psiNL = (complex_d*) fftw_malloc(M*(N+1) * sizeof(complex_d));

    scr.scratch = (complex_d*) fftw_malloc(M * sizeof(complex_d));
    scr.scratch2 = (complex_d*) fftw_malloc(M * sizeof(complex_d));
    scr.scratch3 = (complex_d*) fftw_malloc(M * sizeof(complex_d));
    scr.scratch4 = (complex_d*) fftw_malloc(M * sizeof(complex_d));

    scr.U0 = (complex_d*) fftw_malloc(M * sizeof(complex_d));
    scr.u = (complex_d*) fftw_malloc(M * sizeof(complex_d));
    scr.v = (complex_d*) fftw_malloc(M * sizeof(complex_d));
    scr.udxlplpsi = (complex_d*) fftw_malloc(M * sizeof(complex_d));
    scr.vdylplpsi = (complex_d*) fftw_malloc(M * sizeof(complex_d));
    scr.biharmpsi = (complex_d*) fftw_malloc(M * sizeof(complex_d));
    scr.lplpsi = (complex_d*) fftw_malloc(M * sizeof(complex_d));
    scr.d2yPSI0 = (complex_d*) fftw_malloc(M * sizeof(complex_d));
    scr.d3yPSI0 = (complex_d*) fftw_malloc(M * sizeof(complex_d));
    scr.d2ypsi = (complex_d*) fftw_malloc(M * sizeof(complex_d));
    scr.d3ypsi = (complex_d*) fftw_malloc(M * sizeof(complex_d));
    scr.d4ypsi = (complex_d*) fftw_malloc(M * sizeof(complex_d));
    scr.d4xpsi = (complex_d*) fftw_malloc(M * sizeof(complex_d));
    scr.d2xd2ypsi = (complex_d*) fftw_malloc(M * sizeof(complex_d));

    // Viscoelastic variables

    trC = (complex_d*) fftw_malloc(M*(N+1) * sizeof(complex_d));

    cijOld = (complex_d*) fftw_malloc(3*(N+1)*M * sizeof(complex_d));
    cij = (complex_d*) fftw_malloc(3*(N+1)*M * sizeof(complex_d));
    cijNL = (complex_d*) fftw_malloc(3*(N+1)*M * sizeof(complex_d));

    // temporary Viscoelastic variables

    scr.dxu = (complex_d*) fftw_malloc(M * sizeof(complex_d));
    scr.dyu = (complex_d*) fftw_malloc(M * sizeof(complex_d));
    scr.dxv = (complex_d*) fftw_malloc(M * sizeof(complex_d));
    scr.dyv = (complex_d*) fftw_malloc(M * sizeof(complex_d));
    scr.cxxdxu = (complex_d*) fftw_malloc(M * sizeof(complex_d));
    scr.cxydyu = (complex_d*) fftw_malloc(M * sizeof(complex_d));
    scr.cxy0dyU0 = (complex_d*) fftw_malloc(M * sizeof(complex_d));
    scr.vgradcxx = (complex_d*) fftw_malloc(M * sizeof(complex_d));
    scr.cxydxv = (complex_d*) fftw_malloc(M * sizeof(complex_d));
    scr.cyydyv = (complex_d*) fftw_malloc(M * sizeof(complex_d));
    scr.vgradcyy = (complex_d*) fftw_malloc(M * sizeof(complex_d));
    scr.cxxdxv = (complex_d*) fftw_malloc(M * sizeof(complex_d));
    scr.cyydyu = (complex_d*) fftw_malloc(M * sizeof(complex_d));
    scr.cyy0dyU0 = (complex_d*) fftw_malloc(M * sizeof(complex_d));
    scr.vgradcxy = (complex_d*) fftw_malloc(M * sizeof(complex_d));
    scr.d2ycxy = (complex_d*) fftw_malloc(M * sizeof(complex_d));
    scr.d2xcxy = (complex_d*) fftw_malloc(M * sizeof(complex_d));
    scr.dxycyy_cxx = (complex_d*) fftw_malloc(M * sizeof(complex_d));
    scr.dycxy = (complex_d*) fftw_malloc(M * sizeof(complex_d));
    scr.dycxy0 = (complex_d*) fftw_malloc(M * sizeof(complex_d));
    scr.d2ycxyN = (complex_d*) fftw_malloc(M * sizeof(complex_d));
    scr.d2xcxyN = (complex_d*) fftw_malloc(M * sizeof(complex_d));
    scr.dxycyy_cxxN = (complex_d*) fftw_malloc(M * sizeof(complex_d));
    scr.dycxyN = (complex_d*) fftw_malloc(M * sizeof(complex_d));
    scr.dycxy0N = (complex_d*) fftw_malloc(M * sizeof(complex_d));

    scr.scratchin = (fftw_complex*) fftw_malloc((2*Mf-2) * sizeof(fftw_complex));
    scr.scratchout = (fftw_complex*) fftw_malloc((2*Mf-2) * sizeof(fftw_complex));

    scr.scratchp1 = (complex_d*) fftw_malloc((2*Mf-2) * sizeof(complex_d));
    scr.scratchp2 = (complex_d*) fftw_malloc((2*Mf-2) * sizeof(complex_d));
    scr.scratchp3 = (complex_d*) fftw_malloc((2*Mf-2) * sizeof(complex_d));

    scr.RHSvec = (complex_d*) fftw_malloc(M * sizeof(complex_d));

    // Set up some dft plans
    printf("\n------\nSetting up fftw3 plans\n------\n");
    phys_plan = fftw_plan_dft_1d(2*Mf-2,  scr.scratchin, scr.scratchout,
	    FFTW_BACKWARD, fftwFlag);
    spec_plan = fftw_plan_dft_1d(2*Mf-2,  scr.scratchin, scr.scratchout,
	    FFTW_FORWARD, fftwFlag);

    scr.phys_plan = &phys_plan;
    scr.spec_plan = &spec_plan;

    printf("\n------\nLoading initial streamfunction and operators\n------\n");

    // load the forcing 
    load_hdf5_state("forcing.h5", forcing, params);
    // load the initial field from scipy
    load_hdf5_state_visco("initial_visco.h5", psi, &cij[0], &cij[(N+1)*M],
	    &cij[2*(N+1)*M], params);
    for (i=0; i<3*(N+1)*M; i++)
    {
	cijOld[i] = cij[i];
    }
    for (i=0; i<(N+1)*M; i++)
    {
	forcingN[i] = forcing[i];
    }

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

	sprintf(fn, "./operators/hOp%d.h5", i);
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

	sprintf(fn, "./output/hOp%d.h5", i);
	printf("writing: %s\n", fn);
	for (j=0; j<M*M; j++)
	{
	    tmpop[j] = hopsList[i*M*M + j];
	}

	save_hdf5_arr(fn, &tmpop[0], M*M);
    }
    save_hdf5_state("./output/psi.h5", &psi[0], params);
    save_hdf5_state("./output/cxx.h5", &cij[0], params);
    save_hdf5_state("./output/cyy.h5", &cij[(N+1)*M], params);
    save_hdf5_state("./output/cxy.h5", &cij[2*(N+1)*M], params);
    save_hdf5_state("./output/forcing.h5", &forcing[0], params);
#endif

    // BEGIN TIME STEPPING
    // -------------------

    printf("\n------\nperforming the time iteration\n------\n");
    printf("\nTime:\t\tKE_tot:\t\tKE0:\t\tKE1:\n");

    // calculate and output t=0 quantities
    // u
    single_dy(&psi[ind(0,0)], scr.U0, params);
    single_dy(&psi[ind(1,0)], scr.u, params);

    // v = -dxdpsi
    single_dx(&psi[ind(1,0)], scr.v, 1, params);
    for(j=0; j<M; j++)
    {
	scr.v[j] = -scr.v[j];
    }
    KE0 = calc_cheby_KE_mode(scr.U0, scr.U0, 0, params) * (15.0/ 8.0) * 0.5;
    KE1 = calc_cheby_KE_mode(scr.u, scr.v, 1, params) * (15.0/ 8.0);

    KE_tot = KE0 + KE1;

    if (save_traj==1)
    {
    save_hdf5_snapshot_visco(&hdf5fp, &filetype_id, &datatype_id,
	    psi, &cij[0], &cij[(N+1)*M], &cij[2*(N+1)*M], 0.0, params);
    }

    fprintf(tracefp, "%e\t%e\t%e\t%e\n", 0.0, KE_tot, KE0, KE1);

    printf("%e\t%e\t%e\t%e\t\n", 0.0, KE_tot, KE0, KE1);

    // **** STUPID TEST *****
    //scr.scratch[0] = log(cabs((0.5*M_PI/params.Wi)*cij[2*(N+1)*M + ind(1,1)]));
    //scr.scratch[1] = log(cabs((0.5*M_PI/params.Wi)*cij[ind(1,2)]));
    //printf("%f %20.18f %20.18f\n", 0.0, creal(scr.scratch[0]), creal(scr.scratch[1]));
    //fprintf(tracefp, "%f %20.18f %20.18f\n", 0.0, creal(scr.scratch[0]), creal(scr.scratch[1]));

    for (timeStep=0; timeStep<numTimeSteps; timeStep++)
    {

	time = timeStep*dt;

	// Reset temporary variables for the new timestep
	for (i=0; i<3*(N+1)*M; i++)
	{
	    cijOld[i] = cij[i];
	    cijNL[i] = cij[i];
	}
	for (i=0; i<(N+1)*M; i++)
	{
	    psiOld[i] = psi[i];
	    psiNL[i] = psi[i];
	}


	// OSCILLATING PRESSURE GRADIENT
	// ----------------------------------

	if (params.oscillatory_flow != 0)
	{
	    // Calculate forcing for the half step 

	    periods = floor(initTime/(2.0*M_PI));
	    phase = initTime - 2.0*M_PI*periods;

	    forcing[ind(0,0)] = params.P*cos(time + phase);
	    forcingN[ind(0,0)] = params.P*cos((timeStep+0.5)*dt + phase);

	    // calculate the half-step variables for the nonlinear terms
	    // (the *nl variables)

	    step_conformation_linear_oscil(cijOld, cijNL, psiOld, cijOld,
		    0.5*dt, scr, params);

	    calc_base_cij(cijNL, (timeStep+0.5)*dt, scr, params);

	    step_sf_linear_SI_oscil_visco(psiOld, psiNL, cijOld, cijNL, psiOld,
		    forcing, forcingN, 0.5*dt, timeStep, hopsList, scr, params);

	    calc_base_sf(psiNL, (timeStep+0.5)*dt, scr, params);

#ifdef MYDEBUG 
	    // output when debugging
	    save_hdf5_state("./output/psiStar.h5", &psiNL[0], params);
	    save_hdf5_state("./output/cxxStar.h5", &cijNL[0], params);
	    save_hdf5_state("./output/cyyStar.h5", &cijNL[(N+1)*M], params);
	    save_hdf5_state("./output/cxyStar.h5", &cijNL[2*(N+1)*M], params);

	    save_hdf5_arr("./output/U0.h5",  &scr.U0[0], M);
	    save_hdf5_arr("./output/u.h5",  &scr.u[0], M);
	    save_hdf5_arr("./output/v.h5", &scr.v[0], M);
	    save_hdf5_arr("./output/lplpsi.h5", &scr.lplpsi[0], M);
	    save_hdf5_arr("./output/d2yPSI0.h5", &scr.d2yPSI0[0], M);
	    save_hdf5_arr("./output/d3yPSI0.h5", &scr.d3yPSI0[0], M);
	    save_hdf5_arr("./output/d3ypsi.h5", &scr.d3ypsi[0], M);
	    save_hdf5_arr("./output/d4ypsi.h5", &scr.d4ypsi[0], M);
	    save_hdf5_arr("./output/d2xd2ypsi.h5", &scr.d2xd2ypsi[0], M);
	    save_hdf5_arr("./output/d4xpsi.h5", &scr.d4xpsi[0], M);
	    save_hdf5_arr("./output/biharmpsi.h5", &scr.biharmpsi[0], M);
	    save_hdf5_arr("./output/dycxy0.h5", &scr.dycxy0[0], M);
	    save_hdf5_arr("./output/d2ycxy.h5", &scr.d2ycxy[0], M);
	    save_hdf5_arr("./output/d2xcxy.h5", &scr.d2xcxy[0], M);
	    save_hdf5_arr("./output/dxycyy_cxx.h5", &scr.dxycyy_cxx[0], M);

	    save_hdf5_arr("./output/dycxy0Star.h5", &scr.dycxy0N[0], M);
	    save_hdf5_arr("./output/d2ycxyStar.h5", &scr.d2ycxyN[0], M);
	    save_hdf5_arr("./output/d2xcxyStar.h5", &scr.d2xcxyN[0], M);
	    save_hdf5_arr("./output/dxycyy_cxxStar.h5", &scr.dxycyy_cxxN[0], M);

	    save_hdf5_arr("./output/udxlplpsi.h5", &scr.udxlplpsi[0], M);
	    save_hdf5_arr("./output/vdylplpsi.h5", &scr.vdylplpsi[0], M);

	    save_hdf5_arr("./output/RHSVec1s.h5", &scr.scratch[0], M);

	    save_hdf5_arr("./output/vgradcxx.h5", &scr.vgradcxx[0], M);
	    save_hdf5_arr("./output/vgradcxy.h5", &scr.vgradcxy[0], M);
	    save_hdf5_arr("./output/vgradcyy.h5", &scr.vgradcyy[0], M);

	    save_hdf5_arr("./output/cxydyu.h5", scr.cxydyu, M);
	    save_hdf5_arr("./output/cxxdxu.h5", scr.cxxdxu, M);

	    save_hdf5_arr("./output/cxydxv.h5", scr.cxydxv, M);
	    save_hdf5_arr("./output/cyydyv.h5", scr.cyydyv, M);

	    save_hdf5_arr("./output/cxxdxv.h5", scr.cxxdxv, M);
	    save_hdf5_arr("./output/cyydyu.h5", scr.cyydyu, M);


#endif // MYDEBUG

	    // Calculate forcing for the full step 

	    forcing[ind(0,0)] = params.P*cos(time+0.5*dt + phase);
	    forcingN[ind(0,0)] = params.P*cos((timeStep+1.0)*dt + phase);


	    // Calculate flow at new time step

	    step_conformation_linear_oscil(cijOld, cij, psiNL, cijNL, dt, scr, params);

	    calc_base_cij(cij, (timeStep+1.0)*dt, scr, params);

	    step_sf_linear_SI_oscil_visco(psiOld, psi, cijOld, cij, psiNL,
		    forcing, forcingN, dt, timeStep, opsList, scr, params);

	    calc_base_sf(psi, (timeStep+1.0)*dt, scr, params);

	    //scr.scratch[0] = log(cabs((0.5*M_PI/params.Wi)*cij[2*(N+1)*M + ind(1,1)]));
	    //scr.scratch[1] = log(cabs((0.5*M_PI/params.Wi)*cij[ind(1,2)]));
	    //fprintf(tracefp, "%f %20.18f %20.18f\n", (timeStep+1.0)*dt, creal(scr.scratch[0]), creal(scr.scratch[1]));

#ifdef MYDEBUG 
	    // output when debugging

	    save_hdf5_arr("./output/dycxy0New.h5", &scr.dycxy0N[0], M);
	    save_hdf5_arr("./output/d2ycxyNew.h5", &scr.d2ycxyN[0], M);
	    save_hdf5_arr("./output/d2xcxyNew.h5", &scr.d2xcxyN[0], M);
	    save_hdf5_arr("./output/dxycyy_cxxNew.h5", &scr.dxycyy_cxxN[0], M);

	    save_hdf5_arr("./output/udxlplpsiStar.h5", &scr.udxlplpsi[0], M);
	    save_hdf5_arr("./output/vdylplpsiStar.h5", &scr.vdylplpsi[0], M);

	    save_hdf5_arr("./output/vgradcxxStar.h5", &scr.vgradcxx[0], M);
	    save_hdf5_arr("./output/vgradcxyStar.h5", &scr.vgradcxy[0], M);
	    save_hdf5_arr("./output/vgradcyyStar.h5", &scr.vgradcyy[0], M);

	    save_hdf5_arr("./output/cxydyuStar.h5", scr.cxydyu, M);
	    save_hdf5_arr("./output/cxxdxuStar.h5", scr.cxxdxu, M);

	    save_hdf5_arr("./output/cxydxvStar.h5", scr.cxydxv, M);
	    save_hdf5_arr("./output/cyydyvStar.h5", scr.cyydyv, M);

	    save_hdf5_arr("./output/cxxdxvStar.h5", scr.cxxdxv, M);
	    save_hdf5_arr("./output/cyydyuStar.h5", scr.cyydyu, M);

	    save_hdf5_state("./output/psiNew.h5", &psi[0], params);
	    save_hdf5_state("./output/cxxNew.h5", &cij[0], params);
	    save_hdf5_state("./output/cyyNew.h5", &cij[(N+1)*M], params);
	    save_hdf5_state("./output/cxyNew.h5", &cij[2*(N+1)*M], params);

	    printf("\nFORCE END THE DEBUGGING RUN\n");
	    break;
#endif // MYDEBUG


	} else {
	    // TIME INDEPENDENT PRESSURE GRADIENT
	    // ----------------------------------

	    step_conformation_linear_Crank_Nicolson(cijOld, cijNL, psiOld, cijOld,
		    0.5*dt, scr, params);
	    step_sf_linear_SI_Crank_Nicolson_visco(psiOld, psiNL, cijOld, cijNL, psiOld,
		    forcing, forcingN, 0.5*dt, timeStep, hopsList, scr, params);

#ifdef MYDEBUG 
	    save_hdf5_state("./output/cxxN.h5", &cijNL[0], params);
	    save_hdf5_state("./output/cyyN.h5", &cijNL[(N+1)*M], params);
	    save_hdf5_state("./output/cxyN.h5", &cijNL[2*(N+1)*M], params);

	    printf("\nFORCE END THE DEBUGGING RUN\n");
	    break;
#endif

	    step_conformation_linear_Crank_Nicolson(cijOld, cij, psiNL, cijNL, dt, scr, params);

	    step_sf_linear_SI_Crank_Nicolson_visco(psiOld, psi, cijOld, cij, psiNL,
		    forcing, forcingN, dt, timeStep, opsList, scr, params);

	}

	//=========================================================================================

	// OUTPUT DATA 
	// -----------

	if (((timeStep+1) % stepsPerFrame) == 0 )
	{

	    time = (timeStep + 1.0)*dt;

	    double normPSI1 = 0;
	    double normPSI0 = 0;

	    // u
	    single_dy(&psi[ind(0,0)], scr.U0, params);
	    single_dy(&psi[ind(1,0)], scr.u, params);

	    // v = -dxdpsi
	    single_dx(&psi[ind(1,0)], scr.v, 1, params);

	    for (j=0; j<M; j++)
	    {
		scr.scratch[j] = scr.U0[j];
		scr.scratch2[j] = scr.u[j];
	    }

	    fprintf(trace1mode, "%e\t%e\t%e\t%e\t%e\n", 
		    time, creal(scr.scratch[3]), cimag(scr.scratch[3]),
		    creal(scr.scratch2[3]), cimag(scr.scratch2[3]));

	    for (j=M-1; j>=0; j=j-1)
	    {
		normPSI0 += creal(psi[ind(0,j)]*conj(psi[ind(0,j)])); 
		normPSI1 += creal(psi[ind(1,j)]*conj(psi[ind(1,j)])); 
	    }

	    fprintf(tracePSI, "%e\t%e\t%e\n", time, normPSI0, normPSI1);

	    KE0 = calc_cheby_KE_mode(scr.U0, scr.U0, 0, params) * (15.0/ 8.0) * 0.5;
	    KE1 = calc_cheby_KE_mode(scr.u, scr.v, 1, params) * (15.0/ 8.0);

	    KE_tot = KE0 + KE1;

	    if (save_traj==1)
	    {
		printf("%e\t%e\t%e\t%e\t\n", time, KE_tot, KE0, KE1);
	    }

	    // **** STUPID TESTS ****
	    // scr.scratch[0] = log(cabs((0.5*M_PI/params.Wi)*cij[2*(N+1)*M + ind(1,1)]));
	    // scr.scratch[1] = log(cabs((0.5*M_PI/params.Wi)*cij[ind(1,2)]));
	    // printf("%f %20.18f %20.18f\n", time , creal(scr.scratch[0]), creal(scr.scratch[1]));
	    

	    if (save_traj==1)
	    {
		save_hdf5_snapshot_visco(&hdf5fp, &filetype_id, &datatype_id,
			psi, &cij[0], &cij[(N+1)*M], &cij[2*(N+1)*M], time, params);
	    }


	    fprintf(tracefp, "%e\t%e\t%e\t%e\n", time, KE_tot, KE0, KE1);

	    fflush(tracePSI);
	    fflush(trace1mode);
	    fflush(tracefp);
	    if (save_traj==1)
	    {
		H5Fflush(hdf5fp, H5F_SCOPE_GLOBAL);
	    }

	}

    }

    // save the final state

    hdf5final = H5Fcreate("output/final.h5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    save_hdf5_state_visco(&hdf5final,
	    &filetype_id, &datatype_id, psi, &cij[0], &cij[(N+1)*M], &cij[2*(N+1)*M], params);



    fclose(tracefp);
    fclose(tracePSI);
    fclose(trace1mode);

    // clean up hdf5
    status = H5Tclose(datatype_id);
    status = H5Tclose(filetype_id);
    status = H5Fclose(hdf5final);
    if (save_traj==1) { status = H5Fclose(hdf5fp); }

    // garbage collection
    fftw_destroy_plan(phys_plan);
    fftw_destroy_plan(spec_plan);

    fftw_free(tmpop);
    fftw_free(opsList);
    fftw_free(hopsList);
    fftw_free(psi);
    fftw_free(forcing);
    fftw_free(psiOld);
    fftw_free(psiNL);

    fftw_free(scr.scratch);
    fftw_free(scr.scratch2);
    fftw_free(scr.scratch3);
    fftw_free(scr.scratch4);
    fftw_free(scr.U0);
    fftw_free(scr.d2yPSI0);
    fftw_free(scr.d3yPSI0);
    fftw_free(scr.u);
    fftw_free(scr.v);
    fftw_free(scr.udxlplpsi);
    fftw_free(scr.vdylplpsi);
    fftw_free(scr.lplpsi);
    fftw_free(scr.biharmpsi);
    fftw_free(scr.scratchin);
    fftw_free(scr.scratchout);
    fftw_free(scr.scratchp1);
    fftw_free(scr.scratchp2);
    fftw_free(scr.d2ypsi);
    fftw_free(scr.d3ypsi);
    fftw_free(scr.d4ypsi);
    fftw_free(scr.d4xpsi);
    fftw_free(scr.d2xd2ypsi);
    fftw_free(scr.RHSvec);

    fftw_free(trC );

    fftw_free(cijOld );
    fftw_free(cij );
    fftw_free(cijNL );

    fftw_free(scr.dxu);
    fftw_free(scr.dyu);
    fftw_free(scr.dxv);
    fftw_free(scr.dyv);
    fftw_free(scr.cxxdxu);
    fftw_free(scr.cxydyu);
    fftw_free(scr.vgradcxx);
    fftw_free(scr.cxydxv);
    fftw_free(scr.cyydyv);
    fftw_free(scr.vgradcyy);
    fftw_free(scr.cxxdxv);
    fftw_free(scr.cyydyu);
    fftw_free(scr.vgradcxy);
    fftw_free(scr.d2ycxy);
    fftw_free(scr.d2xcxy);
    fftw_free(scr.dxycyy_cxx);
    fftw_free(scr.dycxy);
    fftw_free(scr.dycxy0);
    fftw_free(scr.d2ycxyN);
    fftw_free(scr.d2xcxyN);
    fftw_free(scr.dxycyy_cxxN);
    fftw_free(scr.dycxyN);
    fftw_free(scr.dycxy0N);


    printf("quitting c program\n");
    return 0;

}
