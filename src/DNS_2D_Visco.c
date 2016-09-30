/* -------------------------------------------------------------------------- *
 *									      *
 *  DNS_2D_Visco.c							      *
 *                                                                            *
 *  Time stepping DNS program for 2D Newtonian fluid.			      *
 *                                                                            *
 *                                                                            *
 * -------------------------------------------------------------------------- */

// Last modified: Fri 30 Sep 14:05:11 2016

/* Program Description:
 *
 * This program is written to work with a python setup program. The setup
 * program will write a series of files containing matrix operators of all
 * Fourier modes in the problem. These will then be imported and this program
 * will perform the timestepping using FFT's for the products and my own
 * functions to perform derivatives. 
 *
 */

// Headers

#include"fields_IO.h"
#include"fields_2D.h"
#include"time_steppers.h"

void read_cline_args(int argc, char **argv, flow_params *params);

void setup_scratch_space(flow_scratch *scr, flow_params params);

void output_macro_state(complex_d *psi, complex_d *cij,  complex_d *trC, double phase, double time,
	FILE *traceKE, FILE *tracePSI, FILE *trace1mode, FILE *traceStressfp, flow_scratch scr, flow_params params);

void debug_output_halfstep_variables(complex_d *psiNL, complex_d *cijNL, flow_scratch scr, flow_params params);

void debug_output_fullstep_variables(complex_d *psi, complex_d *cij, flow_scratch scr, flow_params params);

//int DNS_2D_Visco(flow_params params);
//{
//
//    return DNS_2D_Visco(params);
//}

// Main

int main(int argc, char **argv) 
{
    flow_params params;

    printf("\n------\nReading Command line arguments\n------\n");
    read_cline_args(argc, argv, &params);
    
    int stepsPerFrame = params.stepsPerFrame;
    int numTimeSteps = params.numTimeSteps;
    double dt = params.dt;
    double initTime=params.initTime;
    
    int timeStep = 0;
    int posdefck = 0;

    double time = 0;

    int periods = 0;
    double phase = 0;

    double normPSI1 = 0;
    double normPSI2 = 0;
    double normPSI0 = 0;

    double KE0 = 1.0;
    double KE1 = 0.0;
    double KE2 = 0.0;
    double KE_tot = 0.0;
    double KE_xdepend = 0.0;
    double EE0 = 1.0;
    double EE1 = 0.0;
    double EE2 = 0.0;
    double EE_tot = 0.0;
    double EE_xdepend = 0.0;

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
    

    FILE *tracefp, *tracePSI, *trace1mode, *traceStressfp;
    char *trace_fn, *traj_fn;
    int i, j;
    int N = params.N;
    int M = params.M;
    int Nf = params.Nf;
    int Mf = params.Mf;

    trace_fn = "./output/trace.dat";
    traj_fn = "./output/traj.h5";

    tracefp = fopen(trace_fn, "w");
    traceStressfp = fopen("./output/traceStress.dat", "w");
    tracePSI = fopen("./output/tracePSI.dat", "w");
    trace1mode = fopen("./output/traceMode.dat", "w");

    // Variables for HDF5 output
    hid_t hdf5fp, hdf5final, datatype_id, filetype_id;
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

    complex_d *psi, *psiOld, *psiNL, *psi_lam;
    complex_d *forcing, *forcingN;
    complex_d *cijOld, *cij, *cijNL;
    complex_d *trC;
    complex_d *tmpop;
    complex_d *opsList, *hopsList;

    flow_scratch scr;
    setup_scratch_space(&scr, params); 

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

    psiOld = (complex_d*) fftw_malloc(M*(N+1) * sizeof(complex_d));
    psi = (complex_d*) fftw_malloc(M*(N+1) * sizeof(complex_d));
    psiNL = (complex_d*) fftw_malloc(M*(N+1) * sizeof(complex_d));
    psi_lam = (complex_d*) fftw_malloc(M*(N+1) * sizeof(complex_d));
    forcing = (complex_d*) fftw_malloc(M*(N+1) * sizeof(complex_d));
    forcingN = (complex_d*) fftw_malloc(M*(N+1) * sizeof(complex_d));

    // Viscoelastic variables

    trC = (complex_d*) fftw_malloc(M*(N+1) * sizeof(complex_d));

    cijOld = (complex_d*) fftw_malloc(3*(N+1)*M * sizeof(complex_d));
    cij = (complex_d*) fftw_malloc(3*(N+1)*M * sizeof(complex_d));
    cijNL = (complex_d*) fftw_malloc(3*(N+1)*M * sizeof(complex_d));

    printf("\n------\nLoading initial streamfunction and operators\n------\n");

    // load the initial field from scipy
    load_hdf5_state_visco("initial_visco.h5", psi, &cij[0], &cij[(N+1)*M],
				&cij[2*(N+1)*M], params);
        for (i=0; i<3*(N+1)*M; i++)
	{
	    cijOld[i] = cij[i];
	}
        for (i=0; i<(N+1)*M; i++)
	{
	    psiOld[i] = psi[i];
	}

    load_hdf5_state("forcing.h5", forcing, params);

        for (i=0; i<(N+1)*M; i++)
        {
	    forcingN[i] = forcing[i];
	}

    load_hdf5_state("laminar.h5", psi_lam, params);

    // load the operators from scipy 
    for (i=0; i<N+1; i++) 
    {
	char fn[30];
	sprintf(fn, "./operators/op%d.h5", i);
	#ifdef MYDEBUG
	printf("opening: %s\n", fn);
	#endif
	load_hdf5_operator(fn, tmpop, params);

	for (j=0; j<M*M; j++)
	{
	    opsList[i*M*M + j] = tmpop[j];
	}

	sprintf(fn, "./operators/hOp%d.h5", i);
	#ifdef MYDEBUG
	printf("opening: %s\n", fn);
	#endif
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
    save_hdf5_state("./output/forcing.h5", &forcing[0], params);
    #endif
    
    // set the test.
    printf("\n------\nperforming the stress equilibration\n------\n");
    #ifdef MYDEBUG   
    printf("\nTime\t\tIntegrated TrC");
    #endif

    #ifndef MYDEBUG
    //equilibriate_stress( psiOld, psi_lam, cijOld, cij, cijNL, dt, scr, params,
    //	    		&hdf5fp, &filetype_id, &datatype_id);
    #endif

    periods = floor(initTime/(2.0*M_PI));
    phase = initTime - 2.0*M_PI*periods;


    // Cant get the oscillatory laminar flow working in python, so I did it here
    //if (params.oscillatory_flow != 0)
    //{
    //    calc_base_cij(cij, phase + 0.0, scr, params);
    //    calc_base_sf(psi, phase + 0.0, scr, params);
    //}

    output_macro_state(psi, cij, trC, phase, time, tracefp,
	    tracePSI, trace1mode, traceStressfp, scr, params);

    save_hdf5_snapshot_visco(&hdf5fp, &filetype_id, &datatype_id,
	 psi, &cij[0], &cij[(N+1)*M], &cij[2*(N+1)*M], 0.0, params);

    // perform the time iteration
    printf("\n------\nperforming the time iteration\n------\n");
    printf("\nTime\t\tKE_tot\t\t KE0\t\t KE1\t\t EE0\n");


    for (timeStep=0; timeStep<numTimeSteps; timeStep++)
    {

	time = (timeStep)*dt;
        
        #ifdef MYDEBUG
        if (timeStep==0)
        {
            d2x(psi, scr.scratch, params);
            save_hdf5_state("./output/d2xpsi.h5", &scr.scratch[0], params);
            save_hdf5_state("./output/psi.h5", &psi[0], params);

            save_hdf5_state("./output/cxx.h5", &cij[0], params);
            save_hdf5_state("./output/cyy.h5", &cij[(N+1)*M], params);
            save_hdf5_state("./output/cxy.h5", &cij[2*(N+1)*M], params);

        }
        #endif

        // Step the stresses using 2nd order pc CN method

        // make the half step for the prediction of C and PSI for NL terms
        // C calculation has only NL terms in psi

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

	if (params.oscillatory_flow != 0)
	{
	    // OSCILLATING PRESSURE GRADIENT
	    forcing[ind(0,0)] = params.P*cos((timeStep)*dt + phase);
	    forcingN[ind(0,0)] = params.P*cos((timeStep+0.5)*dt + phase);

	    step_conformation_oscil(cijOld, cijNL, psiOld, cijOld,
						0.5*dt, scr, params);
	    step_sf_SI_oscil_visco(psiOld, psiNL, cijOld, cijNL, psiOld,
			    forcing, forcingN, 0.5*dt, timeStep, hopsList, scr, params);

	    #ifdef MYDEBUG
	    printf("\nFORCE END DEBUGGING RUN\n");
	    exit(1);
	    #endif

	    // calculate forcing on the half step
	    forcing[ind(0,0)] = params.P*cos((timeStep)*dt + phase);
	    forcingN[ind(0,0)] = params.P*cos((timeStep+1.0)*dt + phase);

	    step_conformation_oscil(cijOld, cij, psiNL, cijNL, dt, scr, params);

	    step_sf_SI_oscil_visco(psiOld, psi, cijOld, cij, psiNL,
				forcing, forcingN, dt, timeStep, opsList, scr, params);


	} else {

	    step_conformation_Crank_Nicolson(cijOld, cijNL, psiOld, cijOld,
		    0.5*dt, scr, params);

#ifdef MYDEBUG
	    if(timeStep==0)
	    {
		debug_output_halfstep_variables(psiNL, cijNL, scr, params);
	    }
#endif  // MY_DEBUG

	    //if (timestep%(2*params.Wi/dt)==0)
	    //{
	    //  for(int i=1; i<(N+1)*M; i++)
	//	{
	//	    //forcing[i]	*= 0.5;
	//	    //forcingN[i] *= 0.5;
	//	    forcing[i]	*= 0.0;
	//	    forcingN[i] *= 0.0;
	//	}
	    //printf("\n STEPPING THE FORCING DOWN \n");
	    //}

	    step_sf_SI_Crank_Nicolson_visco(psiOld, psiNL, cijOld, cijNL, psiOld,
		    forcing, forcingN, 0.5*dt, timeStep, hopsList, scr, params);

	    // use the old values plus the values on the half step for the NL terms
	    step_conformation_Crank_Nicolson(cijOld, cij, psiNL, cijNL, dt, scr, params);


	    step_sf_SI_Crank_Nicolson_visco(psiOld, psi, cijOld, cij, psiNL,
		    forcing, forcingN, dt, timeStep, opsList, scr, params);


#ifdef MYDEBUG
	    if (timeStep==0)
	    {

		debug_output_fullstep_variables(psi, cij, scr, params);
	    }
#endif // MY_DEBUG


	}


        // output some information at every frame
        if (((timeStep+1) % stepsPerFrame) == 0 )
        {


	    time = (timeStep + 1)*dt;

	    output_macro_state(psi, cij, trC, phase, time, tracefp,
		    tracePSI, trace1mode, traceStressfp, scr, params);

            save_hdf5_snapshot_visco(&hdf5fp, &filetype_id, &datatype_id,
	    	 psi, &cij[0], &cij[(N+1)*M], &cij[2*(N+1)*M], time, params);
             
            H5Fflush(hdf5fp, H5F_SCOPE_GLOBAL);

         }
    }
    

    // save the final state
    hdf5final = H5Fcreate("output/final.h5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    save_hdf5_state_visco(&hdf5final,
    &filetype_id, &datatype_id, psi, &cij[0], &cij[(N+1)*M], &cij[2*(N+1)*M], params);

    fclose(tracefp);
    fclose(traceStressfp);
    fclose(tracePSI);
    fclose(trace1mode);

    // clean up hdf5
    status = H5Tclose(datatype_id);
    status = H5Tclose(filetype_id);
    status = H5Fclose(hdf5fp);
    status = H5Fclose(hdf5final);

    // garbage collection
    fftw_destroy_plan(scr.act_phys_plan);
    fftw_destroy_plan(scr.act_spec_plan);

    fftw_free(tmpop);
    fftw_free(opsList);
    fftw_free(hopsList);
    fftw_free(forcing);
    fftw_free(forcingN);

    fftw_free(psiOld);
    fftw_free(psi);
    fftw_free(psiNL);
    fftw_free(psi_lam);
    fftw_free(cijOld);
    fftw_free(cij);
    fftw_free(cijNL);
    fftw_free(trC);

    fftw_free(scr.scratch);
    fftw_free(scr.scratch2);
    fftw_free(scr.scratch3);
    fftw_free(scr.scratch4);
    fftw_free(scr.scratch5);
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
    fftw_free(scr.scratchp3);
    fftw_free(scr.dyyypsi);
    fftw_free(scr.d4ypsi);
    fftw_free(scr.d4xpsi);
    fftw_free(scr.d2xd2ypsi);
    fftw_free(scr.dypsi);
    fftw_free(scr.vdyypsi);
    fftw_free(scr.RHSvec);
    fftw_free(scr.dxu);
    fftw_free(scr.dyu);
    fftw_free(scr.dxv);
    fftw_free(scr.dyv);
    fftw_free(scr.cxxdxu);
    fftw_free(scr.cxydyu);
    fftw_free(scr.vgradcxx );
    fftw_free(scr.cxydxv);
    fftw_free(scr.cyydyv);
    fftw_free(scr.vgradcyy );
    fftw_free(scr.cxxdxv);
    fftw_free(scr.cyydyu);
    fftw_free(scr.vgradcxy );
    fftw_free(scr.d2ycxy);
    fftw_free(scr.d2xcxy);
    fftw_free(scr.dxycyy_cxx);
    fftw_free(scr.dycxy);
    fftw_free(scr.d2ycxyN);
    fftw_free(scr.d2xcxyN);
    fftw_free(scr.dxycyy_cxxN);
    fftw_free(scr.dycxyN);

    printf("quitting c program\n");

    return 0;
}

void read_cline_args(int argc, char **argv, flow_params *pParams)
{
    // Read in parameters from cline args.

    int shortArg;
    extern char *optarg;
    extern int optind;

    printf("begin reading arguments using getopt\n");
    while ((shortArg = getopt (argc, argv, "OdN:M:U:k:R:W:b:D:P:t:s:T:i:")) != -1)
	switch (shortArg)
	{
	    case 'N':
		pParams->N = atoi(optarg);
		printf("N %d\n", pParams->N);
		break;
	    case 'M':
		pParams->M = atoi(optarg);
		break;
	    case 'U':
		pParams->U0 = atof(optarg);
		break;
	    case 'k':
		pParams->kx = atof(optarg);
		break;
	    case 'R':
		pParams->Re = atof(optarg);
		break;
	    case 'W':
		pParams->Wi = atof(optarg);
		break;
	    case 'b':
		pParams->beta = atof(optarg);
		break;
	    case 'D':
		pParams->De = atof(optarg);
		break;
	    case 'P':
		pParams->P = atof(optarg);
		break;
	    case 't':
		pParams->dt = atof(optarg);
		break;
	    case 's':
		pParams->stepsPerFrame = atoi(optarg);
		break;
	    case 'T':
		pParams->numTimeSteps = atoi(optarg);
		break;
	    case 'i':
		pParams->initTime = atof(optarg);
		break;
	    case 'd':
		pParams->dealiasing = 1;
		printf("Dealiasing on\n");
		break;
	    case 'O':
		pParams->oscillatory_flow = 1;
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
		exit(1);
	    default:
		printf("aborting\n");
		abort ();
	}

    if (pParams->dealiasing == 1)
    {
	pParams->Nf = (3*pParams->N)/2 + 1;
	pParams->Mf = 2*pParams->M; //(3*pParams->M)/2;
    } else
    {
	pParams->Nf = pParams->N;
	pParams->Mf = pParams->M;
    }

}

void setup_scratch_space(flow_scratch *scr, flow_params params) 
{
    int N = params.N;
    int M = params.M;
    int Nf = params.Nf;
    int Mf = params.Mf;

    //fftw_plan phys_plan, spec_plan;

    unsigned fftwFlag;
#ifdef MYDEBUG 
    fftwFlag = FFTW_ESTIMATE;
#else
    fftwFlag = FFTW_MEASURE;
#endif

    
    // temporary Newtonian variables

    scr->scratch = (complex_d*) fftw_malloc(M*(N+1) * sizeof(complex_d));
    scr->scratch2 = (complex_d*) fftw_malloc(M*(N+1) * sizeof(complex_d));
    scr->scratch3 = (complex_d*) fftw_malloc(M*(N+1) * sizeof(complex_d));
    scr->scratch4 = (complex_d*) fftw_malloc(M*(N+1) * sizeof(complex_d));
    scr->scratch5 = (complex_d*) fftw_malloc(M*(N+1) * sizeof(complex_d));

    scr->u = (complex_d*) fftw_malloc(M*(N+1) * sizeof(complex_d));
    scr->v = (complex_d*) fftw_malloc(M*(N+1) * sizeof(complex_d));
    scr->udxlplpsi = (complex_d*) fftw_malloc(M*(N+1) * sizeof(complex_d));
    scr->vdylplpsi = (complex_d*) fftw_malloc(M*(N+1) * sizeof(complex_d));
    scr->lplpsi = (complex_d*) fftw_malloc(M*(N+1) * sizeof(complex_d));
    scr->biharmpsi = (complex_d*) fftw_malloc(M*(N+1) * sizeof(complex_d));
    scr->dyyypsi = (complex_d*) fftw_malloc(M*(N+1) * sizeof(complex_d));
    scr->d4ypsi = (complex_d*) fftw_malloc(M*(N+1) * sizeof(complex_d));
    scr->d4xpsi = (complex_d*) fftw_malloc(M*(N+1) * sizeof(complex_d));
    scr->d2xd2ypsi = (complex_d*) fftw_malloc(M*(N+1) * sizeof(complex_d));
    scr->dypsi = (complex_d*) fftw_malloc(M*(N+1) * sizeof(complex_d));
    scr->vdyypsi = (complex_d*) fftw_malloc(M*(N+1) * sizeof(complex_d));

    // temporary Viscoelastic variables

    scr->dxu = (complex_d*) fftw_malloc(M*(N+1) * sizeof(complex_d));
    scr->dyu = (complex_d*) fftw_malloc(M*(N+1) * sizeof(complex_d));
    scr->dxv = (complex_d*) fftw_malloc(M*(N+1) * sizeof(complex_d));
    scr->dyv = (complex_d*) fftw_malloc(M*(N+1) * sizeof(complex_d));
    scr->cxxdxu = (complex_d*) fftw_malloc(M*(N+1) * sizeof(complex_d));
    scr->cxydyu = (complex_d*) fftw_malloc(M*(N+1) * sizeof(complex_d));
    scr->vgradcxx = (complex_d*) fftw_malloc(M*(N+1) * sizeof(complex_d));
    scr->cxydxv = (complex_d*) fftw_malloc(M*(N+1) * sizeof(complex_d));
    scr->cyydyv = (complex_d*) fftw_malloc(M*(N+1) * sizeof(complex_d));
    scr->vgradcyy = (complex_d*) fftw_malloc(M*(N+1) * sizeof(complex_d));
    scr->cxxdxv = (complex_d*) fftw_malloc(M*(N+1) * sizeof(complex_d));
    scr->cyydyu = (complex_d*) fftw_malloc(M*(N+1) * sizeof(complex_d));
    scr->vgradcxy = (complex_d*) fftw_malloc(M*(N+1) * sizeof(complex_d));
    scr->d2ycxy = (complex_d*) fftw_malloc(M*(N+1) * sizeof(complex_d));
    scr->d2xcxy = (complex_d*) fftw_malloc(M*(N+1) * sizeof(complex_d));
    scr->dxycyy_cxx = (complex_d*) fftw_malloc(M*(N+1) * sizeof(complex_d));
    scr->dycxy = (complex_d*) fftw_malloc(M*(N+1) * sizeof(complex_d));
    scr->d2ycxyN = (complex_d*) fftw_malloc(M*(N+1) * sizeof(complex_d));
    scr->d2xcxyN = (complex_d*) fftw_malloc(M*(N+1) * sizeof(complex_d));
    scr->dxycyy_cxxN = (complex_d*) fftw_malloc(M*(N+1) * sizeof(complex_d));
    scr->dycxyN = (complex_d*) fftw_malloc(M*(N+1) * sizeof(complex_d));

    scr->scratchin = (fftw_complex*) fftw_malloc((2*Mf-2)*(2*Nf+1) * sizeof(fftw_complex));
    scr->scratchout = (fftw_complex*) fftw_malloc((2*Mf-2)*(2*Nf+1) * sizeof(fftw_complex));

    scr->scratchp1 = (double*) fftw_malloc((2*Mf-2)*(2*Nf+1) * sizeof(double));
    scr->scratchp2 = (double*) fftw_malloc((2*Mf-2)*(2*Nf+1) * sizeof(double));
    scr->scratchp3 = (double*) fftw_malloc((2*Mf-2)*(2*Nf+1) * sizeof(double));

    scr->RHSvec = (complex_d*) fftw_malloc(M * sizeof(complex_d));

    // Set up some dft plans
    printf("\n------\nSetting up fftw3 plans\n------\n");
    scr->act_phys_plan = fftw_plan_dft_2d(2*Nf+1, 2*Mf-2,  scr->scratchin, scr->scratchout,
        		 FFTW_BACKWARD, fftwFlag);
    scr->act_spec_plan = fftw_plan_dft_2d(2*Nf+1, 2*Mf-2,  scr->scratchin, scr->scratchout,
			 FFTW_FORWARD, fftwFlag);
    
    scr->phys_plan = &(scr->act_phys_plan);
    scr->spec_plan = &(scr->act_spec_plan);
}

void output_macro_state(complex_d *psi, complex_d *cij, complex_d *trC, double phase, double time,
	FILE *traceKE, FILE *tracePSI, FILE *trace1mode, FILE *traceStressfp, flow_scratch scr, flow_params params)
{
    int i, j;
    int N = params.N;
    int M = params.M;
    int Nf = params.Nf;
    int Mf = params.Mf;

    int posdefck = 0;
    
    double normPSI1 = 0;
    double normPSI2 = 0;
    double normPSI0 = 0;

    double KE0 = 1.0;
    double KE1 = 0.0;
    double KE2 = 0.0;
    double KE_tot = 0.0;
    double KE_xdepend = 0.0;
    double EE0 = 1.0;
    double EE1 = 0.0;
    double EE2 = 0.0;
    double EE_tot = 0.0;
    double EE_xdepend = 0.0;


    // track one mode 
    for (i=0; i<N+1; i++)
    {
	for (j=0; j<M; j++)
	{
	    scr.scratch[ind(i,j)] = psi[ind(i,j)];
	}
    }
    scr.scratch[ind(0,0)] -= 2.0/3.0;
    scr.scratch[ind(0,1)] -= 3.0/4.0;
    scr.scratch[ind(0,3)] += 1.0/12.0;

    fprintf(trace1mode, "%e\t%e\t%e\t%e\t%e\t%e\t%e\n", 
	    time, creal(scr.scratch[ind(0,6)]), cimag(scr.scratch[ind(0,6)]),
	     creal(scr.scratch[ind(1,6)]), cimag(scr.scratch[ind(1,6)]),
	      creal(scr.scratch[ind(2,6)]), cimag(scr.scratch[ind(2,6)]));

    // calculate norm u excluding the Poiseuille terms
    for (j=M-1; j>=0; j=j-1)
    {
	if (j > 3)
	{
	    normPSI0 += creal(psi[ind(0,j)]*psi[ind(0,j)]); 
	}
	normPSI1 += creal(psi[ind(1,j)]*conj(psi[ind(1,j)])); 
	normPSI2 += creal(psi[ind(2,j)]*conj(psi[ind(2,j)])); 
    }

    fprintf(tracePSI, "%e\t%e\t%e\t%e\t\n", time, normPSI0, normPSI1, normPSI2);


    // u
    dy(&psi[ind(0,0)], scr.u, params);

    // v = -dxdpsi
    dx(&psi[ind(0,0)], scr.v, params);
    for(j=0; j<M; j++)
    {
	scr.v[j] = -scr.v[j];
    }


    KE0 = calc_KE_mode(scr.u, scr.v, 0, params) * (15.0/ 8.0);
    KE1 = calc_KE_mode(scr.u, scr.v, 1, params) * (15.0/ 8.0);
    KE2 = calc_KE_mode(scr.u, scr.v, 2, params) * (15.0/ 8.0);
    KE_xdepend = KE1 + KE2; 
    for (i=3; i<N+1; i++)
    {
	KE_xdepend += calc_KE_mode(scr.u, scr.v, i, params) * (15.0/ 8.0);
    }

    KE_tot = KE0 + KE_xdepend;

    posdefck = trC_tensor(cij, trC, scr, params);

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

    // NOTE: As it stands. EE0 is the total (summed over x and y)
    // Elastic Kinetic energy?
    // I don't think the other components have a meaning yet.

    fprintf(traceStressfp, "%e\t%e\t%d\n", time, EE0, posdefck);

    printf("%e\t%e\t%e\t%e\t%e\n", time, KE_tot, KE0, KE1, EE0);

    fprintf(traceKE, "%e\t%e\t%e\t%e\t%e\t%e\n", time, KE_tot, KE0, KE1, KE2, KE_xdepend);

    fflush(tracePSI);
    fflush(trace1mode);
    fflush(traceKE);
    fflush(traceStressfp);

}

void debug_output_halfstep_variables(complex_d *psiNL, complex_d *cijNL, flow_scratch scr, flow_params params)
{
    int N = params.N;
    int M = params.M;
    int Nf = params.Nf;
    int Mf = params.Mf;

    save_hdf5_state("./output/dxu.h5", &scr.dxu[0], params);
    save_hdf5_state("./output/dyu.h5", &scr.dyu[0], params);
    save_hdf5_state("./output/dxv.h5", &scr.dxv[0], params);
    save_hdf5_state("./output/dyv.h5", &scr.dyv[0], params);

    save_hdf5_state("./output/cxxdxu.h5", &scr.cxxdxu[0], params);
    save_hdf5_state("./output/cxydyu.h5", &scr.cxydyu[0], params);
    save_hdf5_state("./output/cxydxv.h5", &scr.cxydxv[0], params);
    save_hdf5_state("./output/cyydyv.h5", &scr.cyydyv[0], params);
    save_hdf5_state("./output/cxxdxv.h5", &scr.cxxdxv[0], params);
    save_hdf5_state("./output/cyydyu.h5", &scr.cyydyu[0], params);

    save_hdf5_state("./output/vgradcxx.h5", &scr.vgradcxx[0], params);
    save_hdf5_state("./output/vgradcyy.h5", &scr.vgradcyy[0], params);
    save_hdf5_state("./output/vgradcxy.h5", &scr.vgradcxy[0], params);

}

void debug_output_fullstep_variables(complex_d *psi, complex_d *cij, flow_scratch scr, flow_params params)
{
    int N = params.N;
    int M = params.M;
    int Nf = params.Nf;
    int Mf = params.Mf;


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
    save_hdf5_state("./output/vdyypsi.h5", &scr.vdyypsi[0], params);

    save_hdf5_state("./output/d2xcxy.h5", &scr.d2xcxy[0], params);
    save_hdf5_state("./output/d2ycxy.h5", &scr.d2ycxy[0], params);
    save_hdf5_state("./output/dxycyy_cxx.h5", &scr.dxycyy_cxx[0], params);
    save_hdf5_state("./output/dycxy.h5", &scr.dycxy[0], params);

    save_hdf5_state("./output/cxx2.h5", &cij[0], params);
    save_hdf5_state("./output/cyy2.h5", &cij[(N+1)*M], params);
    save_hdf5_state("./output/cxy2.h5", &cij[2*(N+1)*M], params);
    save_hdf5_state("./output/psi2.h5", &psi[0], params);

}


