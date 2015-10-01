/* -------------------------------------------------------------------------- *
 *									      *
 *  test_fields.c							      *
 *                                                                            *
 *  testing for the calculation of 1D derivatives and convolutions	      *
 *                                                                            *
 *                                                                            *
 * -------------------------------------------------------------------------- */

// Last modified: Thu  1 Oct 15:33:25 2015

/* Program Description:
 *
 * This program is written to work with a python setup program. The setup
 * program will output some test vectors, on which this program performs some
 * operations. The python program will then check that the output of this code
 * is consistent with that of fields_2D.py using the function
 * fields_2D.test_c_version_1D function.
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
 * load_operators - loads the spectral operators from a text file generated in
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

// Main

int main() 
{
    flow_params params;

    params.M = 100;
    params.U0 = 0;
    params.kx = 1.31;
    params.Re = 400;
    params.Wi = 1e-05;
    params.beta = 1.0;
    params.dealiasing = 1.0;

    if (params.dealiasing)
    {
	params.Mf = (3*params.M)/2;
    } else
    {
	params.Mf = params.M;
    }

    // Declare variables

    int i=0;
    int j=0;
    int M = params.M;
    int Mf = params.Mf;

    lin_flow_scratch scr;

    fftw_plan phys_plan, spec_plan;
    char infn[20] = "initial.h5";

    int shapefft = (2*Mf-2);

    //dynamically malloc array of complex numbers.
    fftw_complex* arrin = (fftw_complex*) fftw_malloc(M * sizeof(fftw_complex));
    fftw_complex* specout = (fftw_complex*) fftw_malloc(M * sizeof(fftw_complex));
    fftw_complex* specout2 = (fftw_complex*) fftw_malloc(M * sizeof(fftw_complex));
    fftw_complex* derivout = (fftw_complex*) fftw_malloc(M * sizeof(fftw_complex));
    fftw_complex* scratch = (fftw_complex*) fftw_malloc(M * sizeof(fftw_complex));

    complex_d* physout = (complex_d*) fftw_malloc((2*Mf-2) * sizeof(complex_d));
    complex_d* physout2 = (complex_d*) fftw_malloc((2*Mf-2) * sizeof(complex_d));
    complex_d* phystest = (complex_d*) fftw_malloc((2*Mf-2) * sizeof(complex_d));

    // flow scratch space for the field calculations
    scr.scratch = (complex_d*) fftw_malloc(M * sizeof(complex_d));
    scr.scratch2 = (complex_d*) fftw_malloc(M * sizeof(complex_d));
    scr.scratch3 = (complex_d*) fftw_malloc(M * sizeof(complex_d));
    scr.scratch4 = (complex_d*) fftw_malloc(M * sizeof(complex_d));
    scr.scratchin = (fftw_complex*) fftw_malloc((2*Mf-2) * sizeof(fftw_complex));
    scr.scratchout = (fftw_complex*) fftw_malloc((2*Mf-2) * sizeof(fftw_complex));

    scr.scratchp1 = (complex_d*) fftw_malloc((2*Mf-2) * sizeof(complex_d));
    scr.scratchp2 = (complex_d*) fftw_malloc((2*Mf-2) * sizeof(complex_d));

    //Set up some dft plans
    phys_plan = fftw_plan_dft_1d(2*Mf-2,  scr.scratchin, scr.scratchout,
			 FFTW_BACKWARD, FFTW_ESTIMATE);
    spec_plan = fftw_plan_dft_1d(2*Mf-2,  scr.scratchin, scr.scratchout,
			 FFTW_FORWARD, FFTW_ESTIMATE);
    scr.phys_plan = &phys_plan;
    scr.spec_plan = &spec_plan;

    load_hdf5_arr(infn, arrin, M);

    save_hdf5_arr("./output/testSpec.h5", arrin, M);

    // check derivatives

    single_dx(arrin, derivout, 1, params);
    
    save_hdf5_arr("./output/testdx.h5", derivout, M);

    single_dy(arrin, derivout, params);
    save_hdf5_arr("./output/testdy.h5", derivout, M);

    single_dy(derivout, scratch, params);
    single_dy(scratch, derivout, params);
    single_dy(derivout, scratch, params);

    save_hdf5_arr("./output/testd4y.h5", scratch, M);

    // test transform to physical space 
    to_cheby_physical(arrin, physout, scr, params);
    save_hdf5_arr("./output/testPhysicalT.h5", physout, shapefft);

    // test transform to spectral space
    to_cheby_spectral(physout, specout, scr, params);
    save_hdf5_arr("./output/testSpectralT.h5", specout, shapefft);

    // Test a convolution
    fft_cheby_convolve(arrin, arrin, specout, scr, params);

    save_hdf5_arr("./output/fft_convolve.h5", specout, M);

    // Test transforms from physical space array
    for (j=0; j<Mf; j++)
    {
	phystest[j] = tanh(j*M_PI/(Mf-1.));
    }

    to_cheby_spectral(phystest, specout2, scr, params);
    save_hdf5_arr("./output/phystest2.h5", phystest, shapefft);
    save_hdf5_arr("./output/testSpectralT2.h5", specout2, M);

    to_cheby_physical(specout2, physout, scr, params);
    save_hdf5_arr("./output/testPhysT4.h5", physout, shapefft);


    //Test repeated spectral transforms
    for (i=0; i<100; i++)
    {
	to_cheby_physical(arrin, physout, scr, params);
	to_cheby_spectral(physout, arrin, scr, params);
    }

    save_hdf5_arr("./output/testSpectralTR.h5", arrin, M);

    //garbage collection
    fftw_destroy_plan(phys_plan);
    fftw_destroy_plan(spec_plan);

    fftw_free(arrin);
    fftw_free(derivout);
    fftw_free(specout);
    fftw_free(specout2);
    fftw_free(scratch);

    free(physout);
    free(physout2);
    free(phystest);

    fftw_free(scr.scratch);
    fftw_free(scr.scratch2);
    fftw_free(scr.scratch3);
    fftw_free(scr.scratch4);
    fftw_free(scr.scratchin); 
    fftw_free(scr.scratchout);
    fftw_free(scr.scratchp1);
    fftw_free(scr.scratchp2);

    return 0;
}
