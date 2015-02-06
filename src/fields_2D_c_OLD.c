/* -------------------------------------------------------------------------- *
 *									      *
 *  fields_2D_c.c							      *
 *                                                                            *
 *  functions for 2D fields in C					      *
 *                                                                            *
 *                                                                            *
 * -------------------------------------------------------------------------- */

// Last modified: Wed 28 Jan 11:56:25 2015

#include"fields_2D_c.h"

// Functions

void save_state(FILE *fp, fftw_complex *arr,  flow_params cnsts)
{
    /*
     * This function takes a pointer to the first element of a 2D array and
     * uses values from it to fill a a file.  Read these files using the scipy
     * command: genfromtxt(filename, dtype='complex128')
     */

    int N = cnsts.N;
    int M = cnsts.M;
    int rows = (2*N+1);
    int cols = M;
    int i, j;

    for(i=0; i<rows; i++)
    {

	for(j=0; j<cols; j++)
	{
	    fprintf(fp, " (%e18+%e18j) ", creal(arr[ind(i, j)]),
					  cimag(arr[ind(i, j)]));
	}
	fprintf(fp, "\n");

    }
    
}

void load_state(FILE *fp, fftw_complex *arr,  flow_params cnsts)
{
    /* 
     * This function loads a 2D array of complex doubles in scipy format to a
     * 2D array.
     *
     */

    float tmpRe, tmpIm;

    int N = cnsts.N;
    int M = cnsts.M;
    int rows = (2*N+1);
    int cols = M;
    int i, j;

    for(i=0; i<rows; i++)
    {

	for(j=0; j<cols; j++)
	{
	    fscanf(fp, " (%e+%ej) ", &tmpRe, &tmpIm);
	    arr[ind(i, j)] = (double)tmpRe + (double)tmpIm * I;
	    //if(j==1) { printf(" (%e+%ej) ", creal(arr[i][j]), cimag(arr[i][j])); }
	}
	fscanf(fp, "\n");

    }
    
}

void load_operator(FILE *fp, fftw_complex *arr,  flow_params cnsts)
{
    /* 
     * This function loads a 2D array of complex doubles in scipy format to a
     * 2D array of size M by M.
     *
     */

    float tmpRe, tmpIm;

    int M = cnsts.M;
    int rows = M;
    int cols = M;
    int i, j;

    for(i=0; i<rows; i++)
    {

	for(j=0; j<cols; j++)
	{
	    fscanf(fp, " (%e+%ej) ", &tmpRe, &tmpIm);
	    arr[M*i + j] = (double)tmpRe + (double)tmpIm * I;
	    //if(j==1) { printf(" (%e+%ej) ", creal(arr[i][j]), cimag(arr[i][j])); }
	}
	fscanf(fp, "\n");

    }
    
}

void dx(fftw_complex *arrin, fftw_complex *arrout,  flow_params cnsts)
{
    int N = cnsts.N;
    int M = cnsts.M;
    double kx = cnsts.kx;
    int i, j;


    // For positive fourier modes
    for(i=0; i<N+1; i++)
    {
	//For all Chebyshev modes
	for(j=0; j<M; j++)
	{
	    arrout[ind(i, j)] = i*kx*I*arrin[ind(i, j)];
	}

    }

    // For conjugate fourier modes
    for(i=N+1; i<2*N+1; i++)
    {
	//For all Chebyshev modes
	for(j=0; j<M; j++)
	{
	    arrout[ind(i, j)] = -(2*N+1-i)*kx*I*arrin[ind(i, j)];
	}

    }
}

void dy(fftw_complex *arrin, fftw_complex *arrout,  flow_params cnsts)
{
    int N = cnsts.N;
    int M = cnsts.M;
    int i, j;


    // For all Fourier modes
    for(i=0; i<2*N+1; i++)
    {
	// Last 2 modes
	arrout[ind(i, M-1)] = 0;
	arrout[ind(i, M-2)] = 2*(M-1)*arrin[ind(i, M-1)];
	
	// For rest of the Chebyshev modes
	for(j=M-3; j>0; j=j-1)
	{
	    arrout[ind(i, j)] = 2*(j+1)*arrin[ind(i, j+1)] + arrout[ind(i, j+2)];
	}

	// Zeroth mode
	arrout[ind(i, 0)] = arrin[ind(i, 1)] + 0.5*arrout[ind(i, 2)];
    }
    
}

void to_physical(fftw_complex *arrin, fftw_complex *arrout,
		 fftw_complex *scratchin, fftw_complex *scratchout,
		 fftw_plan *phys_plan,  flow_params cnsts)
{
    int N = cnsts.N;
    int M = cnsts.M;
    int i,j;

    // take the complex conjugate, because we need both transforms to be forward.
    // factor of 2*N+1 is just here because my other method I am comparing with
    // has transforms in the opposite direction which pick up a normalisation
    // factor
    if (cnsts.dealiasing)
    {
        for (i=0; i<2*N/3 + 1; i++)
        {
            for (j=0; j<2*M/3; j++)
            {
        	scratch[ind(i,j)] = conj(arrin[ind(i,j)])/(2*N+1);
            }
        }
        for (i=2*N+1 - 2*N/3; i<2*N+1; i++)
        {
            for (j=0; j<2*M/3; j++)
            {
        	scratch[ind(i,j)] = conj(arrin[ind(i,j)])/(2*N+1);
            }
        }
	// zero off the rest of the fourier modes
	for(i=2*N/3 + 1; i<2*N+1 - 2*N/3; i++)
	{
	    for(j=0; j<M; j++)
	    {
		scratch[ind(i,j)] = 0;
	    }
	}

	// zero off the rest of the Chebyshev modes
	for(i=0; i<2*N+1; i++)
	{
	    for(j=2*M/3; j<M; j++)
	    {
		scratch[ind(i,j)] = 0;
	    }
	}
    }
    else
    {
	for (i=0; i<2*N+1; i++)
	{
	    for (j=0; j<M; j++)
	    {
		scratch[ind(i,j)] = conj(arrin[ind(i,j)])/(2*N+1);
	    }
	}
    }


    //out2D[M:, :] = out2D[M-2:0:-1, :]

    for (i=0; i<2*N+1; i++)
    {
	for (j=2; j<M; j++)
	{
	    scratch[ind(i, M-2+j)] = scratch[ind(i, M-j)];
	}

      //out2D[0, :] = 2*out2D[0, :]
	scratch[ind(i, 0)] = 2*scratch[ind(i, 0)];

      //out2D[M-1, :] = 2*out2D[M-1, :]
	scratch[ind(i, M-1)] = 2*scratch[ind(i, M-1)];
    }

    //perform the 2D fft.
    //out2D = 0.5*fftpack.fft2(out2D)
    
    fftw_execute(*phys_plan);
    
    for (i=0; i<2*N+1; i++)
    {
	for (j=0; j<M; j++)
	{
	    arrout[ind(i,j)] = 0.5*arrout[ind(i,j)];
	}
    }


}

void to_spectral(fftw_complex *arrin, fftw_complex *arrout,
		fftw_complex *scratch, fftw_plan *spec_plan,  flow_params cnsts)
{
    int N = cnsts.N;
    int M = cnsts.M;
    int i,j;

    // Perform the FFT across the x direction   

    // The first half contains the vector on the Gauss-Labatto points
    // out2D[:M, :] = real(in2D)
    for (i=0; i<2*N+1; i++)
    {
	for (j=0; j<M; j++)
	{
	    scratch[ind(i,j)] = arrin[ind(i,j)];
	}
    }

    // The second half contains the vector on the Gauss-Labatto points excluding
    // the first and last elements and in reverse order
    // out2D[M:, :] = out2D[M-2:0:-1, :]

    for (i=0; i<2*N+1; i++)
    {
	for (j=2; j<M; j++)
	{
	    scratch[ind(i, M-2+j)] = scratch[ind(i, M-j)];
	}
    }

    // Perform the transformation on this temporary vector
    // out2D = fftpack.fft2(out2D)
    fftw_execute(*spec_plan);

    // Renormalise and divide by c_k to convert to Chebyshev polynomials
    // if dealiasing make sure to zero off the inaccurate modes
    int k;
	for (i=1; i<N+1; i++)
	{

	    k = 2*N+1 - i;
	    printf("i = %d, k = %d", i, k);
	    for (j=0; j<M; j++)
	    {
		printf(" %e ", (creal(arrout[ind(i,j)]) - creal(arrout[ind(k,j)]))/
			        creal(arrout[ind(i,j)]));
		printf(" %ej ", (cimag(arrout[ind(i,j)]) + cimag(arrout[ind(k,j)]))/
				cimag(arrout[ind(i,j)]));
		printf("\t %d ", creal(arrout[ind(k,j)])==creal(arrout[ind(i,j)]));
		printf(" %d ", cimag(arrout[ind(k,j)])==-cimag(arrout[ind(i,j)]));
		printf("\n");
	    }

	}
    
    if (cnsts.dealiasing)
    {

        for (i=0; i<2*N/3 + 1; i++)
        {
	  // out2D[0, :] = (0.5/(M-1.0))*out2D[0, :]
	    arrout[ind(i,0)] = (0.5/(M-1.0)) * arrout[ind(i,0)]; 

            for (j=1; j<2*M/3; j++)
            {
		// out2D[1:M-1, :] = (1.0/(M-1.0))*out2D[1:M-1, :]
		arrout[ind(i,j)] = (1.0/(M-1.0))*arrout[ind(i,j)];
            }
        }

        for (i=2*N+1 - 2*N/3; i<2*N+1; i++)
        {
            for (j=1; j<2*M/3; j++)
            {
		// out2D[1:M-1, :] = (1.0/(M-1.0))*out2D[1:M-1, :]
		arrout[ind(i,j)] = (1.0/(M-1.0))*arrout[ind(i,j)];
		//if (i==8)
		//{
		//printf(" %e+%ej ", creal(arrout[ind(i,j)]), cimag(arrout[ind(i,j)]));
		//printf("\n");
		//}
            }
        }

	// zero off the aliased fourier modes
	for(i=2*N/3 + 1; i<2*N+1 - 2*N/3; i++)
	{
	    for(j=0; j<M; j++)
	    {
		arrout[ind(i,j)] = 0;
	    }
	}

	// zero off the aliased Chebyshev modes
	for(i=0; i<2*N+1; i++)
	{
	    for(j=2*M/3; j<M; j++)
	    {
		arrout[ind(i,j)] = 0;
	    }
	    for(j=0; j<M; j++)
	    {
	    }
	}

    }

    else
    {
	for (i=0; i<2*N+1; i++)
	{
	    // out2D[0, :] = (0.5/(M-1.0))*out2D[0, :]
	    arrout[ind(i,0)] = (0.5/(M-1.0)) * arrout[ind(i,0)]; 

	    for (j=1; j<M-1; j++)
	    {
		// out2D[1:M-1, :] = (1.0/(M-1.0))*out2D[1:M-1, :]
		arrout[ind(i,j)] = (1.0/(M-1.0))*arrout[ind(i,j)];
	    }

	    //out2D[M-1, :] = (0.5/(M-1.0))*out2D[M-1, :]
	    arrout[ind(i,M-1)] = (0.5/(M-1.0)) * arrout[ind(i,M-1)]; 
	}
    }

}
