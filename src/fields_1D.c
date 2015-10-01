/* -------------------------------------------------------------------------- *
 *									      *
 *  fields_2D_c.c							      *
 *                                                                            *
 *  functions for 2D fields in C					      *
 *                                                                            *
 *                                                                            *
 * -------------------------------------------------------------------------- */

// Last modified: Thu  1 Oct 16:50:51 2015

#include"fields_1D.h"

// Functions

void single_dx(fftw_complex *arrin, fftw_complex *arrout, int fou, flow_params cnsts)
{
    int M = cnsts.M;
    double kx = cnsts.kx;
    int j=0;

    //For all Chebyshev modes
    for(j=0; j<M; j++)
    {
	arrout[j] = fou*kx*I*arrin[j];
    }
}

void single_d2x(fftw_complex *arrin, fftw_complex *arrout, int fou, flow_params cnsts)
{
    int M = cnsts.M;
    double kx = cnsts.kx;
    int j=0;


    //For all Chebyshev modes
    for(j=0; j<M; j++)
    {
	arrout[j] = -pow(fou*kx, 2)*arrin[j];
    }
}

void single_d4x(fftw_complex *arrin, fftw_complex *arrout, int fou, flow_params cnsts)
{
    int M = cnsts.M;
    double kx = cnsts.kx;
    int j=0;


    //For all Chebyshev modes
    for(j=0; j<M; j++)
    {
	arrout[j] = pow(fou*kx, 4)*arrin[j];
    }
}

void single_dy(fftw_complex *arrin, fftw_complex *arrout, flow_params cnsts)
{
    int M = cnsts.M;
    int j=0;

    // Last 2 modes
    arrout[M-1] = 0;
    arrout[M-2] = 2.0*(M-1.0)*arrin[M-1];
    
    // For rest of the Chebyshev modes
    for(j=M-3; j>0; j=j-1)
    {
	arrout[j] = 2.0*(j+1.0)*arrin[j+1] + arrout[j+2];
    }

    // Zeroth mode
    arrout[0] = arrin[1] + 0.5*arrout[2];

}

void to_cheby_physical(complex_d *arrin, complex_d *arrout, lin_flow_scratch scr,
	flow_params cnsts)
{

    // takes a 1D vector of Chebyshevs and transforms them to real space
    // preserving the imaginary part.  This uses an inverse FT. sign is +1.
    
    int M = cnsts.M;
    int Mf = cnsts.Mf;
    int j=0;


    // Read in the real part of the input array, transform that first
    for (j=0; j<M; j++)
    {
	scr.scratchin[j] = creal(arrin[j]);
    }

    // zero off the rest of the Chebyshev modes
    for(j=M; j<Mf; j++)
    {
	scr.scratchin[j] = 0;
    }

    //out2D[M:, :] = out2D[M-2:0:-1, :]

    for (j=2; j<Mf; j++)
    {
	scr.scratchin[Mf-2+j] = scr.scratchin[Mf-j];
    }

    //out2D[0, :] = 2*out2D[0, :]
    scr.scratchin[0] = 2*scr.scratchin[0];

    //out2D[Mf-1, :] = 2*out2D[Mf-1, :]
    scr.scratchin[Mf-1] = 2*scr.scratchin[Mf-1];

    ////perform the 2D ifft?.
    ////out2D = 0.5*fftpack.ifft2(out2D)

    fftw_execute(*scr.phys_plan);

    for (j=0; j<Mf; j++)
    {
	arrout[j] = 0.5*creal(scr.scratchout[j]);
    }

    // Now read in the imaginary part of the input array, transform that 
    for (j=0; j<M; j++)
    {
	scr.scratchin[j] = cimag(arrin[j]);
    }

    // zero off the rest of the Chebyshev modes
    for(j=M; j<Mf; j++)
    {
	scr.scratchin[j] = 0;
    }

    //out2D[M:, :] = out2D[M-2:0:-1, :]

    for (j=2; j<Mf; j++)
    {
	scr.scratchin[Mf-2+j] = scr.scratchin[Mf-j];
    }

    //out2D[0, :] = 2*out2D[0, :]
    scr.scratchin[0] = 2*scr.scratchin[0];

    //out2D[Mf-1, :] = 2*out2D[Mf-1, :]
    scr.scratchin[Mf-1] = 2*scr.scratchin[Mf-1];

    ////perform the 2D ifft?.
    ////out2D = 0.5*fftpack.ifft2(out2D)

    fftw_execute(*scr.phys_plan);

    for (j=0; j<Mf; j++)
    {
	arrout[j] += I * 0.5*creal(scr.scratchout[j]);
    }

}

void to_cheby_spectral(complex_d *arrin, complex_d *arrout,
	lin_flow_scratch scr,  flow_params cnsts)
{
    int M = cnsts.M;
    int Mf = cnsts.Mf;
    int j=0;

    // The first half contains the vector on the Gauss-Labatto points
    // out2D[:M, :] = real(in2D)

    // Transform real part of the array
    for (j=0; j<Mf; j++)
    {
	scr.scratchin[j] = creal(arrin[j]);
    }

    // The second half contains the vector on the Gauss-Labatto points excluding
    // the first and last elements and in reverse order
    // out2D[M:, :] = out2D[M-2:0:-1, :]

    for (j=2; j<Mf; j++)
    {
	scr.scratchin[Mf-2+j] = scr.scratchin[Mf-j];
    }

    // Perform the transformation on this temporary vector
    // out2D = fftpack.fft2(out2D)
    fftw_execute(*scr.spec_plan);

    if (cnsts.dealiasing)
    {
	// copy zeroth and positive modes into output

	// out2D[0, :] = (0.5/(M-1.0))*out2D[0, :]
	arrout[0] = (0.5/(Mf-1.0)) * creal(scr.scratchout[0]); 

	// when dealiasing this will be still x 1.0 not 0.5, because it isn't
	// the last element in the transformed array
	for (j=1; j<M; j++)
	{
	    // out2D[1:M-1, :] = (1.0/(M-1.0))*out2D[1:M-1, :]
	    arrout[j] = (1.0/(Mf-1.0))*creal(scr.scratchout[j]);
	}
    }

    else
    {
	// out2D[0, :] = (0.5/(M-1.0))*out2D[0, :]
	arrout[0] = (0.5/(M-1.0)) * creal(scr.scratchout[0]); 

	for (j=1; j<M-1; j++)
	{
	    // out2D[1:M-1, :] = (1.0/(M-1.0))*out2D[1:M-1, :]
	    arrout[j] = (1.0/(M-1.0))*creal(scr.scratchout[j]);
	}

	//out2D[M-1, :] = (0.5/(M-1.0))*out2D[M-1, :]
	arrout[M-1] = (0.5/(M-1.0)) * creal(scr.scratchout[M-1]); 
    }

    // Transform imaginary part of the array
    for (j=0; j<Mf; j++)
    {
	scr.scratchin[j] = cimag(arrin[j]);
    }

    // The second half contains the vector on the Gauss-Labatto points excluding
    // the first and last elements and in reverse order
    // out2D[M:, :] = out2D[M-2:0:-1, :]

    for (j=2; j<Mf; j++)
    {
	scr.scratchin[Mf-2+j] = scr.scratchin[Mf-j];
    }

    // Perform the transformation on this temporary vector
    // out2D = fftpack.fft2(out2D)
    fftw_execute(*scr.spec_plan);


    if (cnsts.dealiasing)
    {
	// copy zeroth and positive modes into output

	// out2D[0, :] = (0.5/(M-1.0))*out2D[0, :]
	arrout[0] += I * (0.5/(Mf-1.0)) * creal(scr.scratchout[0]); 

	// when dealiasing this will be still x 1.0 not 0.5, because it isn't
	// the last element in the transformed array
	for (j=1; j<M; j++)
	{
	    // out2D[1:M-1, :] = (1.0/(M-1.0))*out2D[1:M-1, :]
	    arrout[j] += I * (1.0/(Mf-1.0)) * creal(scr.scratchout[j]);
	}
    }

    else
    {
	// out2D[0, :] = (0.5/(M-1.0))*out2D[0, :]
	arrout[0] += I * (0.5/(M-1.0)) * creal(scr.scratchout[0]); 

	for (j=1; j<M-1; j++)
	{
	    // out2D[1:M-1, :] = (1.0/(M-1.0))*out2D[1:M-1, :]
	    arrout[j] += I * (1.0/(M-1.0)) * creal(scr.scratchout[j]);
	}

	//out2D[M-1, :] = (0.5/(M-1.0))*out2D[M-1, :]
	arrout[M-1] += I * (0.5/(M-1.0)) * creal(scr.scratchout[M-1]); 
    }

}

void fft_cheby_convolve(complex_d *arr1, complex_d *arr2, complex_d *arrout,
		    lin_flow_scratch scr, flow_params cnsts)
{
    // all scratch arrays must be different
    // out array may be the same as one in array

    int Mf = cnsts.Mf;

    int j=0;

    to_cheby_physical(arr1, scr.scratchp1, scr, cnsts);
    to_cheby_physical(arr2, scr.scratchp2, scr, cnsts);

    for(j=0; j<Mf; j++)
    {
	scr.scratchp1[j] = scr.scratchp2[j]*scr.scratchp1[j];
    }

    to_cheby_spectral(scr.scratchp1, arrout, scr, cnsts);
}

double calc_cheby_KE_mode(fftw_complex *u, fftw_complex *v, int n, flow_params cnsts)
{
    double KE = 0;
    int i=0;
    int m=0;
    int p=0;
    complex_d usq=0;
    complex_d vsq=0;
    complex_d tmpu=0;
    complex_d tmpv=0;
    int M=cnsts.M;

	for (i=0; i<M; i+=2)
	{
	    usq = 0;
	    vsq = 0;

	    for (m=i-M+1; m<M; m++)
	    {
		p = abs(i-m);

		tmpu = u[ind(n,p)];
		tmpv = v[ind(n,p)];

		if (p==0)
		{
		    tmpu *= 2.0;
		    tmpv *= 2.0;
		}

		tmpu *= conj(u[ind(n,abs(m))]);
		tmpv *= conj(v[ind(n,abs(m))]);

		if (abs(m)==0)
		{
		    tmpu *= 2.0;
		    tmpv *= 2.0;
		}

		if (i==0)
		{
		    usq += 0.25*tmpu;
		    vsq += 0.25*tmpv;
		} else
		{
		    usq += 0.5*tmpu;
		    vsq += 0.5*tmpv;
		}

	    }

	    KE += (2. / (1.-i*i)) * usq;
	    KE += (2. / (1.-i*i)) * vsq;

	}

	if (n == 0)
	{
	    return 0.5*creal(KE);
	} else {
	    return creal(KE);
	}
}
