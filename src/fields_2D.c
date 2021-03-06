/* -------------------------------------------------------------------------- *
 *									      *
 *  fields_2D_c.c							      *
 *                                                                            *
 *  functions for 2D fields in C					      *
 *                                                                            *
 *                                                                            *
 * -------------------------------------------------------------------------- */

// Last modified: Wed 30 Sep 13:00:57 2015

#include"fields_2D.h"

// Functions

void dx(fftw_complex *arrin, fftw_complex *arrout,  flow_params cnsts)
{
    int N = cnsts.N;
    int M = cnsts.M;
    double kx = cnsts.kx;
    int i, j;


    // For positive fourier modes
    for(i=0; i<N+1; i++)
    {
	//printf("%d\n",i);
	//For all Chebyshev modes
	for(j=0; j<M; j++)
	{
	    arrout[ind(i, j)] = i*kx*I*arrin[ind(i, j)];
	}

    }
}

void d2x(fftw_complex *arrin, fftw_complex *arrout,  flow_params cnsts)
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
	    arrout[ind(i, j)] = -pow(i*kx, 2)*arrin[ind(i, j)];
	}

    }
}

void d4x(fftw_complex *arrin, fftw_complex *arrout,  flow_params cnsts)
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
	    arrout[ind(i, j)] = pow(i*kx, 4)*arrin[ind(i, j)];
	}

    }
}

void dy(fftw_complex *arrin, fftw_complex *arrout,  flow_params cnsts)
{
    int N = cnsts.N;
    int M = cnsts.M;
    int i, j;

    // For all Fourier modes
    for(i=0; i<N+1; i++)
    {
	// Last 2 modes
	arrout[ind(i, M-1)] = 0;
	arrout[ind(i, M-2)] = 2.0*(M-1.0)*arrin[ind(i, M-1)];
	
	// For rest of the Chebyshev modes
	for(j=M-3; j>0; j=j-1)
	{
	    arrout[ind(i, j)] = 2.0*(j+1.0)*arrin[ind(i, j+1)] + arrout[ind(i, j+2)];
	}

	// Zeroth mode
	arrout[ind(i, 0)] = arrin[ind(i, 1)] + 0.5*arrout[ind(i, 2)];
    }
    
}

void matdy(fftw_complex *matarr, flow_params cnsts)
{
    int i, j;
    int M = cnsts.M;

    for (i=0; i<M*M; i++)
    {
	matarr[i] = 0;
    }


    for (j=1; j<M; j=j+2)
    {
	matarr[j] = j;
    }

    for (i=1; i<M; i++)
    {
	for (j=i+1; j<M; j=j+2)
	{
	    matarr[M*i + j] = 2*j;
	}
    }

}

void to_physical_1(fftw_complex *arrin, fftw_complex *arrout,
		 fftw_complex *scratchFin, fftw_complex *scratchFout,
		 fftw_complex *scratchCin, fftw_complex *scratchCout,
		 fftw_plan *phys_fou_plan, fftw_plan *phys_cheb_plan,
		 flow_params cnsts)
{
    int i,j;
    int M = cnsts.M;
    int Mf = cnsts.Mf;
    int N = cnsts.N;
    int Nf = cnsts.Nf;

    // Do the Fourier Transfrom

    // for each Chebyshev mode, fourier transform
    for (j=0; j<M; j++)
    {

        // copy the fourier mode into scratch
        if (cnsts.dealiasing)
        {
            scratchFin[0] = arrin[ind(0,j)];
            for (i=1; i<N+1; i++)
            {
        	scratchFin[i] = arrin[ind(i,j)];
        	scratchFin[2*Nf+1-i] = arrin[ind(2*N+1-i,j)];
            }
            // zero off the rest of the modes
            for(i=N+1; i<2*Nf+1-N; i++)
            {
        	scratchFin[i] = 0;
            }

        } else
        {
            for (i=0; i<2*N+1; i++)
            {
        	scratchFin[i] = arrin[ind(i,j)];
            }
        }

        fftw_execute(*phys_fou_plan);

        // copy the transformed array into the output for now
        for (i=0; i<2*Nf+1; i++)
        {
            arrout[indfft(i,j)] = scratchFout[i];
        }
    
    }

    // For each x column, do the Chebyshev transform

    for(i=0; i<2*Nf+1; i++)
    {

        for (j=0; j<M; j++)
        {
            scratchCin[j] = 0.5*arrout[indfft(i,j)];
        }

        if (cnsts.dealiasing)
        {
            // zero off the rest of the Chebyshev modes
            for(j=M; j<Mf; j++)
            {
        	scratchCin[j] = 0;
            }
        }


        //out2D[M:, :] = out2D[M-2:0:-1, :]

        for (j=2; j<Mf; j++)
        {
            scratchCin[Mf-2+j] = scratchCin[Mf-j];
        }

        //out2D[0, :] = 2*out2D[0, :]
        scratchCin[0] = 2*scratchCin[0];

        //out2D[Mf-1, :] = 2*out2D[Mf-1, :]
        scratchCin[Mf-1] = 2*scratchCin[Mf-1];


        //perform the 2D fft.
        //out2D = fftpack.fft2(out2D)

        fftw_execute(*phys_cheb_plan);

        for (j=0; j<Mf; j++)
        {
            arrout[indfft(i,j)] = scratchCout[j];
        }
    }
}

void to_physical_r(complex_d *arrin, double *arrout, flow_scratch scr,
	flow_params cnsts)
{
    int N = cnsts.N;
    int M = cnsts.M;
    int Nf = cnsts.Nf;
    int Mf = cnsts.Mf;
    int i,j;

    // This uses an inverse FT. sign is +1.
    // factor of 2*N+1 is just here because my other method I am comparing with
    // has transforms in the inverse direction which pick up a normalisation
    // factor. 

    if (cnsts.dealiasing)
    {
        for (j=0; j<M; j++)
        {
            scr.scratchin[indfft(0,j)] = arrin[ind(0,j)];
        }

        for (i=1; i<N+1; i++)
        {
            for (j=0; j<M; j++)
            {
        	scr.scratchin[indfft(i,j)] = arrin[ind(i,j)];
        	scr.scratchin[indfft(2*Nf+1-i, j)] =  conj(arrin[ind(i,j)]);
           }
        }

        // zero off the rest of the fourier modes
        for(i=N+1; i<2*Nf+1-N; i++)
        {
            for(j=0; j<Mf; j++)
            {
        	scr.scratchin[indfft(i,j)] = 0;
            }
        }

        // zero off the rest of the Chebyshev modes
        for(i=0; i<2*Nf+1; i++)
        {
            for(j=M; j<Mf; j++)
            {
        	scr.scratchin[indfft(i,j)] = 0;
            }
        }
    }
    else
    {
        for (j=0; j<M; j++)
        {
            scr.scratchin[indfft(0,j)] = arrin[ind(0,j)];
        }

        for (i=1; i<N+1; i++)
        {
            for (j=0; j<M; j++)
            {
        	scr.scratchin[indfft(i,j)] = arrin[ind(i,j)];
        	scr.scratchin[indfft(2*Nf+1-i, j)] =  conj(arrin[ind(i,j)]);
           }
        }
    }


    //out2D[M:, :] = out2D[M-2:0:-1, :]

    for (i=0; i<2*Nf+1; i++)
    {
        for (j=2; j<Mf; j++)
        {
            scr.scratchin[indfft(i, Mf-2+j)] = scr.scratchin[indfft(i, Mf-j)];
        }

        //out2D[0, :] = 2*out2D[0, :]
        scr.scratchin[indfft(i, 0)] = 2*scr.scratchin[indfft(i, 0)];

        //out2D[Mf-1, :] = 2*out2D[Mf-1, :]
        scr.scratchin[indfft(i, Mf-1)] = 2*scr.scratchin[indfft(i, Mf-1)];
    }

    ////perform the 2D ifft?.
    ////out2D = 0.5*fftpack.ifft2(out2D)

    fftw_execute(*scr.phys_plan);

    for (i=0; i<2*Nf+1; i++)
    {
        for (j=0; j<Mf; j++)
        {
            arrout[indfft(i,j)] = 0.5*creal(scr.scratchout[indfft(i,j)]);
        }
    }
}

void to_spectral_1(fftw_complex *arrin, fftw_complex *arrout, fftw_complex *scratch,
		 fftw_complex *scratchFin, fftw_complex *scratchFout,
		 fftw_complex *scratchCin, fftw_complex *scratchCout,
		 fftw_plan *spec_fou_plan, fftw_plan *spec_cheb_plan,
		 flow_params cnsts)
{
    int N = cnsts.N;
    int M = cnsts.M;
    int Nf = cnsts.Nf;
    int Mf = cnsts.Mf;
    double normalise = 2*Nf+1;
    int i=0;
    int j=0;

    // zero out the interim array

    for (i=0; i<2*Nf+1; i++)
    {
	for (j=0; j<2*Mf-2; j++)
	{
	    scratch[indfft(i,j)] = 0;
	}
    }

    // Perform the Chebyshev transforms
    
    for (i=0; i<2*Nf+1; i++)
    {

	for (j=0; j<Mf; j++)
	{
	    scratchCin[j] = arrin[indfft(i,j)] / normalise;
	    //scratch[indfft(i,j)] = arrin[indfft(i,j)] ;// normalise;
	}

	// The second half contains the vector on the Gauss-Labatto points excluding
	// the first and last elements and in reverse order
	// out2D[M:, :] = out2D[M-2:0:-1, :]

	for (j=2; j<Mf; j++)
	{
	    scratchCin[Mf-2+j] = scratchCin[Mf-j];
	    //scratch[indfft(i,Mf-2+j)] = scratch[indfft(i, Mf-j)];
	}

	// Perform the transformation on this temporary vector
	// out2D = fftpack.fft2(out2D)
	fftw_execute(*spec_cheb_plan);

	if (cnsts.dealiasing)
	{
	    // out2D[0, :] = (0.5/(M-1.0))*out2D[0, :]
	    scratch[indfft(i,0)] = (0.5/(Mf-1.0)) * scratchCout[0];

	    // when dealiasing this will be still x 1.0 not 0.5, because it isn't
	    // the last element in the transformed array
	    for (j=1; j<M; j++)
	    {
		// out2D[1:M-1, :] = (1.0/(M-1.0))*out2D[1:M-1, :]
		scratch[indfft(i,j)] = (1.0/(Mf-1.0))*scratchCout[j];
	    }

	}

	else
	{
	    // out2D[0, :] = (0.5/(M-1.0))*out2D[0, :]
	    scratch[indfft(i,0)] = (0.5/(M-1.0)) * scratchCout[0]; 

	    for (j=1; j<M-1; j++)
	    {
		// out2D[1:M-1, :] = (1.0/(M-1.0))*out2D[1:M-1, :]
		scratch[indfft(i,j)] = (1.0/(M-1.0))*scratchCout[j];
	    }

	    //out2D[M-1, :] = (0.5/(M-1.0))*out2D[M-1, :]
	    scratch[indfft(i,M-1)] = (0.5/(M-1.0)) * scratchCout[M-1]; 
	}
    }

    // Perform the Fourier transform

    for (j=0; j<M; j++)
    {
        // copy the Chebyshev mode from the interim array
        for (i=0; i<2*Nf+1;  i++)
        {
            scratchFin[i] = scratch[indfft(i,j)];
        }

        fftw_execute(*spec_fou_plan);

        // copy the Chebyshev mode into the output

        if (cnsts.dealiasing)
        {
            arrout[ind(0,j)] = scratchFout[0];
            for (i=1; i<N+1; i++)
            {
        	arrout[ind(i,j)] = scratchFout[i];
        	arrout[ind(2*N+1-i,j)] = scratchFout[2*Nf+1-i];
            }

        } else
        {
            for (i=0; i<2*N+1; i++)
            {
        	arrout[ind(i,j)] = scratchFout[i];
            }
        }
    }

    // // skip the Fourier transform, just output the chebyshev transform
    // for (j=0; j<M; j++)
    // {
    //         for (i=0; i<2*N+1; i++)
    //         {
    //     	arrout[ind(i,j)] = scratch[indfft(i,j)];
    //         }
    // }
}

void to_spectral_r(double *arrin, complex_d *arrout,
	flow_scratch scr,  flow_params cnsts)
{
    int N = cnsts.N;
    int M = cnsts.M;
    int Nf = cnsts.Nf;
    int Mf = cnsts.Mf;
    int i,j;

    fftw_complex normalise = 2*Nf+1;

    // Perform the FFT across the x direction   

    // The first half contains the vector on the Gauss-Labatto points
    // out2D[:M, :] = real(in2D)
    // include normalisation here so that spectral space has same normalisation as it	   
    // started with.

    for (i=0; i<2*Nf+1; i++)
    {
	for (j=0; j<Mf; j++)
	{
	    scr.scratchin[indfft(i,j)] = arrin[indfft(i,j)]/normalise;
	}
    }

    // The second half contains the vector on the Gauss-Labatto points excluding
    // the first and last elements and in reverse order
    // out2D[M:, :] = out2D[M-2:0:-1, :]

    for (i=0; i<2*Nf+1; i++)
    {
	for (j=2; j<Mf; j++)
	{
	    scr.scratchin[indfft(i, Mf-2+j)] = scr.scratchin[indfft(i, Mf-j)];
	}
    }

    // Perform the transformation on this temporary vector
    // out2D = fftpack.fft2(out2D)
    fftw_execute(*scr.spec_plan);
    if (cnsts.dealiasing)
    {
	// copy zeroth and positive modes into output

	// out2D[0, :] = (0.5/(M-1.0))*out2D[0, :]
	arrout[ind(0,0)] = (0.5/(Mf-1.0)) * scr.scratchout[indfft(0,0)]; 

	// when dealiasing this will be still x 1.0 not 0.5, because it isn't
	// the last element in the transformed array
	for (j=1; j<M; j++)
	{
	    // out2D[1:M-1, :] = (1.0/(M-1.0))*out2D[1:M-1, :]
	    arrout[ind(0,j)] = (1.0/(Mf-1.0))*scr.scratchout[indfft(0,j)];
	}


	for (i=1; i<N+1; i++)
	{
	    arrout[ind(i,0)] = (0.5/(Mf-1.0)) * scr.scratchout[indfft(i,0)]; 

	    // when dealiasing this will be still x 1.0 not 0.5, because it isn't
	    // the last element in the transformed array
	    for (j=1; j<M; j++)
	    {
		arrout[ind(i,j)] = (1.0/(Mf-1.0))*scr.scratchout[indfft(i,j)];
	    }
	}
    }

    else
    {
	for (i=0; i<N+1; i++)
	{
	    // out2D[0, :] = (0.5/(M-1.0))*out2D[0, :]
	    arrout[ind(i,0)] = (0.5/(M-1.0)) * scr.scratchout[indfft(i,0)]; 

	    for (j=1; j<M-1; j++)
	    {
		// out2D[1:M-1, :] = (1.0/(M-1.0))*out2D[1:M-1, :]
		arrout[ind(i,j)] = (1.0/(M-1.0))*scr.scratchout[indfft(i,j)];
	    }

	    //out2D[M-1, :] = (0.5/(M-1.0))*out2D[M-1, :]
	    arrout[ind(i,M-1)] = (0.5/(M-1.0)) * scr.scratchout[indfft(i,M-1)]; 
	}
    }

}

void fft_convolve_r(complex_d *arr1, complex_d *arr2, complex_d *arrout,
		    flow_scratch scr, flow_params cnsts)
{
    // all scratch arrays must be different
    // out array may be the same as one in array

    int Nf = cnsts.Nf;
    int Mf = cnsts.Mf;

    int i, j;

    to_physical_r(arr1, scr.scratchp1, scr, cnsts);
    to_physical_r(arr2, scr.scratchp2, scr, cnsts);

    for (i=0; i<(2*Nf+1); i++)
    {
	for(j=0; j<Mf; j++)
	{
	    scr.scratchp1[indfft(i,j)] = scr.scratchp2[indfft(i,j)]*scr.scratchp1[indfft(i,j)];
	}
    }

    to_spectral_r(scr.scratchp1, arrout, scr, cnsts);
}

double calc_KE_mode(fftw_complex *u, fftw_complex *v, int n, flow_params cnsts)
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

int trC_tensor(complex_d *cij, complex_d *trC, flow_scratch scr, flow_params cnsts)
{
    // Calculate the trace of the conformation tensor. This is both a measure
    // of the polymer stretch and a useful thing to be able to do for a FENE
    // code.

    int i=0;
    int N = cnsts.N;
    int M = cnsts.M;
    int Nf = cnsts.Nf;
    int Mf = cnsts.Mf;

    int posdefck=1;

    // Form a sum of those stresses at every point in the domain
    to_physical_r(&cij[0], scr.scratchp1, scr, cnsts);
    to_physical_r(&cij[(N+1)*M], scr.scratchp2, scr, cnsts);

    for(i=0;i<(2*Nf+1)*Mf; i++)
    {
	scr.scratchp1[i] += scr.scratchp2[i];
    }

    // check positive definite everywhere

    for (i=0; i<(2*Nf+1)*Mf; i++) 
    {
	if (scr.scratchp1[i]<0) 
	{
	    posdefck = 0;
	}
    }


    // transform trC back to spectral space
    to_spectral_r(scr.scratchp1, trC, scr, cnsts);

    return posdefck;

}

void diagonalised_C(complex_d *cij, complex_d *ecij, double *rcij,
	flow_scratch scr, flow_params cnsts)
{
    // Calculate the trace of the diagonalised conformation tensor. This is 
    // the root of the polymer stretch ??.
    
    int i =0; 
    int N = cnsts.N;
    int M = cnsts.M;
    int Nf = cnsts.Nf;
    int Mf = cnsts.Mf;

    // First calculate the real space stresses
    to_physical_r(&cij[0], &rcij[0], scr, cnsts);
    to_physical_r(&cij[M*(N+1)], &rcij[Mf*(2*Nf+1)], scr, cnsts);
    to_physical_r(&cij[2*M*(N+1)], &rcij[2*Mf*(2*Nf+1)], scr, cnsts);

    // Find the eigenvalues of the stress tensor at every point in the domain.
    for (i=0; i<Mf*(2*Nf+1);i++)
    {
	scr.scratchp1[i] = rcij[i] + rcij[Mf*(2*Nf+1) + i];
	scr.scratchp1[i] += sqrt(
			    pow(rcij[i] + rcij[Mf*(2*Nf+1) + i],2)
			    - 4*(rcij[i] * rcij[Mf*(2*Nf+1) + i] 
				    - rcij[2*Mf*(2*Nf+1) + i]));
	scr.scratchp1[i] *= 0.5;

	scr.scratchp2[i] = rcij[i] + rcij[Mf*(2*Nf+1) + i]; 
	scr.scratchp2[i] -= sqrt(
			    pow(rcij[i] + rcij[Mf*(2*Nf+1) + i],2)
			    - 4*(rcij[i] * rcij[Mf*(2*Nf+1) + i] 
				    - rcij[2*Mf*(2*Nf+1) + i]));
	scr.scratchp2[i] *= 0.5;

    }

    to_spectral_r(scr.scratchp1, &ecij[0], scr, cnsts);
    to_spectral_r(scr.scratchp2, &ecij[M*(N+1)], scr, cnsts);

}

double calc_EE_mode(complex_d *trC, int n, flow_params cnsts)
{
    // Use the (diagonalised) stress (??) to calculate the Trace of C
    // (?) and integrate it over y for the mode in question to get the Elastic
    // energy
    
    int j=0;
    int M=cnsts.M;
    double Wi = cnsts.Wi;

    complex_d EE=0;


    for (j=0; j<M; j+=2)
    {

	EE += (2. / (1.-j*j)) * (trC[ind(n,0)]);
    }

    if (n == 0)
    {
	// subtract the trace of identity matrix integrated over the domain
	return 0.5*creal(EE-4.0);
    } else {
	return creal(EE);
    }


    return EE/Wi;
}
