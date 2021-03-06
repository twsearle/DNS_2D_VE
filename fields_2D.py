###############################################################################
#
#   fields_2D.py 
#   module for 2D fields and associated methods.
#
#
###############################################################################

"""
Module with functions for working with 3D fields.

Layout of a field:
------------------
    
    Fields should be 2D numpy arrays, a[y,x] where y is the Chebyshev
    direction and x is a fourier direction. 

Functions:
----------

set_constants:
    returns a dictionary containing the constants to be passed around the
    program.

dy:
    First Chebyshev direction derivative of a field using Orszag's method

dyy:
    Second Chebyshev direction derivative of a field using Orszag's method

d3y:
    Third Chebyshev direction derivative of a field using Orszag's method

d4y:
    Third Chebyshev direction derivative of a field using Orszag's method

dx: 
    First Fourier derivative.

dxx: 
    Second Fourier derivative.

d3x: 
    Second Fourier derivative.

d4x: 
    Second Fourier derivative.

dxy:
    Chebyshev + Fourier derivative.

dxyy:
    2 Chebyshev + Fourier derivative.

dxxy:
    Chebyshev + 2 Fourier derivative.

dxyyy:
    3 Chebyshev + Fourier derivative.

dxxyy:
    2 Chebyshev + 2 Fourier derivative.

dxxxy:
    Chebyshev + 3 Fourier derivative.

biharmonic:
    Calculate the biharmonic (d4/dx4 + d4/dy4 + 2 d2/d2x2y) of a field.

to_physical:
    Transforms a field to physical space, onto GL + uniform 2D grid ready for
    multiplication.

to_spectral:
    Transforms a field to spectral space, onto Chebyshev + Fourier coeff.

Unit Tests:
-------------

test_diff:
    tests the differentiation methods.

test_prods:
    tests the transform methods for products of fields.

Unit Testing Notes:
-------------------

* There looked like there was a problem with the transformation routines, turns
out that the x grid used for the uniform realspace analytic version was not
quite right - it has to end sligntly before 2pi because kx at 0 = that at 2*pi
so the fft doesn't bother including it. 

"""

### MODULES ###

from scipy import *
from numpy.random import rand
from scipy import optimize, linalg, special
import numpy as np

from numpy.fft import fftshift, ifftshift

import matplotlib.pyplot as plt
#from scipy.fftpack import dct as dct
import subprocess
import cPickle as pickle
import h5py

# IF YOU WANT TO UNCOMMENT THE TESTS, YOU WILL NEED THIS PACKAGE FROM MY
# BITBUCKET ACCOUNT
#import TobySpectralMethods as tsm

### FUNCTIONS ###

def set_constants(M=16, N=16, 
                  kx=pi, Ly=2,
                  Re=400, Wi=1e-5, beta=1.0,
                  epsJ=1e-6, 
                  dealiasing=False):
    """
    returns a dictionary containing the constants to be passed around the
    program.
    """

    if dealiasing:
        Mf = (3*M)/2
        Nf = (3*N)/2 + 1
    else:
        Mf = M
        Nf = N

    Lx = 2*pi / kx

    return {'M':M, 'N':N, 'Mf':Mf, 'Nf':Nf, 'Lx':Lx, 'Ly':Ly, 
            'kx':kx, 'Re':Re, 'Wi':Wi, 'beta':beta,
            'epsJ':epsJ, 'dealiasing':dealiasing}
    
def mk_diff_x(CNSTS):
    """Make matrix to do fourier differentiation wrt x."""

    M = CNSTS['M']
    N = CNSTS['N']
    kx = CNSTS['kx']

    MDX = zeros( ((2*N+1)*M, (2*N+1)*M), dtype='complex')

    n = -N
    for i in range(0, (2*N+1)*M, M):
        MDX[i:i+M, i:i+M] = eye(M, M, dtype='complex')*n*kx*1.j
        n += 1
    del n, i
    return MDX

def single_dy(cSpec, CNSTS):
    """
    Efficient (ignoring the slowness of loops) way of computing the Chebyshev
    derivative of an array of Chebyshev coefficients
    """

    M = CNSTS['M']
    Ly = CNSTS['Ly']

    out = zeros(M, 'complex')

    # Use recurrence relation to calculate each mode in turn, given we know that
    # m>M modes must all be zero

    # m = M-2 special case
    out[M-2] = 2*(M-1)*cSpec[M-1]
    for i in range(3, M):
        m = M - i  
        out[m] = out[m+2] + 2*(m+1)*cSpec[m+1]
    del i
    # apply the normal C function for the zeroth mode
    out[0] = 0.5*(out[2] + 2*cSpec[1])

    return out

def dy(spec2D, CNSTS):
    """
    Orszag's method for doing a y derivative. 

    """

    M = CNSTS['M']
    N = CNSTS['N']

    outSpec = zeros((M, 2*N+1), 'complex')

    # The highest modes calculated separately.
    outSpec[M-1, :] = 0
    outSpec[M-2, :] = 2*(M-1)*spec2D[M-1, :] 

    for i in range(M-3, 0, -1):
        outSpec[i, :] = 2*(i+1)*spec2D[i+1, :] + outSpec[i+2, :]

    # the m = 0 mode is special - the c function.
    outSpec[0, :] = spec2D[1, :] + 0.5*outSpec[2, :]

    return outSpec

def dyy(spec2D, CNSTS):
    """
    Second Chebyshev direction derivative of a field using Orszag's method
    """

    M = CNSTS['M']
    N = CNSTS['N']

    outSpec = zeros((M, 2*N+1), 'complex')

    # The highest modes calculated separately.
    outSpec[M-1, :] = 0
    outSpec[M-2, :] = 0
    p = M-1
    outSpec[M-3, :] = (p**3 - p*(M-3)**2) * spec2D[p, :] 

    #tmpo and tmpe are odd and even sums respectively
    if M%2 == 0:
        tmp1o = p**3 * spec2D[p, :]
        tmp2o = p * spec2D[p, :]
        tmp1e = 0.0
        tmp2e = 0.0
    else:
        tmp1o = 0.0
        tmp2o = 0.0
        tmp1e = p**3 * spec2D[p, :]
        tmp2e = p * spec2D[p, :]

    for i in range(M-4, -1, -1):

            p = i+2

            if ((M+i) % 2) != 0:
                outSpec[i, :] = p * (p**2 - i**2) * spec2D[p, :] \
                                + tmp1o - i**2*tmp2o
                tmp1o += p**3 * spec2D[p, :]
                tmp2o += p * spec2D[p, :]

            else:
                outSpec[i, :] = p * (p**2 - i**2) * spec2D[p, :] \
                                + tmp1e - i**2*tmp2e
                tmp1e += p**3 * spec2D[p, :]
                tmp2e += p * spec2D[p, :]

    # the m = 0 mode is special - the c function.
    outSpec[0, :] = 0.5 * outSpec[0, :]

    return outSpec

def d3y(spec2D, CNSTS):
    """
    Third Chebyshev direction derivative of a field using Orszag's method

    TODO:
        Think I can spped this up by repeating what I did in dyy case above
        and removing the inner loop using a cumulative sum.
    """

    M = CNSTS['M']
    N = CNSTS['N']

    outSpec = zeros((M, 2*N+1), 'complex')

    # the m = 0 mode is special - the c function.
    
    for p in range(3, M, 2):
        outSpec[0, :] += 0.125*spec2D[p, :] * ( 2*p**3 * (p**2 - 2*p + 1)
                                        - p * ( (p**2 - 2*p)**2 + 1) ) 

    for i in range(1, M):
        for p in range(i + 3, M, 2):
            outSpec[i, :] += 0.25*spec2D[p, :] * ( 2*p**3 * (p**2 - 2*p - i**2 + 1)
                                            - p * ( (p**2 - 2*p)**2 
                                                   - (i**2 - 1)**2 ) ) 
    
    return outSpec

def d4y(spec2D, CNSTS):
    """
    Fourth Chebyshev direction derivative of a field using Orszag's method
    """

    M = CNSTS['M']
    N = CNSTS['N']

    outSpec = zeros((M, 2*N+1), 'complex')

    # the m = 0 mode is special - the c function.

    tmpCon = 1. / 24. 
    
    for p in range(4, M, 2):
        outSpec[0, :] += 0.5*tmpCon*spec2D[p, :]*p * (p**2 * (p**2 - 4)**2) 

    for i in range(1, M):
        for p in range(i + 4, M, 2):
            outSpec[i, :] += tmpCon*spec2D[p, :]*p * ( p**2 * (p**2 - 4)**2 
                                               - 3*i**2*p**4 + 3*i**4*p**2
                                               - i**2 * (i**2 - 4)**2 ) 
    
    return outSpec


def dx(spec2D, CNSTS):
    """
    First Fourier derivative.
    NEEDS TO BE CHECKED! SEEMS WRONG for conjugate modes.
    """

    N = CNSTS['N']
    M = CNSTS['M']
    kx = CNSTS['kx']

    deriv2D = zeros((M, 2*N+1), 'complex')

    deriv2D[:,0] = 0

    for n in range(1,N+1):
        deriv2D[:,n] = 1.j*n*kx*spec2D[:,n]      
        deriv2D[:,N+n] = -1.j*(N+1-n)*kx*spec2D[:,N+n] 
    del n

    return deriv2D


def dxx(spec2D, CNSTS):
    """
    Second Fourier derivative.
    """

    N = CNSTS['N']
    M = CNSTS['M']
    kx = CNSTS['kx']

    deriv2D = zeros((M, 2*N+1), 'complex')

    for n in range(N):
        deriv2D[:,n] = -(n*kx)**2 * spec2D[:,n]      
        deriv2D[:,N+1+n] = -((N-n)*kx)**2 * spec2D[:,N+1+n] 
    del n

    return deriv2D

def d3x(spec2D, CNSTS):
    """
    Second Fourier derivative.
    """

    N = CNSTS['N']
    M = CNSTS['M']
    kx = CNSTS['kx']

    deriv2D = zeros((M, 2*N+1), 'complex')

    for n in range(N):
        deriv2D[:,n] = -1.j*(n*kx)**3 * spec2D[:,n]      
        deriv2D[:,N+1+n] = 1.j*((N-n)*kx)**3 * spec2D[:,N+1+n] 
    del n

    return deriv2D

def d4x(spec2D, CNSTS):
    """
    Second Fourier derivative.
    """

    N = CNSTS['N']
    M = CNSTS['M']
    kx = CNSTS['kx']

    deriv2D = zeros((M, 2*N+1), 'complex')

    for n in range(N):
        deriv2D[:,n] = (n*kx)**4 * spec2D[:,n]      
        deriv2D[:,N+1+n] = ((N-n)*kx)**4 * spec2D[:,N+1+n] 
    del n

    deriv2D[:,2*N] = kx**4 * spec2D[:,2*N]

    return deriv2D

def dxy():
    """
    Chebyshev + Fourier derivative.
    """

    pass

def dxyy():
    """
    2 Chebyshev + Fourier derivative.
    """

    pass

def dxxy():
    """
    Chebyshev + 2 Fourier derivative.
    """

    pass

def dxyyy():
    """
    3 Chebyshev + Fourier derivative.
    """

    pass

def dxxyy():
    """
    2 Chebyshev + 2 Fourier derivative.
    """

    pass

def dxxxy():
    """
    Chebyshev + 3 Fourier derivative.
    """

    pass

def biharmonic():
    """
    Calculate the biharmonic (d4/dx4 + d4/dy4 + 2 d2/d2x2y) of a field.
    """

    pass

def to_physical(in2D, CNSTS):
    """
    Full 2 dimensional transform from spectral to real space. 
    
    First use the Fast fourier transform, then use my Chebyshev transform on
    the result in the y direction.

    I have attempted to minimize the number of 2D arrays created. 

    Note: dealiasing removes a third of the effective degrees of freedom. The
    true resolution is then much lower than that assumed by N,M this ought to
    be fixed in future versions as it will be a huge waste of computation.
         
    """

    M = CNSTS['M']
    N = CNSTS['N']
    Mf = CNSTS['Mf']
    Nf = CNSTS['Nf']

    tmp = zeros((M, 2*Nf+1), dtype='complex')
    tmp[:,:N+1] = in2D[:,:N+1]
    tmp[:,2*Nf+1-N:] = in2D[:,N+1:]

    # Perform the FFT across the x and z directions   

    _realtmp = zeros((2*Mf-2, 2*Nf+1), dtype='double')
    out2D = zeros((2*Mf-2, 2*Nf+1), dtype='complex')

    out2D[:M, :] = np.fft.fftpack.ifft(tmp, axis=-1)

    # test imaginary part of the fft is zero
    normImag = linalg.norm(imag(out2D))
    if normImag > 1e-12:
        print "output of ifft in to_physical is not real, norm = ", normImag 
        print 'highest x,modes:'
        print imag(out2D)[0, N-3:N+1]

    _realtmp = real(out2D)
    
    # Perform the Chebyshev transformation across the y direction

    # The second half contains the vector on the Gauss-Labatto points excluding
    # the first and last elements and in reverse order
    # do this before filling out the first half! 
    _realtmp[Mf:, :] = _realtmp[Mf-2:0:-1, :]

    # The first half contains the vector on the Gauss-Labatto points * c_k
    _realtmp[0, :] = 2*_realtmp[0, :]
    _realtmp[Mf-1, :] = 2*_realtmp[Mf-1, :]

    # Perform the transformation
    out2D = 0.5*np.fft.fftpack.rfft(_realtmp, axis=0 )

    normImag = linalg.norm(imag(out2D[0:M, :]))
    if normImag > 1e-12:
        print "output after Cheb transform in to_physical is not real, norm = ", normImag

    out2D = real(out2D)
    
    return out2D[0:Mf, :] * (2*Nf+1)

def to_physical_2(in2D, CNSTS):
    """
    Full 2 dimensional transform from spectral to real space using a single 2D
    complex fft.
        - PROBABLY MUCH SLOWER:
            Parallelism might speed it up a bit, but you need to a full rather
            than a real transform in y dir, and you need to do twice the number
            of x transforms => 4* the cost. so for N = 20000 it is 8e5 rather
            than 2e5 flops. Is that a big enough difference?

        - PROBABLY MUCH EASIER TO PROGRAM IN C:
            don't know how to plan all the necessary transforms otherwise!

    To get both transforms to be forward transforms, need to flip Fourier
    modes and renormalise.

    Note: dealiasing removes a third of the effective degrees of freedom. The
    true resolution is then much lower than that assumed by N,M this ought to
    be fixed in future versions as it will be a huge waste of computation.
        
    """

    M = CNSTS['M']
    N = CNSTS['N']

    Mf = CNSTS['Mf']
    Nf = CNSTS['Nf']

    # Prepare the field.

    out2D = zeros((2*Mf-2, 2*Nf+1), dtype='complex')

    # take complex conjugate (because actually want to do the inverse FFT) and
    # renormalise because only the ifft does renormalisation for you 
    # move renormalisation to to_spectral. that way we should be able to keep
    # the spectra with the same normalisation as the matrix code.

    out2D[:M, 0] = conj(in2D[:,0]) #/ (2*Nf+1)
    out2D[:M, 1:N+1] = conj(in2D[:,1:N+1]) #/ (2*Nf+1)
    out2D[:M, 2*Nf+1-N:] = conj(in2D[:,N+1:]) #/ (2*Nf+1)

    #if CNSTS['dealiasing']:
    #    out2D[:, 2*N/3 + 1 : 2*N+1 - 2*N/3] = 0 
    #    out2D[2*M/3:, :] = 0


    #if CNSTS['dealiasing']:
    #    out2D[:, 2*N/3 + 2 : 2*N+1 - 2*N/3] = 0 

    #out2D[:M, :] = conj(np.fft.fftpack.fft(out2D[:M, :], axis=-1))

    # The second half contains the vector on the Gauss-Labatto points excluding
    # the first and last elements and in reverse order
    # do this before filling out the first half! 
    out2D[2*Mf-M:, :] = out2D[M-2:0:-1, :]

    # The first half contains the vector on the Gauss-Labatto points * c_k
    out2D[0, :] = 2*out2D[0, :]
    out2D[Mf-1, :] = 2*out2D[Mf-1, :]

    # Perform the FFT across the x and z directions   

    out2D = 0.5*np.fft.fftpack.fft2(out2D)

    #out2D = real(out2D)
    
    return out2D[0:Mf, :]

def to_physical_3(in2D, CNSTS):
    """
       Use the ifft this time.  
    """

    M = CNSTS['M']
    N = CNSTS['N']

    Mf = CNSTS['Mf']
    Nf = CNSTS['Nf']

    # Prepare the field.

    out2D = zeros((2*Mf-2, 2*Nf+1), dtype='complex')
    scratch2D = zeros((2*Mf-2, 2*Nf+1), dtype='complex')

    out2D[:M, 0] = in2D[:,0] 
    out2D[:M, 1:N+1] = in2D[:,1:N+1] 
    out2D[:M, 2*Nf+1-N:] = in2D[:,N+1:] 

    # The second half contains the vector on the Chebyshev modes excluding
    # the first and last elements and in reverse order
    # do this before filling out the first half! 
    scratch2D[2*Mf-M:, :] = out2D[M-2:0:-1, :]

    # The first half contains the vector on the Chebyshev modes * ck/2
    scratch2D[0, :] = 2*out2D[0, :]
    scratch2D[1:Mf-1, :] = out2D[1:Mf-1, :]
    scratch2D[Mf-1, :] = 2*out2D[Mf-1, :]

    # Perform the iFFT across the x and z directions   

    out2D = 0.5*np.fft.fftpack.ifft2(scratch2D) 

    #out2D = real(out2D)
    
    return out2D[0:Mf, :] * (2*Mf-2) * (2*Nf+1)

def to_spectral(in2D, CNSTS): 
    """
    Full 2 dimensional transform from real space to spectral space.

    Note: dealiasing removes a third of the effective degrees of freedom. The
    true resolution is then much lower than that assumed by N,M this ought to
    be fixed in future versions as it seems like a waste of computation in the
    derivative and addition steps.
    """

    M = CNSTS['M']
    N = CNSTS['N']
    Mf = CNSTS['Mf']
    Nf = CNSTS['Nf']


    # Perform the FFT across the x direction   
    _realtmp = zeros((2*Mf-2, 2*Nf+1), dtype='double')
    out2D = zeros((M, 2*N+1), dtype='complex')

    # The first half contains the vector on the Gauss-Labatto points
    _realtmp[:Mf, :] = real(in2D)

    # The second half contains the vector on the Gauss-Labatto points excluding
    # the first and last elements and in reverse order
    _realtmp[Mf:, :] = _realtmp[Mf-2:0:-1, :]

    # Perform the transformation on this temporary vector
    # TODO: Think about antialiasing here
    _realtmp = np.fft.fftpack.rfft(_realtmp, axis=0)

    # Renormalise and divide by c_k to convert to Chebyshev polynomials
    _realtmp[0, :] = (0.5/(Mf-1.0))*_realtmp[0, :]
    _realtmp[1:Mf-1, :] = (1.0/(Mf-1.0))*_realtmp[1:Mf-1, :]
    _realtmp[Mf-1, :] = (0.5/(Mf-1.0))*_realtmp[Mf-1, :]

    # test imaginary part of the fft is zero
    normImag = linalg.norm(imag(_realtmp))
    if normImag > 1e-12:
        print "output of cheb transform in to_spectral is not real, norm = ", normImag 
        print 'highest x, z modes:'
        print imag(_realtmp)[0, N-3:N+1]

    _realtmp[:Mf, :] = np.fft.fftpack.fft(_realtmp[:Mf, :])

    out2D[:, :N+1] = _realtmp[:M, :N+1]
    out2D[:, N+1:] = _realtmp[:M, 2*Nf+1-N:]

    return out2D / (2*Nf+1)

def to_spectral_2(in2D, CNSTS): 
    """
    Full 2 dimensional transform from real space to spectral space using single
    2D transform.

    Note: dealiasing removes a third of the effective degrees of freedom. The
    true resolution is then much lower than that assumed by N,M this ought to
    be fixed in future versions as it seems like a waste of computation in the
    derivative and addition steps.

    Bear in mind, fftpack renormalises its transforms but fftw does not. This
    means the c code will have an extra factor about the place.
    """

    M = CNSTS['M']
    N = CNSTS['N']

    Mf = CNSTS['Mf']
    Nf = CNSTS['Nf']

    # Perform the FFT across the x direction   
    tmp = zeros((2*Mf-2, 2*Nf+1), dtype='double')

    out2D = zeros((M, 2*N+1), dtype='complex')

    # The first half contains the vector on the Gauss-Labatto points
    tmp[:Mf, :] = real(in2D)

    # The second half contains the vector on the Gauss-Labatto points excluding
    # the first and last elements and in reverse order
    tmp[Mf:, :] = tmp[Mf-2:0:-1, :]

    # Perform the transformation on this temporary vector
    # TODO: Think about antialiasing here
    #_realtmp = np.fft.fftpack.rfft(_realtmp, axis=0)
    tmp = np.fft.fftpack.fft2(tmp)

    ## Renormalise and divide by c_k to convert to Chebyshev polynomials
    tmp[0, :] = (0.5/(Mf-1.))*tmp[0, :]
    tmp[1:Mf-1, :] = (1.0/(Mf-1.))*tmp[1:Mf-1, :]
    tmp[Mf-1, :] = (0.5/(Mf-1.))*tmp[Mf-1, :]

    ## remove the aliased modes and copy into output
    out2D[:, :N+1] = tmp[:M, :N+1]
    out2D[:, N+1:] = tmp[:M, 2*Nf+1-N:]
    #print "is the temp matrix spectrum of real space?"
    #print allclose(tmp[:Mf, 1:Nf+1], conj(tmp[:Mf, 2*Nf+1:Nf:-1])) 
    #print "is the output matrix spectrum of real space?",
    #print allclose(out2D[:, 1:N+1], conj(out2D[:M, 2*N+1:N:-1])) 

    return out2D / (2*Nf+1)

def forward_cheb_transform(GLcmplx, CNSTS):
    """
    Use a real FFT to transform a single array from the Gauss-Labatto grid to
    Chebyshev polynomials.

    Note, this uses a real FFT therefore you must apply the transformations in
    the other directions before this one, otherwise you will loose the data from
    the imaginary parts.
    """

    M = CNSTS['M']
    Mf = CNSTS['Mf']
    dealiasing = CNSTS['dealiasing']

    # Define the temporary vector for the transformation
    tmp = zeros(2*Mf-2)

    # The first half contains the vector on the Gauss-Labatto points
    tmp[:Mf] = real(GLcmplx)

    # The second half contains the vector on the Gauss-Labatto points excluding
    # the first and last elements and in reverse order
    tmp[Mf:] = real(GLcmplx[Mf-2:0:-1])

    #savez('tmp.npz', tmp=tmp, consts=CNSTS)

    # Perform the transformation on this temporary vector
    # TODO: Think about antialiasing here
    tmp = real(np.fft.fftpack.rfft(tmp))

    out = zeros(M, dtype='complex')

    # Renormalise and divide by c_k to convert to Chebyshev polynomials
    out[0] = (0.5/(Mf-1.0)) * tmp[0]
    out[1:M-1] = (1.0/(Mf-1.0)) * tmp[1:M-1]
    if dealiasing:
        out[M-1] = (1.0/(Mf-1.0)) * tmp[M-1]
    else:
        out[M-1] = (0.5/(Mf-1.0)) * tmp[M-1]

    # Define the temporary vector for the transformation
    tmp = zeros(2*Mf-2)

    # The first half contains the vector on the Gauss-Labatto points
    tmp[:Mf] = imag(GLcmplx)

    # The second half contains the vector on the Gauss-Labatto points excluding
    # the first and last elements and in reverse order
    tmp[Mf:] = imag(GLcmplx[Mf-2:0:-1])

    # Perform the transformation on this temporary vector
    tmp = real(np.fft.fftpack.rfft(tmp))

    # Renormalise and divide by c_k to convert to Chebyshev polynomials
    out[0] += 1.j * (0.5/(Mf-1.0)) * tmp[0]
    out[1:M-1] += 1.j * (1.0/(Mf-1.0)) * tmp[1:M-1]
    if dealiasing:
        out[M-1] += 1.j * (1.0/(Mf-1.0)) * tmp[M-1]
    else:
        out[M-1] += 1.j * (0.5/(Mf-1.0)) * tmp[M-1]

    return out

def backward_cheb_transform(cSpec, CNSTS):
    M = CNSTS['M']
    Mf = CNSTS['Mf']

    out = zeros((Mf), dtype='complex')

    _realtmp = zeros((2*Mf-2), dtype='double')

    _realtmp[:M] = real(cSpec[:])
    
    # Perform the Chebyshev transformation across the y direction

    # The second half contains the vector on the Gauss-Labatto points excluding
    # the first and last elements and in reverse order
    # do this before filling out the first half! 
    _realtmp[Mf:] = _realtmp[Mf-2:0:-1]

    # The first half contains the vector on the Gauss-Labatto points * c_k
    _realtmp[0] = 2*_realtmp[0]
    _realtmp[Mf-1] = 2*_realtmp[Mf-1]

    # Perform the transformation
    #print shape(np.fft.fftpack.rfft(r_[1:5]))
    out += 0.5*real(np.fft.fftpack.rfft(_realtmp))

    _realtmp[:] = 0.0
    _realtmp[:M] = imag(cSpec[:])
    
    # Perform the Chebyshev transformation across the y direction

    # The second half contains the vector on the Gauss-Labatto points excluding
    # the first and last elements and in reverse order
    # do this before filling out the first half! 
    _realtmp[Mf:] = _realtmp[Mf-2:0:-1]

    # The first half contains the vector on the Gauss-Labatto points * c_k
    _realtmp[0] = 2*_realtmp[0]
    _realtmp[Mf-1] = 2*_realtmp[Mf-1]

    # Perform the transformation
    out += 0.5*1.j*real(np.fft.fftpack.rfft(_realtmp))

    return out[0:Mf]

def backward_cheb_transform_2(cSpec, CNSTS):
    """
    Use a DCT to transform a single array of Chebyshev polynomials to the
    Gauss-Labatto grid.
    """
    # cleverer way, now works!
    M = CNSTS['M']
    Mf = CNSTS['Mf']

    # Define the temporary vector for the transformation
    tmp = zeros(Mf)

    # The first half contains the vector on the Gauss-Labatto points * c_k
    tmp[0] = real(cSpec[0])
    tmp[1:M] = 0.5*real(cSpec[1:M])
    tmp[Mf-1] = 2*tmp[Mf-1]

    out = zeros(Mf, dtype='complex')
    out = dct(tmp, type=1).astype('complex') 

    tmp[0] = imag(cSpec[0])
    tmp[1:M] = 0.5*imag(cSpec[1:M])
    tmp[Mf-1] = 2*tmp[Mf-1]

    out += dct(tmp, type=1) * 1.j

    return out[0:Mf]
#
#    Mf = CNSTS['Mf']
#    M = CNSTS['M']
#
#    # Define the temporary vector for the transformation
#    tmp = zeros(Mf)
#    out = zeros(Mf, dtype='complex')
#
#    # The first half contains the vector on the Gauss-Labatto points * c_k
#    tmp[0] = real(cSpec[0])
#    tmp[1:M] = 0.5*real(cSpec[1:M])
#    tmp[Mf-1] = 2*tmp[Mf-1]
#
#    # Perform the transformation via a dct
#    out[:] = real(dct(tmp, type=1))
#
#    # Define the temporary vector for the transformation
#    tmp = zeros(Mf)
#
#    # The first half contains the vector on the Gauss-Labatto points * c_k
#    tmp[0] = imag(cSpec[0])
#    tmp[1:M] = 0.5*imag(cSpec[1:M])
#    tmp[Mf-1] = 2*tmp[Mf-1]
#
#    # Perform the transformation for the imaginary part via a dct
#    out += 1.j*real(dct(tmp, type=1))
#
#    return out[0:Mf]

def increase_resolution(vec, NOld, MOld, CNSTS):
    """increase resolution from Nold, Mold to N, M and return the higher res
    vector"""
    N = CNSTS["N"]
    M = CNSTS["M"]

    highMres = zeros((2*NOld+1)*M, dtype ='complex')

    for n in range(2*NOld+1):
        highMres[n*M:n*M + MOld] = vec[n*MOld:(n+1)*MOld]
    del n
    fullres = zeros((2*N+1)*M, dtype='complex')
    fullres[(N-NOld)*M:(N-NOld)*M + M*(2*NOld+1)] = highMres[0:M*(2*NOld+1)]
    return fullres

def decrease_resolution(vec, NOld, MOld, CNSTS):
    """ 
    decrease both the N and M resolutions
    """
    N = CNSTS["N"]
    M = CNSTS["M"]

    lowMvec = zeros((2*NOld+1)*M, dtype='complex')
    for n in range(2*NOld+1):
        lowMvec[n*M:(n+1)*M] = vec[n*MOld:n*MOld + M]
    del n

    lowNMvec = zeros((2*N+1)*M, dtype='D')
    lowNMvec = lowMvec[(NOld-N)*M:(NOld-N)*M + (2*N+1)*M]

    return lowNMvec

def decide_resolution(vec, NOld, MOld, CNSTS):
    """
    Choose to increase or decrease resolution depending on values of N,M
    NOld,MOld.
    """
    N = CNSTS["N"]
    M = CNSTS["M"]
    if N >= NOld and M >= MOld:
        ovec = increase_resolution(vec, NOld, MOld, CNSTS)

    elif N <= NOld and M <= MOld:
        ovec = decrease_resolution(vec, NOld, MOld, CNSTS)

    return ovec

def stupid_transform(GLreal, CNSTS):
    """
    apply the Chebyshev transform the stupid way.
    """

    M = CNSTS['M']
    Ly = CNSTS['Ly']

    out = zeros(M)

    for i in range(M):
        out[i] += (1./(M-1.))*GLreal[0]
        for j in range(1,M-1):
            out[i] += (2./(M-1.))*GLreal[j]*cos(pi*i*j/(M-1))
        out[i] += (1./(M-1.))*GLreal[M-1]*cos(pi*i)
    del i,j

    out[0] = out[0]/2.
    out[M-1] = out[M-1]/2.

    return out

def stupid_transform_i(GLspec, CNSTS):
    """
    apply the Chebyshev transform the stupid way.
    """

    M = CNSTS['M']
    Mf = CNSTS['Mf']
    Ly = CNSTS['Ly']

    out = zeros(Mf)

    for i in range(Mf):
        out[i] += GLspec[0]
        for j in range(1,M-1):
            out[i] += GLspec[j]*cos(pi*i*j/(Mf-1))
        out[i] += GLspec[M-1]*cos(pi*i)
    del i,j

    return out

def load_hdf5_state(filename):
    f = h5py.File(filename, "r")
    inarr = array(f["psi"])
    f.close()
    return inarr

#def test_roll_profile(CNSTS):
#    """
#    Use the roll profile from the SSP to check that differentiation and
#    transformation are working correctly.
#    """
#
#    M = CNSTS['M']
#    N = CNSTS['N']
#    Mf = CNSTS['Mf']
#    Nf = CNSTS['Nf']
#    Lx = CNSTS['Lx']
#    Ly = CNSTS['Ly']
#    kx = CNSTS['kx']
#
#    gamma = pi / Ly
#    p = optimize.fsolve(lambda p: p*tan(p) + gamma*tanh(gamma), 2)
#    oneOverC = ones(M)
#    oneOverC[0] = 1. / 2.
#
#    V = zeros((M, 2*N+1), dtype = 'complex')
#
#    for m in range(0,M,2):
#        V[m, 1] = 2*oneOverC[m]*( ((-1)**(m/2))*(special.jv(m,p)/cos(p)) - 
#                    special.iv(m,gamma)/cosh(gamma) )
#        V[m, 2*N] = 2*oneOverC[m]*( ((-1)**(m/2))*(special.jv(m,p)/cos(p)) - 
#                    special.iv(m,gamma)/cosh(gamma) )
#    del m        
#
#    Normal = ( cos(p)*cosh(gamma) ) / ( cosh(gamma) - cos(p) )
#    V = 0.5 * Normal * V
#    actualSpec = V
#
#    y_points = cos(pi*arange(Mf)/(Mf-1))
#    #x_points = linspace(0, 2.-(2./(2*Nf+1)), 2*Nf+1)
#    xlen = 2*pi / kx
#    x_points = linspace(0, xlen-(xlen/(2*Nf+1)), 2*Nf+1)
#
#    GLreal = zeros((Mf, 2*Nf+1), 'complex')
#
#    for i in range(2*Nf+1):
#        # y dependence
#        GLreal[:,i] = Normal*cos(p*y_points) / cos(p)
#        GLreal[:,i] += - Normal*cosh(gamma*y_points)/ cosh(gamma)
#        # x dependence
#        GLreal[:,i] = GLreal[:, i]*cos(kx*x_points[i])
#
#    print 'values at realspace endpoints x: ', GLreal[0,0], GLreal[0,2*Nf]
#
#    actualRYderiv = zeros((Mf, 2*Nf+1), 'complex')
#
#    for i in range(2*Nf+1):
#        # y dependence
#        actualRYderiv[:,i] = - Normal*p*sin(p*y_points) / cos(p)
#        actualRYderiv[:,i] += - Normal*gamma*sinh(gamma*y_points) / cosh(gamma)
#        # x dependence
#        actualRYderiv[:,i] = actualRYderiv[:, i]*cos(kx*x_points[i])
#
#    #plt.imshow(real(GLreal), origin='lower')
#    #plt.imshow(real(actualRYderiv), origin='lower')
#    #show()
#
#
#    print """
#    -----------------------
#    Test Transformations:
#    -----------------------
#    """
#    print """
#    --------------
#    Orthogonality 
#    --------------
#    """
#
#    ## transform is inverse of inverse transform?
#    inverseTest1 = to_spectral(to_physical(actualSpec, CNSTS), CNSTS)
#    inverseTest2 = to_spectral_2(to_physical_2(actualSpec, CNSTS), CNSTS)
#    inverseTest3 = to_spectral_2(to_physical_2(inverseTest2, CNSTS), CNSTS)
#
#
#    print 'transform is inverse of inverse transform? '
#    print 'two 1D transforms method', allclose(actualSpec, inverseTest1)
#    print '1 2D transform method', allclose(actualSpec, inverseTest2)
#    print '1 2D transform method', allclose(actualSpec, inverseTest3)
#
#    print 'if you start from real space? ', allclose(GLreal, to_physical(to_spectral(GLreal, CNSTS), CNSTS))
#    print '1 2D transform', allclose(GLreal,
#                                     to_physical_2(to_spectral_2(GLreal, CNSTS), CNSTS))
#
#
#    print 'to physical ifft is same as fft?'
#    result1 =  to_physical(actualSpec, CNSTS)
#    result2 = to_physical_3(actualSpec, CNSTS)
#    print allclose(result1,result2)
#    #print linalg.norm( (result1-result2))
#    #plt.imshow(real(result2), origin='lower')
#    #plt.colorbar()
#    #plt.show()
#    #plt.imshow(real(result1), origin='lower')
#    #plt.colorbar()
#    #plt.show()
#
#
#    ## Backwards Test ##
#    print """
#    --------------------------------------
#    Test transformation to physical space.
#    --------------------------------------
#    """
#
#    stupid = stupid_transform_i(2*actualSpec[:,1], CNSTS)
#    print 'stupid transfrom the same as analytic GLpoints'
#    print allclose(GLreal[:,0], stupid)
#
#    physicalTest = to_physical(actualSpec, CNSTS)
#    physicalTest2 = real(to_physical_2(actualSpec,CNSTS))
#    physicalTest3 = real(to_physical_3(actualSpec,CNSTS))
#
#
#    print 'actual real space = transformed analytic spectrum?', allclose(GLreal,
#                                                                         physicalTest)
#
#    print '2D transform is the same as 2 1D transforms with conj fft?', allclose(physicalTest, 
#                                                                   physicalTest2)
#
#    print '2D transform is the same as 2 1D transforms with ifft?', allclose(physicalTest, 
#                                                                   physicalTest3)
#    
#    #print 'difference: ', linalg.norm(physicalTest2-physicalTest)
#    #print 'difference Fourier dir: ', (physicalTest2-physicalTest)[M/2,:]
#    #print 'difference Cheby dir: ', (physicalTest2-physicalTest)[:,N/2]
#
#    plt.plot(y_points, real(physicalTest[:,10]), 'b')
#    plt.plot(y_points, real(GLreal[:,10]), 'r+')
#    plt.show()
#
#    plt.plot(x_points, real(physicalTest[M/2,:]), 'b')
#    plt.plot(x_points, real(GLreal[M/2,:]), 'r+')
#    plt.show()
#
#    plt.imshow(real(GLreal),  origin='lower')
#    plt.colorbar()
#    plt.show()
#    plt.imshow(real(physicalTest),  origin='lower')
#    plt.colorbar()
#    plt.show()
#
#    #print 'the maximum difference in the arrays ', amax(real(GLreal) -real(physicalTest))
#
#    ## Forwards test ##
#    print """
#    --------------------------------------
#    Test transformation to spectral space.
#    --------------------------------------
#    """
#    cSpec = to_spectral(GLreal, CNSTS)
#
#    print 'analytic spectrum = transformed GL spectrum?', allclose(actualSpec,
#                                                                   cSpec)
#    #plt.plot(real(cSpec[:,1]), 'b')
#    #plt.plot(real(actualSpec[:,1]), 'r+')
#    #plt.show()
#
#    #plt.plot(real(cSpec[2,:]), 'b')
#    #plt.plot(real(actualSpec[2,:]), 'r+')
#    #plt.show()
#
#    #plt.plot(y_points, GLreal)
#    #plt.plot(y_points, physical_test, '+')
#    #plt.show()
#
#    SpectralTest2 = to_spectral_2(GLreal, CNSTS)
#    print '2D transform is the same as 2 1D transforms?', allclose(cSpec, 
#                                        SpectralTest2)
#
#    #print 'difference: ', linalg.norm(SpectralTest2-cSpec)
#    #print 'difference Fourier dir: ', (SpectralTest2-cSpec)[1,:]
#    #print 'difference Cheby dir: ', (SpectralTest2-cSpec)[:,1]
#
#    # Products
#    tsm.initTSM(N_=N, M_=M, kx_=kx)
#    
#    flatSpec = fftshift(actualSpec, axes=1)
#    flatSpec = flatSpec.T.flatten()
#    matprod = dot(tsm.prod_mat(flatSpec), flatSpec)
#    matprod = matprod.reshape(2*N+1, M).T 
#    matprod = ifftshift(matprod, axes=-1)
#
#    print 'compare matrix product code with python fft products'
#
#    pyprod = to_spectral(physicalTest2*physicalTest2, CNSTS) # * (2*Nf+1)**2
#    #plt.imshow(real(physicalTest2*physicalTest2))
#    #plt.colorbar()
#    #plt.show()
#    #plt.imshow(real(to_physical(matprod, CNSTS)))
#    #plt.colorbar()
#    #plt.show()
#    
#    print allclose(pyprod, matprod)
#    #print linalg.norm(pyprod - matprod)
#    #plt.imshow(real(pyprod -matprod))
#    #plt.colorbar()
#    #plt.show()
#
#    print """
#    -----------------------
#    Test y derivatives:
#    -----------------------
#    """
#
#    yDerivTest = dy(actualSpec, CNSTS)
#    yyDerivTest = dy(yDerivTest, CNSTS)
#    yyyDerivTest = dy(yyDerivTest, CNSTS)
#    yyyyDerivTest = dy(yyyDerivTest, CNSTS)
#
#    print 'dy of the spectrum in real space = real space analytic derivative? ',\
#            allclose(to_physical(yDerivTest, CNSTS), actualRYderiv)
#    print 'difference ' , linalg.norm(to_physical(yDerivTest, CNSTS) -
#                                      actualRYderiv)
#    print 'dy by python = dy by matrix multiplication? '
#    MDY = tsm.mk_diff_y()
#    matdy = dot(MDY, flatSpec)
#    matdy = matdy.reshape(2*N+1, M).T
#    matdy = ifftshift(matdy, axes=-1)
#    print allclose(matdy, yDerivTest)
#    print linalg.norm(matdy - yDerivTest)
#    print 'Chebyshev modes check'
#    for m in range(M):
#        print 'mode', m, linalg.norm(matdy[m,:] - yDerivTest[m,:])
#
#    tmpTest = dyy(actualSpec, CNSTS)
#
#    print 'dyy of the spectrum is the same as a double application of dy? ',\
#            allclose(to_physical(yyDerivTest, CNSTS), to_physical(tmpTest, CNSTS))
#
#    tmpTest = d3y(actualSpec, CNSTS)
#
#    print 'd3y of the spectrum is the same as a triple application of dy? ',\
#            allclose(to_physical(yyyDerivTest, CNSTS), to_physical(tmpTest, CNSTS))
#
#    tmpTest = d4y(actualSpec, CNSTS)
#
#    print 'd4y of the spectrum is the same as quadruple application of dy? ',\
#            allclose(to_physical(yyyyDerivTest, CNSTS), to_physical(tmpTest, CNSTS))
#
#    print """
#    -----------------------
#    Test x derivatives:
#    -----------------------
#    """
#
#    xDerivTest = dx(actualSpec, CNSTS)
#    xxDerivTest = dx(xDerivTest, CNSTS)
#    xxxDerivTest = dx(xxDerivTest, CNSTS)
#    xxxxDerivTest = dx(xxxDerivTest, CNSTS)
#
#    tmpTest = dxx(actualSpec, CNSTS)
#
#    print 'dxx of the spectrum is the same as a double application of dx? ',\
#            allclose(to_physical(xxDerivTest, CNSTS), to_physical(tmpTest, CNSTS))
#
#    tmpTest = d3x(actualSpec, CNSTS)
#
#    print 'd3x of the spectrum is the same as a triple application of dx? ',\
#            allclose(to_physical(xxxDerivTest, CNSTS), to_physical(tmpTest, CNSTS))
#
#    tmpTest = d4x(actualSpec, CNSTS)
#
#    print 'd4x of the spectrum is the same as a quadruple application of dx? ',\
#            allclose(to_physical(xxxxDerivTest, CNSTS), to_physical(tmpTest, CNSTS))
#
#
#    print """
#    -----------------------
#    Test mixed derivatives:
#    -----------------------
#    """

#def test_prods(CNSTS):
#    """
#    tests the transform methods for products of fields.
#    """
#
#    M = CNSTS['M']
#    N = CNSTS['N']
#    Mf = CNSTS['Mf']
#    Nf = CNSTS['Nf']
#    Lx = CNSTS['Lx']
#    Ly = CNSTS['Ly']
#
#    print '100 products of random matrices:'
#
#    As = zeros((M, 2*N+1), dtype='complex')
#    Bs = zeros((M, 2*N+1), dtype='complex')
#
#    for i in range(1,2*N/3+1):
#        As[:2*M/3,i] = rand(2*M/3) + rand(2*M/3)*1.j
#        As[:2*M/3,2*N+1-i] = conj(As[:2*M/3,i])
#        Bs[:2*M/3,i] = rand(2*M/3) + rand(2*M/3)*1.j
#        Bs[:2*M/3,2*N+1-i] = conj(Bs[:2*M/3,i])
#
#    Aold = copy(As)
#    Bold = copy(Bs)
#
#    for i in range(1000):
#
#        A = to_physical_2(As, CNSTS)
#        B = to_physical_2(Bs, CNSTS)
#
#        C = A*B
#
#        As = to_spectral_2(A, CNSTS)
#        Bs = to_spectral_2(B, CNSTS)
#
#    print allclose(Aold,As), allclose(Bold,Bs)
#    print linalg.norm(Aold)/linalg.norm(As)
#
def test_arrays_equal(arr1, arr2, tol=1e-12):
    testBool = allclose(arr1, arr2)
    print testBool
    if not testBool:
        print 'difference', linalg.norm(arr1-arr2)

        if linalg.norm(arr1-arr2)>tol:
            print 'relative difference', linalg.norm(arr1-arr2)

            print "max difference", amax(arr1-arr2)
            print "max difference arg", argmax(arr1-arr2)

            if shape(arr1) == (M, 2*N+1):
                print "mode 0", linalg.norm(arr1[:,0]-arr2[:,0])
                for n in range(1,N+1):
                    print "mode", n, linalg.norm(arr1[:, n]-arr2[:, n])
                    print "mode", -n,linalg.norm(arr1[:, n]-arr2[:, n])

            if shape(arr1) == ((2*N+1)*M):
                print "mode 0", linalg.norm(arr1[N*M:(N+1)*M]-arr2[N*M:(N+1)*M])
                for n in range(1,N+1):
                    print "mode", n, linalg.norm(arr1[(N+n)*M:(N+n+1)*M]-arr2[(N+n)*M:(N+n+1)*M])
                    print "mode", -n,linalg.norm(arr1[(N-n)*M:(N+1-n)*M]-arr2[(N-n)*M:(N+1-n)*M])


            #plt.imshow(real(ctestSpec3), origin='lower')
            #plt.colorbar()
            #plt.show()
            #plt.imshow(real(pythonSpec3), origin='lower')
            #plt.colorbar()
            #plt.show()

            print 'FAIL'

            exit(1)

#def test_c_version(CNSTS):
#    """
#    Test the C version of the code. Make sure constants are the same across
#    codes until I implement passing of this info back and forth.
#
#    Tests will be performed by comparing the results of the C code with the
#    results of this code, rather than the true results. this buys some time
#    before I have ot do the boring job of working out how to do the transform in
#    C.
#
#    """
#    M = CNSTS['M']
#    N = CNSTS['N']
#    Mf = CNSTS['Mf']
#    Nf = CNSTS['Nf']
#    Lx = CNSTS['Lx']
#    Ly = CNSTS['Ly']
#    kx = CNSTS['kx']
#
#    outputdir = './output/'
#
#    gamma = pi / Ly
#    p = optimize.fsolve(lambda p: p*tan(p) + gamma*tanh(gamma), 2)
#    oneOverC = ones(M)
#    oneOverC[0] = 1. / 2.
#
#    actualSpec, _ = pickle.load(open('pf-N5-M40-kx1.31-Re3000.0.pickle', 'r'))
#    actualSpec = decide_resolution(actualSpec, 5, 40, CNSTS)
#
#    actualSpec = actualSpec.reshape(2*N+1, M).T
#    actualSpec = ifftshift(actualSpec, axes=1)
#
#    # insert stupider spectrum
#    #actualSpec = zeros((M,2*N+1), dtype='complex')
#    #actualSpec = ones((M,2*N+1), dtype='complex')
#    #actualSpec[:M/3,5] = r_[M/3:0:-1]
#    #actualSpec[:M/3, 0] = r_[M/3:0:-1]
#    #actualSpec[2*M/3:,0] = actualSpec[2*M/3:,0] * 1e-6 * rand(M-2*M/3)*1.j
#    #actualSpec[2*M/3:,5] = actualSpec[2*M/3:,5] * 1e-6 * rand(M-2*M/3)*1.j
#    #actualSpec[:,2*N-4] = r_[0:M]
#    #actualSpec[:2*M/3,1:2*N/3 + 1] = rand(2*M/3, 2*N/3) + 1.j*rand(2*M/3, 2*N/3)
#    #actualSpec[:, N+1:] = conj(fliplr(actualSpec[:, 1:N+1]))
#    #actualSpec[:M, 1:N + 1] = rand(M, N) + 1.j*rand(M, N)
#    #actualSpec[:, N+1:] = conj(fliplr(actualSpec[:, 1:N+1]))
#
#    # save the initial state
#    f = h5py.File("initial.h5", "w")
#    dset = f.create_dataset("psi", ((2*N+1)*M,), dtype='complex')
#    dset[...] = actualSpec.T.flatten()
#    f.close()
#
#    # call the c program
#    subprocess.call(["./test_fields"])
#
#    # Read in the c programs output Reshape is because fft insists on 1D double
#    # complex arrays.  T is because this program uses fortran order not c order
#    # for y and x.  slice is because the rest of the array is junk I carry round
#    # the c program to speed up the transforms.
#    
#    ctestSpec = load_hdf5_state(outputdir + "testSpec.h5")
#    ctestSpec = ctestSpec.reshape(2*N+1, M).T
#
#    ctestdxSpec = load_hdf5_state(outputdir + "testdx.h5").reshape(2*N+1, M).T
#    ctestdySpec = load_hdf5_state(outputdir + "testdy.h5").reshape(2*N+1, M).T
#
#    # Compare python code and C code
#
#    ChkBool = allclose(ctestSpec, actualSpec)
#    print "Python and C code have the same initial spectra?: ", ChkBool
#    if not ChkBool:
#        print linalg.norm(ctestSpec-actualSpec)
#
#    print """
#    -------------------
#    Test dy
#    -------------------
#    """
#    dySpec = dy(actualSpec, CNSTS)
#
#    print "Python and C code give the same derivative: "
#    test_arrays_equal(ctestdySpec, dySpec)
#
#    d4ySpec = dy(dy(dy(dySpec, CNSTS), CNSTS), CNSTS)
#    ctestd4ySpec = load_hdf5_state(outputdir + "testd4y.h5").reshape(2*N+1, M).T
#
#    print "Python and C code give the same d4y: "
#    test_arrays_equal(ctestd4ySpec,d4ySpec)
#
#
#    flatspec = fftshift(actualSpec, axes=1)
#    flatspec = flatspec.T.flatten()
#
#    tsm.initTSM(N_=N, M_=M, kx_=kx)
#
#    MDX = tsm.mk_diff_x()
#    MDY = tsm.mk_diff_y()
#
#    matd4ySpec = dot(MDY, dot(MDY, dot(MDY, dot(MDY, flatspec)))) 
#    matd4ySpec = matd4ySpec.reshape(2*N+1, M).T
#    matd4ySpec = ifftshift(matd4ySpec, axes=-1)
#
#    print "Python and matrix method give the same d4y: "
#    test_arrays_equal(matd4ySpec,d4ySpec)
#
#
#    print """
#    -------------------
#    Test dx
#    -------------------
#    """
#
#    dxSpec = dx(actualSpec, CNSTS)
#
#    print "Python and C code give the same derivative: "
#    test_arrays_equal(ctestdxSpec,dxSpec)
#
#    print "test matrix multiplication method and compare with python sum"
#
#    flatspec = fftshift(actualSpec, axes=1)
#    flatspec = flatspec.T.flatten()
#    matdx = dot(MDX, flatspec)
#
#    matdyypsi = dot( dot(MDY, MDY), flatspec)
#
#    matdxpsi2D = matdx.reshape(2*N+1, M).T
#    matdxpsi2D = ifftshift(matdxpsi2D,axes=-1)
#
#    print 'checking matrix deriv is the same as the python looped derivative'
#    test_arrays_equal(matdxpsi2D, dxSpec)
#
#    print """
#
#    -----------------------
#    Test Transformations:
#    -----------------------
#    """
#
#    # remember the normalisation factor
#
#    ctestPhys = load_hdf5_state(outputdir + "testPhysicalT.h5").reshape(2*Nf+1, 2*Mf-2).T[:Mf, :]
#
#    actualPhys = real(to_physical(actualSpec,CNSTS))
#
#    print "Physical Transform: C transform is the same as python transform?"
#    test_arrays_equal(actualPhys, real(ctestPhys))
#
#    ctestSpec = load_hdf5_state(outputdir + "testSpectralT.h5").reshape(2*N+1, M).T 
#    python2spec = to_spectral_2(actualPhys, CNSTS)
#
#    print "Spectral Transform: C transform is the same as python transform?"
#    test_arrays_equal(python2spec, ctestSpec)
#
#    phystest = zeros((Mf, 2*Nf+1), dtype='complex')
#
#    for i in range(2*Nf+1):
#        for j in range(Mf):
#	    phystest[j,i] =  cos(i*pi/(2.*Nf)) * tanh(j*pi/(Mf-1.))
#
#    pythonSpec3 = to_spectral(phystest, CNSTS)
#    #ctestSpec3 = load_hdf5_state(outputdir + "testSpectralT2.h5").reshape(2*Nf+1, 2*Mf-2).T
#    ctestSpec3 = load_hdf5_state(outputdir + "testSpectralT2.h5").reshape(2*N+1, M).T 
#    cphystest = load_hdf5_state(outputdir + "phystest2.h5").reshape(2*Nf+1, 2*Mf-2).T[:Mf, :]
#
#    print 'Spectral Transform: '
#    print 'c code has same physical space array to test ?'
#    test_arrays_equal(cphystest, phystest)
#
#    print 'From real space problem to spectral space, comparision of python and C'
#    test_arrays_equal(pythonSpec3, ctestSpec3)
#
#    #ctestSpec_1D = load_hdf5_state(outputdir + "testSpec_1D.h5").reshape(2*N+1, M).T 
#
#    print 'Spectral Transform: '
#    pythonPhys4 = to_physical(pythonSpec3, CNSTS)
#    ctestPhys4 = load_hdf5_state(outputdir + "testPhysT4.h5").reshape(2*Nf+1, (2*Mf-2)).T[:Mf, :] 
#    print 'From real space problem to spectral space and back again, comparision of python and C'
#    test_arrays_equal(pythonPhys4, ctestPhys4)
#
#
#    python2specR = copy(python2spec)
#    for i in range(100):
#        pythonPhys = to_physical_2(python2specR, CNSTS)
#        python2specR = to_spectral_2(pythonPhys, CNSTS)
#    del i
#
#    test_arrays_equal(python2spec, python2specR)
#
#    print 'checking python fft products are equal to matrix method products'
#    print 'vdyypsi'
#
#    matvdyypsi = dot(tsm.prod_mat(-matdx), matdyypsi)
#
#
#    physv = to_physical(-dxSpec, CNSTS)
#    physdyy = to_physical(dy(dy(actualSpec, CNSTS), CNSTS), CNSTS)
#    vdyypsi = to_spectral(physv*physdyy, CNSTS)
#
#    matdx = fftshift(dxSpec, axes=-1)
#    matdx = matdx.T.flatten()
#    matdyypsi = fftshift(dy(dy(actualSpec, CNSTS), CNSTS), axes=-1)
#    matdyypsi = matdyypsi.T.flatten()
#    matvdyypsi = dot(tsm.prod_mat(-matdx), matdyypsi)
#    matvdyypsi2D = matvdyypsi.reshape(2*N+1, M).T
#    matvdyypsi2D = ifftshift(matvdyypsi2D, axes=-1) 
#
#    test_arrays_equal(matvdyypsi2D, vdyypsi)
#
#
#    print 'Check matrix and python fft methods both convolve the same: psipsi'
#    psiR = real(to_physical(actualSpec, CNSTS))
#    psipsi = to_spectral(psiR*psiR, CNSTS)
#    matpsipsi = dot(tsm.prod_mat(flatspec), flatspec)
#    matpsipsi2D = matpsipsi.reshape(2*N+1, M).T
#    matpsipsi2D = ifftshift(matpsipsi2D, axes=-1) 
#
#    test_arrays_equal(matpsipsi2D, psipsi)
#
#    psipsic = load_hdf5_state(outputdir + "fft_convolve.h5").reshape(2*N+1, M).T
#
#    print 'Checking c products are the same as python products'
#    test_arrays_equal(psipsic, psipsi)
#
#    ctestSpecR = load_hdf5_state(outputdir + "testSpectralTR.h5").reshape(2*N+1, M).T 
#
#    print "Repeated Transforms: C field is stable after 100 transforms?"
#    test_arrays_equal(ctestSpecR, ctestSpec)
#
#    print """
#    ==============================================================================
#
#        2D FIELDS TESTS PASSED! 
#
#    ==============================================================================
#    """

#def test_c_version_1D(CNSTS):
#    """
#    Test the C version of the code. Make sure constants are the same across
#    codes until I implement passing of this info back and forth.
#
#    Tests will be performed by comparing the results of the C code with the
#    results of this code, rather than the true results. this buys some time
#    before I have ot do the boring job of working out how to do the transform in
#    C.
#
#    """
#    M = CNSTS['M']
#    N = CNSTS['N']
#    Mf = CNSTS['Mf']
#    Nf = CNSTS['Nf']
#    Lx = CNSTS['Lx']
#    Ly = CNSTS['Ly']
#    kx = CNSTS['kx']
#
#    outputdir = './output/'
#
#    gamma = pi / Ly
#    p = optimize.fsolve(lambda p: p*tan(p) + gamma*tanh(gamma), 2)
#    oneOverC = ones(M)
#    oneOverC[0] = 1. / 2.
#
#    actualSpec, _ = pickle.load(open('pf-N5-M40-kx1.31-Re3000.0.pickle', 'r'))
#    actualSpec = decide_resolution(actualSpec, 5, 40, CNSTS)
#
#    actualSpec = actualSpec.reshape(2*N+1, M).T
#    actualSpec = ifftshift(actualSpec, axes=1)
#
#    actualSpec = actualSpec[:,1]
#
#    # save the initial state
#    f = h5py.File("initial.h5", "w")
#    dset = f.create_dataset("psi", (M,), dtype='complex')
#    dset[...] = actualSpec.T.flatten()
#    f.close()
#
#    # call the c program
#    subprocess.call(["./test_fields_1D"])
#
#    # Read in the c programs output Reshape is because fft insists on 1D double
#    # complex arrays.  T is because this program uses fortran order not c order
#    # for y and x.  slice is because the rest of the array is junk I carry round
#    # the c program to speed up the transforms.
#    
#    ctestSpec = load_hdf5_state(outputdir + "testSpec.h5")
#
#    ctestdxSpec = load_hdf5_state(outputdir + "testdx.h5")
#    ctestdySpec = load_hdf5_state(outputdir + "testdy.h5")
#
#    # Compare python code and C code
#
#    print "Python and C code have the same initial spectra?: "
#    test_arrays_equal(ctestSpec, actualSpec)
#
#    print """
#    -------------------
#    Test dy
#    -------------------
#    """
#    dySpec = single_dy(actualSpec, CNSTS)
#
#    print "Python and C code give the same derivative: "
#    test_arrays_equal(ctestdySpec, dySpec)
#
#    d4ySpec = single_dy(single_dy(single_dy(dySpec, CNSTS), CNSTS), CNSTS)
#    ctestd4ySpec = load_hdf5_state(outputdir + "testd4y.h5")
#
#    print "Python and C code give the same d4y: "
#    test_arrays_equal(ctestd4ySpec, d4ySpec)
#
#
#    flatspec = copy(actualSpec)
#
#    tsm.initTSM(N_=N, M_=M, kx_=kx)
#
#    MDX = 1.j*kx*eye(M, dtype='complex')
#    MDY = tsm.mk_single_diffy()
#
#    matd4ySpec = dot(MDY, dot(MDY, dot(MDY, dot(MDY, flatspec)))) 
#
#    print "Python and matrix method give the same d4y: "
#    test_arrays_equal(matd4ySpec, d4ySpec)
#
#
#    print """
#    -------------------
#    Test dx
#    -------------------
#    """
#
#    dxSpec = 1.j*kx*actualSpec
#
#    print "Python and C code give the same derivative: "
#    test_arrays_equal(ctestdxSpec,dxSpec)
#
#    print "test matrix multiplication method and compare with python sum"
#
#    flatspec = copy(actualSpec)
#    matdx = dot(MDX, flatspec)
#
#    matdyypsi = dot( dot(MDY, MDY), flatspec)
#
#    print 'checking matrix deriv is the same as the python looped derivative'
#    test_arrays_equal(matdx, dxSpec)
#
#    print """
#
#    -----------------------
#    Test Transformations:
#    -----------------------
#    """
#
#    # remember the normalisation factor
#
#    ctestPhys = load_hdf5_state(outputdir + "testPhysicalT.h5")[:Mf]
#
#    actualPhys = backward_cheb_transform(actualSpec, CNSTS)
#
#    print "Physical Transform: C transform is the same as python transform?"
#    test_arrays_equal(actualPhys, ctestPhys)
#
#    ctestSpec = load_hdf5_state(outputdir + "testSpectralT.h5")
#    python2spec = forward_cheb_transform(actualPhys, CNSTS)
#
#    tmp = zeros((Mf,2*Nf+1), dtype='complex')
#    tmp[:,0] = actualPhys
#    python2spec2 = to_spectral_2(real(tmp), CNSTS)[:,0]
#    python2spec2 += 1.j*to_spectral_2(imag(tmp), CNSTS)[:,0]
#    python2spec2 *= (2*Nf+1)
#    
#    print "Spectral Transform: Python transforms are consistent?"
#    test_arrays_equal(python2spec2, python2spec)
#
#    print "Spectral Transform: C transform is the same as python transform?"
#    test_arrays_equal(python2spec, ctestSpec)
#
#    phystest = zeros(Mf, dtype='complex')
#
#    for j in range(Mf):
#        phystest[j] =  tanh(j*pi/(Mf-1.))
#
#    pythonSpec3 = forward_cheb_transform(phystest, CNSTS)
#    ctestSpec3 = load_hdf5_state(outputdir + "testSpectralT2.h5") 
#    cphystest = load_hdf5_state(outputdir + "phystest2.h5")[:Mf]
#
#    print 'Spectral Transform: '
#    print 'c code has same physical space array to test ?'
#    test_arrays_equal(cphystest, phystest)
#
#    print 'From real space problem to spectral space, comparision of python and C'
#    test_arrays_equal(pythonSpec3, ctestSpec3)
#
#    print 'Spectral Transform: '
#    pythonPhys4 = backward_cheb_transform(pythonSpec3, CNSTS)
#    ctestPhys4 = load_hdf5_state(outputdir + "testPhysT4.h5")[:Mf] 
#    print 'From real space problem to spectral space and back again, comparision of python and C'
#    test_arrays_equal(pythonPhys4, ctestPhys4)
#
#
#    python2specR = copy(python2spec)
#    for i in range(100):
#        pythonPhys = backward_cheb_transform(python2specR, CNSTS)
#        python2specR = forward_cheb_transform(pythonPhys, CNSTS)
#    del i
#
#    test_arrays_equal(python2spec, python2specR)
#
#    print 'checking python fft products are equal to matrix method products'
#    print 'vdyypsi'
#
#    matvdyypsi = dot(tsm.cheb_prod_mat(-matdx), matdyypsi)
#
#
#    physv = backward_cheb_transform(-dxSpec, CNSTS)
#    physdyy = backward_cheb_transform(single_dy(single_dy(actualSpec, CNSTS), CNSTS), CNSTS)
#    vdyypsi = forward_cheb_transform(physv*physdyy, CNSTS)
#
#    matdx =copy(dxSpec)
#    matdyypsi = single_dy(single_dy(actualSpec, CNSTS), CNSTS)
#    matvdyypsi = dot(tsm.cheb_prod_mat(-matdx), matdyypsi)
#
#    test_arrays_equal(matvdyypsi, vdyypsi)
#
#
#    print 'Check matrix and python fft methods both convolve the same: psipsi'
#    psiR = backward_cheb_transform(actualSpec, CNSTS)
#    psipsi = forward_cheb_transform(psiR*psiR, CNSTS)
#    matpsipsi = dot(tsm.cheb_prod_mat(flatspec), flatspec)
#
#    test_arrays_equal(matpsipsi, psipsi)
#
#    psipsic = load_hdf5_state(outputdir + "fft_convolve.h5")
#
#    print 'Checking c products are the same as python products'
#    test_arrays_equal(psipsic, psipsi)
#
#    ctestSpecR = load_hdf5_state(outputdir + "testSpectralTR.h5")
#
#    print "Repeated Transforms: C field is stable after 100 transforms?"
#    test_arrays_equal(ctestSpecR, ctestSpec)
#
#    print """
#    ==============================================================================
#
#        1D FIELDS TESTS PASSED! 
#
#    ==============================================================================
#    """

### MAIN ###

if __name__ == "__main__":

    CNSTS = set_constants(M=100, N=5, kx=1.31, dealiasing=True)

    #test_roll_profile(CNSTS)

    #test_diff(CNSTS, testFunc=lambda x: 1-x**2)

    #test_prods(CNSTS)

#    test_c_version(CNSTS)

#   test_c_version_1D(CNSTS)

#    test = rand(100)+ 1.j*rand(100)
#    a = backward_cheb_transform(test,CNSTS)
#    b = backward_cheb_transform_2(test,CNSTS)
#    test_arrays_equal(a,b)

