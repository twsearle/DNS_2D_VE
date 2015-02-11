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
from scipy import fftpack, optimize, linalg, special
import subprocess
import cPickle as pickle
import h5py

from pylab import *

import TobySpectralMethods as tsm

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

def dy(spec2D, CNSTS):
    """
    Orszag's method for doing a single y derivative. 

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

    out2D[:M, :] = fftpack.ifft(tmp, axis=-1)

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
    out2D = 0.5*fftpack.rfft(_realtmp, axis=0 )

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

    #out2D[:M, :] = conj(fftpack.fft(out2D[:M, :], axis=-1))

    # The second half contains the vector on the Gauss-Labatto points excluding
    # the first and last elements and in reverse order
    # do this before filling out the first half! 
    out2D[2*Mf-M:, :] = out2D[M-2:0:-1, :]

    # The first half contains the vector on the Gauss-Labatto points * c_k
    out2D[0, :] = 2*out2D[0, :]
    out2D[Mf-1, :] = 2*out2D[Mf-1, :]

    # Perform the FFT across the x and z directions   

    out2D = 0.5*fftpack.fft2(out2D)

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

    out2D = 0.5*fftpack.ifft2(scratch2D) 

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
    _realtmp = fftpack.rfft(_realtmp, axis=0)

    # Renormalise and divide by c_k to convert to Chebyshev polynomials
    _realtmp[0, :] = (0.5/(Mf-1.0))*_realtmp[0, :]
    _realtmp[1:Mf-1, :] = (1.0/(Mf-1.0))*_realtmp[1:Mf-1, :]
    _realtmp[Mf-1, :] = (0.5/(Mf-1.0))*_realtmp[Mf-1, :]

    # test imaginary part of the fft is zero
    normImag = linalg.norm(imag(_realtmp))
    if normImag > 1e-12:
        print "output of cheb transform in to_spectral is not real, norm = ", normImag 
        print 'highest x, z modes:'
        print imag(_realtmp)[0, N-3:N+1, L-3:L+1]

    _realtmp[:Mf, :] = fftpack.fft(_realtmp[:Mf, :])

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
    #_realtmp = fftpack.rfft(_realtmp, axis=0)
    tmp = fftpack.fft2(tmp)

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

def forward_cheb_transform(GLreal, CNSTS):
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

    Ly = CNSTS['Ly']

    # Define the temporary vector for the transformation
    tmp = zeros(2*Mf-2)

    # The first half contains the vector on the Gauss-Labatto points
    tmp[:Mf] = real(GLreal)

    # The second half contains the vector on the Gauss-Labatto points excluding
    # the first and last elements and in reverse order
    tmp[Mf:] = real(GLreal[Mf-2:0:-1])

    # Perform the transformation on this temporary vector
    # TODO: Think about antialiasing here
    tmp = real(fftpack.rfft(tmp))

    out = zeros(M, dtype='complex')
    # Renormalise and divide by c_k to convert to Chebyshev polynomials
    out[0] = (0.5/(Mf-1.0)) * tmp[0]
    out[1:M-1] = (1.0/(Mf-1.0)) * tmp[1:M-1]
    if dealiasing:
        out[M-1] = (1.0/(Mf-1.0)) * tmp[M-1]
    else:
        out[M-1] = (0.5/(Mf-1.0)) * tmp[M-1]

    return out

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

def test_roll_profile(CNSTS):

    """
    Use the roll profile from the SSP to check that differentiation and
    transformation are working correctly.
    """

    M = CNSTS['M']
    N = CNSTS['N']
    Mf = CNSTS['Mf']
    Nf = CNSTS['Nf']
    Lx = CNSTS['Lx']
    Ly = CNSTS['Ly']
    kx = CNSTS['kx']

    gamma = pi / Ly
    p = optimize.fsolve(lambda p: p*tan(p) + gamma*tanh(gamma), 2)
    oneOverC = ones(M)
    oneOverC[0] = 1. / 2.

    V = zeros((M, 2*N+1), dtype = 'complex')

    for m in range(0,M,2):
        V[m, 1] = 2*oneOverC[m]*( ((-1)**(m/2))*(special.jv(m,p)/cos(p)) - 
                    special.iv(m,gamma)/cosh(gamma) )
        V[m, 2*N] = 2*oneOverC[m]*( ((-1)**(m/2))*(special.jv(m,p)/cos(p)) - 
                    special.iv(m,gamma)/cosh(gamma) )
    del m        

    Normal = ( cos(p)*cosh(gamma) ) / ( cosh(gamma) - cos(p) )
    V = 0.5 * Normal * V
    actualSpec = V

    y_points = cos(pi*arange(Mf)/(Mf-1))
    #x_points = linspace(0, 2.-(2./(2*Nf+1)), 2*Nf+1)
    xlen = 2*pi / kx
    x_points = linspace(0, xlen-(xlen/(2*Nf+1)), 2*Nf+1)

    GLreal = zeros((Mf, 2*Nf+1), 'complex')

    for i in range(2*Nf+1):
        # y dependence
        GLreal[:,i] = Normal*cos(p*y_points) / cos(p)
        GLreal[:,i] += - Normal*cosh(gamma*y_points)/ cosh(gamma)
        # x dependence
        GLreal[:,i] = GLreal[:, i]*cos(kx*x_points[i])

    print 'values at realspace endpoints x: ', GLreal[0,0], GLreal[0,2*Nf]

    actualRYderiv = zeros((Mf, 2*Nf+1), 'complex')

    for i in range(2*Nf+1):
        # y dependence
        actualRYderiv[:,i] = - Normal*p*sin(p*y_points) / cos(p)
        actualRYderiv[:,i] += - Normal*gamma*sinh(gamma*y_points) / cosh(gamma)
        # x dependence
        actualRYderiv[:,i] = actualRYderiv[:, i]*cos(kx*x_points[i])

    #imshow(real(GLreal), origin='lower')
    #imshow(real(actualRYderiv), origin='lower')
    #show()


    print """
    -----------------------
    Test Transformations:
    -----------------------
    """
    print """
    --------------
    Orthogonality 
    --------------
    """

    ## transform is inverse of inverse transform?
    inverseTest1 = to_spectral(to_physical(actualSpec, CNSTS), CNSTS)
    inverseTest2 = to_spectral_2(to_physical_2(actualSpec, CNSTS), CNSTS)
    inverseTest3 = to_spectral_2(to_physical_2(inverseTest2, CNSTS), CNSTS)


    print 'transform is inverse of inverse transform? '
    print 'two 1D transforms method', allclose(actualSpec, inverseTest1)
    print '1 2D transform method', allclose(actualSpec, inverseTest2)
    print '1 2D transform method', allclose(actualSpec, inverseTest3)

    print 'if you start from real space? ', allclose(GLreal, to_physical(to_spectral(GLreal, CNSTS), CNSTS))
    print '1 2D transform', allclose(GLreal,
                                     to_physical_2(to_spectral_2(GLreal, CNSTS), CNSTS))


    print 'to physical ifft is same as fft?'
    result1 =  to_physical(actualSpec, CNSTS)
    result2 = to_physical_3(actualSpec, CNSTS)
    print allclose(result1,result2)
    #print linalg.norm( (result1-result2))
    #imshow(real(result2), origin='lower')
    #colorbar()
    #show()
    #imshow(real(result1), origin='lower')
    #colorbar()
    #show()


    ## Backwards Test ##
    print """
    --------------------------------------
    Test transformation to physical space.
    --------------------------------------
    """

    stupid = stupid_transform_i(2*actualSpec[:,1], CNSTS)
    print 'stupid transfrom the same as analytic GLpoints'
    print allclose(GLreal[:,0], stupid)

    physicalTest = to_physical(actualSpec, CNSTS)
    physicalTest2 = real(to_physical_2(actualSpec,CNSTS))
    physicalTest3 = real(to_physical_3(actualSpec,CNSTS))


    print 'actual real space = transformed analytic spectrum?', allclose(GLreal,
                                                                         physicalTest)

    print '2D transform is the same as 2 1D transforms with conj fft?', allclose(physicalTest, 
                                                                   physicalTest2)

    print '2D transform is the same as 2 1D transforms with ifft?', allclose(physicalTest, 
                                                                   physicalTest3)
    
    #print 'difference: ', linalg.norm(physicalTest2-physicalTest)
    #print 'difference Fourier dir: ', (physicalTest2-physicalTest)[M/2,:]
    #print 'difference Cheby dir: ', (physicalTest2-physicalTest)[:,N/2]

    plot(y_points, real(physicalTest[:,10]), 'b')
    plot(y_points, real(GLreal[:,10]), 'r+')
    show()

    plot(x_points, real(physicalTest[M/2,:]), 'b')
    plot(x_points, real(GLreal[M/2,:]), 'r+')
    show()

    imshow(real(GLreal),  origin='lower')
    colorbar()
    show()
    imshow(real(physicalTest),  origin='lower')
    colorbar()
    show()

    #print 'the maximum difference in the arrays ', amax(real(GLreal) -real(physicalTest))

    ## Forwards test ##
    print """
    --------------------------------------
    Test transformation to spectral space.
    --------------------------------------
    """
    cSpec = to_spectral(GLreal, CNSTS)

    print 'analytic spectrum = transformed GL spectrum?', allclose(actualSpec,
                                                                   cSpec)
    #plot(real(cSpec[:,1]), 'b')
    #plot(real(actualSpec[:,1]), 'r+')
    #show()

    #plot(real(cSpec[2,:]), 'b')
    #plot(real(actualSpec[2,:]), 'r+')
    #show()

    #plot(y_points, GLreal)
    #plot(y_points, physical_test, '+')
    #show()

    SpectralTest2 = to_spectral_2(GLreal, CNSTS)
    print '2D transform is the same as 2 1D transforms?', allclose(cSpec, 
                                        SpectralTest2)

    #print 'difference: ', linalg.norm(SpectralTest2-cSpec)
    #print 'difference Fourier dir: ', (SpectralTest2-cSpec)[1,:]
    #print 'difference Cheby dir: ', (SpectralTest2-cSpec)[:,1]

    # Products
    tsm.initTSM(N_=N, M_=M, kx_=kx)
    
    flatSpec = fftshift(actualSpec, axes=1)
    flatSpec = flatSpec.T.flatten()
    matprod = dot(tsm.prod_mat(flatSpec), flatSpec)
    matprod = matprod.reshape(2*N+1, M).T 
    matprod = ifftshift(matprod, axes=-1)

    print 'compare matrix product code with python fft products'

    pyprod = to_spectral(physicalTest2*physicalTest2, CNSTS) # * (2*Nf+1)**2
    #imshow(real(physicalTest2*physicalTest2))
    #colorbar()
    #show()
    #imshow(real(to_physical(matprod, CNSTS)))
    #colorbar()
    #show()
    
    print allclose(pyprod, matprod)
    #print linalg.norm(pyprod - matprod)
    #imshow(real(pyprod -matprod))
    #colorbar()
    #show()

    print """
    -----------------------
    Test y derivatives:
    -----------------------
    """

    yDerivTest = dy(actualSpec, CNSTS)
    yyDerivTest = dy(yDerivTest, CNSTS)
    yyyDerivTest = dy(yyDerivTest, CNSTS)
    yyyyDerivTest = dy(yyyDerivTest, CNSTS)

    print 'dy of the spectrum in real space = real space analytic derivative? ',\
            allclose(to_physical(yDerivTest, CNSTS), actualRYderiv)

    tmpTest = dyy(actualSpec, CNSTS)

    print 'dyy of the spectrum is the same as a double application of dy? ',\
            allclose(to_physical(yyDerivTest, CNSTS), to_physical(tmpTest, CNSTS))

    tmpTest = d3y(actualSpec, CNSTS)

    print 'd3y of the spectrum is the same as a triple application of dy? ',\
            allclose(to_physical(yyyDerivTest, CNSTS), to_physical(tmpTest, CNSTS))

    tmpTest = d4y(actualSpec, CNSTS)

    print 'd4y of the spectrum is the same as quadruple application of dy? ',\
            allclose(to_physical(yyyyDerivTest, CNSTS), to_physical(tmpTest, CNSTS))

    print """
    -----------------------
    Test x derivatives:
    -----------------------
    """

    xDerivTest = dx(actualSpec, CNSTS)
    xxDerivTest = dx(xDerivTest, CNSTS)
    xxxDerivTest = dx(xxDerivTest, CNSTS)
    xxxxDerivTest = dx(xxxDerivTest, CNSTS)

    tmpTest = dxx(actualSpec, CNSTS)

    print 'dxx of the spectrum is the same as a double application of dx? ',\
            allclose(to_physical(xxDerivTest, CNSTS), to_physical(tmpTest, CNSTS))

    tmpTest = d3x(actualSpec, CNSTS)

    print 'd3x of the spectrum is the same as a triple application of dx? ',\
            allclose(to_physical(xxxDerivTest, CNSTS), to_physical(tmpTest, CNSTS))

    tmpTest = d4x(actualSpec, CNSTS)

    print 'd4x of the spectrum is the same as a quadruple application of dx? ',\
            allclose(to_physical(xxxxDerivTest, CNSTS), to_physical(tmpTest, CNSTS))


    print """
    -----------------------
    Test mixed derivatives:
    -----------------------
    """

def test_prods(CNSTS):
    """
    tests the transform methods for products of fields.
    """

    M = CNSTS['M']
    N = CNSTS['N']
    Mf = CNSTS['Mf']
    Nf = CNSTS['Nf']
    Lx = CNSTS['Lx']
    Ly = CNSTS['Ly']

    print '100 products of random matrices:'

    As = zeros((M, 2*N+1), dtype='complex')
    Bs = zeros((M, 2*N+1), dtype='complex')

    for i in range(1,2*N/3+1):
        As[:2*M/3,i] = rand(2*M/3) + rand(2*M/3)*1.j
        As[:2*M/3,2*N+1-i] = conj(As[:2*M/3,i])
        Bs[:2*M/3,i] = rand(2*M/3) + rand(2*M/3)*1.j
        Bs[:2*M/3,2*N+1-i] = conj(Bs[:2*M/3,i])

    Aold = copy(As)
    Bold = copy(Bs)

    for i in range(1000):

        A = to_physical_2(As, CNSTS)
        B = to_physical_2(Bs, CNSTS)

        C = A*B

        As = to_spectral_2(A, CNSTS)
        Bs = to_spectral_2(B, CNSTS)

    print allclose(Aold,As), allclose(Bold,Bs)
    print linalg.norm(Aold)/linalg.norm(As)

def test_c_version(CNSTS):

    """
    Test the C version of the code. Make sure constants are the same across
    codes until I implement passing of this info back and forth.

    Tests will be performed by comparing the results of the C code with the
    results of this code, rather than the true results. this buys some time
    before I have ot do the boring job of working out how to do the transform in
    C.

    """
    M = CNSTS['M']
    N = CNSTS['N']
    Mf = CNSTS['Mf']
    Nf = CNSTS['Nf']
    Lx = CNSTS['Lx']
    Ly = CNSTS['Ly']
    kx = CNSTS['kx']

    gamma = pi / Ly
    p = optimize.fsolve(lambda p: p*tan(p) + gamma*tanh(gamma), 2)
    oneOverC = ones(M)
    oneOverC[0] = 1. / 2.

    #V = zeros((M, 2*N+1), dtype = 'complex')

    #for m in range(0,M,2):
    #    V[m, 1] = 2*oneOverC[m]*( ((-1)**(m/2))*(special.jv(m,p)/cos(p)) - 
    #                special.iv(m,gamma)/cosh(gamma) )
    #    V[m, 2*N] = 2*oneOverC[m]*( ((-1)**(m/2))*(special.jv(m,p)/cos(p)) - 
    #                special.iv(m,gamma)/cosh(gamma) )
    #del m        

    #Normal = ( cos(p)*cosh(gamma) ) / ( cosh(gamma) - cos(p) )
    #V = 0.5 * Normal * V
    #actualSpec = V

    actualSpec, _ = pickle.load(open('pf-N5-M40-kx1.31-Re3000.0.pickle', 'r'))
    actualSpec = decide_resolution(actualSpec, 5, 40, CNSTS)
    actualSpec = actualSpec.reshape(2*N+1, M).T
    actualSpec = ifftshift(actualSpec, axes=1)


    # insert stupider spectrum
    #actualSpec = zeros((M,2*N+1), dtype='complex')
    #actualSpec[:M/3,5] = r_[M/3:0:-1]
    #actualSpec[:M/3, 0] = r_[M/3:0:-1]
    #actualSpec[2*M/3:,0] = actualSpec[2*M/3:,0] * 1e-6 * rand(M-2*M/3)*1.j
    #actualSpec[2*M/3:,5] = actualSpec[2*M/3:,5] * 1e-6 * rand(M-2*M/3)*1.j
    #actualSpec[:,2*N-4] = r_[0:M]
    #actualSpec[:2*M/3,1:2*N/3 + 1] = rand(2*M/3, 2*N/3) + 1.j*rand(2*M/3, 2*N/3)
    #actualSpec[:, N+1:] = conj(fliplr(actualSpec[:, 1:N+1]))
    #actualSpec[:M, 1:N + 1] = rand(M, N) + 1.j*rand(M, N)
    #actualSpec[:, N+1:] = conj(fliplr(actualSpec[:, 1:N+1]))

    # save the initial state
    f = h5py.File("initial.h5", "w")
    dset = f.create_dataset("psi", ((2*N+1)*M,), dtype='complex')
    dset[...] = actualSpec.T.flatten()
    f.close()

    # call the c program
    subprocess.call(["./test_fields"])
    subprocess.call(["./test_fields_1"])

    # Read in the c programs output
    # Reshape is because fft insists on 1D double complex arrays.
    # T is because this program uses fortran order not c order for y and x.
    # slice is because the rest of the array is junk I carry round the c program
    # to speed up the transforms.
    
    #ctestSpec = load("testSpec.npy").reshape(2*N+1, 2*M-2).T[:M, :]
    #ctestdxSpec = load("testdx.npy").reshape(2*N+1, 2*M-2).T[:M, :]
    #ctestdySpec = load("testdy.npy").reshape(2*N+1, 2*M-2).T[:M, :]

    #ctestSpec = load("testSpec.npy")
    ctestSpec = load_hdf5_state("testSpec.h5")
    ctestSpec = ctestSpec.reshape(2*N+1, M).T
    #print ctestSpec[:,1]

    ctestdxSpec = load_hdf5_state("testdx.h5").reshape(2*N+1, M).T
    ctestdySpec = load_hdf5_state("testdy.h5").reshape(2*N+1, M).T

    # Compare python code and C code
    print "Python and C code have the same initial spectra: ", allclose(ctestSpec, actualSpec)
    print linalg.norm(ctestSpec-actualSpec)
    
    #print "c-python: ", (ctestSpec - actualSpec)[:,1]
    #print "c: ", (ctestSpec)[:,1]
    #print "python: ", (actualSpec)[:,1]
    #print "all should be zero: ", linalg.norm(ctestSpec[:,0]),
    #linalg.norm(ctestSpec[:,2:N+1]), linalg.norm(ctestSpec[:,N+1:2*N])

    print """
    -------------------
    Test dy
    -------------------
    """
    dySpec = dy(actualSpec, CNSTS)

    print "Python and C code give the same derivative: ", allclose(ctestdySpec,
                                                                  dySpec)
    d4ySpec = dy(dy(dy(dySpec, CNSTS), CNSTS), CNSTS)
    ctestd4ySpec = load_hdf5_state("testd4y.h5").reshape(2*N+1, M).T
    print "Python and C code give the same d4y: ", allclose(ctestd4ySpec,
                                                                  d4ySpec)
    if not allclose(ctestd4ySpec, d4ySpec):
        print "fourier modes"
        print "mode", 0, linalg.norm(ctestd4ySpec[:,0] - d4ySpec[:, 0])
        for n in range(1,N):
            print "mode", n, linalg.norm(ctestd4ySpec[:, n] - d4ySpec[:, n])
            print "mode", -n, linalg.norm(ctestd4ySpec[:, 2*N+1-n] - d4ySpec[:, 2*N+1-n])

        print "zeroth fourier diff", (ctestd4ySpec[:, 0] - d4ySpec[:, 0]) /d4ySpec[:,0]

        print "chebyshev modes"
        for m in range(M):
            print "mode", m, linalg.norm(ctestd4ySpec[m, :] - d4ySpec[m, :])



    flatspec = fftshift(actualSpec, axes=1)
    flatspec = flatspec.T.flatten()

    tsm.initTSM(N_=N, M_=M, kx_=kx)

    MDX = tsm.mk_diff_x()
    MDY = tsm.mk_diff_y()

    matd4ySpec = dot(MDY, dot(MDY, dot(MDY, dot(MDY, flatspec)))) 
    matd4ySpec = matd4ySpec.reshape(2*N+1, M).T
    matd4ySpec = ifftshift(matd4ySpec, axes=-1)

    print "Python and matrix method give the same d4y: ", allclose(matd4ySpec,
                                                                  d4ySpec)

    if not allclose(matd4ySpec, d4ySpec):
        print "fourier modes"
        print "mode", 0, linalg.norm(matd4ySpec[:,0] - d4ySpec[:, 0])
        for n in range(1,N):
            print "mode", n, linalg.norm(matd4ySpec[:, n] - d4ySpec[:, n])
            print "mode", -n, linalg.norm(matd4ySpec[:, 2*N+1-n] - d4ySpec[:, 2*N+1-n])

        print "chebyshev modes"
        for m in range(M):
            print "mode", m, linalg.norm(matd4ySpec[m, :] - d4ySpec[m, :])

    print """
    -------------------
    Test dx
    -------------------
    """

    dxSpec = dx(actualSpec, CNSTS)

    print "Python and C code give the same derivative: ", allclose(ctestdxSpec,
                                                                  dxSpec)
    if not (allclose(ctestdxSpec,dxSpec)):
        for n in range(1,N+1):
            print "mode", n, linalg.norm(ctestdxSpec[:, n] - dxSpec[:, n])
            print "mode", -n, linalg.norm(ctestdxSpec[:, 2*N+1-n] - dxSpec[:, 2*N+1-n])
        print allclose(ctestdxSpec,actualSpec)
        print ctestdxSpec[:,5]
        print dxSpec[:,5]

        print ctestdxSpec[:,2*N-4]
        print dxSpec[:,2*N-4]

    print "test matrix multiplication method and compare with python sum"

    flatspec = fftshift(actualSpec, axes=1)
    flatspec = flatspec.T.flatten()
    matdx = dot(MDX, flatspec)

    matdyypsi = dot( dot(MDY, MDY), flatspec)

    matdxpsi2D = matdx.reshape(2*N+1, M).T
    matdxpsi2D = ifftshift(matdxpsi2D,axes=-1)

    print 'checking matrix deriv is the same as the python looped derivative'
    print allclose(matdxpsi2D, dxSpec)
    if not allclose(matdxpsi2D, dxSpec):
        for n in range(1,N):
            print "mode", n, linalg.norm(matdx[(N+n)*M:(N+1+n)*M] - dxSpec[:, n])
            print "mode", -n, linalg.norm(matdx[(N-n)*M: (N+1-n)*M] - dxSpec[:, 2*N+1-n])

    print """
    -----------------------
    Test Transformations:
    -----------------------
    """

    print (2*Nf+1)*(2*Mf-2) 
    # the normalisation factor comes in becuase we are doing a single 2D fft
    ctestPhys = load_hdf5_state("testPhysicalT.h5").reshape(2*Nf+1, 2*Mf-2).T[:Mf, :]

    actualPhys = real(to_physical(actualSpec,CNSTS))
    testBool = allclose(actualPhys, real(ctestPhys))
    print "Physical Transform: C transform is the same as python transform?",testBool

    if testBool == False:
        print "\ndifference: ", linalg.norm(ctestPhys - actualPhys)
        print "complex residue", linalg.norm(imag(ctestPhys))

        print 'max diff', amax(ctestPhys - actualPhys)
        print 'argmax diff', argmax(ctestPhys - actualPhys)

        imshow(real(ctestPhys - actualPhys), origin='lower')
        colorbar()
        show()
        #imshow(real(ctestPhys), origin='lower')
        #show()
    ctestPhys1D = load_hdf5_state("testPhys_1D.h5").reshape(2*Nf+1, 2*Mf-2).T[:Mf, :]
    testBool = allclose(actualPhys, ctestPhys1D)
    print "Physical Transform: C 1D transform is the same as python transform?",testBool

    ctestSpec = load_hdf5_state("testSpectralT.h5").reshape(2*N+1, M).T 
    python2spec = to_spectral_2(actualPhys, CNSTS)
    testBool = allclose(python2spec, ctestSpec)
    print "Spectral Transform: C transform is the same as python transform?",testBool

    if testBool == False:
        print "\ndifference: ", linalg.norm(ctestSpec - python2spec)

        print 'max diff', amax(ctestSpec - actualSpec)
        print 'argmax diff', argmax(ctestSpec - actualSpec)

        print "\nmodal differences\n"
        print "mode", 0, linalg.norm(ctestSpec[:, 0] - python2spec[:, 0])
        for n in range(1,N+1):
            print "mode", n, linalg.norm(ctestSpec[:, n] - python2spec[:, n])
            print "mode", -n, linalg.norm(ctestSpec[:, 2*N+1-n] - python2spec[:, 2*N+1-n])
            print "python real?", allclose(python2spec[:, n], 
                                           conj(python2spec[:,2*N+1-n]))
            print "c real?", allclose(ctestSpec[:, n], 
                                           conj(ctestSpec[:,2*N+1-n]))
            print "ratio of terms", real(ctestSpec[0,n]/python2spec[0,n])
            print "ratio of terms", real(ctestSpec[1,n]/python2spec[1,n])
            print "ratio of terms", real(ctestSpec[2,n]/python2spec[2,n])


    phystest = zeros((Mf, 2*Nf+1), dtype='complex')

    for i in range(2*Nf+1):
        for j in range(Mf):
	    phystest[j,i] =  cos(i*pi/(2.*Nf)) * tanh(j*pi/(Mf-1.))
	    #phystest[j,i] = i + j

    pythonSpec3 = to_spectral(phystest, CNSTS)
    #ctestSpec3 = load_hdf5_state("testSpectralT2.h5").reshape(2*Nf+1, 2*Mf-2).T
    ctestSpec3 = load_hdf5_state("testSpectralT2.h5").reshape(2*N+1, M).T 
    cphystest = load_hdf5_state("phystest2.h5").reshape(2*Nf+1, 2*Mf-2).T[:Mf, :]

    print 'Spectral Transform: '
    print 'c code has same test?', allclose(cphystest, phystest)
    print 'From real space problem to spectral space, comparision of python and C'
    testBool =  allclose(pythonSpec3, ctestSpec3)
    print testBool
    if testBool == False:
        print "\ndifference: ", linalg.norm(pythonSpec3 - ctestSpec3)
        print "\nmodal differences\n"
        print "mode", 0, linalg.norm(ctestSpec3[:, 0] - pythonSpec3[:, 0])
        for n in range(1,N+1):
            print "mode", n, linalg.norm(ctestSpec3[:, n] - pythonSpec3[:, n])
            print "mode", -n, linalg.norm(ctestSpec3[:, 2*N+1-n] - pythonSpec3[:, 2*N+1-n])

        print "mode", 0, "difference", ctestSpec3[:, 0] - pythonSpec3[:, 0]
        print pythonSpec3[M-1, 0]
        print ctestSpec3[M-1, 0]
        print 2*ctestSpec3[M-1, 0] - pythonSpec3[M-1,0]

        imshow(real(ctestSpec3), origin='lower')
        colorbar()
        show()
        imshow(real(pythonSpec3), origin='lower')
        colorbar()
        show()

    ctestSpec_1D = load_hdf5_state("testSpec_1D.h5").reshape(2*N+1, M).T 

    pythonSpecCheb = zeros((Mf, 2*Nf+1), dtype='complex')
    pythonSpecCheb = to_spectral(phystest,CNSTS)
    #for i in range(2*Nf+1):
    #    pythonSpecCheb[:,i] = forward_cheb_transform(phystest[:,i], CNSTS)

    testBool =  allclose(pythonSpecCheb, ctestSpec_1D)

    print 'Spectral Transform: '
    print 'From real space problem to spectral space, comparision of python and C 1D'
    print testBool
    if not testBool:
        print 'c'
        imshow(real(ctestSpec_1D))
        colorbar()
        show()
        print 'p'
        imshow(real(pythonSpecCheb))
        colorbar()
        show()
        print 'p-c'
        imshow(real(pythonSpecCheb-ctestSpec_1D))
        colorbar()
        show()

    print 'Spectral Transform: '
    print 'From real space problem to spectral space and back again, comparision of python and C'
    pythonPhys4 = to_physical(pythonSpec3, CNSTS)
    ctestPhys4 = load_hdf5_state("testPhysT4.h5").reshape(2*Nf+1, (2*Mf-2)).T[:Mf, :] 
    testBool = allclose(pythonPhys4, ctestPhys4)

    print testBool

    python2specR = copy(python2spec)
    for i in range(100):
        pythonPhys = to_physical_2(python2specR, CNSTS)
        python2specR = to_spectral_2(pythonPhys, CNSTS)
    del i

    testBool = allclose(python2spec, python2specR)

    print 'checking python fft products are equal to matrix method products'
    print 'vdyypsi'

    matvdyypsi = dot(tsm.prod_mat(-matdx), matdyypsi)


    physv = to_physical(-dxSpec, CNSTS)
    physdyy = to_physical(dy(dy(actualSpec, CNSTS), CNSTS), CNSTS)

    matdx = fftshift(dxSpec, axes=-1)
    matdx = matdx.T.flatten()
    matdyypsi = fftshift(dy(dy(actualSpec, CNSTS), CNSTS), axes=-1)
    matdyypsi = matdyypsi.T.flatten()

    matvdyypsi = dot(tsm.prod_mat(-matdx), matdyypsi)

    vdyypsi = to_spectral(physv*physdyy, CNSTS)

    #print "mode", 0, linalg.norm(matvdyypsi[(N)*M:(N+1)*M] - vdyypsi[:, 0])
    #for n in range(1,N+1):
    #    print "mode", n, linalg.norm(matvdyypsi[(N+n)*M:(N+1+n)*M] - vdyypsi[:, n])
    #    print "mode", -n, linalg.norm(matvdyypsi[(N-n)*M: (N+1-n)*M] - vdyypsi[:, 2*N+1-n])

    matvdyypsi2D = matvdyypsi.reshape(2*N+1, M).T
    matvdyypsi2D = ifftshift(matvdyypsi2D, axes=-1) 

    print allclose(matvdyypsi2D, vdyypsi)
    print linalg.norm(matvdyypsi2D - vdyypsi)


    print 'psipsi'
    psiR = real(to_physical(actualSpec, CNSTS))
    psipsi = to_spectral(psiR*psiR, CNSTS)
    matpsipsi = dot(tsm.prod_mat(flatspec), flatspec)
    matpsipsi2D = matpsipsi.reshape(2*N+1, M).T
    matpsipsi2D = ifftshift(matpsipsi2D, axes=-1) 

    print allclose(matpsipsi2D, psipsi)

    if not allclose(matpsipsi2D, psipsi):
        print "mode", 0, linalg.norm(matpsipsi[(N)*M:(N+1)*M] - psipsi[:, 0])
        for n in range(1,N+1):
            print "mode", n, linalg.norm(matpsipsi[(N+n)*M:(N+1+n)*M] - psipsi[:, n])
            print "mode", -n, linalg.norm(matpsipsi[(N-n)*M: (N+1-n)*M] - psipsi[:, 2*N+1-n])

        matpsipsi2D = matpsipsi.reshape(2*N+1, M).T
        matpsipsi2D = ifftshift(matpsipsi2D, axes=-1)
        
        diff = matpsipsi2D - psipsi
        print linalg.norm(diff)
        print diff.flatten()[argmax(diff)]

        imshow(real(psiR*psiR), origin='lower')
        colorbar()
        show()
        rsmat =real(to_physical_2(matpsipsi2D, CNSTS))
        imshow(rsmat, origin='lower')
        colorbar()
        show()
        print 'difference between physical space representations', linalg.norm(rsmat-psiR*psiR)

    print 'check c products by fft_convolve function are same as normal c products'

    psipsic = load_hdf5_state("psipsi.h5").reshape(2*N+1, M).T
    psipsic2 = load_hdf5_state("fft_convolve.h5").reshape(2*N+1, M).T
    testBool = allclose(psipsic, psipsic2)
    print testBool
    if not testBool:
        print linalg.norm(psipsic-psipsic2)


    print 'Checking c products are the same as python products'

    psipsicR = load_hdf5_state("psipsiR.h5").reshape(2*Nf+1, 2*Mf-2).T[:Mf,:]

    print allclose(psipsic, psipsi)
    print 'difference ', linalg.norm(psipsic-psipsi)
    if not allclose(psipsic, psipsi):
        print linalg.norm(psipsic-psipsi)
        print 'difference between their physical representations '
        print linalg.norm(psipsicR - (psiR*psiR))
        imshow(real(psipsicR), origin='lower')
        colorbar()
        show()


    print "Repeated Transforms: python field is stable after 100 transforms?",testBool
    if testBool == False:
        print "\ndifference: ", linalg.norm(python2spec - python2specR)


    ctestSpecR = load_hdf5_state("testSpectralTR.h5").reshape(2*N+1, M).T 
    testBool = allclose(ctestSpecR, ctestSpec)
    print "Repeated Transforms: C field is stable after 100 transforms?",testBool
    if testBool == False:
        print "\ndifference: ", linalg.norm(ctestSpec - ctestSpecR)
        print "\nmodal differences\n"
        print "mode", 0, linalg.norm(ctestSpec[:, 0] - ctestSpecR[:, 0])
        for n in range(1,N+1):
            print "mode", n, linalg.norm(ctestSpec[:, n] - ctestSpecR[:, n])
            print "mode", -n, linalg.norm(ctestSpec[:, 2*N+1-n] - ctestSpecR[:, 2*N+1-n])

        print linalg.norm(ctestSpec[M-1,0] - ctestSpecR[M-1,0 ])

### MAIN ###

if __name__ == "__main__":

    CNSTS = set_constants(M=50, N=5, kx=1.31, dealiasing=True)

    #test_roll_profile(CNSTS)

    #test_diff(CNSTS, testFunc=lambda x: 1-x**2)

    #test_prods(CNSTS)

    test_c_version(CNSTS)

