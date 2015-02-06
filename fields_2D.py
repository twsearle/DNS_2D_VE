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
        Mf = 3*M/2
        Nf = 3*N/2
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

    tmp = copy(in2D)
    
    if CNSTS['dealiasing']:
        tmp[:, 2*N/3 + 2 : 2*N+1 - 2*N/3] = 0 
        tmp[2*M/3:, :] = 0

    # Perform the FFT across the x and z directions   

    _realtmp = zeros((2*M-2, 2*N+1), dtype='double')
    out2D = zeros((2*M-2, 2*N+1), dtype='complex')

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
    _realtmp[M:, :] = _realtmp[M-2:0:-1, :]

    # The first half contains the vector on the Gauss-Labatto points * c_k
    _realtmp[0, :] = 2*_realtmp[0, :]
    _realtmp[M-1, :] = 2*_realtmp[M-1, :]

    # Perform the transformation
    out2D = 0.5*fftpack.rfft(_realtmp, axis=0 )

    normImag = linalg.norm(imag(out2D[0:M, :]))
    if normImag > 1e-12:
        print "output after Cheb transform in to_physical is not real, norm = ", normImag

    out2D = real(out2D)
    
    return out2D[0:M, :]

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

    out2D[:M, :N+1] = conj(in2D[:,:N+1]) / (2*Nf+1)
    out2D[:M, 2*Nf+1-N:] = conj(in2D[:,N+1:]) / (2*Nf+1)

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


    # Perform the FFT across the x direction   
    _realtmp = zeros((2*M-2, 2*N+1), dtype='double')
    out2D = zeros((M, 2*N+1), dtype='complex')

    # The first half contains the vector on the Gauss-Labatto points
    _realtmp[:M, :] = real(in2D)

    # The second half contains the vector on the Gauss-Labatto points excluding
    # the first and last elements and in reverse order
    _realtmp[M:, :] = _realtmp[M-2:0:-1, :]

    # Perform the transformation on this temporary vector
    # TODO: Think about antialiasing here
    _realtmp = fftpack.rfft(_realtmp, axis=0)

    out2D = _realtmp[:M, :]

    # Renormalise and divide by c_k to convert to Chebyshev polynomials
    out2D[0, :] = (0.5/(M-1.0))*out2D[0, :]
    out2D[1:M-1, :] = (1.0/(M-1.0))*out2D[1:M-1, :]
    out2D[M-1, :] = (0.5/(M-1.0))*out2D[M-1, :]

    # test imaginary part of the fft is zero
    normImag = linalg.norm(imag(out2D))
    if normImag > 1e-12:
        print "output of cheb transform in to_spectral is not real, norm = ", normImag 
        print 'highest x, z modes:'
        print imag(out2D)[0, N-3:N+1, L-3:L+1]

    out2D = fftpack.fft(out2D)

    if CNSTS['dealiasing']:

        # zero modes for Fourier dealiasing

        out2D[:, 2*N/3 + 2 : 2*N+1 - 2*N/3] = 0 

        # TODO: zero modes for Chebyshev dealiasing?

    return out2D

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
    tmp[0, :] = (0.5/(Mf-1.0))*tmp[0, :]
    tmp[1:Mf-1, :] = (1.0/(Mf-1.0))*tmp[1:Mf-1, :]
    tmp[Mf-1, :] = (0.5/(Mf-1.0))*tmp[Mf-1, :]

    ## remove the aliased modes and copy into output
    out2D[:, :N+1] = tmp[:M, :N+1]
    out2D[:, N+1:] = tmp[:M, 2*Nf+1-N:]
    print "is the temp matrix spectrum of real space?"
    print allclose(tmp[:Mf, 1:Nf+1], conj(tmp[:Mf, 2*Nf+1:Nf:-1])) 
    print "is the output matrix spectrum of real space?",
    print allclose(out2D[:, 1:N+1], conj(out2D[:M, 2*N+1:N:-1])) 

    return out2D

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
    actualSpec = (2*Nf+1) * V

    y_points = cos(pi*arange(Mf)/(Mf-1))
    x_points = linspace(0, 2.-(2./(2*Nf+1)), 2*Nf+1)

    GLreal = zeros((Mf, 2*Nf+1), 'complex')

    for i in range(2*Nf+1):
        # y dependence
        GLreal[:,i] = Normal*cos(p*y_points) / cos(p)
        GLreal[:,i] += - Normal*cosh(gamma*y_points)/ cosh(gamma)
        # x dependence
        GLreal[:,i] = GLreal[:, i]*cos(kx*x_points[i])

    actualRYderiv = zeros((Mf, 2*Nf+1), 'complex')

    for i in range(2*N+1):
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

    ## transform is inverse of inverse transform?
    inverseTest1 = to_spectral(to_physical(actualSpec, CNSTS), CNSTS)
    inverseTest2 = to_spectral_2(to_physical_2(actualSpec, CNSTS), CNSTS)
    inverseTest3 = to_spectral_2(to_physical_2(inverseTest2, CNSTS), CNSTS)


    print 'transform is inverse of inverse transform? '
    print 'two 1D transforms method', allclose(actualSpec, inverseTest1)
    print '1 2D transform method', allclose(actualSpec, inverseTest2)
    print '1 2D transform method', allclose(actualSpec, inverseTest3)

    print inverseTest2


    if CNSTS['dealiasing'] == False:
        print 'and if you start from real space? ', allclose(GLreal, to_physical(to_spectral(GLreal, CNSTS), CNSTS))
        print '1 2D transform', allclose(GLreal,
                                         to_physical_2(to_spectral_2(GLreal, CNSTS), CNSTS))
    else:
        print 'and if you start from real space? '
        print """
        doesnt make sense to do this: real space contains information that we
        zero out when we dealias
        """

    ## Backwards Test ##

    physicalTest = to_physical(actualSpec, CNSTS)

    print 'actual real space = transformed analytic spectrum?', allclose(GLreal,
                                                                         physicalTest)
    # To get both transforms to be forward transforms, need to flip Fourier
    # modes and renormalise
    physicalTest2 = real(to_physical_2(actualSpec,CNSTS))
    print '2D transform is the same as 2 1D transforms?', allclose(physicalTest, 
                                        physicalTest2)
    
    #print 'difference: ', linalg.norm(physicalTest2-physicalTest)
    #print 'difference Fourier dir: ', (physicalTest2-physicalTest)[M/2,:]
    #print 'difference Cheby dir: ', (physicalTest2-physicalTest)[:,N/2]

    #plot(real(physicalTest[:,100]), 'b')
    #plot(real(GLreal[:,100]), 'r+')
    #show()

    #plot(real(physicalTest[M/2,:]), 'b')
    #plot(real(GLreal[M/2,:]), 'r+')
    #show()

    #imshow(real(GLreal) - real(physicalTest), origin='lower')
    #colorbar()
    #show()

    #print 'the maximum difference in the arrays ', amax(real(GLreal) -real(physicalTest))

    ## Forwards test ##
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

    print (tmpTest - yyDerivTest)[:, 1]
    print dyy(ones((M,2*N+1)), CNSTS)[:,1]

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





def test_diff(CNSTS, testFunc=lambda x: 1-x**2):

    """
    Check correctness of x,y differentiation.

    I differentiate the profile in all directions in turn: x, y

    Then I move onto a mixed profile. a superposition of 2 functions in the 2
    directions and check that differentiation works.

    """

    M = CNSTS['M']
    N = CNSTS['N']
    Lx = CNSTS['Lx']
    Ly = CNSTS['Ly']

    print '\n----------------------------------------'
    print '2D differentiation of a function'
    print '-----------------------------------------'

    # y
    print '\n-------------------'
    print 'testing y direction'
    print '-------------------'
    testVec = zeros((M, 2*N+1), dtype='complex')

    y_points = cos(pi*arange(M) / (M-1))

    testVec[:,0] = testFunc(y_points)

    testVec = to_spectral(testVec, CNSTS)
    yParabola = testVec[:,0]


    derivVec = dy(testVec, CNSTS)
    realVec = to_physical(derivVec, CNSTS)

    fig = figure()
    fig.suptitle('y direction')
    xlabel('y')
    plot(y_points, realVec[:, 0], 'g+')
    show()

    # x
    #print '\n-------------------'
    #print 'testing x direction'
    #print '-------------------'
    #testVec = zeros((M, 2*N+1), dtype='complex')

    #x_points = Lx*arange(2*N+1) / (2*N) - Lx/2.
    #xParabola = fftpack.fft(testFunc(x_points))

    #testVec[0,:] = xParabola

    #derivVec = dx(testVec, CNSTS)

    #realVec = to_physical(derivVec, CNSTS)

    #fig = figure()
    #fig.suptitle('x direction')
    #xlabel('x')
    #plot(x_points, realVec[0, :], 'g+')
    #plot(x_points, realVec[2, :], 'r.')
    #show()

    # Parabolas in both directions at the same time
    print '\n--------------------------------------------------------'
    print 'testing parabolas in both directions at the same time'
    print '----------------------------------------------------------'

    print 'not implemented yet!'

    #testVec = zeros((M, 2*N+1), dtype='complex')

    #testVec[:,0] = (2*N+1) * yParabola
    #testVec[0,:] += xParabola

    #realdy = dy(testVec, CNSTS)
    #realdx = dx(testVec, CNSTS)

    #print 'transform y derivative'
    #realdy = to_physical(realdy, CNSTS)
    #print 'transform x derivative'
    #realdx = to_physical(realdx, CNSTS)

    #_fail = False

    #for _m in range(M):
    #    for _n in range(2*N+1):
    #        if not allclose(realdx[_m, _n, :], realdx[0, 0, :], atol=1e-6):
    #            print '(y, z)', _m, _n
    #            _fail = True
    #            break

    #if _fail:
    #    print 'Error: the partial derivative w.r.t x is not independent'
    #    print '\tof the other directions'
    #    _fail = False
    #else: 
    #    print 'the partial derivative w.r.t x is y,z independent'

    #for _n in range(2*N+1):
    #    for _i in range(2*L+1):
    #        if not allclose(realdy[:, _n, _i], realdy[:, 0, 0], atol=1e-6):
    #            print '(z, x)', _n, _i
    #            _fail = True
    #            break

    #if _fail:
    #    print 'Error: the partial derivative w.r.t y is not independent'
    #    print '\tof the other directions'
    #    _fail = False
    #else: 
    #    print 'the partial derivative w.r.t y is z,x independent'


    #fig = figure()
    #fig.suptitle('partial derivatives of $f = 1-x^2 + 1-y^2 + 1-z^2$')
    #xlabel('$x_i$')
    #ylabel('$\partial_{i} f $')
    #plot(x_points, -2*x_points, 'b')
    #plot(x_points, realdx[0, 0, :], 'b+')
    #plot(y_points, realdy[:, 0, 0], 'g.')
    #plot(z_points, realdz[0, :, :], 'rx')
    #show()

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

    # V = zeros((M, 2*N+1), dtype = 'complex')

    # for m in range(0,M,2):
    #     V[m, 1] = 2*oneOverC[m]*( ((-1)**(m/2))*(special.jv(m,p)/cos(p)) - 
    #                 special.iv(m,gamma)/cosh(gamma) )
    #     V[m, 2*N] = 2*oneOverC[m]*( ((-1)**(m/2))*(special.jv(m,p)/cos(p)) - 
    #                 special.iv(m,gamma)/cosh(gamma) )
    # del m        

    # Normal = ( cos(p)*cosh(gamma) ) / ( cosh(gamma) - cos(p) )
    # V = 0.5 * Normal * V
    # actualSpec = (2*N+1) * V

    #actualSpec, _ = pickle.load(open('pf-N5-M40-kx1.31-Re3000.0.pickle', 'r'))
    #actualSpec = actualSpec.reshape(2*N+1, M).T
    #actualSpec = ifftshift(actualSpec, axes=1)
    #actualSpec[:,5] = 0
    #actualSpec[:,2*N-4] = 0


    # insert stupider spectrum
    actualSpec = zeros((M,2*N+1), dtype='complex')
    #actualSpec[:,5] = r_[0:M]
    actualSpec[:M/3, 0] = r_[0:M/3]
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

    # Read in the c programs output
    # Reshape is because fft insists on 1D double complex arrays.
    # T is because this program uses fortran order not c order for y and x.
    # TODO: Fix fortran order
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

    print 'checking matrix deriv is the same as the python looped derivative'
    for n in range(1,N):
        print "mode", n, linalg.norm(matdx[(N+n)*M:(N+1+n)*M] - dxSpec[:, n])
        print "mode", -n, linalg.norm(matdx[(N-n)*M: (N+1-n)*M] - dxSpec[:, 2*N+1-n])

    print """
    -----------------------
    Test Transformations:
    -----------------------
    """
    print 'checking python fft products are equal to matrix method products'
    # print 'vdyypsi'

    # matvdyypsi = dot(tsm.prod_mat(-matdx), matdyypsi)

    # physv = to_physical_2(-dxSpec, CNSTS)
    # physdyy = to_physical_2(dy(dy(actualSpec, CNSTS), CNSTS), CNSTS)
    # vdyypsi = to_spectral_2(physv*physdyy, CNSTS)

    # print "mode", 0, linalg.norm(matvdyypsi[(N)*M:(N+1)*M] - vdyypsi[:, 0])
    # for n in range(1,N+1):
    #     print "mode", n, linalg.norm(matvdyypsi[(N+n)*M:(N+1+n)*M] - vdyypsi[:, n])
    #     print "mode", -n, linalg.norm(matvdyypsi[(N-n)*M: (N+1-n)*M] - vdyypsi[:, 2*N+1-n])

    # matvdyypsi2D = matvdyypsi.reshape(2*N+1, M).T
    # matvdyypsi2D = ifftshift(matvdyypsi2D, axes=-1)

    # print allclose(matvdyypsi2D, vdyypsi)
    # print linalg.norm(matvdyypsi2D - vdyypsi)
    # print matvdyypsi2D[:, 2]
    # print vdyypsi[:, 2]

    print 'psipsi'
    psiR = to_physical_2(actualSpec, CNSTS)
    psipsi = to_spectral_2(psiR*psiR, CNSTS)
    matpsipsi = dot(tsm.prod_mat(flatspec), flatspec)
    matpsipsi2D = matpsipsi.reshape(2*N+1, M).T
    matpsipsi2D = ifftshift(matpsipsi2D, axes=-1)

    print "mode", 0, linalg.norm(matpsipsi[(N)*M:(N+1)*M] - psipsi[:, 0])
    for n in range(1,N+1):
        print "mode", n, linalg.norm(matpsipsi[(N+n)*M:(N+1+n)*M] - psipsi[:, n])
        print "mode", -n, linalg.norm(matpsipsi[(N-n)*M: (N+1-n)*M] - psipsi[:, 2*N+1-n])

    matpsipsi2D = matpsipsi.reshape(2*N+1, M).T
    matpsipsi2D = ifftshift(matpsipsi2D, axes=-1)

    print allclose(matpsipsi2D, psipsi)
    print linalg.norm(matpsipsi2D - psipsi)
    print "pick on one mode 0,2", psipsi[0,1], matpsipsi2D[0,1]
    print matpsipsi2D
    print psipsi
    exit(1)


    # the normalisation factor comes in becuase we are doing a single 2D fft
    #ctestPhys = load_hdf5_state("testPhysicalT.h5").reshape(2*N+1, 2*M-2).T[:M, :] 
    ctestPhys = load_hdf5_state("testPhysicalT.h5").reshape(2*Nf+1, 2*Mf-2).T[:Mf, :]

    actualPhys = real(to_physical_2(actualSpec,CNSTS))
    testBool = allclose(actualPhys, real(ctestPhys))
    print "Physical Transform: C transform is the same as python transform?",testBool

    if testBool == False:
        print "\ndifference: ", linalg.norm(ctestPhys - actualPhys)
        print "complex residue", linalg.norm(imag(ctestPhys))

        imshow(real(actualPhys), origin='lower')
        show()
        imshow(real(ctestPhys), origin='lower')
        show()
    
    #ctestSpec = load_hdf5_state("testSpectralT.h5").reshape(2*N+1, 2*M-2).T[:M, :] 
    #ctestSpec = load_hdf5_state("testSpectralT.h5").reshape(2*Nf+1, 2*Mf-2).T
    ctestSpec = load_hdf5_state("testSpectralT.h5").reshape(2*N+1, M).T 
    python2spec = to_spectral_2(actualPhys, CNSTS)
    testBool = allclose(python2spec, ctestSpec)
    print "Spectral Transform: C transform is the same as python transform?",testBool

    if testBool == False:
        print "\ndifference: ", linalg.norm(ctestSpec - python2spec)
        print "\nmodal differences\n"
        print "mode", 0, linalg.norm(ctestSpec[:, 0] - python2spec[:, 0])
        for n in range(1,N+1):
            print "mode", n, linalg.norm(ctestSpec[:, n] - python2spec[:, n])
            print "mode", -n, linalg.norm(ctestSpec[:, 2*N+1-n] - python2spec[:, 2*N+1-n])
            print "python real?", allclose(python2spec[:, n], 
                                           conj(python2spec[:,2*N+1-n]))
            print "c real?", allclose(ctestSpec[:, n], 
                                           conj(ctestSpec[:,2*N+1-n]))


    phystest = zeros((Mf, 2*Nf+1), dtype='complex')

    for i in range(2*Nf+1):
        for j in range(Mf):
	    phystest[j,i] = cos(i*pi/(2.*Nf)) * tanh(j*pi/(Mf-1.))
	    #phystest[j,i] = i + j

    pythonSpec3 = to_spectral_2(phystest, CNSTS)
    #ctestSpec3 = load_hdf5_state("testSpectralT2.h5").reshape(2*Nf+1, 2*Mf-2).T
    ctestSpec3 = load_hdf5_state("testSpectralT2.h5").reshape(2*N+1, M).T 
    cphystest = load_hdf5_state("phystest2.h5").reshape(2*Nf+1, 2*Mf-2).T[:Mf, :]

    print allclose(cphystest, phystest)
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

    print 'From real space problem to spectral space and back again, comparision of python and C'
    pythonPhys4 = to_physical_2(pythonSpec3, CNSTS)
    ctestPhys4 = load_hdf5_state("testPhysT4.h5").reshape(2*Nf+1, (2*Mf-2)).T[:Mf, :] 
    testBool = allclose(pythonPhys4, ctestPhys4)

    print testBool

    python2specR = copy(python2spec)
    for i in range(100):
        pythonPhys = to_physical_2(python2specR, CNSTS)
        python2specR = to_spectral_2(pythonPhys, CNSTS)
    del i

    testBool = allclose(python2spec, python2specR)

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

    CNSTS = set_constants(M=40, N=5, kx=1.31, dealiasing=False)

    #test_roll_profile(CNSTS)

    #test_diff(CNSTS, testFunc=lambda x: 1-x**2)

    #test_prods(CNSTS)

    test_c_version(CNSTS)

