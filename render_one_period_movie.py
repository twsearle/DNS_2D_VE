#------------------------------------------------------------------------------
#   Colour map movies for the Oscillatory flow problem
#
#   Last modified: Wed 22 Jun 10:56:49 2016
#
#------------------------------------------------------------------------------

#MODULES
import sys
from scipy import *
from scipy import linalg
from scipy import fftpack
import numpy as np
from numpy.fft import fftshift, ifftshift
from scipy import interpolate, linalg
import h5py
from scipy.fftpack import dct as dct


import cPickle as pickle
import ConfigParser
from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib import ticker
from matplotlib import animation
import brewer2mpl 

import TobySpectralMethods as tsm

#import RStransform

#SETTINGS----------------------------------------

config = ConfigParser.RawConfigParser()
fp = open('config.cfg')
config.readfp(fp)
N = config.getint('General', 'N')
M = config.getint('General', 'M')
Re = config.getfloat('General', 'Re')
beta = config.getfloat('General', 'beta')
Wi   = config.getfloat('General', 'Wi')
kx = config.getfloat('General', 'kx')
De = config.getfloat('Oscillatory Flow', 'De')
dt = config.getfloat('Time Iteration', 'dt')
totTime = config.getfloat('Time Iteration', 'totTime')
numFrames = config.getint('Time Iteration', 'numFrames')
dealiasing = config.getboolean('Time Iteration', 'Dealiasing')

Mf = 2*M
Nf = 12*N

numYs = Mf
numXs = 2*Nf+1

## Choose a point to  set the phase so that initially ystar is pure cosine
ystar = 0.44
## Set the scale of the mean flow subtracted flow
scale = 1 

fp.close()

print "Settings:"
print """------------------------------------
N \t= {N}
M \t= {M}              
Wi \t= {Wi}        
Re \t= {Re}         
beta \t= {beta}
kx \t= {kx}
------------------------------------
        """.format(N=N, M=M, kx=kx, Re=Re, beta=beta, Wi=Wi)


CNSTS = {'N':N, 'M':M, 'Nf':Nf, 'Mf':Mf, 'kx':kx, 'Re':Re, 'b':beta, 'Wi':Wi, 'dt':dt, 
         'numFrames':numFrames,'totTime':totTime}
inFileName = "output/traj.h5"

#------------------------------------------------

# FUNCTIONS

def load_hdf5_flatform(fp, time):

    dataset_id = "/t{0:f}".format(time)
    print dataset_id

    psi = array(fp[dataset_id+"/psi"])
    cxx = array(fp[dataset_id+"/cxx"])
    cyy = array(fp[dataset_id+"/cyy"])
    cxy = array(fp[dataset_id+"/cxy"])

    psi = reshape_field(psi)
    cxx = reshape_field(cxx)
    cyy = reshape_field(cyy)
    cxy = reshape_field(cxy)

    return psi, cxx, cyy, cxy

def interpolate_GL_to_uniform_grid(vec):

    ygl = zeros(numYs,dtype='d')
    for m in range(numYs):
        ygl[m] = cos(pi*m/(numYs-1))

    f = interpolate.interp1d(ygl[::-1],vec[::-1], bounds_error=False,
                         kind='linear')
    return f(y_points)

def FC_FFT_transform(inarr, CNSTS):
    """
       Use the ifft to do a 2D fourier-Chebyshev transform.  
    """

    M = CNSTS['M']
    N = CNSTS['N']

    Mf = CNSTS['Mf']
    Nf= CNSTS['Nf']

    # Prepare the field.

    in2D = inarr.reshape(2*N+1, M).T
    in2D = ifftshift(in2D, axes=1)

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

def backward_cheb_transform(cSpec, CNSTS):
    """
    Use a DCT to transform a single array of Chebyshev polynomials to the
    Gauss-Labatto grid.
    """
    # cleverer way, now works!
    M = CNSTS['M']
    Mf = numYs#CNSTS['Mf']

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

def stupid_transform_i(GLspec):
    """
    apply the Chebyshev transform the stupid way.
    """

    Mf = numYs

    out = zeros(Mf)

    for i in range(Mf):
        out[i] += GLspec[0]
        for j in range(1,M-1):
            out[i] += GLspec[j]*cos(pi*i*j/(Mf-1))
        out[i] += GLspec[M-1]*cos(pi*i)
    del i,j

    return out

def apply_phase_factor(Psi, Cxx, Cxy, Cyy, phase_factor):

    # apply the phase factor
    for n in range(1,N+1):
        Psi[(N+n)*M:(N+n+1)*M] = phase_factor**n*Psi[(N+n)*M:(N+n+1)*M]
        Psi[(N-n)*M:(N-n+1)*M] = conj(phase_factor**n)*Psi[(N-n)*M:(N-n+1)*M]

        Cxx[(N+n)*M:(N+n+1)*M] = phase_factor**n*Cxx[(N+n)*M:(N+n+1)*M]
        Cxx[(N-n)*M:(N-n+1)*M] = conj(phase_factor**n)*Cxx[(N-n)*M:(N-n+1)*M]

        Cyy[(N+n)*M:(N+n+1)*M] = phase_factor**n*Cyy[(N+n)*M:(N+n+1)*M]
        Cyy[(N-n)*M:(N-n+1)*M] = conj(phase_factor**n)*Cyy[(N-n)*M:(N-n+1)*M]

        Cxy[(N+n)*M:(N+n+1)*M] = phase_factor**n*Cxy[(N+n)*M:(N+n+1)*M]
        Cxy[(N-n)*M:(N-n+1)*M] = conj(phase_factor**n)*Cxy[(N-n)*M:(N-n+1)*M]


    # remove zeroth mode
    Psi[N*M:(N+1)*M] = 0
    Cxx[N*M:(N+1)*M] = 0
    Cxy[N*M:(N+1)*M] = 0
    Cyy[N*M:(N+1)*M] = 0
    print 'removing zeroth mode'

    return Psi, Cxx, Cyy, Cxy

def transform_all_fields(Psi, Cxx, Cxy, Cyy):

    U = dot(MDY, Psi) 
    V = -dot(MDX, Psi) 

    # Perform transformation
    Psi2D = real(FC_FFT_transform(Psi, CNSTS))
    U2D = real(FC_FFT_transform(U, CNSTS))
    V2D = real(FC_FFT_transform(V, CNSTS))
    Cxx2D = real(FC_FFT_transform(Cxx, CNSTS))
    Cyy2D = real(FC_FFT_transform(Cyy, CNSTS))
    Cxy2D = real(FC_FFT_transform(Cxy, CNSTS))

    for field in [Psi2D, Cxx2D, Cyy2D, Cxy2D]:
        for xColNum in range(len(field[0,:])):
            xCol = field[:, xColNum] 
            field[:, xColNum] = interpolate_GL_to_uniform_grid(xCol)

    return Psi2D, Cxx2D, Cxy2D, Cyy2D, U2D, V2D 

def reshape_field(field):

    tmp = field.reshape((N+1, M)).T
    field = zeros((M, 2*N+1), dtype='complex')
    field[:, :N+1] = tmp
    for n in range(1, N+1):
        field[:, 2*N+1 - n] = conj(field[:, n])
    field = fftshift(field, axes=1)
    field = field.T.flatten()

    return field

def real_space_oscillatory_flow(time):
    """
    Calculate the base flow at t =0 for the oscillatory flow problem in real
    space.
    """

    Mf = numYs

    y = cos(pi*arange(Mf)/(Mf-1))

    Re = Wi / 1182.44

    tmp = beta + (1-beta) / (1 + 1.j*De)
    #print 'tmp', tmp
    alpha = sqrt( (1.j*pi*Re*De) / (2*Wi*tmp) )
    #print 'alpha', alpha
    Chi = real( (1-1.j)*(1 - tanh(alpha) / alpha) )
    #print 'Chi', Chi 

    Psi_B = zeros((Mf, 2*Nf+1), dtype='d')
    U_B = zeros((Mf, 2*Nf+1), dtype='d')
    Cxy_B = zeros((Mf, 2*Nf+1), dtype='d')
    Cxx_B = zeros((Mf, 2*Nf+1), dtype='d')
    Cyy_B = zeros((Mf, 2*Nf+1), dtype='d')

    for i in range(Mf):
        for j in range(2*Nf+1):
            psi_im = pi/(2.j*Chi)*(y[i] - sinh(alpha*y[i])/(alpha*cosh(alpha))
                                        + sinh(alpha*-1)/(alpha*cosh(alpha))
                                   )
            Psi_B[i,j] = real(psi_im*exp(1.j*time))

            u_cmplx = pi/(2.j*Chi) * (1. - cosh(alpha*y[i])/(cosh(alpha)))
            U_B[i,j] = real(u_cmplx*exp(1.j*time))

            dyu_cmplx = pi/(2.j*Chi) *(-alpha*sinh(alpha*y[i])/(cosh(alpha)))
            cxy_cmplx = (1.0/(1.0+1.j*De)) * ((2*Wi/pi) * dyu_cmplx) 

            Cxy_B[i,j] = real( cxy_cmplx*exp(1.j*time) )

            cxx_cmplx = (1.0/(1.0+2.j*De))*(Wi/pi)*(cxy_cmplx*dyu_cmplx*exp(2.j*time))
            cxx_cmplx += (1.0/(1.0-2.j*De))*(Wi/pi)*(conj(cxy_cmplx)*conj(dyu_cmplx))*exp(-2.j*time)

            cxx_cmplx += 1. + (Wi/pi)*( cxy_cmplx*conj(dyu_cmplx) +
                                       conj(cxy_cmplx)*dyu_cmplx ) 
            Cxx_B[i,j] = real(cxx_cmplx)

    del i, j

    Cyy_B[:,0] = 1


    return U_B, Cxx_B, Cyy_B, Cxy_B

def read_base_flow(Psi, Cxx, Cxy, Cyy, time):

    U = dot(MDY, Psi)

    UB1d = backward_cheb_transform(U[N*M:(N+1)*M], CNSTS)
    CxxB1d = backward_cheb_transform(Cxx[N*M:(N+1)*M], CNSTS)
    CyyB1d = backward_cheb_transform(Cyy[N*M:(N+1)*M], CNSTS)
    CxyB1d = backward_cheb_transform(Cxy[N*M:(N+1)*M], CNSTS)

    UB = zeros((numYs, numXs))
    CxxB = zeros((numYs, numXs))
    CyyB = zeros((numYs, numXs))
    CxyB = zeros((numYs, numXs))

    for i in range(numXs):
        UB[:,i] = UB1d
        CxxB[:,i] = CxxB1d
        CyyB[:,i] = CyyB1d
        CxyB[:,i] = CxyB1d

    return real(UB), real(CxxB), real(CyyB), real(CxyB)

def calculate_piston_phase(hdf5filename):
    """
    Consider 2*pi worth of base flow trajectory data, and the same of the base
    flow calculation in order to calculate the shift we need to apply to the
    time, the phase factor, to make the trajectory time and the simulation time
    match up again.
    """

    UB, _, _, _ = real_space_oscillatory_flow(0.0)
    UB = UB[:,0]
    
    frames_per_t = numFrames / totTime
    t_per_frame = totTime / numFrames
    initTime = 0
    finalTime = floor(frames_per_t * 2.*pi ) * t_per_frame

    timeArray = r_[initTime:finalTime+t_per_frame:t_per_frame]

    checkArray = zeros((len(timeArray),2), dtype='d')

    for i, time in enumerate(timeArray):
        Psi, _, _, _ = load_hdf5_flatform(hdf5filename, time)
        U_ti = dot(MDY, Psi)

        U_ti_B = real(backward_cheb_transform(U_ti[N*M:(N+1)*M], CNSTS))

        checkArray[i,0] = time 
        checkArray[i,1] =  linalg.norm(abs(U_ti_B - UB))

    time_shift =  checkArray[argmin(checkArray[:,1]),0]

    print 'piston_phase', time_shift

    return time_shift

# MAIN

tsm.initTSM(N_=N, M_=M, kx_=kx)
MDY = tsm.mk_diff_y()
MDX = tsm.mk_diff_x()

y_points = zeros(numYs,dtype='d')
for yIndx in range(numYs):
    y_points[yIndx] = (2.0*yIndx)/(numYs-1.0) - 1.0 
del yIndx

y_c_points = cos(pi*arange(numYs)/(numYs-1))

# Read in the data
f = h5py.File(inFileName, "r")

piston_phase = calculate_piston_phase(f)
print piston_phase

lastPeriod = floor(CNSTS['totTime'] / (2*pi))
initTimeExact = (lastPeriod-2.)*2*pi
frames_per_t = numFrames / totTime
t_per_frame = totTime / numFrames

initTime = floor(frames_per_t * initTimeExact) * t_per_frame
finalTime = floor(frames_per_t * (initTimeExact + 2*pi)) * t_per_frame

initTime += piston_phase
finalTime += piston_phase


Psi, Cxx, Cyy, Cxy = load_hdf5_flatform(f, initTime)

# Choose the value of the streamfunction at a point for 1st mode, 
# psi_1(0) = 1
PSIr1 =  stupid_transform_i(real(Psi[(N+1)*M:(N+2)*M])) +\
         stupid_transform_i(imag(Psi[(N+1)*M:(N+2)*M]))*1.j


hi_yindx = argmin(abs(y_c_points-ystar*ones(numYs)))

# calculate a phase factor 1 / (psi_1(0)) such that the streamfunction is real
phase_factor = 1./PSIr1[hi_yindx]

# scale the phase factor so that it is just a phase with no amplitude,
phase_factor = phase_factor / sqrt(phase_factor*conj(phase_factor))

timesList = r_[initTime:finalTime+t_per_frame:t_per_frame]

UBArray = zeros((len(timesList), numYs), dtype='d')
CxxBArray= zeros((len(timesList), numYs), dtype='d')
CyyBArray= zeros((len(timesList), numYs), dtype='d')
CxyBArray= zeros((len(timesList), numYs), dtype='d')

Psi2DArray= zeros((len(timesList), numYs, numXs), dtype='d')
Cxx2DArray= zeros((len(timesList), numYs, numXs), dtype='d')
Cxy2DArray= zeros((len(timesList), numYs, numXs), dtype='d')
Cyy2DArray= zeros((len(timesList), numYs, numXs), dtype='d')
U2DArray= zeros((len(timesList), numYs, numXs), dtype='d')
V2DArray = zeros((len(timesList), numYs, numXs), dtype='d')

minpsi, maxpsi, minUB, maxUB, mincxx, maxcxx, mincxy, maxcxy = 0,0,0,0,1,1,0,0

for step, time in enumerate(timesList):

    Psi, Cxx, Cyy, Cxy = load_hdf5_flatform(f, time)

    UB, CxxB, CyyB, CxyB = read_base_flow(Psi, Cxx, Cxy, Cyy,
                                                  time)

    UBArray[step,:] = UB[:,0]
    CxxBArray[step,:] = CxxB[:,0]
    CyyBArray[step,:] = CyyB[:,0]
    CxyBArray[step,:] = CxyB[:,0] 

    Psi, Cxx, Cxy, Cyy = apply_phase_factor(Psi, Cxx, Cxy, Cyy, phase_factor)


    Psi2DArray[step,:, :], Cxx2DArray[step,:, :], \
    Cxy2DArray[step,:, :], Cyy2DArray[step,:, :], \
    U2DArray[step,:, :], V2DArray[step,:, :] = transform_all_fields(scale*Psi,
                                                                    scale*Cxx,
                                                                    scale*Cxy,
                                                                    scale*Cyy) 

    thisminpsi = amin(Psi2DArray[step,:, :])
    if minpsi > thisminpsi:
        minpsi = thisminpsi

    thismaxpsi = amax(Psi2DArray[step,:, :])
    if maxpsi < thismaxpsi:
        maxpsi = thismaxpsi

    thismincxx = amin(Cxx2DArray[step,:, :])
    if mincxx > thismincxx:
        mincxx = thismincxx
        maxcxx = - mincxx

    thisminUB = amin(UBArray[step,:])
    if minUB > thisminUB:
        minUB = thisminUB

    thismaxUB = amax(UBArray[step,:])
    if maxUB < thismaxUB:
        maxUB = thismaxUB


diffUB = maxUB-minUB

bmap = brewer2mpl.get_map('Spectral', 'Diverging', 11, reverse=True)
extent_ = [0,2.*pi/kx,-1,1]

#fig = plt.figure(figsize=(5.73,8.65))
fig = plt.figure(figsize=(5.73,3))
#fig, axes = plt.subplots(1,2,figsize=(5.73,3))

axes = [ plt.subplot2grid((2,2), (0,0), rowspan=2),
         plt.subplot2grid((2,2), (0,1) ),
         plt.subplot2grid((2,2), (1,1) )]

axes[0].set_xlim((minUB-0.1*diffUB, maxUB+0.1*diffUB))
axes[0].set_xticks(linspace(minUB, maxUB, 3))
axes[0].axhline(color='gray', linewidth=0.5, linestyle='--')
axes[0].axvline(color='gray', linewidth=0.5, linestyle='--')

axes[1].set_xticks(linspace(0, 2*pi/kx, 3))
axes[2].set_xticks(linspace(0, 2*pi/kx, 3))

plt.subplots_adjust(right=0.75)
psi_cbar_ax = fig.add_axes([0.82, 0.6, 0.02, 0.25])
cxx_cbar_ax = fig.add_axes([0.82, 0.2, 0.02, 0.25])

Uline, = axes[0].plot(real(UBArray[0, :]), y_c_points, color='#1b9e77')

psiIm = axes[1].imshow(real(Psi2DArray[0,:,:]), origin='lower', extent=extent_,
               aspect=1, cmap=bmap.mpl_colormap, vmin=minpsi, vmax=maxpsi )

cxxIm = axes[2].imshow(real(Cxx2DArray[0,:,:]), origin='lower', extent=extent_,
               aspect=1, cmap=bmap.mpl_colormap, vmin=mincxx, vmax=maxcxx )

cbarpsi = fig.colorbar(psiIm, orientation='vertical', cax=psi_cbar_ax,
                       label=r'$\psi$')
cbarcxx = fig.colorbar(cxxIm, orientation='vertical', cax=cxx_cbar_ax,
                       label=r'$C_{xx}$')

cbarpsi.set_ticks(linspace(minpsi, maxpsi, 3))
cbarcxx.set_ticks(linspace(mincxx, maxcxx, 3))


ims = []

for step, time in enumerate(timesList):

    #Uim = Ubaseax.plot(real(UBArray[step, :]), y_c_points, color='#1b9e77')

    Uline.set_data(real(UBArray[step, :]), y_c_points)

    psiIm.set_data(real(Psi2DArray[step,:,:]) )

    cxxIm.set_data(real(Cxx2DArray[step,:,:]) )

    outFileName = 'snapshots/step{0:04d}.png'.format(step)
    plt.savefig(outFileName, dpi=400)


