#------------------------------------------------------------------------------
#   colour map plotter for 2D coherent state finder
#
#   Last modified: Mon 30 May 14:17:57 2016
#
#------------------------------------------------------------------------------
#TODO check that the axes are the right way up?

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

numYs = 2*M
numXs = 12*N


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


CNSTS = {'N':N, 'M':M, 'kx':kx, 'Re':Re, 'b':beta, 'Wi':Wi, 'dt':dt, 
         'numFrames':numFrames,'totTime':totTime}
inFileName = "output/traj.h5"

#------------------------------------------------

# FUNCTIONS

def load_hdf5_flatform(fp, time):

    dataset_id = "/t{0:f}".format(time)
    print dataset_id

    psi = array(f[dataset_id+"/psi"])
    cxx = array(f[dataset_id+"/cxx"])
    cyy = array(f[dataset_id+"/cyy"])
    cxy = array(f[dataset_id+"/cxy"])

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

    Mf = numYs
    Nf = numXs

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
    Nf = numXs

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
x_points = zeros(numXs,dtype='d') 

for xIndx in range(numXs):
    #               2.lambda     * fractional position
    #x_points[xIndx] = (4.*pi/kx) * ((1.*xIndx)/numXs)
    x_points[xIndx] = (2.*pi/kx) * ((1.*xIndx)/numXs)
del xIndx

y_points = zeros(numYs,dtype='d')
for yIndx in range(numYs):
    y_points[yIndx] = (2.0*yIndx)/(numYs-1.0) - 1.0 
del yIndx

y_c_points = cos(pi*arange(numYs)/(numYs-1))

# Read in
f = h5py.File(inFileName, "r")

piston_phase = calculate_piston_phase(f)
print piston_phase

lastPeriod = floor(CNSTS['totTime'] / (2*pi))
initTime = (lastPeriod-1.)*2*pi #- pi/2.

frames_per_t = numFrames / totTime
t_per_frame = totTime / numFrames
stepTime = pi/6.
time1 = floor(frames_per_t * initTime) * t_per_frame
time2 = floor(frames_per_t * (initTime + stepTime)) * t_per_frame
time3 = floor(frames_per_t * (initTime + 2*stepTime)) * t_per_frame
time4 = floor(frames_per_t * (initTime + 3*stepTime)) * t_per_frame

time1 += piston_phase
time2 += piston_phase
time3 += piston_phase
time4 += piston_phase


print time1
print time2
print time3
print time4

Psi, Cxx, Cyy, Cxy = load_hdf5_flatform(f, time1)

# Choose the value of the streamfunction at a point for 1st mode, 
# psi_1(0) = 1
PSIr1 =  stupid_transform_i(real(Psi[(N+1)*M:(N+2)*M])) +\
         stupid_transform_i(imag(Psi[(N+1)*M:(N+2)*M]))*1.j

# calculate a phase factor 1 / (psi_1(0)) such that the streamfunction is real
phase_factor = 1./PSIr1[numYs/2]

# scale the phase factor so that it is just a phase with no amplitude,
phase_factor = phase_factor / sqrt(phase_factor*conj(phase_factor))


Psi2, Cxx2, Cyy2, Cxy2 = load_hdf5_flatform(f, time2)
Psi3, Cxx3, Cyy3, Cxy3 = load_hdf5_flatform(f, time3)
Psi4, Cxx4, Cyy4, Cxy4 = load_hdf5_flatform(f, time4)

UB, CxxB, CyyB, CxyB     = read_base_flow(Psi, Cxx, Cxy, Cyy,
                                              time1)
UB2, CxxB2, CyyB2, CxyB2 = read_base_flow(Psi2, Cxx2, Cxy2, Cyy2,
                                              time2)
UB3, CxxB3, CyyB3, CxyB3 = read_base_flow(Psi3, Cxx3, Cxy3, Cyy3,
                                              time3)
UB4, CxxB4, CyyB4, CxyB4 = read_base_flow(Psi4, Cxx4, Cxy4, Cyy4,
                                              time4)

Psi, Cxx, Cxy, Cyy = apply_phase_factor(Psi, Cxx, Cxy, Cyy, phase_factor)
Psi2, Cxx2, Cxy2, Cyy2 = apply_phase_factor(Psi2, Cxx2, Cxy2, Cyy2, phase_factor)
Psi3, Cxx3, Cxy3, Cyy3 = apply_phase_factor(Psi3, Cxx3, Cxy3, Cyy3, phase_factor)
Psi4, Cxx4, Cxy4, Cyy4 = apply_phase_factor(Psi4, Cxx4, Cxy4, Cyy4, phase_factor)

scale = 1e8 

Psi2D, Cxx2D, Cxy2D, Cyy2D, U2D, V2D = transform_all_fields(scale*Psi,
                                                            scale*Cxx,
                                                            scale*Cxy,
                                                            scale*Cyy) 

Psi2D2, Cxx2D2, Cxy2D2, Cyy2D2, U2D2, V2D2 = transform_all_fields(scale*Psi2,
                                                                  scale*Cxx2,
                                                                  scale*Cxy2,
                                                                  scale*Cyy2) 

Psi2D3, Cxx2D3, Cxy2D3, Cyy2D3, U2D3, V2D3 = transform_all_fields(scale*Psi3,
                                                                  scale*Cxx3,
                                                                  scale*Cxy3,
                                                                  scale*Cyy3) 

Psi2D4, Cxx2D4, Cxy2D4, Cyy2D4, U2D4, V2D4 = transform_all_fields(scale*Psi4,
                                                                  scale*Cxx4,
                                                                  scale*Cxy4,
                                                                  scale*Cyy4)

#UB, CxxB, CyyB, CxyB = real_space_oscillatory_flow(time1)
#UB2, CxxB2, CyyB2, CxyB2 = real_space_oscillatory_flow(time2)
#UB3, CxxB3, CyyB3, CxyB3 = real_space_oscillatory_flow(time3)
#UB4, CxxB4, CyyB4, CxyB4 = real_space_oscillatory_flow(time4)

# make meshes
grid_x, grid_y = meshgrid(x_points, y_points)

minpsi = amin(vstack((Psi2D, Psi2D2, Psi2D3, Psi2D4)) )
maxpsi = amax(vstack((Psi2D, Psi2D2, Psi2D3, Psi2D4)) )

mincxx = amin(vstack((Cxx2D, Cxx2D2, Cxx2D3, Cxx2D4)) )
#maxcxx = amax(vstack((Cxx2D, Cxx2D2, Cxx2D3, Cxx2D4)) )
maxcxx = - mincxx

minUB = amin(vstack((UB, UB2, UB3, UB4)) )
maxUB = amax(vstack((UB, UB2, UB3, UB4)) )
diffUB = maxUB-minUB

#fig, axes = plt.subplots(4,3,figsize=(5.73,8.65), sharex=True, sharey=True)
#fig = plt.figure(figsize=(5.73,8.65))
fig = plt.figure(figsize=(5.73,7.65))

psiax = fig.add_subplot(4,3,2)
psiax.set_xticks(linspace(min(x_points), max(x_points), 3))
psiax.set_yticks([-1.0,0.0,1.0])

Ubaseax = fig.add_subplot(4,3,1, sharey=psiax)
Ubaseax.set_xlim((minUB-0.1*diffUB, maxUB+0.1*diffUB))
Ubaseax.set_xticks(linspace(minUB, maxUB, 3))

axes = [Ubaseax,
        psiax,
        fig.add_subplot(4,3,3, sharex=psiax, sharey=psiax),
        fig.add_subplot(4,3,4, sharex=Ubaseax, sharey=psiax),
        fig.add_subplot(4,3,5, sharex=psiax, sharey=psiax),
        fig.add_subplot(4,3,6, sharex=psiax, sharey=psiax),
        fig.add_subplot(4,3,7, sharex=Ubaseax, sharey=psiax),
        fig.add_subplot(4,3,8, sharex=psiax, sharey=psiax),
        fig.add_subplot(4,3,9, sharex=psiax, sharey=psiax),
        fig.add_subplot(4,3,10, sharex=Ubaseax, sharey=psiax),
        fig.add_subplot(4,3,11, sharex=psiax, sharey=psiax),
        fig.add_subplot(4,3,12, sharex=psiax, sharey=psiax) ] 

bmap = brewer2mpl.get_map('Spectral', 'Diverging', 11, reverse=True)

extent_ = [0,2.*pi/kx,-1,1]

axes[0].set_ylabel('$y$')
axes[3].set_ylabel('$y$')
axes[6].set_ylabel('$y$')
axes[9].set_ylabel('$y$')

#for ax in axes:
#    ax.set_xticklabels([])
#    ax.set_yticklabels([])

axes[9].set_xlabel('$U$')
axes[10].set_xlabel('$x$')
axes[11].set_xlabel('$x$')

Uplt1 = axes[0].plot(real(UB[:,0]), y_c_points )
axes[0].axhline(color='gray', linewidth=0.5, linestyle='--')
axes[0].axvline(color='gray', linewidth=0.5, linestyle='--')

psiIm1 = axes[1].imshow(real(Psi2D), origin='lower', extent=extent_,
               aspect=1, cmap=bmap.mpl_colormap, vmin=minpsi, vmax=maxpsi )

cxxIm1 = axes[2].imshow(real(Cxx2D), origin='lower', extent=extent_,
               aspect=1, cmap=bmap.mpl_colormap, vmin=mincxx, vmax=maxcxx )
axes[2].axhline(y=-0.44, color='gray', linewidth=0.5, linestyle=':')
axes[2].axhline(y=0.44, color='gray', linewidth=0.5, linestyle=':')

Uplt2 = axes[3].plot(real(UB2[:,0]), y_c_points )
axes[3].axhline(color='gray', linewidth=0.5, linestyle='--')
axes[3].axvline(color='gray', linewidth=0.5, linestyle='--')

psiIm2 = axes[4].imshow(real(Psi2D2), origin='lower', extent=extent_,
               aspect=1, cmap=bmap.mpl_colormap, vmin=minpsi, vmax=maxpsi )

cxxIm2 = axes[5].imshow(real(Cxx2D2), origin='lower', extent=extent_,
               aspect=1, cmap=bmap.mpl_colormap, vmin=mincxx, vmax=maxcxx )
axes[5].axhline(y=-0.44, color='gray', linewidth=0.5, linestyle=':')
axes[5].axhline(y=0.44, color='gray', linewidth=0.5, linestyle=':')

Uplt3 = axes[6].plot(real(UB3[:,0]), y_c_points )
axes[6].axhline(color='gray', linewidth=0.5, linestyle='--')
axes[6].axvline(color='gray', linewidth=0.5, linestyle='--')

psiIm3 = axes[7].imshow(real(Psi2D3), origin='lower', extent=extent_,
               aspect=1, cmap=bmap.mpl_colormap, vmin=minpsi, vmax=maxpsi )

cxxIm3 = axes[8].imshow(real(Cxx2D3), origin='lower', extent=extent_,
               aspect=1, cmap=bmap.mpl_colormap, vmin=mincxx, vmax=maxcxx )
axes[8].axhline(y=-0.44, color='gray', linewidth=0.5, linestyle=':')
axes[8].axhline(y=0.44, color='gray', linewidth=0.5, linestyle=':')

Uplt4 = axes[9].plot(real(UB4[:,0]), y_c_points )
axes[9].axhline(color='gray', linewidth=0.5, linestyle='--')
axes[9].axvline(color='gray', linewidth=0.5, linestyle='--')

psiIm4 = axes[10].imshow(real(Psi2D4), origin='lower', extent=extent_,
               aspect=1, cmap=bmap.mpl_colormap, vmin=minpsi, vmax=maxpsi )

cxxIm4 = axes[11].imshow(real(Cxx2D4), origin='lower', extent=extent_,
               aspect=1, cmap=bmap.mpl_colormap, vmin=mincxx, vmax=maxcxx )
axes[11].axhline(y=-0.44, color='gray', linewidth=0.5, linestyle=':')
axes[11].axhline(y=0.44, color='gray', linewidth=0.5, linestyle=':')



fig.subplots_adjust(left=0.15, right=0.85, hspace=0.1)

psi_cbar_ax = fig.add_axes([0.15, 0.1, 0.25, 0.02])
cxx_cbar_ax = fig.add_axes([0.65, 0.1, 0.25, 0.02])


cbarpsi = fig.colorbar(psiIm1, orientation='horizontal', cax=psi_cbar_ax,
                       label=r'$\psi$')
cbarcxx = fig.colorbar(cxxIm1, orientation='horizontal', cax=cxx_cbar_ax,
                       label=r'$C_{xx}$')

cbarpsi.set_ticks(linspace(minpsi, maxpsi, 3))
cbarcxx.set_ticks(linspace(mincxx, maxcxx, 3))

outFileName = 'cmap_period_panels.pdf'
plt.savefig(outFileName)
