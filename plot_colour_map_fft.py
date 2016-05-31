#------------------------------------------------------------------------------
#   colour map plotter for 2D coherent state finder
#
#   Last modified: Wed 18 May 12:04:25 2016
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


import cPickle as pickle
import ConfigParser
from matplotlib import pyplot as plt
from matplotlib import rc
import brewer2mpl 

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
numYs = 2*M
numXs = 4*N

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


CNSTS = {'N':N, 'M':M, 'kx':kx, 'Re':Re, 'b':beta, 'Wi':Wi}
inFileName = "pf_sl-N{N}-M{M}-kx{kx}-Re{Re}-b{b}-Wi{Wi}.pickle".format(**CNSTS)

#------------------------------------------------

# FUNCTIONS

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

# MAIN

# Read in
#inFileName = "pf-N{N}-M{M}-kx{kx}-Re{Re}.pickle".format(N=N, M=M, kx=kx, Re=Re)
#(Psi, Nu) = pickle.load(open(inFileName, 'r'))

(Psi, Cxx, Cyy, Cxy, Nu) = pickle.load(open(inFileName, 'r'))

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

# Choose the value of the streamfunction at a point for 1st mode, 
# psi_1(0) = 1
PSIr1 =  stupid_transform_i(real(Psi[(N+1)*M:(N+2)*M])) +\
         stupid_transform_i(imag(Psi[(N+1)*M:(N+2)*M]))*1.j

# calculate a phase factor 1 / (psi_1(0)) such that the streamfunction is real
phase_factor = 1./PSIr1[numYs/2]

# scale the phase factor so that it is just a phase with no amplitude,
phase_factor = phase_factor / sqrt(phase_factor*conj(phase_factor))

# apply the phase factor
for n in range(1,N+1):
    Psi[(N+n)*M:(N+n+1)*M] = phase_factor**n*Psi[(N+n)*M:(N+n+1)*M]
    Psi[(N-n)*M:(N-n+1)*M] = phase_factor**(-n)*Psi[(N-n)*M:(N-n+1)*M]
    Cxx[(N+n)*M:(N+n+1)*M] = phase_factor**n*Cxx[(N+n)*M:(N+n+1)*M]
    Cxx[(N-n)*M:(N-n+1)*M] = phase_factor**(-n)*Cxx[(N-n)*M:(N-n+1)*M]
    Cyy[(N+n)*M:(N+n+1)*M] = phase_factor**n*Cyy[(N+n)*M:(N+n+1)*M]
    Cyy[(N-n)*M:(N-n+1)*M] = phase_factor**(-n)*Cyy[(N-n)*M:(N-n+1)*M]
    Cxy[(N+n)*M:(N+n+1)*M] = phase_factor**n*Cxy[(N+n)*M:(N+n+1)*M]
    Cxy[(N-n)*M:(N-n+1)*M] = phase_factor**(-n)*Cxy[(N-n)*M:(N-n+1)*M]

# remove zeroth mode
Psi[N*M:(N+1)*M] = 0
Cxx[N*M:(N+1)*M] = 0
Cxy[N*M:(N+1)*M] = 0
Cyy[N*M:(N+1)*M] = 0

# Perform transformation
Psi2D = real(FC_FFT_transform(Psi, CNSTS))
Cxx2D = real(FC_FFT_transform(Cxx, CNSTS))
Cyy2D = real(FC_FFT_transform(Cyy, CNSTS))
Cxy2D = real(FC_FFT_transform(Cxy, CNSTS))

for field in [Psi2D, Cxx2D, Cyy2D, Cxy2D]:
    for xColNum in range(len(field[0,:])):
        xCol = field[:, xColNum] 
        field[:, xColNum] = interpolate_GL_to_uniform_grid(xCol)

# make meshes
grid_x, grid_y = meshgrid(x_points, y_points)

fig = plt.figure(figsize=(10.0,6.0))

bmap = brewer2mpl.get_map('Spectral', 'Diverging', 11, reverse=True)

extent_ = [0,2.*pi/kx,-1,1]

ax1 = fig.add_subplot(2,2,1)
im1 = ax1.imshow(real(Psi2D), origin='lower', extent=extent_,
               aspect=1, cmap=bmap.mpl_colormap)
plt.colorbar(im1, orientation='horizontal')
ax1.set_title('psi')
ax1.set_xlabel('x')
ax1.set_ylabel('y')

ax2 = fig.add_subplot(2,2,2)
im2 = ax2.imshow(real(Cxx2D), origin='lower', extent=extent_,
               aspect=1, cmap=bmap.mpl_colormap )
plt.colorbar(im2, orientation='horizontal')
ax2.set_title('Cxx')
ax2.set_xlabel('x')
ax2.set_ylabel('y')

ax3 = fig.add_subplot(2,2,3)
im3 = ax3.imshow(real(Cyy2D), origin='lower', extent=extent_,
               aspect=1, cmap=bmap.mpl_colormap )
plt.colorbar(im3, orientation='horizontal')
ax3.set_title('Cyy')
ax3.set_xlabel('x')
ax3.set_ylabel('y')

ax4 = fig.add_subplot(2,2,4)
im4 = ax4.imshow(real(Cxy2D), origin='lower', extent=extent_,
               aspect=1, cmap=bmap.mpl_colormap )
plt.colorbar(im4, orientation='horizontal')
ax4.set_title('Cxy')
ax4.set_xlabel('x')
ax4.set_ylabel('y')

outFileName = 'cmap_' + inFileName[:-7] + '.pdf'
plt.savefig(outFileName)
