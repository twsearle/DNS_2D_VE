from scipy import *
from scipy import linalg
from scipy import fftpack
from numpy.fft import fftshift, ifftshift
import subprocess
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


from pyevtk.hl import gridToVTK 
import cPickle as pickle

import ConfigParser
import h5py

import fields_2D as f2d

argparser = argparse.ArgumentParser()
argparser.add_argument("-Newt", 
                help = 'Examine newtonian ECS',
                       action="store_true")
argparser.add_argument("-p", "--path", type=str, default=".", 
                help='specify the directory containing the data')

args = argparser.parse_args()

config = ConfigParser.RawConfigParser()
fp = open(args.path + '/config.cfg')
config.readfp(fp)
N = config.getint('General', 'N')
M = config.getint('General', 'M')
Re = config.getfloat('General', 'Re')
Wi = config.getfloat('General', 'Wi')
beta = config.getfloat('General', 'beta')
kx = config.getfloat('General', 'kx')
Nf = 2*N
Mf = 2*M

De = config.getfloat('Oscillatory Flow', 'De')

dt = config.getfloat('Time Iteration', 'dt')
totTime = config.getfloat('Time Iteration', 'totTime')
numFrames = config.getint('Time Iteration', 'numFrames')

dealiasing = config.getboolean('Time Iteration', 'Dealiasing')

fp.close()

numTimeSteps = int(totTime / dt)

kwargs = {'N': N, 'M': M, 'Nf': Nf, 'Mf':Mf, 
          'Re': Re,'Wi': Wi, 'De':De, 'beta': beta, 'kx': kx,'time':
          totTime, 'dt':dt, 'dealiasing':dealiasing }

if args.path == '.':
    inFileName = args.path + "/output/traj.h5".format()
else:
    inFileName = args.path + "/traj.h5".format()


CNSTS = kwargs


class Flow(object):
    def __init__(self, fig, data0, data1):

        self.data0 = data0
        self.data1 = data1

        self.ax0 = fig.add_subplot(121)

        self.ax0.set_xlim([-1, 1]) 
        lo_lim = amin(data0[:,:])
        up_lim = amax(data0[:,:])
        if lo_lim == up_lim:
            lo_lim = lo_lim - 0.5*up_lim
            up_lim = up_lim + 0.5*up_lim

        self.ax0.set_ylim([lo_lim, up_lim]) 

        self.line0, = self.ax0.plot([], [], lw=2)
        self.zeroline0 = self.ax0.plot([-1,1], [0,0], linewidth=0.5,
                                       linestyle='--',color='gray')

        self.line0.set_data([], [])

        self.ax1 = fig.add_subplot(122)

        self.ax1.set_xlim([-1, 1]) 
        lo_lim = amin(data1[:,:])
        up_lim = amax(data1[:,:])
        self.ax1.set_ylim([lo_lim, up_lim]) 

        self.line1, = self.ax1.plot([], [], lw=2)
        self.zeroline1 = self.ax1.plot([-1,1], [0,0], linewidth=0.5,
                                       linestyle='--',color='gray')

        self.line1.set_data([], [])

    def plot_step(self, i):

        self.line0.set_data(y, self.data0[i, :])
        self.line1.set_data(y, self.data1[i, :])



def load_hdf5_snapshot(fp, time):

    dataset_id = "/t{0:f}".format(time)
    print dataset_id

    inarr = array(f[dataset_id])

    return inarr

def load_hdf5_snapshot_visco(fp, time):

    dataset_id = "/t{0:f}".format(time)
    print dataset_id

    psi = array(f[dataset_id+"/psi"])
    cxx = array(f[dataset_id+"/cxx"])
    cyy = array(f[dataset_id+"/cyy"])
    cxy = array(f[dataset_id+"/cxy"])

    return psi, cxx, cyy, cxy

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

def calc_laminar_flow(y_points, t):
    tmp = beta + (1-beta) / (1 + 1.j*De)
    alpha = sqrt( (1.j*pi*Re*De) / (2*Wi*tmp) )

    Chi = real( (1-1.j)*(1 - tanh(alpha) / alpha) )

    U0 = zeros((Mf), dtype='d')
    Cxx0 = zeros((Mf), dtype='d')
    Cxy0 = zeros((Mf), dtype='d')

    for i in range(Mf):
        y =y_points[i]

        u_im = pi/(2.j*Chi) *(1-cosh(alpha*y)/(cosh(alpha)))*exp(1.j*t)

        U0[i] = real(u_im)

        dyu_cmplx = pi/(2.j*Chi) *(-alpha*sinh(alpha*y)/(cosh(alpha)))
        cxy_cmplx = (1.0/(1.0+1.j*De)) * ((2*Wi/pi) * dyu_cmplx) 

        Cxy0[i] = real( cxy_cmplx *exp(1.j*t))

        Cxx0tmp = (1.0/(1.0+2.j*De))*(Wi/pi)*(cxy_cmplx*dyu_cmplx)*exp(2.j*t)
        Cxx0tmp += (1.0/(1.0-2.j*De))*(Wi/pi)*(conj(cxy_cmplx)*conj(dyu_cmplx))*exp(-2.j*t) 

        Cxx0tmp += 1. + (Wi/pi)*( cxy_cmplx*conj(dyu_cmplx) + conj(cxy_cmplx)*dyu_cmplx ) 
        Cxx0[i] = real(Cxx0tmp)

    del y, i
    return U0, Cxx0, Cxy0

def plot_snapshot(data0, data1, tStep, varName):
    fig = plt.figure(figsize=(5.0, 3.0))

    data0 = data0
    data1 = data1

    ax0 = fig.add_subplot(121)

    ax0.set_xlim([-1, 1]) 
    lo_lim = amin(data0[:,:])
    up_lim = amax(data0[:,:])
    if lo_lim == up_lim:
        lo_lim = lo_lim - 0.5*up_lim
        up_lim = up_lim + 0.5*up_lim

    ax0.set_ylim([lo_lim, up_lim]) 

    line0, = ax0.plot([], [], lw=2)
    zeroline0 = ax0.plot([-1,1], [0,0], linewidth=0.5,
                                   linestyle='--',color='gray')

    line0.set_data([], [])

    ax1 = fig.add_subplot(122)

    ax1.set_xlim([-1, 1]) 
    lo_lim = amin(data1[:,:])
    up_lim = amax(data1[:,:])
    ax1.set_ylim([lo_lim, up_lim]) 

    line1, = ax1.plot([], [], lw=2)
    zeroline1 = ax1.plot([-1,1], [0,0], linewidth=0.5,
                                   linestyle='--',color='gray')

    line1.set_data([], [])

    line0.set_data(y, data0[tstep, :])
    line1.set_data(y, data1[tstep, :])

    plt.savefig('{varName}_snapshot.pdf'.format(varName=varName))
    plt.close()


##### MAIN ######

print"=====================================\n"
print "Settings:"
print """------------------------------------
N \t\t= {N}
M \t\t= {M}              
Re \t\t= {Re}         
kx \t\t= {kx}
dt\t\t= {dt}
totTime\t\t= {t}
NumTimeSteps\t= {NT}
------------------------------------
""".format(N=N, M=M, kx=kx, Re=Re, dt=dt, NT=numTimeSteps, t=totTime)


f = h5py.File(inFileName, "r")

# Coordinates 

y = cos(pi*arange(Mf)/(Mf-1))

frames_per_t = numFrames / totTime
low_frame = int(numFrames - floor(2*pi*frames_per_t))
numSteps = numFrames - low_frame

uReal   = zeros((numSteps, Mf), dtype='double')
cxxReal = zeros((numSteps, Mf), dtype='double')
cxyReal = zeros((numSteps, Mf), dtype='double')
cyyReal = zeros((numSteps, Mf), dtype='double')

U0   = zeros((numSteps, Mf), dtype='double')
Cxx0 = zeros((numSteps, Mf), dtype='double')
Cxy0 = zeros((numSteps, Mf), dtype='double')
Cyy0 = ones((numSteps, Mf), dtype='double')

for frameNum in range(low_frame,numFrames):
    time = (totTime / numFrames) * frameNum
    psi, cxx, cyy, cxy = load_hdf5_snapshot_visco(f, time)

    psi = psi.reshape((N+1, M)).T
    # plot only the 1st mode
    psi[:,0] = 0
    psi = hstack((psi, conj(psi[:, N:0:-1])))
    cxx = cxx.reshape((N+1, M)).T
    cxx[:,0] = 0
    cxx = hstack((cxx, conj(cxx[:, N:0:-1])))
    cxy = cxy.reshape((N+1, M)).T
    cxy[:,0] = 0
    cxy = hstack((cxy, conj(cxy[:, N:0:-1])))
    cyy = cyy.reshape((N+1, M)).T
    cyy[:,0] = 0
    cyy = hstack((cyy, conj(cyy[:, N:0:-1])))

    u = f2d.dy(psi, CNSTS) 

    tstep = frameNum - low_frame
    uReal[tstep,:] = real(to_physical_2(u, CNSTS).T)[0,:]
    cxxReal[tstep,:] = real(to_physical_2(cxx, CNSTS).T)[0,:]
    cxyReal[tstep,:] = real(to_physical_2(cxy, CNSTS).T)[0,:]
    cyyReal[tstep,:] = real(to_physical_2(cyy, CNSTS).T)[0,:]

    U0[tstep,:], Cxx0[tstep,:], Cxy0[tstep,:] = calc_laminar_flow(y, time)

f.close()

# save plots of the first frame.

plot_snapshot(U0, uReal, 0, varName='U')
plot_snapshot(Cxx0, cxxReal, 0, varName='Cxx')
plot_snapshot(Cxy0, cxyReal, 0, varName='Cxy')
plot_snapshot(Cyy0, cyyReal, 0, varName='Cyy')

# plot animations

fig = plt.figure(figsize=(5.0,3.0))
Uobj = Flow(fig, U0, uReal)
anim = animation.FuncAnimation(fig, Uobj.plot_step,
                               frames=numSteps, interval=100, blit=False)
anim.save('uReal.mp4', fps=20, extra_args=['-vcodec', 'libx264'])
fig.clf()

Cxxobj = Flow(fig, Cxx0, cxxReal)
anim = animation.FuncAnimation(fig, Cxxobj.plot_step,
                               frames=numSteps, interval=100, blit=False)
anim.save('cxxReal.mp4', fps=20, extra_args=['-vcodec', 'libx264'])
fig.clf()

Cxyobj = Flow(fig, Cxy0, cxyReal)
anim = animation.FuncAnimation(fig, Cxyobj.plot_step,
                               frames=numSteps, interval=100, blit=False)
anim.save('cxyReal.mp4', fps=20, extra_args=['-vcodec', 'libx264'])
fig.clf()

Cyyobj = Flow(fig, Cyy0, cyyReal)
anim = animation.FuncAnimation(fig, Cyyobj.plot_step,
                               frames=numSteps, interval=100, blit=False)
anim.save('cyyReal.mp4', fps=20, extra_args=['-vcodec', 'libx264'])
fig.clf()




