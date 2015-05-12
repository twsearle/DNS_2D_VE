from scipy import *
from scipy import linalg
from scipy import fftpack
from numpy.fft import fftshift, ifftshift
import argparse

from pyevtk.hl import gridToVTK 
import cPickle as pickle

import ConfigParser
import h5py

import fields_2D as f2d


config = ConfigParser.RawConfigParser()
fp = open('config.cfg')
config.readfp(fp)
N = config.getint('General', 'N')
M = config.getint('General', 'M')
Re = config.getfloat('General', 'Re')
Wi = 0.0
beta = 1.0
kx = config.getfloat('General', 'kx')
Nf = 50
Mf = 100

dt = config.getfloat('Time Iteration', 'dt')
totTime = config.getfloat('Time Iteration', 'totTime')
numFrames = config.getint('Time Iteration', 'numFrames')

dealiasing = config.getboolean('Time Iteration', 'Dealiasing')

fp.close()

argparser = argparse.ArgumentParser()
argparser.add_argument("-Newt", 
                help = 'Examine pure newtonian ECS',
                       action="store_true")

args = argparser.parse_args()

numTimeSteps = int(totTime / dt)

kwargs = {'N': N, 'M': M, 'Nf': Nf, 'Mf':Mf, 
          'Re': Re,'Wi': Wi, 'beta': beta, 'kx': kx,'time':
          totTime, 'dt':dt, 'dealiasing':dealiasing }

inFileName = "./output/traj.h5".format()

CNSTS = kwargs

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

fpvd = open("traj.pvd", 'w')

fpvd.write('''<?xml version="1.0"?>
<VTKFile type="Collection" version="0.1"
         compressor="vtkZLibDataCompressor">
  <Collection>
''')

f = h5py.File(inFileName, "r")

# Dimensions 

nx, ny, nz = 2*Nf, Mf-1, 1 
ncells = nx * ny * nz 

# Coordinates 
lx = 2*pi / kx
ly = 2.0
dx = lx / (nx)
x = arange(0, lx + dx, dx, dtype='float64') 
y = cos(pi*arange(Mf)/(Mf-1))
z = array([0.0, 1.0])

if args.Newt:

    for frameNum in range(numFrames):
        time = (totTime / numFrames) * frameNum
        arr = load_hdf5_snapshot(f, time)

        arr = arr.reshape((N+1, M)).T
        arr = hstack((arr, conj(arr[:, N:0:-1])))

        tmp = to_physical_2(arr, CNSTS).T

        psiReal = zeros((2*Nf+1, Mf, 2))

        psiReal[:,:,0] = real(tmp)
        psiReal[:,:,1] = real(tmp)

        filename = "vtktraj/t{0}.vtr".format(time)

        gridToVTK(filename[:-4], x, y, z, pointData = {"psi" : psiReal}) 

         
        fpvd.write('<DataSet timestep="{0}" group="" part="0"\n'.format(time))
        fpvd.write('file="{0}"/>'.format(filename))

else:

    for frameNum in range(1,numFrames):
        time = (totTime / numFrames) * frameNum
        psi, cxx, cyy, cxy = load_hdf5_snapshot_visco(f, time)

        psi = psi.reshape((N+1, M)).T
        psi = hstack((psi, conj(psi[:, N:0:-1])))

        tmp = to_physical_2(psi, CNSTS).T
        psiReal = zeros((2*Nf+1, Mf, 2))
        psiReal[:,:,0] = real(tmp)
        psiReal[:,:,1] = real(tmp)

        cxx = cxx.reshape((N+1, M)).T
        cxx = hstack((cxx, conj(cxx[:, N:0:-1])))

        tmp = to_physical_2(cxx, CNSTS).T
        cxxReal = zeros((2*Nf+1, Mf, 2))
        cxxReal[:,:,0] = real(tmp)
        cxxReal[:,:,1] = real(tmp)

        cyy = cyy.reshape((N+1, M)).T
        cyy = hstack((cyy, conj(cyy[:, N:0:-1])))

        tmp = to_physical_2(cyy, CNSTS).T
        cyyReal = zeros((2*Nf+1, Mf, 2))
        cyyReal[:,:,0] = real(tmp)
        cyyReal[:,:,1] = real(tmp)

        cxy = cxy.reshape((N+1, M)).T
        cxy = hstack((cxy, conj(cxy[:, N:0:-1])))

        tmp = to_physical_2(cxy, CNSTS).T
        cxyReal = zeros((2*Nf+1, Mf, 2))
        cxyReal[:,:,0] = real(tmp)
        cxyReal[:,:,1] = real(tmp)

        filename = "vtktraj/t{0}.vtr".format(time)

        gridToVTK(filename[:-4], x, y, z, pointData = {"psi": psiReal, "cxx": cxxReal,
                                                       "cyy": cyyReal, "cxy": cxyReal}) 

         
        fpvd.write('<DataSet timestep="{0}" group="" part="0"\n'.format(time))
        fpvd.write('file="{0}"/>'.format(filename))


fpvd.write('''
  </Collection>
</VTKFile>
''')


fpvd.close()
f.close()
