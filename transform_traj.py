

from scipy import *
from scipy import linalg
from scipy import fftpack
from numpy.fft import fftshift, ifftshift

import cPickle as pickle

import ConfigParser
import h5py

config = ConfigParser.RawConfigParser()
fp = open('config.cfg')
config.readfp(fp)
N = config.getint('General', 'N')
M = config.getint('General', 'M')
Re = config.getfloat('General', 'Re')
Wi = 0.0
beta = 1.0
kx = config.getfloat('General', 'kx')

dt = config.getfloat('Time Iteration', 'dt')
totTime = config.getfloat('Time Iteration', 'totTime')
numFrames = config.getint('Time Iteration', 'numFrames')

dealiasing = config.getboolean('Time Iteration', 'Dealiasing')

fp.close()

numTimeSteps = int(totTime / dt)

kwargs = {'N': N, 'M': M, 'Re': Re,'Wi': Wi, 'beta': beta, 'kx': kx,'time':
          totTime, 'dt':dt, 'dealiasing':dealiasing }

inFileName = "./output/traj_psi.h5".format()

CNSTS = kwargs

def load_hdf5_snapshot(fp, time):

    dataset_id = "/t{0:f}".format(time)
    print dataset_id

    inarr = array(f[dataset_id])

    return inarr

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

frameNum = 73 
time = (totTime / numFrames) * frameNum
arr = load_hdf5_snapshot(f, time)

print time 
print arr

f.close()
