#-----------------------------------------------------------------------------
#   2D spectral direct numerical simulator
#
#   Last modified: Fri 27 Feb 16:31:30 2015
#
#-----------------------------------------------------------------------------

"""

Simulation of plane Poiseuille flow. In the past I have used a numerical scheme
which uses the streamfunction at the wrong time in the Runge-Kutta method. Here
I attempt to correct this by calculating the streamfunction twice, Once at t
+ dt/2 and once at t + dt.

Simulation of 2D Newtonian flow.

TODO:
    
    * Test BLAS against loops for random matrices.

    * Test BLAS against loops with openMP

    * Write 1st, 2nd, 3rd, 4th derivative in fortran for Chebyshev modes

    * Write Fourier derivatives, 1st, 2nd, 4th order

Outline:
    
    * read in data

    * Form operators for semi-implicit crank-nicolson

    * for all times do:

        * solve for Psi at current time based on previous time

    until: timeup



"""

# MODULES
from scipy import *
from scipy import linalg
from numpy.fft import fftshift, ifftshift
from numpy.random import rand

import cPickle as pickle

import ConfigParser
import subprocess
import h5py

# SETTINGS---------------------------------------------------------------------

config = ConfigParser.RawConfigParser()
fp = open('config.cfg')
config.readfp(fp)
N = config.getint('General', 'N')
M = config.getint('General', 'M')
Re = config.getfloat('General', 'Re')
Wi = config.getfloat('General', 'Wi')
beta = config.getfloat('General', 'beta')
kx = config.getfloat('General', 'kx')

dt = config.getfloat('Time Iteration', 'dt')
totTime = config.getfloat('Time Iteration', 'totTime')
numFrames = config.getint('Time Iteration', 'numFrames')
dealiasing = config.getboolean('Time Iteration', 'Dealiasing')

fp.close()

if dealiasing:
    Nf = (3*N)/2 + 1
    Mf = (3*M)/2
else:
    Nf = N
    Mf = M

numTimeSteps = int(totTime / dt)
assert totTime % dt, "non-integer number of time steps!"
assert Wi != 0.0, "cannot have Wi = 0!"

NOld = N 
MOld = M
kwargs = {'N': N, 'M': M, 'Nf':Nf, 'Mf':Mf, 'Re': Re, 'Wi': Wi, 'beta': beta,
          'kx': kx,'time': totTime, 'dt':dt, 'dealiasing':dealiasing}
baseFileName  = "-N{N}-M{M}-kx{kx}-Re{Re}.pickle".format(**kwargs)
outFileName  = "pf{0}".format(baseFileName)
outFileNameTrace = "trace{0}.dat".format(baseFileName[:-7])
outFileNameTime = "series-pf{0}".format(baseFileName)
#inFileName = "pf-N{N}-M{M}-kx{kx}-Re{Re}.pickle".format(N=NOld, M=MOld, 
#                                                        kx=kx, Re=Re)
inFileName = "pf-N{N}-M{M}-kx{kx}-Re{Re}.pickle".format(**kwargs)

CNSTS = kwargs

# -----------------------------------------------------------------------------

def mk_single_diffy():
    """Makes a matrix to differentiate a single vector of Chebyshev's, 
    for use in constructing large differentiation matrix for whole system"""
    # make matrix:
    mat = zeros((M, M), dtype='d')
    for m in range(M):
        for p in range(m+1, M, 2):
            mat[m,p] = 2*p*oneOverC[m]

    return mat

def mk_cheb_int():
    integrator = zeros(M, dtype='d')
    for m in range(0,M,2):
        integrator[m] = 2. / (1.-m*m)
    del m
    
    return integrator

def append_save_array(array, fp):

    (rows, cols) = shape(array)
    for i in range(rows):
        for j in range(cols):
            fp.write('{0:15.8g}'.format(array[i,j]))
        fp.write('\n')


def load_hdf5_state(filename):
    f = h5py.File(filename, "r")
    inarr = array(f["psi"])
    f.close()
    return inarr


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


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

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

# SET UP

vecLen = (2*N+1)*M

oneOverRe = 1. / Re
assert oneOverRe != infty, "Can't set Reynold's to zero!"

# Set the oneOverC function: 1/2 for m=0, 1 elsewhere:
oneOverC = ones(M)
oneOverC[0] = 1. / 2.

# single mode Operators
SMDY = mk_single_diffy()
SMDYY = dot(SMDY, SMDY)
SMDYYY = dot(SMDY, SMDYY)

INTY = mk_cheb_int()

# Identity
SII = eye(M, M, dtype='complex')

# Boundary arrays
BTOP = ones(M)
BBOT = ones(M)
BBOT[1:M:2] = -1

DERIVTOP = zeros((M), dtype='complex')
DERIVBOT = zeros((M), dtype='complex')
for j in range(M):
    DERIVTOP[j] = dot(BTOP, SMDY[:,j]) 
    DERIVBOT[j] = dot(BBOT, SMDY[:,j])
del j

#### The initial stream-function
PSI = zeros((2*N+1)*M,dtype='complex')

# This is Poiseuille flow 
#PSI[N*M]   += 2.0/3.0
#PSI[N*M+1] += 3.0/4.0
#PSI[N*M+2] += 0.0
#PSI[N*M+3] += -1.0/12.0

#PSI[(N-1)*M + 3] += 1e-11
#PSI[(N-2)*M + 3] += 1e-11
#PSI[(N+1)*M:(N+2)*M] = conj(PSI[(N-1)*M:N*M])

#print 'performing linear stability of Poiseuille flow test'

# Read in stream function from file
#(PSI, Nu) = pickle.load(open(inFileName,'r'))

PSI[N*M]   += 2.0/3.0
PSI[N*M+1] += 3.0/4.0
PSI[N*M+3] += -1.0/12.0
PSI[N*M:N*M+4]   += 0.01*rand(4) + 0j
PSI[N*M+2] += 0.0

PSI[(N-1)*M + 2] += 1e-2 - 1e-2j
PSI[(N-1)*M + 4] += 1e-2 - 1e-2j
PSI[(N-1)*M + 6] += 1e-3 - 1e-3j
PSI[(N-1)*M + 8] += 1e-4 - 1e-4j
PSI[(N-1)*M + 10] += 1e-4 - 1e-4j

PSI[(N-2)*M + 3] += 1e-4 - 1e-4j
PSI[(N-2)*M + 5] += 1e-4 - 1e-4j

PSI[(N+1)*M:(N+2)*M] = conj(PSI[(N-1)*M:N*M])
PSI[(N+2)*M:(N+3)*M] = conj(PSI[(N-2)*M:(N-1)*M])


# Form the operators
PsiOpInvList = []

# zeroth mode
Psi0thOp = zeros((M,M), dtype='complex')
Psi0thOp = SMDY - 0.5*dt*oneOverRe*SMDYYY + 0j

# Apply BCs

# dypsi0(+-1) = 0
Psi0thOp[M-3, :] = DERIVTOP
Psi0thOp[M-2, :] = DERIVBOT
# psi0(-1) =  0
Psi0thOp[M-1, :] = BBOT

PsiOpInvList.append(linalg.inv(Psi0thOp))

for i in range(1, N+1):
    n = i

    PSIOP = zeros((2*M, 2*M), dtype='complex')
    SLAPLAC = -n*n*kx*kx*SII + SMDYY

    PSIOP[0:M, 0:M] = 0
    PSIOP[0:M, M:2*M] = SII - 0.5*oneOverRe*dt*SLAPLAC

    PSIOP[M:2*M, 0:M] = SLAPLAC
    PSIOP[M:2*M, M:2*M] = -SII

    # Apply BCs
    # dypsi(+-1) = 0
    PSIOP[M-2, :] = concatenate((DERIVTOP, zeros(M, dtype='complex')))
    PSIOP[M-1, :] = concatenate((DERIVBOT, zeros(M, dtype='complex')))
    
    # dxpsi(+-1) = 0
    PSIOP[2*M-2, :] = concatenate((BTOP, zeros(M, dtype='complex')))
    PSIOP[2*M-1, :] = concatenate((BBOT, zeros(M, dtype='complex')))

    # store the inverse of the relevent part of the matrix
    PSIOP = linalg.inv(PSIOP)
    PSIOP = PSIOP[0:M, 0:M]

    PsiOpInvList.append(PSIOP)

del PSIOP

# zeroth mode
Psi0thOp = zeros((M,M), dtype='complex')
Psi0thOp = SMDY - 0.5*dt*oneOverRe*SMDYYY + 0j

# Apply BCs

# dypsi0(+-1) = 0
Psi0thOp[M-3, :] = DERIVTOP
Psi0thOp[M-2, :] = DERIVBOT
# psi0(-1) =  0
Psi0thOp[M-1, :] = BBOT

PsiOpInvList.append(linalg.inv(Psi0thOp))

PsiOpInvList = array(PsiOpInvList)

#### SAVE THE OPERATORS AND INITIAL STATE FOR THE C CODE

for i in range(N+1):
    # operator order in list is 0->N
    n = i
    print n
    opFn = "./operators/op{0}.h5".format(n)
    print "writing ", opFn
    f = h5py.File(opFn, "w")
    dset = f.create_dataset("op", (M*M,), dtype='complex')
    dset[...] = PsiOpInvList[i].flatten()

    f.close()

    #savetxt("./operators/op{0}.dat".format(abs(n)),PsiOpInvList[n])
del i

# make PSI 2D
PSI = PSI.reshape(2*N+1, M).T
# put PSI into FFT ordering.
PSI = ifftshift(PSI, axes=1)

print "writing initial state to initial.h5"

f = h5py.File("initial.h5", "w")
dset = f.create_dataset("psi", ((2*N+1)*M,), dtype='complex')
dset[...] = PSI.T.flatten()
f.close()

#### TIME ITERATE 

stepsPerFrame = numTimeSteps/numFrames

# Run program in C

# pass the flow variables and the time iteration settings to the C code
if dealiasing:
    cargs = ["./DNS_2D_Newt", "-N", "{0:d}".format(CNSTS["N"]), "-M",
             "{0:d}".format(CNSTS["M"]), "-L", "2.0", "-k", "{0:e}".format(CNSTS["kx"]),
             "-R", "{0:e}".format(CNSTS["Re"]), "-W", "{0:e}".format(CNSTS["Wi"]), "-b",
             "{0:e}".format(CNSTS["beta"]), "-t", "{0:e}".format(CNSTS["dt"]), "-s",
             "{0:d}".format(stepsPerFrame), "-T", "{0:d}".format(numTimeSteps),
            "-d"]
    print "./DNS_2D_Newt", "-N", "{0:d}".format(CNSTS["N"]), "-M", \
          "{0:d}".format(CNSTS["M"]), "-L", "2.0", "-k", \
            "{0:e}".format(CNSTS["kx"]),"-R", \
            "{0:e}".format(CNSTS["Re"]), "-W", "{0:e}".format(CNSTS["Wi"]),\
            "-b", "{0:e}".format(CNSTS["beta"]), "-t", \
            "{0:e}".format(CNSTS["dt"]), "-s",\
          "{0:d}".format(stepsPerFrame), "-T", "{0:d}".format(numTimeSteps),"-d"

else:
    cargs = ["./DNS_2D_Newt", "-N", "{0:d}".format(CNSTS["N"]), "-M",
             "{0:d}".format(CNSTS["M"]), "-L", "2.0", "-k", "{0:e}".format(CNSTS["kx"]),
             "-R", "{0:e}".format(CNSTS["Re"]), "-W", "{0:e}".format(CNSTS["Wi"]), "-b",
             "{0:e}".format(CNSTS["beta"]), "-t", "{0:e}".format(CNSTS["dt"]), "-s",
             "{0:d}".format(stepsPerFrame), "-T", "{0:d}".format(numTimeSteps)]
    print "./DNS_2D_Newt", "-N", "{0:d}".format(CNSTS["N"]), "-M", \
          "{0:d}".format(CNSTS["M"]), "-L", "2.0", "-k", \
            "{0:e}".format(CNSTS["kx"]),"-R", \
            "{0:e}".format(CNSTS["Re"]), "-W", "{0:e}".format(CNSTS["Wi"]),\
            "-b", "{0:e}".format(CNSTS["beta"]), "-t", \
            "{0:e}".format(CNSTS["dt"]), "-s",\
          "{0:d}".format(stepsPerFrame), "-T", "{0:d}".format(numTimeSteps)

subprocess.call(cargs)

# Read in data from the C code
print 'done'
