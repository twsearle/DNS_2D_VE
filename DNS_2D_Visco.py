#-----------------------------------------------------------------------------
#   2D spectral direct numerical simulator
#
#   Last modified: Tue 24 Mar 14:31:58 2015
#
#-----------------------------------------------------------------------------

"""

Simulation of Viscoelastic Oldroyd-B plane Poiseuille flow.

TODO:

    * predictor-corrector ABM for the streamfunction. Test if it is faster/
    better.
    
Outline:
    
    * read in data

    * Form operators for semi-implicit crank-nicolson

    * Form half step operators

    * set up the initial streamfunction and stresses

    * Pass to C code 

    * for the first three time steps:

        * solve for Psi at current time based on previous time

        * solve for Psi on the half step 

        * solve for the stresses using 4th order Runge-Kutta 

    * for all other times do:

        * solve for Psi at current time based on previous time

        * Solve for the stresses using 4th Order Adams-Bashforth
        predictor-corrector.j

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
    Mf = 2*M
else:
    Nf = N
    Mf = M

numTimeSteps = int(totTime / dt)
assert (totTime / dt) - float(numTimeSteps) == 0, "Non-integer number of timesteps"
assert Wi != 0.0, "cannot have Wi = 0!"

NOld = N 
MOld = M
kwargs = {'N': N, 'M': M, 'Nf':Nf, 'Mf':Mf,'U0':0, 'Re': Re, 'Wi': Wi, 'beta': beta,
          'kx': kx,'time': totTime, 'dt':dt, 'dealiasing':dealiasing}

inFileName = "pf-N{N}-M{M}-kx{kx}-Re{Re}-b{beta}-Wi{Wi}.pickle".format(**kwargs)

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

def form_operators(dt):
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

    PsiOpInvList = array(PsiOpInvList)
    return PsiOpInvList

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
Cxx = zeros((2*N+1)*M,dtype='complex')
Cyy = zeros((2*N+1)*M,dtype='complex')
Cxy = zeros((2*N+1)*M,dtype='complex')


# Read in stream function from file
#(PSI, Cxx, Cyy, Cxy, Nu) = pickle.load(open(inFileName,'r'))

# --------------- POISEUILLE -----------------

PSI[N*M]   += 2.0/3.0
PSI[N*M+1] += 3.0/4.0
PSI[N*M+2] += 0.0
PSI[N*M+3] += -1.0/12.0

perAmp = 1e-2

for n in range(1,N+1):
    if (n % 2) == 0:
        PSI[(N-n)*M + 1:(N-n)*M + M/2 :2] += (0.1**(n-1))*perAmp*(rand(M/4) + 1.j*rand(M/4))
        Cxx[(N-n)*M + 1:(N-n)*M + M/2 :2] += (0.1**(n-1))*perAmp*(rand(M/4) + 1.j*rand(M/4))
        Cyy[(N-n)*M + 1:(N-n)*M + M/2 :2] += (0.1**(n-1))*perAmp*(rand(M/4) + 1.j*rand(M/4))
        Cxy[(N-n)*M + 1:(N-n)*M + M/2 :2] += (0.1**(n-1))*perAmp*(rand(M/4) + 1.j*rand(M/4))
    else:
        PSI[(N-n)*M:(N-n)*M + M/2 - 1 :2] += (0.1**(n-1))*perAmp*(rand(M/4) + 1.j*rand(M/4))
        Cxx[(N-n)*M:(N-n)*M + M/2 - 1 :2] += (0.1**(n-1))*perAmp*(rand(M/4) + 1.j*rand(M/4))
        Cyy[(N-n)*M:(N-n)*M + M/2 - 1 :2] += (0.1**(n-1))*perAmp*(rand(M/4) + 1.j*rand(M/4))
        Cxy[(N-n)*M:(N-n)*M + M/2 - 1 :2] += (0.1**(n-1))*perAmp*(rand(M/4) + 1.j*rand(M/4))

    PSI[(N+n)*M:(N+n+1)*M] = conj(PSI[(N-n)*M:(N-n+1)*M])
    Cxx[(N+n)*M:(N+n+1)*M] = conj(Cxx[(N-n)*M:(N-n+1)*M])
    Cyy[(N+n)*M:(N+n+1)*M] = conj(Cyy[(N-n)*M:(N-n+1)*M])
    Cxy[(N+n)*M:(N+n+1)*M] = conj(Cxy[(N-n)*M:(N-n+1)*M])
 

#print 'performing linear stability of Poiseuille flow test'

#PSI[N*M]   += 2.0/3.0
#PSI[N*M+1] += 3.0/4.0
#PSI[N*M+3] += -1.0/12.0

#PSI[N*M+1: (N+1)*M - M/2 :2] += perAmp*rand(M/4) 

#PSI[(N-1)*M + 4] += 1e-2 - 1e-2j
#PSI[(N-1)*M + 6] += 1e-3 - 1e-3j
#PSI[(N-1)*M + 8] += 1e-4 - 1e-4j
#PSI[(N-1)*M + 10] += 1e-4 - 1e-4j
#
#PSI[(N-2)*M + 3] += 1e-4 - 1e-4j
#PSI[(N-2)*M + 5] += 1e-4 - 1e-4j
#

#PSI[(N-1)*M:N*M-M/2 -1:2] += perAmp*rand(M/4) - perAmp*1.j*rand(M/4)
#PSI[(N-2)*M+1: (N-1)*M - M/2 :2] += 0.1*perAmp*rand(M/4) - 0.1*perAmp*1.j*rand(M/4)

#PSI[(N+1)*M:(N+2)*M] = conj(PSI[(N-1)*M:N*M])
#PSI[(N+2)*M:(N+3)*M] = conj(PSI[(N-2)*M:(N-1)*M])

forcing = zeros((M,2*N+1), dtype='complex')
forcing[0,0] = 2.0/Re

#KE = 0
#SMDY =  mk_single_diffy()
#u0 = dot(SMDY, PSI[N*M: (N+1)*M])
#u0sq = zeros(M, dtype='complex')
#for n in range(0,M,2):
#    for m in range(n-M+1, M):
#
#        p = abs(n-m)
#        if (p==0):
#            tmp = 2.0*u0[p]
#        else:
#            tmp = u0[p]
#
#        if (abs(m)==0):
#            tmp *= 2.0*conj(u0[abs(m)])
#        else:
#            tmp *= conj(u0[abs(m)])
#
#        if (n==0):
#            u0sq[n] += 0.25*tmp
#        else:
#            u0sq[n] += 0.5*tmp
#
#    KE += (2. / (1.-n*n)) * u0sq[n];
#    print KE
#
#u0sq2 = dot(cheb_prod_mat(u0), u0)
#print u0sq2-u0sq
#
#print 'KE0', KE*(15./8.)*0.5




# --------------- SHEAR LAYER -----------------
#
#y_points = cos(pi*arange(Mf)/(Mf-1))
#delta = 0.1
#
## Set initial streamfunction
#PSI = zeros((Mf, 2*Nf+1), dtype='d')
#
#for i in range(Mf):
#    y =y_points[i]
#    for j in range(2*Nf+1):
#        PSI[i,j] = delta * (1./tanh(1./delta)) * log(cosh(y/delta))
#
#del y, i, j
#
#PSI = f2d.to_spectral(PSI, CNSTS)
#
##test = f2d.dy(PSI, CNSTS) 
##test = f2d.to_physical(test, CNSTS)
##savetxt('U.dat', vstack((y_points,test[:,0])).T)
##PSI = f2d.to_physical(PSI, CNSTS)
##savetxt('PSI.dat', vstack((y_points,PSI[:,0])).T)
##exit(1)
#
#PSI = fftshift(PSI, axes=1)
#PSI = PSI.T.flatten()
#
## set forcing
#forcing = zeros((Mf, 2*Nf+1), dtype='d')
#test = zeros((Mf, 2*Nf+1), dtype='d')
#
#for i in range(Mf):
#    y =y_points[i]
#    for j in range(2*Nf+1):
#        forcing[i,j] = ( 2.0/tanh(1.0/delta)) * (1.0/cosh(y/delta)**2) * tanh(y/delta)
#        forcing[i,j] *= 1.0/(Re * delta**2) 
#
#del y, i, j
#forcing = f2d.to_spectral(forcing, CNSTS)
#forcing[:, 1:] = 0
## set BC
#CNSTS['U0'] = 1.0

# ----------------------------------------------------------------------------


f = h5py.File("forcing.h5", "w")
dset = f.create_dataset("psi", ((2*N+1)*M,), dtype='complex')
dset[...] = forcing.T.flatten()
f.close()

# Form the operators
PsiOpInvList = form_operators(dt)
PsiOpInvListHalf = form_operators(dt/2.0)

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

for i in range(N+1):
    # operator order in list is 0->N
    n = i
    print n
    opFn = "./operators/hOp{0}.h5".format(n)
    print "writing ", opFn
    f = h5py.File(opFn, "w")
    dset = f.create_dataset("op", (M*M,), dtype='complex')
    dset[...] = PsiOpInvListHalf[i].flatten()

    f.close()

    #savetxt("./operators/op{0}.dat".format(abs(n)),PsiOpInvList[n])
del i


PSI = PSI.reshape(2*N+1, M).T
PSI = ifftshift(PSI, axes=1)

Cxx = Cxx.reshape(2*N+1, M).T
Cxx = ifftshift(Cxx, axes=1)

Cyy = Cyy.reshape(2*N+1, M).T
Cyy = ifftshift(Cyy, axes=1)

Cxy = Cxy.reshape(2*N+1, M).T
Cxy = ifftshift(Cxy, axes=1)

f = h5py.File("initial_visco.h5", "w")
psih = f.create_dataset("psi", ((2*N+1)*M,), dtype='complex')
psih[...] = PSI.T.flatten()
cxxh = f.create_dataset("cxx", ((2*N+1)*M,), dtype='complex')
cxxh[...] = Cxx.T.flatten()
cyyh = f.create_dataset("cyy", ((2*N+1)*M,), dtype='complex')
cyyh[...] = Cyy.T.flatten()
cxyh = f.create_dataset("cxy", ((2*N+1)*M,), dtype='complex')
cxyh[...] = Cxy.T.flatten()
f.close()


#### TIME ITERATE 

stepsPerFrame = numTimeSteps/numFrames

# Run program in C

# pass the flow variables and the time iteration settings to the C code
if dealiasing:
    cargs = ["./DNS_2D_Visco", "-N", "{0:d}".format(CNSTS["N"]), "-M",
             "{0:d}".format(CNSTS["M"]),"-U", "{0:e}".format(CNSTS["U0"]), "-k",
             "{0:e}".format(CNSTS["kx"]), "-R", "{0:e}".format(CNSTS["Re"]),
             "-W", "{0:e}".format(CNSTS["Wi"]), "-b",
             "{0:e}".format(CNSTS["beta"]), "-t", "{0:e}".format(CNSTS["dt"]),
             "-s", "{0:d}".format(stepsPerFrame), "-T",
             "{0:d}".format(numTimeSteps), "-d"]
    print "./DNS_2D_Visco", "-N", "{0:d}".format(CNSTS["N"]), "-M", \
             "{0:d}".format(CNSTS["M"]),"-U", "{0:e}".format(CNSTS["U0"]), "-k",\
             "{0:e}".format(CNSTS["kx"]), "-R", "{0:e}".format(CNSTS["Re"]),\
             "-W", "{0:e}".format(CNSTS["Wi"]), "-b",\
             "{0:e}".format(CNSTS["beta"]), "-t", "{0:e}".format(CNSTS["dt"]),\
             "-s", "{0:d}".format(stepsPerFrame), "-T",\
             "{0:d}".format(numTimeSteps), "-d"

else:
    cargs = ["./DNS_2D_Visco", "-N", "{0:d}".format(CNSTS["N"]), "-M",
             "{0:d}".format(CNSTS["M"]),"-U", "{0:e}".format(CNSTS["U0"]), "-k",
             "{0:e}".format(CNSTS["kx"]), "-R", "{0:e}".format(CNSTS["Re"]),
             "-W", "{0:e}".format(CNSTS["Wi"]), "-b",
             "{0:e}".format(CNSTS["beta"]), "-t", "{0:e}".format(CNSTS["dt"]),
             "-s", "{0:d}".format(stepsPerFrame), "-T",
             "{0:d}".format(numTimeSteps)]
    print "./DNS_2D_Visco", "-N", "{0:d}".format(CNSTS["N"]), "-M", \
             "{0:d}".format(CNSTS["M"]),"-U", "{0:e}".format(CNSTS["U0"]), "-k",\
             "{0:e}".format(CNSTS["kx"]), "-R", "{0:e}".format(CNSTS["Re"]),\
             "-W", "{0:e}".format(CNSTS["Wi"]), "-b",\
             "{0:e}".format(CNSTS["beta"]), "-t", "{0:e}".format(CNSTS["dt"]),\
             "-s", "{0:d}".format(stepsPerFrame), "-T",\
             "{0:d}".format(numTimeSteps)

subprocess.call(cargs)

# Read in data from the C code
print 'done'
