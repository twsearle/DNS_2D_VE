#-----------------------------------------------------------------------------
#   2D spectral direct numerical simulator
#
#   Last modified: Thu 19 Mar 16:26:58 2015
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
from scipy import fftpack
from numpy.fft import fftshift, ifftshift

import cPickle as pickle
import matplotlib.pyplot as plt

import ConfigParser
import subprocess
import h5py
import TobySpectralMethods as tsm

# SETTINGS---------------------------------------------------------------------

config = ConfigParser.RawConfigParser()
fp = open('config.cfg')
config.readfp(fp)
N = config.getint('General', 'N')
M = config.getint('General', 'M')
Re = config.getfloat('General', 'Re')
Wi = 0.0
beta = 1.0
kx = config.getfloat('General', 'kx')

dealiasing = True

if dealiasing:
    Nf = (3*N)/2 + 1
    Mf = 2*M
else:
    Nf = N
    Mf = M

dt = 1e-6#config.getfloat('Time Iteration', 'dt')
totTime = 2e-6#config.getfloat('Time Iteration', 'totTime')
numFrames = 1#config.getint('Time Iteration', 'numFrames')
fp.close()

numTimeSteps = int(totTime / dt)
assert not (totTime % dt), "non-integer number of time steps!"

NOld = 5
MOld = 40 
kwargs = {'N': N, 'M': M, 'U0': 0, 'Re': Re,'Wi': Wi, 'beta': beta, 'kx': kx,'time':
          totTime, 'dt':dt, 'dealiasing':dealiasing }
baseFileName  = "-N{N}-M{M}-kx{kx}-Re{Re}-b{beta}-Wi{Wi}-dt{dt}.pickle".format(**kwargs)
outFileName  = "pf{0}".format(baseFileName)
outFileNameTrace = "trace{0}.dat".format(baseFileName[:-7])
outFileNameTime = "series-pf{0}".format(baseFileName)
inFileName = "pf-N{N}-M{M}-kx{kx}-Re{Re}.pickle".format(N=NOld, M=MOld, 
                                                        kx=kx, Re=Re)
#inFileName = "pf-N{N}-M{M}-kx{kx}-Re{Re}-b{beta}-Wi{Wi}.pickle".format(**kwargs)

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

def mk_diff_y():
    """Make the matrix to differentiate a velocity vector wrt y."""
    D = mk_single_diffy()
    MDY = zeros( ((2*N+1)*M,  (2*N+1)*M) )
     
    for cheb in range(0,(2*N+1)*M,M):
        MDY[cheb:cheb+M, cheb:cheb+M] = D
    del cheb
    return MDY

def mk_diff_x():
    """Make matrix to do fourier differentiation wrt x."""
    MDX = zeros( ((2*N+1)*M, (2*N+1)*M), dtype='complex')

    n = -N
    for i in range(0, (2*N+1)*M, M):
        MDX[i:i+M, i:i+M] = eye(M, M, dtype='complex')*n*kx*1.j
        n += 1
    del n, i
    return MDX

def cheb_prod_mat(velA):
    """Function to return a matrix for left-multiplying two Chebychev vectors"""

    D = zeros((M, M), dtype='complex')

    for n in range(M):
        for m in range(-M+1,M):     # Bottom of range is inclusive
            itr = abs(n-m)
            if (itr < M):
                D[n, abs(m)] += 0.5*oneOverC[n]*CFunc[itr]*CFunc[abs(m)]*velA[itr]
    del m, n, itr
    return D

def prod_mat(velA):
    """Function to return a matrix ready for the left dot product with another
    velocity vector"""
    MM = zeros(((2*N+1)*M, (2*N+1)*M), dtype='complex')

    #First make the middle row
    midMat = zeros((M, (2*N+1)*M), dtype='complex')
    for n in range(2*N+1):       # Fourier Matrix is 2*N+1 cheb matricies
        yprodmat = cheb_prod_mat(velA[n*M:(n+1)*M])
        endind = 2*N+1-n
        midMat[:, (endind-1)*M:endind*M] = yprodmat
    del n

    #copy matrix into MM, according to the matrix for spectral space
    # top part first
    for i in range(0, N):
        MM[i*M:(i+1)*M, :(N+1+i)*M] = midMat[:, (N-i)*M:]
    del i
    # middle
    MM[N*M:(N+1)*M, :] = midMat
    #  bottom - beware! This thing is pretty horribly written. i = 0 is actually
    #  row index N+1
    for i in range(0, N):
        MM[(i+N+1)*M:(i+2+N)*M, (i+1)*M:] = midMat[:, :(2*N-i)*M]
    del i

    return MM

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

    else: 
        print 'error in deciding resolution'
        exit(1)

    return ovec

def form_operators(dt):
    opinvlist = []

    # zeroth mode
    Psi0thOp = zeros((M,M), dtype='complex')
    Psi0thOp = SMDY - 0.5*dt*oneOverRe*SMDYYY + 0j

    # Apply BCs

    # dypsi0(+-1) = 0
    Psi0thOp[M-3, :] = DERIVTOP
    Psi0thOp[M-2, :] = DERIVBOT
    # psi0(-1) =  0
    Psi0thOp[M-1, :] = BBOT

    opinvlist.append(linalg.inv(Psi0thOp))

    for i in range(1, N+1):
        n = i

        OP = zeros((2*M, 2*M), dtype='complex')
        SLAPLAC = -n*n*kx*kx*SII + SMDYY

        OP[0:M, 0:M] = 0
        OP[0:M, M:2*M] = SII - 0.5*oneOverRe*dt*SLAPLAC

        OP[M:2*M, 0:M] = SLAPLAC
        OP[M:2*M, M:2*M] = -SII

        # Apply BCs
        # dypsi(+-1) = 0
        OP[M-2, :] = concatenate((DERIVTOP, zeros(M, dtype='complex')))
        OP[M-1, :] = concatenate((DERIVBOT, zeros(M, dtype='complex')))
        
        # dxpsi(+-1) = 0
        OP[2*M-2, :] = concatenate((BTOP, zeros(M, dtype='complex')))
        OP[2*M-1, :] = concatenate((BBOT, zeros(M, dtype='complex')))

        # store the inverse of the relevent part of the matrix
        OP = linalg.inv(OP)
        OP = OP[0:M, 0:M]

        opinvlist.append(OP)

    del OP

    opinvlist = array(opinvlist)
    return opinvlist

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

#set up the CFunc function: 2 for m=0, 1 elsewhere:
CFunc = ones(M)
CFunc[0] = 2.

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

# this is Poiseuille flow 
#PSI[N*M]   += 2.0/3.0
#PSI[N*M+1] += 3.0/4.0
#PSI[N*M+2] += 0.0
#PSI[N*M+3] += -1.0/12.0

# some junk to put in as a test.
#PSI[(N-2)*M+ 0]   += 1
#PSI[(N-2)*M+ 1] += 1
#PSI[(N-2)*M+ 2] += 2.0
#PSI[(N-2)*M+ 3] += 1
#PSI[(N-2)*M+ 4] += 3
# 
#PSI[(N+2)*M:(N+3)*M] = conj(PSI[(N-2)*M:(N-1)*M])

# Read in stream function from file
(PSI, Nu) = pickle.load(open(inFileName,'r'))
PSI = decide_resolution(PSI, NOld, MOld, CNSTS)

for n in range(1,N+1):
    PSI[(N-n)*M: (N-n+1)*M] = conj(PSI[(N+n)*M: (N+n+1)*M])

#for n in range(1,N+1):
#    PSI[(N+n)*M: (N+n+1)*M] = r_[0:M:1]
#    PSI[(N-n)*M: (N-n+1)*M] = r_[0:M:1]

#PSI[:] = rand((2*N+1)*M) + 1.j*rand((2*N+1)*M)
#PSI[N*M:(N+1)*M] = rand(M)

#for i in range(2*N+1-2):
#    PSI[(i+1)*M:(i+2)*M] = r_[0:M][::-1] /(1.*M)


# Form the operators
PsiOpInvListHalf = form_operators(dt/2.0)
PsiOpInvList = form_operators(dt)

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

# make PSI 2D
PSI = PSI.reshape(2*N+1, M).T
# put PSI into FFT ordering.
PSI = ifftshift(PSI, axes=1)

# savetxt("initial.dat", PSI.T, fmt='%.18e')

print "writing initial state to initial.h5"

f = h5py.File("initial.h5", "w")
dset = f.create_dataset("psi", ((2*N+1)*M,), dtype='complex')
dset[...] = PSI.T.flatten()
print shape(dset)
f.close()

#### TIME ITERATE 

stepsPerFrame = numTimeSteps/numFrames


tsm.initTSM(N_=N,M_=M,kx_=kx)
MDY = mk_diff_y()
MDX = mk_diff_x()

# Run program in C

# pass the flow variables and the time iteration settings to the C code
if dealiasing:
    cargs = ["./DNS_2D_Newt", "-N", "{0:d}".format(CNSTS["N"]), "-M",
             "{0:d}".format(CNSTS["M"]),"-U", "{0:e}".format(CNSTS["U0"]), "-k",
             "{0:e}".format(CNSTS["kx"]), "-R", "{0:e}".format(CNSTS["Re"]),
             "-W", "{0:e}".format(CNSTS["Wi"]), "-b",
             "{0:e}".format(CNSTS["beta"]), "-t", "{0:e}".format(CNSTS["dt"]),
             "-s", "{0:d}".format(stepsPerFrame), "-T",
             "{0:d}".format(numTimeSteps), "-d"]

    print "./DNS_2D_Newt", "-N", "{0:d}".format(CNSTS["N"]), "-M", \
             "{0:d}".format(CNSTS["M"]),"-U", "{0:e}".format(CNSTS["U0"]), "-k",\
             "{0:e}".format(CNSTS["kx"]), "-R", "{0:e}".format(CNSTS["Re"]),\
             "-W", "{0:e}".format(CNSTS["Wi"]), "-b",\
             "{0:e}".format(CNSTS["beta"]), "-t", "{0:e}".format(CNSTS["dt"]),\
             "-s", "{0:d}".format(stepsPerFrame), "-T",\
             "{0:d}".format(numTimeSteps), "-d"

else:
    cargs = ["./DNS_2D_Newt", "-N", "{0:d}".format(CNSTS["N"]), "-M",
             "{0:d}".format(CNSTS["M"]),"-U", "{0:e}".format(CNSTS["U0"]), "-k",
             "{0:e}".format(CNSTS["kx"]), "-R", "{0:e}".format(CNSTS["Re"]),
             "-W", "{0:e}".format(CNSTS["Wi"]), "-b",
             "{0:e}".format(CNSTS["beta"]), "-t", "{0:e}".format(CNSTS["dt"]),
             "-s", "{0:d}".format(stepsPerFrame), "-T",
             "{0:d}".format(numTimeSteps)]
    print "./DNS_2D_Newt", "-N", "{0:d}".format(CNSTS["N"]), "-M", \
             "{0:d}".format(CNSTS["M"]),"-U", "{0:e}".format(CNSTS["U0"]), "-k",\
             "{0:e}".format(CNSTS["kx"]), "-R", "{0:e}".format(CNSTS["Re"]),\
             "-W", "{0:e}".format(CNSTS["Wi"]), "-b",\
             "{0:e}".format(CNSTS["beta"]), "-t", "{0:e}".format(CNSTS["dt"]),\
             "-s", "{0:d}".format(stepsPerFrame), "-T",\
             "{0:d}".format(numTimeSteps)

subprocess.call(cargs)

# Read in data from the C code

print "check variables are properly calculated in c code"

print shape(load_hdf5_state("./output/psi.h5"))
print shape(PSI)
psic = load_hdf5_state("./output/psi.h5").reshape(2*N+1, M).T 

print 'initial streamfunction?', allclose(PSI,psic)
print 'difference', linalg.norm(psic-PSI)
if linalg.norm(psic-PSI) > 0.0:
    print ( PSI)[0,:]
    print (psic)[0,:]
    print "FAIL"
    exit(1)
#print 'zeroth mode c-python', psic[:M]- PSI[:M]

# switch PSI back to normal ordering for F modes
PSI = fftshift(PSI, axes=1)
PSI = PSI.T.flatten()
PSIbkp = copy(PSI)

# u
U = dot(MDY, PSI)
USQ = dot(tsm.prod_mat(U), U)

INTY = mk_cheb_int()
print 'KE0 (U^2 only): ', (15.0/16.0)*dot(INTY, USQ[N*M:(N+1)*M])

# read in and make it 2D to get rid of the junk Cheby modes.
# Then take transpose and flatten to return to 2*N+1 chunks of M length.
Uc = load_hdf5_state("./output/u.h5").reshape(2*N+1, M).T 
Uc = fftshift(Uc, axes=1)
Uc = Uc.T.flatten()

print 'U ?', allclose(U, Uc)
print 'difference', linalg.norm(U-Uc)
if linalg.norm(U-Uc) > 1e-12:
    print 'U0', U[N*M: (N+1)*M]
    print 'U0c', Uc[N*M: (N+1)*M]
    print 'FAIL'
    exit(1)

# v
V = -dot(MDX, PSI)

Vc = load_hdf5_state("./output/v.h5").reshape(2*N+1, M).T 
pickle.dump(Vc, open('testVc.pickle','w'))
Vc = fftshift(Vc, axes=1)
Vc = Vc.T.flatten()

print 'V ?', allclose(V, Vc)
print 'difference', linalg.norm(V-Vc)

if not allclose(V, Vc):
    print 'difference', linalg.norm(V-Vc)
    #print 'V1', V[M: 2*M]
    #print 'V1c', Vc[M: 2*M]
    print 'difference', (V-Vc)[N*M+38::M]
    

lplc = load_hdf5_state("./output/lplpsi.h5").reshape(2*N+1, M).T 
lplc = fftshift(lplc, axes=1)
lplc = lplc.T.flatten()

# THE ORDER OF OPERATIONS HERE MATTERS!
#LPLPSI = dot(dot(MDY,MDY) + dot(MDX,MDX), PSI)
LPLPSI = dot(dot(MDY,MDY), PSI) + dot(dot(MDX,MDX), PSI)

print 'laplacian psi ?', allclose(LPLPSI, lplc)
print 'difference', linalg.norm(LPLPSI-lplc)
if not allclose(LPLPSI, lplc):
    #print 'LPLPSI1', LPLPSI[M: 2*M]
    #print 'LPLPSI1c', lplc[M: 2*M]
    print 'difference', (LPLPSI-lplc)[N*M+38::M]

d2yc = load_hdf5_state("./output/d2ypsi.h5").reshape(2*N+1, M).T 
d2yc = fftshift(d2yc, axes=1)
d2yc = d2yc.T.flatten()

D2YPSI = dot(MDY, dot(MDY,PSI) )
D2YPSIud = dot(MDY[:, ::-1], dot(MDY[:, ::-1],PSI[::-1])[::-1] )

print 'd2y psi ?', allclose(D2YPSI, d2yc)
print 'difference', linalg.norm(D2YPSI-d2yc)
print 'ud difference', linalg.norm(D2YPSIud-d2yc)

d3yc = load_hdf5_state("./output/d3ypsi.h5").reshape(2*N+1, M).T 
d3yc = fftshift(d3yc, axes=1)
d3yc = d3yc.T.flatten()

D3YPSI = dot(MDY, dot(MDY, dot(MDY,PSI) ) )

print 'd3y psi ?', allclose(D3YPSI, d3yc)
print 'difference', linalg.norm(D3YPSI-d3yc)
if linalg.norm(D3YPSI-d3yc) > 1e-12:
    #print 'd3y0', D3YPSI[N*M:(N+1)*M]
    #print 'd3yc0', d3yc[N*M:(N+1)*M]
    print 'FAIL'
    exit(1)

d4yc = load_hdf5_state("./output/d4ypsi.h5").reshape(2*N+1, M).T 
d4yc = fftshift(d4yc, axes=1)
d4yc = d4yc.T.flatten()

D4YPSI = dot(MDY, dot(MDY, dot(MDY, dot(MDY,PSI) ) ) )

print 'd4y psi ?', allclose(D4YPSI, d4yc)
print 'difference', linalg.norm(D4YPSI-d4yc)
print "mode 0", linalg.norm(D4YPSI[N*M:(N+1)*M]-d4yc[N*M:(N+1)*M])
for n in range(1,N+1):
    print "mode", n, linalg.norm(D4YPSI[(N+n)*M:(N+n+1)*M]-d4yc[(N+n)*M:(N+n+1)*M])
    print "mode", -n,linalg.norm(D4YPSI[(N-n)*M:(N+1-n)*M]-d4yc[(N-n)*M:(N+1-n)*M])
    
d2xc = load_hdf5_state("./output/d2xpsi.h5").reshape(2*N+1, M).T 
d2xc = fftshift(d2xc, axes=1)
d2xc = d2xc.T.flatten()

D2XPSI = dot(MDX, dot(MDX,PSI) )
#print 'difference', linalg.norm(D2XPSI-PSIbkp)

print 'd2x psi ?', allclose(D2XPSI, d2xc)
print 'difference', linalg.norm(D2XPSI-d2xc)

print "mode 0", linalg.norm(D2XPSI[N*M:(N+1)*M]-d2xc[N*M:(N+1)*M])
for n in range(1,N+1):
    print "mode", n, linalg.norm(D2XPSI[(N+n)*M:(N+n+1)*M]-d2xc[(N+n)*M:(N+n+1)*M])
    print "mode", -n,linalg.norm(D2XPSI[(N-n)*M:(N+1-n)*M]-d2xc[(N-n)*M:(N+1-n)*M])

d4xc = load_hdf5_state("./output/d4xpsi.h5").reshape(2*N+1, M).T 
d4xc = fftshift(d4xc, axes=1)
d4xc = d4xc.T.flatten()

D4XPSI = dot(MDX, dot(MDX, dot(MDX, dot(MDX,PSI) ) ) )

print 'd4x psi ?', allclose(D4XPSI, d4xc)
print 'difference', linalg.norm(D4XPSI-d4xc)

print "mode 0", linalg.norm(D4XPSI[N*M:(N+1)*M]-d4xc[N*M:(N+1)*M])
for n in range(1,N+1):
    print "mode", n, linalg.norm(D4XPSI[(N+n)*M:(N+n+1)*M]-d4xc[(N+n)*M:(N+n+1)*M])
    print "mode", -n,linalg.norm(D4XPSI[(N-n)*M:(N+1-n)*M]-d4xc[(N-n)*M:(N+1-n)*M])
    
d2xd2yc = load_hdf5_state("./output/d2xd2ypsi.h5").reshape(2*N+1, M).T 
d2xd2yc = fftshift(d2xd2yc, axes=1)
d2xd2yc = d2xd2yc.T.flatten()

D2XD2YPSI = dot(MDX, dot(MDX, dot(MDY, dot(MDY,PSI) ) ) )

print 'd2xd2y psi ?', allclose(D2XD2YPSI, d2xd2yc)
print 'difference', linalg.norm(D2XD2YPSI-d2xd2yc)
    
biharmc = load_hdf5_state("./output/biharmpsi.h5").reshape(2*N+1, M).T 
biharmc = fftshift(biharmc, axes=1)
biharmc = biharmc.T.flatten()

# ORDER OF OPERATIONS MATTERS!!!!!
# BIHARMPSI = dot(dot(MDY,MDY) + dot(MDX,MDX), LPLPSI)
#BIHARMPSI = dot(dot(MDY,MDY), LPLPSI) + dot(dot(MDX,MDX), LPLPSI)
BIHARMPSI = D4XPSI + D4YPSI + 2*D2XD2YPSI 

print 'biharm psi ?', allclose(BIHARMPSI, biharmc)
print 'difference', linalg.norm(BIHARMPSI-biharmc)
if not allclose(BIHARMPSI, biharmc):
    #print 'BIHARMPSI1', BIHARMPSI[M: 2*M]
    #print 'BIHARMPSI1c', biharmc[M: 2*M]
    print 'difference', (BIHARMPSI-biharmc)[N*M+38::M]

dxlplc = load_hdf5_state("./output/dxlplpsi.h5").reshape(2*N+1, M).T 
dxlplc = fftshift(dxlplc, axes=1)
dxlplc = dxlplc.T.flatten()

DXLPLPSI = dot(MDX, LPLPSI)

print "dxlplpsi ", allclose(DXLPLPSI, dxlplc)
print 'difference', linalg.norm(DXLPLPSI-dxlplc)

if not allclose(DXLPLPSI, dxlplc):

    print "mode 0", linalg.norm(DXLPLPSI[N*M:(N+1)*M]-dxlplc[N*M:(N+1)*M])
    for n in range(1,N+1):
        print "mode", n, linalg.norm(DXLPLPSI[(N+n)*M:(N+n+1)*M]-dxlplc[(N+n)*M:(N+n+1)*M])
        print "mode", -n,linalg.norm(DXLPLPSI[(N-n)*M:(N+1-n)*M]-dxlplc[(N-n)*M:(N+1-n)*M])

    print "mode 0", linalg.norm(dxlplc[N*M:(N+1)*M])
    for n in range(1,N+1):
        print "mode", n, linalg.norm(dxlplc[(N+n)*M:(N+n+1)*M])
        print "mode", -n,linalg.norm(dxlplc[(N-n)*M:(N+1-n)*M])

udxlplc = load_hdf5_state("./output/udxlplpsi.h5").reshape(2*N+1, M).T 
udxlplc = fftshift(udxlplc, axes=1)
udxlplc = udxlplc.T.flatten()

UDXLPLPSI = dot(prod_mat(U), dot(MDX, LPLPSI))

print 'udxlolpsi ?', allclose(UDXLPLPSI, udxlplc)
print 'difference', linalg.norm(UDXLPLPSI-udxlplc)
if not allclose(UDXLPLPSI, udxlplc):
    print 'relative difference', linalg.norm(UDXLPLPSI-udxlplc)

    print "max difference", amax(UDXLPLPSI-udxlplc)
    print "max difference arg", argmax(UDXLPLPSI-udxlplc)

    print "mode 0", linalg.norm(UDXLPLPSI[N*M:(N+1)*M]-udxlplc[N*M:(N+1)*M])
    for n in range(1,N+1):
        print "mode", n, linalg.norm(UDXLPLPSI[(N+n)*M:(N+n+1)*M]-udxlplc[(N+n)*M:(N+n+1)*M])
        print "mode", -n,linalg.norm(UDXLPLPSI[(N-n)*M:(N+1-n)*M]-udxlplc[(N-n)*M:(N+1-n)*M])

dylplc = load_hdf5_state("./output/dylplpsi.h5").reshape(2*N+1, M).T 
dylplc = fftshift(dylplc, axes=1)
dylplc = dylplc.T.flatten()

DYLPLPSI = dot(MDY, LPLPSI)

print "dylplpsi ", allclose(DYLPLPSI, dylplc)
print 'difference', linalg.norm(DYLPLPSI-dylplc)

vdylplc = load_hdf5_state("./output/vdylplpsi.h5").reshape(2*N+1, M).T 
vdylplc = fftshift(vdylplc, axes=1)
vdylplc = vdylplc.T.flatten()

VDYLPLPSI = dot(prod_mat(V), dot(MDY, LPLPSI))

print 'vdylplpsi ?', allclose(VDYLPLPSI, vdylplc)
print 'difference', linalg.norm(VDYLPLPSI-vdylplc)
if not allclose(VDYLPLPSI, vdylplc):
    print 'relative difference', linalg.norm(VDYLPLPSI-vdylplc)
    #print 'VDYLPLPSI1', VDYLPLPSI[M: 2*M]
    #print 'VDYLPLPSI1c', vdylplc[M: 2*M]
    print 'difference', (VDYLPLPSI-vdylplc)[N*M+38::M]

vdyyc = load_hdf5_state("./output/vdyypsi.h5").reshape(2*N+1, M).T 
vdyyc = fftshift(vdyyc, axes=1)
vdyyc = vdyyc.T.flatten()

VDYU = dot(prod_mat(V), dot(MDY, dot(MDY, PSI))) 

print 'vdyypsi ?', allclose(VDYU, vdyyc)
print 'difference', linalg.norm(VDYU-vdyyc)
if not allclose(VDYU, vdyyc):
    #print 'VDYU1', VDYU[M: 2*M]
    #print 'VDYU1c', vdyyc[M: 2*M]
    print ' high mode difference', (VDYU-vdyyc)[N*M+38::M]

print """
--------------
Operator check
--------------
"""
op0c = load_hdf5_state("./output/op0.h5")#.reshape(M, M).T 

print 'operator 0', linalg.norm(op0c-PsiOpInvList[0].flatten())

for i in range(1,N+1):
    opc = load_hdf5_state("./output/op{0}.h5".format(i))
    print 'operator ',i, linalg.norm(opc - PsiOpInvList[i].flatten())
    if linalg.norm(opc - PsiOpInvList[i].flatten()) > 0.0 :
        exit(1)

hop0c = load_hdf5_state("./output/hOp0.h5")#.reshape(M, M).T 
print 'half operator 0', linalg.norm(hop0c-PsiOpInvListHalf[0].flatten())

for i in range(1,N+1):
    hopc = load_hdf5_state("./output/hOp{0}.h5".format(i))
    print 'half operator ',i, linalg.norm(hopc - PsiOpInvListHalf[i].flatten())
    if linalg.norm(hopc - PsiOpInvListHalf[i].flatten()) > 0.0 :
        exit(1)

DYYYPSI = dot(MDY, dot(MDY, dot(MDY, PSI)))

RHSVec = dt*0.5*oneOverRe*BIHARMPSI \
        + LPLPSI \
        - dt*UDXLPLPSI \
        - dt*VDYLPLPSI 

#RHSVec = dt*0.5*oneOverRe*biharmc \
#        + lplc \
#        - dt*udxlplc \
#        - dt*vdylplc 


# Zeroth mode
RHSVec[N*M:(N+1)*M] = 0
RHSVec[N*M:(N+1)*M] = dt*0.5*oneOverRe*DYYYPSI[N*M:(N+1)*M] \
        + U[N*M:(N+1)*M] \
        - dt*VDYU[N*M:(N+1)*M]
RHSVec[N*M] += dt*2*oneOverRe

#RHSVec[N*M:(N+1)*M] = 0
#RHSVec[N*M:(N+1)*M] = dt*0.5*oneOverRe*d3yc[N*M:(N+1)*M] \
#        + Uc[N*M:(N+1)*M] \
#        - dt*vdyyc[N*M:(N+1)*M]
#RHSVec[N*M] += dt*2*oneOverRe

# Apply BC's

for n in range (N+1): 
    # dyPsi(+-1) = 0  
    # Only impose the BC which is actually present in the inverse operator
    # we are dealing with. Remember that half the Boundary Conditions were
    # imposed on phi, which was accounted for implicitly when we ignored it.
    RHSVec[(N+n)*M + M-2] = 0
    RHSVec[(N+n)*M + M-1] = 0
del n

# dyPsi0(+-1) = 0
RHSVec[N*M + M-3] = 0
RHSVec[N*M + M-2] = 0

# Psi0(-1) = 0
RHSVec[N*M + M-1] = 0

for i in range(0,N+1):
    RHSvecc = load_hdf5_state("./output/RHSVec{0}.h5".format(i))
    print "RHSvec for mode ", i, allclose(RHSVec[(N+i)*M:(N+1+i)*M], RHSvecc)
    print 'difference', linalg.norm(RHSVec[(N+i)*M:(N+1+i)*M]- RHSvecc)
    print 'max difference', amax(RHSVec[(N+i)*M:(N+1+i)*M]- RHSvecc)
    maxarg_ = argmax(RHSVec[(N+i)*M:(N+1+i)*M]- RHSvecc)
    print 'argmax difference', maxarg_
    #print RHSVec[(N+i)*M+maxarg_]
    #print RHSvecc[maxarg_]

    #if i ==0:
    #    print 'c', RHSvecc
    #    print 'mat', RHSVec[N*M:(N+1)*M]
    #    print RHSvecc - RHSVec[N*M:(N+1)*M]
    #    print 'RHSVecC - RHSvec components'
    #    print RHSvecc - Uc[N*M:(N+1)*M] - 0.5*dt*oneOverRe*d3yc[N*M:(N+1)*M] \
    #            -2*dt*oneOverRe
    #    print RHSvecc - U[N*M:(N+1)*M] - 0.5*dt*oneOverRe*D3YPSI[N*M:(N+1)*M] \
    #            -2*dt*oneOverRe
    #    print 'RHSVec - RHSvec components'
    #    print RHSVec[N*M:(N+1)*M] - U[N*M:(N+1)*M] - 0.5*dt*oneOverRe*D3YPSI[N*M:(N+1)*M] \
    #           -2*dt*oneOverRe

psi2c = load_hdf5_state("./output/psi2.h5").reshape(2*N+1, M).T 

#PSI[N*M:(N+1)*M] = dot(PsiOpInvList[0], RHSVec[N*M:(N+1)*M])
for i in range(M):
    PSI[(N)*M + i] = 0
    #for j in range(M-1,-1,-1):
    for j in range(M):
        PSI[(N)*M + i] += PsiOpInvList[0][i,j] * RHSVec[(N)*M + j]

#for n in range(1,N+1):
#    PSI[(N+n)*M:(N+n+1)*M] = dot(PsiOpInvList[n], RHSVec[(N+n)*M:(N+n+1)*M])
#    del n  
for n in range(1,N+1):
    for i in range(M):
        PSI[(N+n)*M + i] = 0
        for j in range(M):
            PSI[(N+n)*M + i] += PsiOpInvList[n][i,j] * RHSVec[(N+n)*M + j]
    del n  

for n in range(0,N):
    PSI[n*M:(n+1)*M] = conj(PSI[(2*N-n)*M:(2*N+1-n)*M])

#PSI = pickle.load(open('./output/PSI10test.pickle', 'r')) 

PSI22D = PSI.reshape(2*N+1, M).T
PSI22D = ifftshift(PSI22D, axes=-1)

print 'psi2 = psi2c?', allclose(PSI22D, psi2c)

print 'mode', 0, allclose(PSI22D[:, 0], psi2c[:, 0])
print 'difference', linalg.norm(PSI22D[:, 0]-psi2c[:, 0])
print 'max difference', amax(PSI22D[:, 0]-psi2c[:, 0])
maxarg_ = argmax(PSI22D[:, 0]-psi2c[:, 0])
print 'argmax difference', maxarg_
print PSI22D[maxarg_, 0]
print psi2c[maxarg_, 0]

#if not allclose(PSI22D, psi2c):
for i in range(1,N+1):
    print 'mode', i, allclose(PSI22D[:, i], psi2c[:, i])
    print 'difference', linalg.norm(PSI22D[:, i]-psi2c[:, i])
    print 'mode', -i, allclose(PSI22D[:, 2*N+1-i], psi2c[:, 2*N+1-i])
    print 'difference', linalg.norm(PSI22D[:, 2*N+1-i]-psi2c[:, 2*N+1-i])

    print 'conjugation', allclose(PSI22D[:,i], conj(PSI22D[:, 2*N+1-i]))
    print 'conjugation', allclose(psi2c[:,i], conj(psi2c[:, 2*N+1-i]))

