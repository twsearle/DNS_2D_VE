#-----------------------------------------------------------------------------
#   2D spectral direct numerical simulator
#
#   Last modified: Tue  1 Dec 17:53:30 2015
#
#-----------------------------------------------------------------------------

"""


Simulation of 2D Newtonian flow.

Outline:
    
    * read in data

    * Form operators for linear semi-implicit crank-nicolson timestepping

    * RUN THE C PROGRAM:

        * for all times do:

            * solve for PSI0 at current time based on previous time

            * solve for psi0 at current time based on previous time

        until: timeup



"""

# MODULES
from scipy import *
from scipy import linalg
from scipy import optimize
from numpy.linalg import cond 
from numpy.fft import fftshift, ifftshift
from numpy.random import rand
import matplotlib.pyplot as plt

import cPickle as pickle

import ConfigParser
import argparse
import subprocess
import h5py

import fields_2D as f2d
import TobySpectralMethods as tsm

# SETTINGS---------------------------------------------------------------------

config = ConfigParser.RawConfigParser()
fp = open('config.cfg')
config.readfp(fp)
M = config.getint('General', 'M')
Re = config.getfloat('General', 'Re')
Wi = config.getfloat('General', 'Wi')
beta = config.getfloat('General', 'beta')
kx = config.getfloat('General', 'kx')

De = config.getfloat('Oscillatory Flow', 'De')

dt = config.getfloat('Time Iteration', 'dt')
totTime = config.getfloat('Time Iteration', 'totTime')
numFrames = config.getint('Time Iteration', 'numFrames')
dealiasing = config.getboolean('Time Iteration', 'Dealiasing')

N = 1

fp.close()

argparser = argparse.ArgumentParser()

tmp = """simulation type, 
            0: Poiseuille
            1: Shear Layer
            2: Oscillatory"""
argparser.add_argument("-flow_type", type=int, default=3, 
                       help=tmp)

args = argparser.parse_args()
initTime =0.0

if dealiasing:
    Nf = (3*N)/2 + 1
    Mf = 2*M
else:
    Nf = N
    Mf = M

numTimeSteps = int(totTime / dt)
assert (totTime / dt) - float(numTimeSteps) == 0, "Non-integer number of timesteps"
assert args.flow_type < 3, "flow type unspecified!" 

NOld = N 
MOld = M
CNSTS = {'NOld': NOld, 'MOld': MOld, 'N': N, 'M': M, 'Nf':Nf, 'Mf':Mf,'U0':0,
          'Re': Re, 'Wi': Wi, 'beta': beta, 'De':De, 'kx': kx,'time': totTime,
         'dt':dt, 'P': 1.0,
          'dealiasing':dealiasing}
baseFileName  = "-N{N}-M{M}-kx{kx}-Re{Re}.pickle".format(**CNSTS)
outFileName  = "pf{0}".format(baseFileName)
outFileNameTrace = "trace{0}.dat".format(baseFileName[:-7])
outFileNameTime = "series-pf{0}".format(baseFileName)
#inFileName = "pf-N{N}-M{M}-kx{kx}-Re{Re}.pickle".format(N=NOld, M=MOld, 
#                                                        kx=kx, Re=Re)
inFileName = "pf-N{N}-M{M}-kx{kx}-Re{Re}.pickle".format(**CNSTS)


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

def x_independent_profile(PSI):
    """
     I think these are the equations for the x independent stresses from the base
     profile.
    """

    dyu = dot(SMDYY, PSI[N*M:(N+1)*M])
    Cyy = zeros(vecLen, dtype='complex')
    Cyy[N*M] += 1.0
    Cxy = zeros(vecLen, dtype='complex')
    Cxy[N*M:(N+1)*M] = Wi*dyu
    Cxx = zeros(vecLen, dtype='complex')
    Cxx[N*M:(N+1)*M] = 2*Wi*Wi*dot(cheb_prod_mat(dyu), dyu)
    Cxx[N*M] += 1.0

    return (Cxx, Cyy, Cxy)

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
    Psi0thOp = SMDY - 0.5*dt*oneOverRe*beta*SMDYYY + 0j

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
        PSIOP[0:M, M:2*M] = SII - 0.5*oneOverRe*beta*dt*SLAPLAC

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

def form_oscil_operators(dt):
    PsiOpInvList = []

    B = (pi*Re*De) / (2*Wi)

    # zeroth mode
    Psi0thOp = zeros((M,M), dtype='complex')
    Psi0thOp = B*SMDY - 0.5*dt*beta*SMDYYY + 0j

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
        PSIOP[0:M, M:2*M] = B*SII - 0.5*oneOverRe*beta*dt*SLAPLAC

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

def oscillatory_flow():
    """
    Some flow variables must be calculated in realspace and then transformed
    spectral space, Cyy =1.0 so it is easy.
    """

    y_points = cos(pi*arange(Mf)/(Mf-1))

    tmp = beta + (1-beta) / (1 + 1.j*De)
    alpha = sqrt( (1.j*pi*Re*De) / (2*Wi*tmp) )

    Chi = real( (1-1.j)*(1 - tanh(alpha) / alpha) )

    # the coefficient for the forcing
    P = (0.5*pi)**2 * (Re*De) / (Chi*Wi)

    PSI = zeros((Mf, 2*Nf+1), dtype='d')
    Cxx = zeros((Mf, 2*Nf+1), dtype='d')
    Cxy = zeros((Mf, 2*Nf+1), dtype='d')

    for i in range(Mf):
        y =y_points[i]
        for j in range(2*Nf+1):
            psi_im = pi/(2.j*Chi) *(y-sinh(alpha*y)/(alpha*cosh(alpha))\
                                     + sinh(alpha*-1)/(alpha*cosh(alpha)) )
            PSI[i,j] = real(psi_im)

            dyu_cmplx = pi/(2.j*Chi) *(-alpha*sinh(alpha*y)/(cosh(alpha)))
            cxy_cmplx = (1.0/(1.0+1.j*De)) * ((2*Wi/pi) * dyu_cmplx) 

            Cxy[i,j] = real( cxy_cmplx )

            Cxx[i,j] = (1.0/(1.0+2.j*De))*(Wi/pi)*(cxy_cmplx*dyu_cmplx)
            Cxx[i,j] += (1.0/(1.0-2.j*De))*(Wi/pi)*(conj(cxy_cmplx)*conj(dyu_cmplx)) 

            Cxx[i,j] += 1. + (Wi/pi)*( cxy_cmplx*conj(dyu_cmplx) +
                                       conj(cxy_cmplx)*dyu_cmplx ) 
            Cxx[i,j] = real(Cxx[i,j])

    del y, i, j

    # transform to spectral space.
    PSI = f2d.to_spectral(PSI, CNSTS)
    PSI = fftshift(PSI, axes=1)
    PSI = PSI.T.flatten()
    Cxx = f2d.to_spectral(Cxx, CNSTS)
    Cxx = fftshift(Cxx, axes=1)
    Cxx = Cxx.T.flatten()
    Cxy = f2d.to_spectral(Cxy, CNSTS)
    Cxy = fftshift(Cxy, axes=1)
    Cxy = Cxy.T.flatten()

    Cyy = zeros((2*N+1)*M, dtype='complex')
    Cyy[N*M] = 1

    forcing = zeros((M,2*N+1), dtype='complex')
    forcing[0,0] = P

    return PSI, Cxx, Cyy, Cxy, forcing, P

def stupid_transform(GLreal, CNSTS):
    """
    apply the Chebyshev transform the stupid way.
    """

    M = CNSTS['M']

    out = zeros(M)

    for i in range(M):
        out[i] += (1./(M-1.))*GLreal[0]
        for j in range(1,M-1):
            out[i] += (2./(M-1.))*GLreal[j]*cos(pi*i*j/(M-1))
        out[i] += (1./(M-1.))*GLreal[M-1]*cos(pi*i)
    del i,j

    out[0] = out[0]/2.
    out[M-1] = out[M-1]/2.

    return out

def stupid_transform_i(GLspec, CNSTS):
    """
    apply the Chebyshev transform the stupid way.
    """

    M = CNSTS['M']
    Mf = CNSTS['Mf']

    out = zeros(Mf, dtype='complex')

    for i in range(Mf):
        out[i] += GLspec[0]
        for j in range(1,M-1):
            out[i] += GLspec[j]*cos(pi*i*j/(Mf-1))
        out[i] += GLspec[M-1]*cos(pi*i)
    del i,j

    return out

def test_arrays_equal(arr1, arr2, tol=1e-12):
    testBool = allclose(arr1, arr2)
    print testBool
    if not testBool:
        print 'difference', linalg.norm(arr1-arr2)

        if linalg.norm(arr1-arr2)>tol:
            print 'relative difference', linalg.norm(arr1-arr2)

            print "max difference", amax(arr1-arr2)
            print "max difference arg", argmax(arr1-arr2)

            if shape(arr1) == (M, 2*N+1):
                print "mode 0", linalg.norm(arr1[:,0]-arr2[:,0])
                for n in range(1,N+1):
                    print "mode", n, linalg.norm(arr1[:, n]-arr2[:, n])
                    print "mode", -n,linalg.norm(arr1[:, n]-arr2[:, n])

            if shape(arr1) == ((2*N+1)*M):
                print "mode 0", linalg.norm(arr1[N*M:(N+1)*M]-arr2[N*M:(N+1)*M])
                for n in range(1,N+1):
                    print "mode", n, linalg.norm(arr1[(N+n)*M:(N+n+1)*M]-arr2[(N+n)*M:(N+n+1)*M])
                    print "mode", -n,linalg.norm(arr1[(N-n)*M:(N+1-n)*M]-arr2[(N-n)*M:(N+1-n)*M])


            plt.plot(real(arr1.flatten()))
            plt.plot(real(arr2.flatten()), linewidth=2.0)
            #plt.semilogy( abs(arr1-arr2).flatten() )
            plt.show(block=True)

            print 'FAIL'

            exit(1)

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

print"=====================================\n"
print "Settings:"
print """------------------------------------
M \t\t= {M}              
Re \t\t= {Re}         
kx \t\t= {kx}
dt\t\t= {dt}
totTime\t\t= {t}
NumTimeSteps\t= {NT}
------------------------------------
        """.format(M=M, kx=kx, Re=Re, dt=dt, NT=numTimeSteps, t=totTime)

# SET UP

vecLen = (2*N+1)*M

oneOverRe = 1. / Re
assert oneOverRe != infty, "Can't set Reynold's to zero!"

CFunc = ones(M)
CFunc[0] = 2.0
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

#### The base flow and initial perturbation flow stream-functions
PSI = zeros((2*N+1)*M, dtype='complex')


if args.flow_type==0:
    # --------------- POISEUILLE -----------------


    plugAmp = 0.00 #* (M/32.0)

    PSI[N*M]   +=  2.0/3.0
    PSI[N*M+1] +=  3.0/4.0
    PSI[N*M+2] +=  0.0
    PSI[N*M+3] +=  -1.0/12.0

    PSI[(N+1)*M:(N+1)*M + M/2] = 1e-2*(rand(M/2) + 1.j*rand(M/2))
    PSI[(N-1)*M:(N)*M] = conj(PSI[(N+1)*M:(N+2)*M])

    ## set initial stress guess based on laminar flow
    Cxx, Cyy, Cxy = x_independent_profile(PSI)

    psiLam = copy(PSI)

    forcing = zeros((M,2*N+1), dtype='complex')
    forcing[0,0] = 2.0/Re

elif args.flow_type==2:
    # --------------- OSCILLATORY FLOW -----------------
    PSI, Cxx, Cyy, Cxy, forcing, CNSTS['P'] = oscillatory_flow()
    psiLam = copy(PSI)

f = h5py.File("forcing.h5", "w")
dset = f.create_dataset("psi", (3*M,), dtype='complex')
dset[...] = forcing.T.flatten()
f.close()

f = h5py.File("laminar.h5", "w")
dset = f.create_dataset("psi", ((2*N+1)*M,), dtype='complex')
psiLam = psiLam.reshape(2*N+1, M).T
psiLam = ifftshift(psiLam, axes=1)
dset[...] = psiLam.T.flatten()
f.close()

# Form the operators
if args.flow_type==2:
    PsiOpInvList = form_oscil_operators(dt)
    PsiOpInvListHalf = form_oscil_operators(dt/2.0)
else:
    PsiOpInvList = form_operators(dt)
    PsiOpInvListHalf = form_operators(dt/2.0)

#### SAVE THE OPERATORS AND INITIAL STATE FOR THE C CODE

for i in range(2):
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

for i in range(2):
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

print "writing initial state to initial.h5"

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
    print "./DNS_2D_linear_Visco", "-N", "{0:d}".format(CNSTS["N"]), "-M",\
             "{0:d}".format(CNSTS["M"]),"-U", "{0:e}".format(CNSTS["U0"]), "-k",\
             "{0:e}".format(CNSTS["kx"]), "-R", "{0:e}".format(CNSTS["Re"]),\
             "-W", "{0:e}".format(CNSTS["Wi"]), "-b",\
             "{0:e}".format(CNSTS["beta"]), "-D", "{0:e}".format(CNSTS["De"]),\
             "-P", "{0:e}".format(CNSTS["P"]), \
             "-t", "{0:e}".format(CNSTS["dt"]),\
             "-s", "{0:d}".format(stepsPerFrame), "-T",\
             "{0:d}".format(numTimeSteps), "-i", "{0:e}".format(initTime), "-d"

    cargs = ["./DNS_2D_linear_Visco", "-N", "{0:d}".format(CNSTS["N"]), "-M",
             "{0:d}".format(CNSTS["M"]),"-U", "{0:e}".format(CNSTS["U0"]), "-k",
             "{0:e}".format(CNSTS["kx"]), "-R", "{0:e}".format(CNSTS["Re"]),
             "-W", "{0:e}".format(CNSTS["Wi"]), "-b",
             "{0:e}".format(CNSTS["beta"]), "-D",
             "{0:e}".format(CNSTS["De"]),
             "-P", "{0:e}".format(CNSTS["P"]),
             "-t", "{0:e}".format(CNSTS["dt"]),
             "-s", "{0:d}".format(stepsPerFrame), "-T",
             "{0:d}".format(numTimeSteps), "-i", "{0:e}".format(initTime), "-d"]

else:
    cargs = ["./DNS_2D_linear_Visco", "-N", "{0:d}".format(CNSTS["N"]), "-M",
             "{0:d}".format(CNSTS["M"]),"-U", "{0:e}".format(CNSTS["U0"]), "-k",
             "{0:e}".format(CNSTS["kx"]), "-R", "{0:e}".format(CNSTS["Re"]),
             "-W", "{0:e}".format(CNSTS["Wi"]), "-b",
             "{0:e}".format(CNSTS["beta"]), "-D",
             "{0:e}".format(CNSTS["De"]),
             "-P", "{0:e}".format(CNSTS["P"]),
             "-t", "{0:e}".format(CNSTS["dt"]),
             "-s", "{0:d}".format(stepsPerFrame), "-T",
             "{0:d}".format(numTimeSteps), "-i", "{0:e}".format(initTime)]

    print "./DNS_2D_linear_Visco", "-N", "{0:d}".format(CNSTS["N"]), "-M",\
             "{0:d}".format(CNSTS["M"]),"-U", "{0:e}".format(CNSTS["U0"]), "-k",\
             "{0:e}".format(CNSTS["kx"]), "-R", "{0:e}".format(CNSTS["Re"]),\
             "-W", "{0:e}".format(CNSTS["Wi"]), "-b",\
             "{0:e}".format(CNSTS["beta"]), "-De",\
            "{0:e}".format(CNSTS[""]),\
             "-P", "{0:e}".format(CNSTS["P"]), \
             "-t", "{0:e}".format(CNSTS["dt"]),\
             "-s", "{0:d}".format(stepsPerFrame), "-T",\
             "{0:d}".format(numTimeSteps), "-i", "{0:e}".format(initTime)

subprocess.call(cargs)

# Read in data from the C code

tsm.initTSM(N_=N,M_=M,kx_=kx)
MDY = tsm.mk_diff_y()
MDX = tsm.mk_diff_x()

print """
----------------
Input variables
----------------
"""

psic = load_hdf5_state("./output/psi.h5").reshape(2*N+1, M).T 
print 'initial streamfunction?'
test_arrays_equal(PSI, psic)

cxxc = load_hdf5_state("./output/cxx.h5").reshape(2*N+1, M).T 
print 'Cxx'
test_arrays_equal(Cxx, cxxc)

cxyc = load_hdf5_state("./output/cxy.h5").reshape(2*N+1, M).T 
print 'Cxy'
test_arrays_equal(Cxy, cxyc)

cyyc = load_hdf5_state("./output/cyy.h5").reshape(2*N+1, M).T 
print 'Cyy'
test_arrays_equal(Cyy, cyyc)

# switch back to normal ordering for F modes
PSI = fftshift(PSI, axes=1)
PSI = PSI.T.flatten()
PSIbkp = copy(PSI)

Cxx = fftshift(Cxx, axes=1)
Cxx = Cxx.T.flatten()
Cyy = fftshift(Cyy, axes=1)
Cyy = Cyy.T.flatten()
Cxy = fftshift(Cxy, axes=1)
Cxy = Cxy.T.flatten()

print """
---------------------
derivatives and terms 
---------------------
"""
# u
U = dot(MDY, PSI)
USQ = dot(tsm.prod_mat(U), U)

INTY = mk_cheb_int()
print 'KE0 (U^2 only): ', (15.0/16.0)*dot(INTY, USQ[N*M:(N+1)*M])

# read in and make it 2D to get rid of the junk Cheby modes.
# Then take transpose and flatten to return to 2*N+1 chunks of M length.
U0c = load_hdf5_state("./output/U0.h5") 

print 'U0 ?'
test_arrays_equal(U[N*M:(N+1)*M], U0c)

uc = load_hdf5_state("./output/u.h5") 
print 'u ?'
test_arrays_equal(U[(N+1)*M:(N+2)*M], uc)

# v
V = -dot(MDX, PSI)

vc = load_hdf5_state("./output/v.h5")

print 'V ?'
test_arrays_equal(V[(N+1)*M:(N+2)*M], vc)

lplc = load_hdf5_state("./output/lplpsi.h5")

# THE ORDER OF OPERATIONS HERE MATTERS!
#LPLPSI = dot(dot(MDY,MDY) + dot(MDX,MDX), PSI)
LPLPSI = dot(dot(MDY,MDY), PSI) + dot(dot(MDX,MDX), PSI)

print 'laplacian psi ?'
test_arrays_equal(LPLPSI[(N+1)*M:(N+2)*M], lplc)


d2yc = load_hdf5_state("./output/d2ypsi.h5")

D2YPSI = dot(MDY, dot(MDY,PSI) )
D2YPSIud = dot(MDY[:, ::-1], dot(MDY[:, ::-1],PSI[::-1])[::-1] )

print 'd2y psi ?'
test_arrays_equal(D2YPSI[(N+1)*M:(N+2)*M], d2yc)

d3ypsi0c = load_hdf5_state("./output/d2ypsi0.h5")
print 'd2y PSI0 ?'
test_arrays_equal(D2YPSI[(N)*M:(N+1)*M], d3ypsi0c)

d3yc = load_hdf5_state("./output/d3ypsi.h5")

D3YPSI = dot(MDY, dot(MDY, dot(MDY,PSI) ) )

print 'd3y psi ?'
test_arrays_equal(D3YPSI[(N+1)*M:(N+2)*M], d3yc)

d3ypsi0c = load_hdf5_state("./output/d3ypsi0.h5")
print 'd3yPSI0 ?'
test_arrays_equal(D3YPSI[(N)*M:(N+1)*M], d3ypsi0c)

d4yc = load_hdf5_state("./output/d4ypsi.h5")

D4YPSI = dot(MDY, dot(MDY, dot(MDY, dot(MDY,PSI) ) ) )

print 'd4y psi ?'
test_arrays_equal(D4YPSI[(N+1)*M:(N+2)*M], d4yc)
    
d2xc = load_hdf5_state("./output/d2xpsi.h5")

D2XPSI = dot(MDX, dot(MDX,PSI) )

print 'd2x psi ?'
test_arrays_equal(D2XPSI[(N+1)*M:(N+2)*M], d2xc)

d4xc = load_hdf5_state("./output/d4xpsi.h5")

D4XPSI = dot(MDX, dot(MDX, dot(MDX, dot(MDX,PSI) ) ) )

print 'd4x psi ?'
test_arrays_equal(D4XPSI[(N+1)*M:(N+2)*M], d4xc)

d2xd2yc = load_hdf5_state("./output/d2xd2ypsi.h5")

D2XD2YPSI = dot(MDX, dot(MDX, dot(MDY, dot(MDY,PSI) ) ) )

print 'd2xd2y psi ?'
test_arrays_equal(D2XD2YPSI[(N+1)*M:(N+2)*M], d2xd2yc)
    
biharmc = load_hdf5_state("./output/biharmpsi.h5")

# ORDER OF OPERATIONS MATTERS!!!!!
# BIHARMPSI = dot(dot(MDY,MDY) + dot(MDX,MDX), LPLPSI)
#BIHARMPSI = dot(dot(MDY,MDY), LPLPSI) + dot(dot(MDX,MDX), LPLPSI)
BIHARMPSI = D4XPSI + D4YPSI + 2*D2XD2YPSI 

print 'biharm psi ?'
test_arrays_equal(BIHARMPSI[(N+1)*M:(N+2)*M], biharmc)

dxlplc = load_hdf5_state("./output/dxlplpsi.h5")

DXLPLPSI = dot(MDX, LPLPSI)

print "dxlplpsi ?"
test_arrays_equal(DXLPLPSI[(N+1)*M:(N+2)*M], dxlplc)

udxlplc = load_hdf5_state("./output/udxlplpsi.h5")

UDXLPLPSI = dot(tsm.prod_mat(U), dot(MDX, LPLPSI))

print 'udxlplpsi ?'
test_arrays_equal(UDXLPLPSI[(N+1)*M:(N+2)*M], udxlplc)

dylplc = load_hdf5_state("./output/dylplpsi.h5")

DYLPLPSI = dot(MDY, LPLPSI)
print "dylplPSI0 ?" 
test_arrays_equal(DYLPLPSI[(N)*M:(N+1)*M], dylplc)

vdylplc = load_hdf5_state("./output/vdylplpsi.h5")

VDYLPLPSI = dot(tsm.cheb_prod_mat(V[(N+1)*M:(N+2)*M]), dot(SMDY, LPLPSI[(N)*M:(N+1)*M]))

print 'vdylplPSI0 ?' 
test_arrays_equal(VDYLPLPSI, vdylplc)


dycxy0c = load_hdf5_state("./output/dycxy0.h5")
DYCXY0 = dot(SMDY, Cxy[N*M:(N+1)*M])
print 'DYCXY0 ?' 
test_arrays_equal(DYCXY0, dycxy0c)

d2xcxyc = load_hdf5_state("./output/d2xcxy.h5")
D2XCXY = -kx**2 * Cxy[(N+1)*M:(N+2)*M]
print 'D2XCXY ?' 
test_arrays_equal(D2XCXY, d2xcxyc)

d2ycxyc = load_hdf5_state("./output/d2ycxy.h5")
D2YCXY = dot(SMDY, dot(SMDY, Cxy[(N+1)*M:(N+2)*M]))
print 'D2YCXY ?' 
test_arrays_equal(D2YCXY, d2ycxyc)

dxycyy_cxxc = load_hdf5_state("./output/dxycyy_cxx.h5")
DXYCYY_CXX = 1.j*kx*dot(SMDY, (Cyy[(N+1)*M:(N+2)*M] - Cxx[(N+1)*M:(N+2)*M]))
print 'DXYCYY_CXX ?' 
test_arrays_equal(DXYCYY_CXX, dxycyy_cxxc)

cyy0dyu0c = load_hdf5_state("./output/cyy0dyu0.h5")
CYY0DYU0 = dot(cheb_prod_mat(Cyy[N*M:(N+1)*M]), dot(SMDY, U[N*M:(N+1)*M]))
print 'CYY0DYU0 ?' 
test_arrays_equal(CYY0DYU0, cyy0dyu0c)

cxy0dyu0c = load_hdf5_state("./output/cxy0dyu0.h5")
CXY0DYU0 = dot(cheb_prod_mat(Cxy[N*M:(N+1)*M]), dot(SMDY, U[N*M:(N+1)*M]))
print 'CXY0DYU0 ?' 
test_arrays_equal(CXY0DYU0, cxy0dyu0c)

print """
-----------------------
Updated stresses check
-----------------------
"""

DXYPSI = zeros((2*N+1)*M, dtype='complex')
DXYPSI[(N+1)*M:(N+2)*M] = 1.j*kx*U[(N+1)*M:(N+2)*M]
DXYPSI[(N-1)*M:N*M] = conj(DXYPSI[(N+1)*M:(N+2)*M])

if args.flow_type ==2: # Oscillatory flow 
    CxxN = zeros((2*N+1)*M, dtype='complex')
    CxxN[(N+1)*M:(N+2)*M] = Cxx[(N+1)*M:(N+2)*M] \
            - 0.25*dt*(1./Wi)*Cxx[(N+1)*M:(N+2)*M] \
            + dt*dot(tsm.cheb_prod_mat(Cxy[(N+1)*M:(N+2)*M]), D2YPSI[N*M:(N+1)*M]) \
            + dt*dot(tsm.cheb_prod_mat(Cxy[N*M:(N+1)*M]), D2YPSI[(N+1)*M:(N+2)*M]) \
            + dt*dot(tsm.cheb_prod_mat(Cxx[N*M:(N+1)*M]), DXYPSI[(N+1)*M:(N+2)*M]) \
            - 0.5*dt*dot(dot(cheb_prod_mat(V[(N+1)*M:(N+2)*M]),SMDY), Cxx[N*M:(N+1)*M]) \
            - 0.5*dt*dot(1.j*kx*cheb_prod_mat(U[N*M:(N+1)*M]), Cxx[(N+1)*M:(N+2)*M])

    CxxN[N*M:(N+1)*M] = (1.0 - 0.25*dt*(1./Wi)) * Cxx[N*M:(N+1)*M] \
                         + dt*CXY0DYU0
    CxxN[N*M] += (.5*dt/Wi)

    CxxN = CxxN * (Wi/(Wi+dt*0.25))

    CyyN = zeros((2*N+1)*M, dtype='complex')
    CyyN[(N+1)*M:(N+2)*M] = Cyy[(N+1)*M:(N+2)*M] \
            - 0.25*dt*(1./Wi)*Cyy[(N+1)*M:(N+2)*M] \
            + dt*dot(tsm.cheb_prod_mat(Cxy[N*M:(N+1)*M]), -D2XPSI[(N+1)*M:(N+2)*M]) \
            + dt*dot(tsm.cheb_prod_mat(Cyy[N*M:(N+1)*M]), -DXYPSI[(N+1)*M:(N+2)*M]) \
            - 0.5*dt*dot(dot(cheb_prod_mat(V[(N+1)*M:(N+2)*M]),SMDY), Cyy[N*M:(N+1)*M]) \
            - 0.5*dt*dot(1.j*kx*cheb_prod_mat(U[N*M:(N+1)*M]), Cyy[(N+1)*M:(N+2)*M])

    CyyN[N*M:(N+1)*M] = (1.0 - 0.25*dt*(1./Wi))*Cyy[N*M:(N+1)*M]
    CyyN[N*M] += (.5*dt/Wi)

    CyyN = CyyN * (Wi/(Wi+dt*0.25))

    CxyN = zeros((2*N+1)*M, dtype='complex')
    CxyN[(N+1)*M:(N+2)*M] = Cxy[(N+1)*M:(N+2)*M] \
            - 0.25*dt*(1./Wi)*Cxy[(N+1)*M:(N+2)*M] \
            - 0.5*dt*dot(tsm.cheb_prod_mat(Cxx[N*M:(N+1)*M]), D2XPSI[(N+1)*M:(N+2)*M]) \
            + 0.5*dt*dot(tsm.cheb_prod_mat(Cyy[N*M:(N+1)*M]), D2YPSI[(N+1)*M:(N+2)*M]) \
            + 0.5*dt*dot(tsm.cheb_prod_mat(Cyy[(N+1)*M:(N+2)*M]), D2YPSI[N*M:(N+1)*M]) \
            - 0.5*dt*dot(dot(cheb_prod_mat(V[(N+1)*M:(N+2)*M]), SMDY), Cxy[N*M:(N+1)*M]) \
            - 0.5*dt*dot(1.j*kx*cheb_prod_mat(U[N*M:(N+1)*M]), Cxy[(N+1)*M:(N+2)*M])

    CxyN[N*M:(N+1)*M] = (1.0 - 0.25*dt*(1./Wi))*Cxy[N*M:(N+1)*M] \
                         + 0.5*dt*CYY0DYU0

    CxyN = CxyN * (Wi/(Wi+dt*0.25))

else: # Time independent forcing flow
    CxxN = zeros((2*N+1)*M, dtype='complex')
    CxxN[(N+1)*M:(N+2)*M] = Cxx[(N+1)*M:(N+2)*M] \
            - 0.25*dt*(1./Wi)*Cxx[(N+1)*M:(N+2)*M] \
            + dt*dot(tsm.cheb_prod_mat(Cxy[(N+1)*M:(N+2)*M]), D2YPSI[N*M:(N+1)*M]) \
            + dt*dot(tsm.cheb_prod_mat(Cxy[N*M:(N+1)*M]), D2YPSI[(N+1)*M:(N+2)*M]) \
            + dt*dot(tsm.cheb_prod_mat(Cxx[N*M:(N+1)*M]), DXYPSI[(N+1)*M:(N+2)*M]) \
            - 0.5*dt*dot(dot(cheb_prod_mat(V[(N+1)*M:(N+2)*M]),SMDY), Cxx[N*M:(N+1)*M]) \
            - 0.5*dt*dot(1.j*kx*cheb_prod_mat(U[N*M:(N+1)*M]), Cxx[(N+1)*M:(N+2)*M])

    CxxN[N*M:(N+1)*M] = (1.0 - 0.25*dt*(1./Wi)) * Cxx[N*M:(N+1)*M] \
                         + dt*CXY0DYU0
    CxxN[N*M] += (.5*dt/Wi)

    CxxN = CxxN * (Wi/(Wi+dt*0.25))

    CyyN = zeros((2*N+1)*M, dtype='complex')
    CyyN[(N+1)*M:(N+2)*M] = Cyy[(N+1)*M:(N+2)*M] \
            - 0.25*dt*(1./Wi)*Cyy[(N+1)*M:(N+2)*M] \
            + dt*dot(tsm.cheb_prod_mat(Cxy[N*M:(N+1)*M]), -D2XPSI[(N+1)*M:(N+2)*M]) \
            + dt*dot(tsm.cheb_prod_mat(Cyy[N*M:(N+1)*M]), -DXYPSI[(N+1)*M:(N+2)*M]) \
            - 0.5*dt*dot(dot(cheb_prod_mat(V[(N+1)*M:(N+2)*M]),SMDY), Cyy[N*M:(N+1)*M]) \
            - 0.5*dt*dot(1.j*kx*cheb_prod_mat(U[N*M:(N+1)*M]), Cyy[(N+1)*M:(N+2)*M])

    CyyN[N*M:(N+1)*M] = (1.0 - 0.25*dt*(1./Wi))*Cyy[N*M:(N+1)*M]
    CyyN[N*M] += (.5*dt/Wi)

    CyyN = CyyN * (Wi/(Wi+dt*0.25))

    CxyN = zeros((2*N+1)*M, dtype='complex')
    CxyN[(N+1)*M:(N+2)*M] = Cxy[(N+1)*M:(N+2)*M] \
            - 0.25*dt*(1./Wi)*Cxy[(N+1)*M:(N+2)*M] \
            - 0.5*dt*dot(tsm.cheb_prod_mat(Cxx[N*M:(N+1)*M]), D2XPSI[(N+1)*M:(N+2)*M]) \
            + 0.5*dt*dot(tsm.cheb_prod_mat(Cyy[N*M:(N+1)*M]), D2YPSI[(N+1)*M:(N+2)*M]) \
            + 0.5*dt*dot(tsm.cheb_prod_mat(Cyy[(N+1)*M:(N+2)*M]), D2YPSI[N*M:(N+1)*M]) \
            - 0.5*dt*dot(dot(cheb_prod_mat(V[(N+1)*M:(N+2)*M]), SMDY), Cxy[N*M:(N+1)*M]) \
            - 0.5*dt*dot(1.j*kx*cheb_prod_mat(U[N*M:(N+1)*M]), Cxy[(N+1)*M:(N+2)*M])

    CxyN[N*M:(N+1)*M] = (1.0 - 0.25*dt*(1./Wi))*Cxy[N*M:(N+1)*M] \
                         + 0.5*dt*CYY0DYU0

    CxyN = CxyN * (Wi/(Wi+dt*0.25))

cxxNc = load_hdf5_state("./output/cxxN.h5").reshape(2*N+1, M).T 
cxyNc = load_hdf5_state("./output/cxyN.h5").reshape(2*N+1, M).T 
cyyNc = load_hdf5_state("./output/cyyN.h5").reshape(2*N+1, M).T 
print '0:'
print 'CxxN'
test_arrays_equal(CxxN[N*M:(N+1)*M], cxxNc[:,0])
print 'CyyN'
test_arrays_equal(CyyN[N*M:(N+1)*M], cyyNc[:,0])
print 'CxyN'
test_arrays_equal(CxyN[N*M:(N+1)*M], cxyNc[:,0])

print '1:'
print 'CxxN'
test_arrays_equal(CxxN[(N+1)*M:(N+2)*M], cxxNc[:,1])
print 'CyyN'
test_arrays_equal(CyyN[(N+1)*M:(N+2)*M], cyyNc[:,1])
print 'CxyN'
test_arrays_equal(CxyN[(N+1)*M:(N+2)*M], cxyNc[:,1])


dycxy0Nc = load_hdf5_state("./output/dycxy0N.h5")
DYCXY0N = dot(SMDY, CxyN[N*M:(N+1)*M])
print 'DYCXY0N ?' 
test_arrays_equal(DYCXY0N, dycxy0Nc)

d2xcxyNc = load_hdf5_state("./output/d2xcxyN.h5")
D2XCXYN = -kx**2 * CxyN[(N+1)*M:(N+2)*M]
print 'D2XCXYN ?' 
test_arrays_equal(D2XCXYN, d2xcxyNc)

d2ycxyNc = load_hdf5_state("./output/d2ycxyN.h5")
D2YCXYN = dot(SMDY, dot(SMDY, CxyN[(N+1)*M:(N+2)*M]))
print 'D2YCXYN ?' 
test_arrays_equal(D2YCXYN, d2ycxyNc)

dxycyy_cxxNc = load_hdf5_state("./output/dxycyy_cxxN.h5")
DXYCYY_CXXN = 1.j*kx*dot(SMDY, (CyyN[(N+1)*M:(N+2)*M] - CxxN[(N+1)*M:(N+2)*M]))
print 'DXYCYY_CXXN ?' 
test_arrays_equal(DXYCYY_CXXN, dxycyy_cxxNc)

print """
--------------
Operator check
--------------
"""
op0c = load_hdf5_state("./output/op0.h5")#.reshape(M, M).T 

print 'operator 0'
test_arrays_equal(op0c, PsiOpInvList[0].flatten())

for i in range(1,N+1):
    opc = load_hdf5_state("./output/op{0}.h5".format(i))
    print 'operator ',i
    test_arrays_equal(opc, PsiOpInvList[i].flatten())

hop0c = load_hdf5_state("./output/hOp0.h5")#.reshape(M, M).T 
print 'half operator 0'
test_arrays_equal(hop0c, PsiOpInvListHalf[0].flatten())

for i in range(1,N+1):
    hopc = load_hdf5_state("./output/hOp{0}.h5".format(i))
    print 'half operator ',i
    test_arrays_equal(hopc, PsiOpInvListHalf[i].flatten())


DYYYPSI = dot(MDY, dot(MDY, dot(MDY, PSI)))

RHSVec = zeros((2*N+1)*M, dtype='complex')
if args.flow_type == 2:
    RHSVec[(N+1)*M:(N+2)*M] = dt*0.25*oneOverRe*beta*BIHARMPSI[(N+1)*M:(N+2)*M] \
                            + LPLPSI[(N+1)*M:(N+2)*M] \
                            - dt*0.5*UDXLPLPSI[(N+1)*M:(N+2)*M] \
                            - dt*0.5*VDYLPLPSI \
                            - dt*0.25*(1.0-beta)*oneOverRe*(1./Wi)*D2XCXY \
                            - dt*0.25*(1.0-beta)*oneOverRe*(1./Wi)*DXYCYY_CXX \
                            + dt*0.25*(1.0-beta)*oneOverRe*(1./Wi)*D2YCXY \
                            - dt*0.25*(1.0-beta)*oneOverRe*(1./Wi)*D2XCXYN \
                            - dt*0.25*(1.0-beta)*oneOverRe*(1./Wi)*DXYCYY_CXXN \
                            + dt*0.25*(1.0-beta)*oneOverRe*(1./Wi)*D2YCXYN

    # Zeroth mode (dt/2 because that is how it appears in the method)

    RHSVec[N*M:(N+1)*M] = 0
    RHSVec[N*M:(N+1)*M] = dt*0.25*beta*oneOverRe*D3YPSI[N*M:(N+1)*M] \
                          + U[N*M:(N+1)*M] \
                          + dt*0.25*(1.-beta)*oneOverRe*(1./Wi)*DYCXY0 \
                          + dt*0.25*(1.-beta)*oneOverRe*(1./Wi)*DYCXY0N

    RHSVec[N*M] += dt*oneOverRe


else:
    RHSVec[(N+1)*M:(N+2)*M] = dt*0.25*oneOverRe*beta*BIHARMPSI[(N+1)*M:(N+2)*M] \
                            + LPLPSI[(N+1)*M:(N+2)*M] \
                            - dt*0.5*UDXLPLPSI[(N+1)*M:(N+2)*M] \
                            - dt*0.5*VDYLPLPSI \
                            - dt*0.25*(1.0-beta)*oneOverRe*(1./Wi)*D2XCXY \
                            - dt*0.25*(1.0-beta)*oneOverRe*(1./Wi)*DXYCYY_CXX \
                            + dt*0.25*(1.0-beta)*oneOverRe*(1./Wi)*D2YCXY \
                            - dt*0.25*(1.0-beta)*oneOverRe*(1./Wi)*D2XCXYN \
                            - dt*0.25*(1.0-beta)*oneOverRe*(1./Wi)*DXYCYY_CXXN \
                            + dt*0.25*(1.0-beta)*oneOverRe*(1./Wi)*D2YCXYN

    # Zeroth mode (dt/2 because that is how it appears in the method)

    RHSVec[N*M:(N+1)*M] = 0
    RHSVec[N*M:(N+1)*M] = dt*0.25*beta*oneOverRe*D3YPSI[N*M:(N+1)*M] \
                          + U[N*M:(N+1)*M] \
                          + dt*0.25*(1.-beta)*oneOverRe*(1./Wi)*DYCXY0 \
                          + dt*0.25*(1.-beta)*oneOverRe*(1./Wi)*DYCXY0N

    RHSVec[N*M] += dt*oneOverRe

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
    print "RHSvec for mode ", i
    #print RHSvecc - RHSVec[(N+i)*M:(N+1+i)*M]

    test_arrays_equal(RHSVec[(N+i)*M:(N+1+i)*M], RHSvecc)


# calculate the updated psi
for i in range(M):
    PSI[(N)*M + i] = 0
    #for j in range(M-1,-1,-1):
    for j in range(M):
        PSI[(N)*M + i] += PsiOpInvList[0][i,j] * RHSVec[(N)*M + j]

for n in range(1,N+1):
    for i in range(M):
        PSI[(N+n)*M + i] = 0
        for j in range(M):
            PSI[(N+n)*M + i] += PsiOpInvList[n][i,j] * RHSVec[(N+n)*M + j]
    del n  

for n in range(0,N):
    PSI[n*M:(n+1)*M] = conj(PSI[(2*N-n)*M:(2*N+1-n)*M])

print """
==============================================================================

ALL TESTS PASSED! 

==============================================================================
"""
