#-----------------------------------------------------------------------------
#   2D spectral direct numerical simulator
#
#   Last modified: Thu  7 May 14:31:07 2015
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
from scipy import optimize
from numpy.linalg import cond 
from numpy.fft import fftshift, ifftshift
from numpy.random import rand

import cPickle as pickle

import ConfigParser
import subprocess
import h5py

import fields_2D as f2d

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

Omega = config.getfloat('Oscillatory Flow', 'Omega')

dt = config.getfloat('Time Iteration', 'dt')
totTime = config.getfloat('Time Iteration', 'totTime')
numFrames = config.getint('Time Iteration', 'numFrames')
dealiasing = config.getboolean('Time Iteration', 'Dealiasing')
shear_layer_flag = config.getboolean('Time Iteration', 'Shear Layer')

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
          'kx': kx, 'Omega':Omega, 'time': totTime, 'dt':dt, 'dealiasing':dealiasing}
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

    print "condition numbers for dt = {0} operators".format(dt)
    print "mode 0, condition number {0:e}".format(cond(Psi0thOp))

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

        # Calculate condition number before taking the inverse
        conditionNum = cond(PSIOP)
        print "mode {0}, condition number {1:e}".format(n, conditionNum)

        # store the inverse of the relevent part of the matrix
        PSIOP = linalg.inv(PSIOP)
        PSIOP = PSIOP[0:M, 0:M]

        PsiOpInvList.append(PSIOP)

    del PSIOP

    PsiOpInvList = array(PsiOpInvList)
    return PsiOpInvList

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

def perturb(psi_, totEnergy, perKEestimate, sigma, gam):
    """
    calculate the KE for a perturbation of amplitude 1 and then choose a
    perturbation amplitude which gives the desired perturbation KE.
    Then use this perturbation KE to calculate a reduction in the base profile
    such that there is the correct total energy
    """

    SMDY =  mk_single_diffy()

    pscale = optimize.fsolve(lambda pscale: pscale*tan(pscale) + gam*tanh(gam), 2)

    perAmp = 1.0
    
    rn = zeros((N,5))
    for n in range(N):
        rn[n,:] = (10.0**(-n))*(0.5-rand(5))

    for j in range(2):

        for n in range(1,N+1):
            if (n % 2) == 0:
                ##------------- PURE RANDOM PERTURBATIONS -------------------
                ## Make sure you satisfy the optimum symmetry for the
                ## perturbation
                #psi_[(N-n)*M + 1:(N-n)*M + M/2 :2] = (10.0**(-n+1))*perAmp*0.5*(1-rand(M/4) + 1.j*rand(M/4))
                #psi_[(N-n)*M + 1:(N-n)*M + 6 :2] =\
                #(10.0**((n+1)))*perAmp*0.5*(1-rand(3) + 1.j*rand(3))
                #psi_[(N-n)*M + 1:(N-n)*M + M/2 :2] = perAmp*0.5*(rand(M/4) + 1.j*rand(M/4))

                ##------------- PERTURBATIONS WHICH SATISFY BCS -------------------
                rSpace = zeros(M, dtype='complex')
                y = 2.0*arange(M)/(M-1.0) -1.0
                ## exponentially decaying sinusoid
                #rSpace = cos(1.0 * 2.0*pi * y) * exp(-(sigma*pi*y)**2)#  * rn[0]
                #rSpace += cos(2.0 * 2.0*pi * y) * exp(-(sigma*pi*y)**2) * rn[1]
                #rSpace += cos(3.0 * 2.0*pi * y) * exp(-(sigma*pi*y)**2) * rn[2]
                #rSpace += cos(4.0 * 2.0*pi * y) * exp(-(sigma*pi*y)**2) * rn[3]
                #rSpace += cos(5.0 * 2.0*pi * y) * exp(-(sigma*pi*y)**2) * rn[4]

                ## sinusoidal
                rSpace = perAmp*cos(1.0 * 2.0*pi * y) * rn[n-1,0]
                rSpace += perAmp*cos(2.0 * 2.0*pi * y) * rn[n-1,1]
                rSpace += perAmp*cos(3.0 * 2.0*pi * y) * rn[n-1,2]

                ## low order eigenfunction of biharmonic operator
                #rSpace = (cos(pscale*y)/cos(pscale) - cosh(gam*y)/(cosh(gam))) * rn[0]

                #savetxt('p{0}.dat'.format(n), vstack((y,real(rSpace))).T)

                psi_[(N+n)*M:(N+n+1)*M] = stupid_transform(rSpace, CNSTS)*1.j


            else:
                ##------------- PURE RANDOM PERTURBATIONS -------------------
                ## Make sure you satisfy the optimum symmetry for the
                ## perturbation
                #psi_[(N-n)*M:(N-n)*M + M/2 - 1 :2] = (10.0**(-n+1))*perAmp*0.5*(1-rand(M/4) + 1.j*rand(M/4))
                #psi_[(N-n)*M:(N-n)*M + 6 - 1 :2] =\
                #(10.0**((n+1)))*perAmp*0.5*(1-rand(3) + 1.j*rand(3))
                #psi_[(N-n)*M:(N-n)*M + M/2 - 1 :2] = perAmp*0.5*(rand(M/4) + 1.j*rand(M/4))

                ##------------- PERTURBATIONS WHICH SATISFY BCS -------------------
                rSpace = zeros(M, dtype='complex')
                y = 2.0*arange(M)/(M-1.0) -1.0
                ## exponentially decaying sinusoid
                #rSpace = sin(1.0 * 2.0*pi * y) * exp(-(sigma*pi*y)**2)#  * rn[0]
                #rSpace += sin(2.0 * 2.0*pi * y) * exp(-(sigma*pi*y)**2) * rn[1]
                #rSpace += sin(3.0 * 2.0*pi * y) * exp(-(sigma*pi*y)**2) * rn[2]
                #rSpace += sin(4.0 * 2.0*pi * y) * exp(-(sigma*pi*y)**2) * rn[3]
                #rSpace += sin(5.0 * 2.0*pi * y) * exp(-(sigma*pi*y)**2) * rn[4]

                ## sinusoidal
                rSpace =  perAmp*sin(1.0 * 2.0*pi * y) * rn[n-1,0]
                rSpace += perAmp*sin(2.0 * 2.0*pi * y) * rn[n-1,1]
                rSpace += perAmp*sin(3.0 * 2.0*pi * y) * rn[n-1,2]

                ## low order eigenfunction of biharmonic operator
                #rSpace = (sin(pscale * y)/(pscale*cos(pscale)) - sinh(gam*y)/(gam*cosh(gam))) * rn[0]

                #savetxt('p{0}.dat'.format(n), vstack((y,real(rSpace))).T)

                psi_[(N+n)*M:(N+n+1)*M] =stupid_transform(rSpace, CNSTS)

            psi_[(N-n)*M:(N-n+1)*M] = conj(psi_[(N+n)*M:(N+n+1)*M])
            del y


        KERest = 0
        KERest2 = 0
        for i in range(1,N+1):
            u = dot(SMDY, psi_[(N+i)*M: (N+i+1)*M])
            KE = 0
            for n in range(0,M,2):
                usq = 0
                for m in range(n-M+1, M):

                    p = abs(n-m)
                    if (p==0):
                        tmp = 2.0*u[p]
                    else:
                        tmp = u[p]

                    if (abs(m)==0):
                        tmp *= 2.0*conj(u[abs(m)])
                    else:
                        tmp *= conj(u[abs(m)])

                    if (n==0):
                        usq += 0.25*tmp
                    else:
                        usq += 0.5*tmp

                KE += (2. / (1.-n*n)) * usq;

            KERest += (15.0/8.0) * KE 
            u = dot(cheb_prod_mat(u), conj(u))
            KERest2 += (15.0/8.0) * dot(INTY, u) 

        # Want KERest = 0.3
        # perAmp^2 ~ 0.3/KERest
        if j==0:
            perAmp = real(sqrt(perKEestimate/KERest))
            print 'perAmp = ', perAmp


    print 'Initial Energy of the perturbation, ', KERest

    # KE_tot = KE_0 + KE_Rest
    # KE_0 = KE_tot - KE_Rest
    # scale_fac^2 = 0.5*(KE_tot-KERest)
    # scale_fac^2 = 0.5*(1/2-KERest) 

    energy_rescale = sqrt((totEnergy - real(KERest)))
    psi_[N*M:(N+1)*M] = energy_rescale*psi_[N*M:(N+1)*M]
    u = dot(SMDY, psi_[N*M: (N+1)*M])
    u = dot(cheb_prod_mat(u), u)
    KE0 = 0.5*(15./8.)*dot(INTY, u)
    print 'Rescaled zeroth KE = ', KE0
    print 'total KE = ', KE0 + KERest2

    return psi_

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

#### The initial stream-function
PSI = zeros((2*N+1)*M,dtype='complex')

# --------------- TWS -----------------

# Read in stream function from file
#(PSI, Nu) = pickle.load(open(inFileName,'r'))


# --------------- POISEUILLE -----------------


#plugAmp = 0.02 #* (M/32.0)

#PSI[N*M]   += (1.-plugAmp) * 2.0/3.0
#PSI[N*M+1] += (1.-plugAmp) * 3.0/4.0
#PSI[N*M+2] += (1.-plugAmp) * 0.0
#PSI[N*M+3] += (1.-plugAmp) * -1.0/12.0
#
#
## --------------- PLUG  -----------------
#
#PSI[N*M]   += (plugAmp) * (5.0/8.0) * 4.0/5.0
#PSI[N*M+1] += (plugAmp) * (5.0/8.0) * 7.0/8.0
#PSI[N*M+3] += (plugAmp) * (5.0/8.0) * -1.0/16.0
#PSI[N*M+5] += (plugAmp) * (5.0/8.0) * -1.0/80.0
#
##PSI[N*M:] = 0
##PSI[:(N+1)*M] = 0
#
#perKEestimate = 0.2
#totEnergy = 0.8
#sigma = 0.1
#gam = 2
#
#PSI = perturb(PSI, totEnergy, perKEestimate, sigma, gam)
#
#forcing = zeros((M,2*N+1), dtype='complex')
#forcing[0,0] = 2.0/Re
#

# --------------- SHEAR LAYER -----------------

#if not shear_layer_flag:
#    y_points = cos(pi*arange(Mf)/(Mf-1))
#    delta = 0.1
#else:
#    slscale = 10.0
#    delta = 0.1
#    y_points = slscale * arctanh(cos(pi*arange(Mf)/(Mf-1)))
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
#
## set BC
#CNSTS['U0'] = 1.0

# --------------- OSCILLATORY FLOW -----------------
#

# PSI

forcing = zeros((M,2*N+1), dtype='complex')
forcing[0,0] = Omega / Re

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
    opFn = "./operators/op{0}.h5".format(n)
    #print "writing ", opFn
    f = h5py.File(opFn, "w")
    dset = f.create_dataset("op", (M*M,), dtype='complex')
    dset[...] = PsiOpInvList[i].flatten()
    f.close()

    #savetxt("./operators/op{0}.dat".format(abs(n)),PsiOpInvList[n])
del i

for i in range(N+1):
    # operator order in list is 0->N
    n = i
    opFn = "./operators/hOp{0}.h5".format(n)
    #print "writing ", opFn
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

#print "writing initial state to initial.h5"

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
             "{0:d}".format(CNSTS["M"]),"-U", "{0:e}".format(CNSTS["U0"]), "-k",
             "{0:e}".format(CNSTS["kx"]), "-R", "{0:e}".format(CNSTS["Re"]),
             "-W", "{0:e}".format(CNSTS["Wi"]), "-b",
             "{0:e}".format(CNSTS["beta"]), "-w",
             "{0:e}".format(CNSTS["Omega"]),
             "-t", "{0:e}".format(CNSTS["dt"]),
             "-s", "{0:d}".format(stepsPerFrame), "-T",
             "{0:d}".format(numTimeSteps), "-d"]

    print "./DNS_2D_Newt", "-N", "{0:d}".format(CNSTS["N"]), "-M",\
             "{0:d}".format(CNSTS["M"]),"-U", "{0:e}".format(CNSTS["U0"]), "-k",\
             "{0:e}".format(CNSTS["kx"]), "-R", "{0:e}".format(CNSTS["Re"]),\
             "-W", "{0:e}".format(CNSTS["Wi"]), "-b",\
             "{0:e}".format(CNSTS["beta"]), "-w", "{0:e}".format(CNSTS["Omega"]),\
             "-t", "{0:e}".format(CNSTS["dt"]),\
             "-s", "{0:d}".format(stepsPerFrame), "-T",\
             "{0:d}".format(numTimeSteps), "-d"

    if shear_layer_flag:
        cargs = ["./DNS_2D_Newt", "-N", "{0:d}".format(CNSTS["N"]), "-M",
                 "{0:d}".format(CNSTS["M"]),"-U", "{0:e}".format(CNSTS["U0"]), "-k",
                 "{0:e}".format(CNSTS["kx"]), "-R", "{0:e}".format(CNSTS["Re"]),
                 "-W", "{0:e}".format(CNSTS["Wi"]), "-b",
                 "{0:e}".format(CNSTS["beta"]), "-w",
                 "{0:e}".format(CNSTS["Omega"]),
                 "-t", "{0:e}".format(CNSTS["dt"]),
                 "-s", "{0:d}".format(stepsPerFrame), "-T",
                 "{0:d}".format(numTimeSteps), "-d", "-n"]

        print "./DNS_2D_Newt", "-N", "{0:d}".format(CNSTS["N"]), "-M",\
                 "{0:d}".format(CNSTS["M"]),"-U", "{0:e}".format(CNSTS["U0"]), "-k",\
                 "{0:e}".format(CNSTS["kx"]), "-R", "{0:e}".format(CNSTS["Re"]),\
                 "-W", "{0:e}".format(CNSTS["Wi"]), "-b",\
                 "{0:e}".format(CNSTS["beta"]), "-w", "{0:e}".format(CNSTS["Omega"]),\
                 "-t", "{0:e}".format(CNSTS["dt"]),\
                 "-s", "{0:d}".format(stepsPerFrame), "-T",\
                 "{0:d}".format(numTimeSteps), "-d", "-n"


else:
    cargs = ["./DNS_2D_Newt", "-N", "{0:d}".format(CNSTS["N"]), "-M",
             "{0:d}".format(CNSTS["M"]),"-U", "{0:e}".format(CNSTS["U0"]), "-k",
             "{0:e}".format(CNSTS["kx"]), "-R", "{0:e}".format(CNSTS["Re"]),
             "-W", "{0:e}".format(CNSTS["Wi"]), "-b",
             "{0:e}".format(CNSTS["beta"]), "-w",
             "{0:e}".format(CNSTS["Omega"]),
             "-t", "{0:e}".format(CNSTS["dt"]),
             "-s", "{0:d}".format(stepsPerFrame), "-T",
             "{0:d}".format(numTimeSteps)]

    print "./DNS_2D_Newt", "-N", "{0:d}".format(CNSTS["N"]), "-M",\
             "{0:d}".format(CNSTS["M"]),"-U", "{0:e}".format(CNSTS["U0"]), "-k",\
             "{0:e}".format(CNSTS["kx"]), "-R", "{0:e}".format(CNSTS["Re"]),\
             "-W", "{0:e}".format(CNSTS["Wi"]), "-b",\
             "{0:e}".format(CNSTS["beta"]), "-w",\
    "{0:e}".format(CNSTS["Omega"]),\
             "-t", "{0:e}".format(CNSTS["dt"]),\
             "-s", "{0:d}".format(stepsPerFrame), "-T",\
             "{0:d}".format(numTimeSteps)

    if shear_layer_flag:
        print "turn on dealiasing first!"
        exit(1)

subprocess.call(cargs)

# Read in data from the C code
print 'done'

