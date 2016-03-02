#-----------------------------------------------------------------------------
#   2D spectral linear time stepping code
#
#   Last modified: Tue 23 Feb 12:11:48 2016
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
from scipy import optimize
from scipy.special import iv
from numpy.linalg import cond 
from numpy.fft import fftshift, ifftshift
from numpy.random import rand

import cPickle as pickle

import ConfigParser
import argparse
import subprocess
import h5py

import fields_2D as f2d

# SETTINGS---------------------------------------------------------------------

config = ConfigParser.RawConfigParser()
fp = open('config.cfg')
config.readfp(fp)
N = 1
M = config.getint('General', 'M')
Re = config.getfloat('General', 'Re')
Wi = config.getfloat('General', 'Wi')
beta = config.getfloat('General', 'beta')
kx = config.getfloat('General', 'kx')

delta = config.getfloat('Shear Layer', 'delta')

De = config.getfloat('Oscillatory Flow', 'De')

dt = config.getfloat('Time Iteration', 'dt')
totTime = config.getfloat('Time Iteration', 'totTime')
numFrames = config.getint('Time Iteration', 'numFrames')
dealiasing = config.getboolean('Time Iteration', 'Dealiasing')

fp.close()

argparser = argparse.ArgumentParser()

argparser.add_argument("-N", type=int, default=N, 
                help='Override Number of Fourier modes given in the config file')
argparser.add_argument("-M", type=int, default=M, 
                help='Override Number of Chebyshev modes in the config file')
argparser.add_argument("-Re", type=float, default=Re, 
                help="Override Reynold's number in the config file") 
argparser.add_argument("-b", type=float, default=beta, 
                help='Override beta of the config file')
argparser.add_argument("-De", type=float, default=De, 
                help='Override Deborah number of the config file')
argparser.add_argument("-Wi", type=float, default=Wi, 
                help='Override Weissenberg number of the config file')
argparser.add_argument("-kx", type=float, default=kx, 
                help='Override wavenumber of the config file')
argparser.add_argument("-initTime", type=float, default=0.0, 
                help='Start simulation from a different time')
tmp = """simulation type, 
            0: Poiseuille
            1: Shear Layer
            2: Oscillatory"""
argparser.add_argument("-flow_type", type=int, default=3, 
                       help=tmp)

args = argparser.parse_args()
N = args.N 
M = args.M
Re = args.Re
beta = args.b
Wi = args.Wi
De = args.De
kx = args.kx
initTime = args.initTime

print 'CHANGING THE REYNOLDS NUMBER!!!!'
Re = Wi / 1182.44
print Re

Re = float("{0:.6e}".format(Re))

if dealiasing:
    #Nf = (3*N)/2 + 1
    Nf = 4*N
    Mf = 2*M
else:
    Nf = N
    Mf = M

numTimeSteps = int(ceil(totTime / dt))
assert (totTime / dt) - float(numTimeSteps) == 0, "Non-integer number of timesteps"
assert Wi != 0.0, "cannot have Wi = 0!"
assert args.flow_type < 3, "flow type unspecified!" 

NOld = 3 
MOld = 40
CNSTS = {'NOld': NOld, 'MOld': MOld, 'N': N, 'M': M, 'Nf':Nf, 'Mf':Mf,'U0':0,
          'Re': Re, 'Wi': Wi, 'beta': beta, 'De':De, 'kx': kx,'time': totTime,
         'dt':dt, 'P': 1.0,
          'dealiasing':dealiasing}


#outPath1 = "./Re{Re}_b{beta}_Wi{Wi}/".format(**CNSTS)
#outPath2 = "kx_{kx}_M{M}_dt{dt}".format(**CNSTS)
#subprocess.call(["mkdir", outPath1])
#subprocess.call(["mkdir", outPath1+outPath2])

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

        #PSIOP = zeros((2*M, 2*M), dtype='complex')
        #SLAPLAC = -n*n*kx*kx*SII + SMDYY

        #PSIOP[0:M, 0:M] = 0
        #PSIOP[0:M, M:2*M] = B*SII - 0.5*beta*dt*SLAPLAC

        #PSIOP[M:2*M, 0:M] = SLAPLAC
        #PSIOP[M:2*M, M:2*M] = -SII

        ## Apply BCs
        ## dypsi(+-1) = 0
        #PSIOP[M-2, :] = concatenate((DERIVTOP, zeros(M, dtype='complex')))
        #PSIOP[M-1, :] = concatenate((DERIVBOT, zeros(M, dtype='complex')))
        #
        ## dxpsi(+-1) = 0
        #PSIOP[2*M-2, :] = concatenate((BTOP, zeros(M, dtype='complex')))
        #PSIOP[2*M-1, :] = concatenate((BBOT, zeros(M, dtype='complex')))

        ## store the inverse of the relevent part of the matrix
        #PSIOP = linalg.inv(PSIOP)
        #PSIOP = PSIOP[0:M, 0:M]

        PSIOP = zeros((2*M, 2*M), dtype='complex')
        SLAPLAC = -n*n*kx*kx*SII + SMDYY

        PSIOP[0:M, 0:M] = 0

        PSIOP[0:M, 0:M] = SLAPLAC
        PSIOP[0:M, M:2*M] = -SII
        PSIOP[M:2*M, M:2*M] = - 0.5*beta*dt*SLAPLAC + B*SII 


        # Apply BCs
        # dypsi(+-1) = 0
        PSIOP[M-2, :] = concatenate((DERIVTOP, zeros(M, dtype='complex')))
        PSIOP[M-1, :] = concatenate((DERIVBOT, zeros(M, dtype='complex')))
        
        # dxpsi(+-1) = 0
        PSIOP[2*M-2, :] = concatenate((BTOP, zeros(M, dtype='complex')))
        PSIOP[2*M-1, :] = concatenate((BBOT, zeros(M, dtype='complex')))

        # store the inverse of the relevent part of the matrix
        PSIOP = linalg.inv(PSIOP)
        PSIOP = PSIOP[0:M, M:2*M]

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

def poiseuille_flow():
    PSI = zeros((2*N+1)*M, dtype='complex')

    PSI[N*M]   += 2.0/3.0
    PSI[N*M+1] += 3.0/4.0
    PSI[N*M+2] += 0.0
    PSI[N*M+3] += -1.0/12.0
    Cxx, Cyy, Cxy = x_independent_profile(PSI)
    forcing = zeros((M,2*N+1), dtype='complex')
    forcing[0,0] = 2./Re

    return PSI, Cxx, Cyy, Cxy, forcing

def plug_like_flow():
    PSI = zeros((2*N+1)*M, dtype='complex')

    PSI[N*M]   += (5.0/8.0) * 4.0/5.0
    PSI[N*M+1] += (5.0/8.0) * 7.0/8.0
    PSI[N*M+3] += (5.0/8.0) * -1.0/16.0
    PSI[N*M+5] += (5.0/8.0) * -1.0/80.0

    PSI[N*M:] = 0
    PSI[:(N+1)*M] = 0
    Cxx, Cyy, Cxy = x_independent_profile(PSI)

    forcing = zeros((M,2*N+1), dtype='complex')
    forcing[0,0] = 2./Re

    return PSI, Cxx, Cyy, Cxy, forcing

def shear_layer_flow(delta=0.1):
    y_points = cos(pi*arange(Mf)/(Mf-1))

    # Set initial streamfunction
    PSI = zeros((Mf, 2*Nf+1), dtype='d')

    for i in range(Mf):
        y =y_points[i]
        for j in range(2*Nf+1):
            PSI[i,j] = delta * (1./tanh(1./delta)) * log(cosh(y/delta))

    del y, i, j

    PSI = f2d.to_spectral(PSI, CNSTS)
    PSI = fftshift(PSI, axes=1)
    PSI = PSI.T.flatten()

    # set forcing
    forcing = zeros((Mf, 2*Nf+1), dtype='d')
    
    for i in range(Mf):
        y =y_points[i]
        for j in range(2*Nf+1):
            forcing[i,j] = ( 2.0/tanh(1.0/delta)) * (1.0/cosh(y/delta)**2) * tanh(y/delta)
            forcing[i,j] *= 1.0/(Re * delta**2) 
    
    del y, i, j
    forcing = f2d.to_spectral(forcing, CNSTS)

    Cxx, Cyy, Cxy = x_independent_profile(PSI)

    return PSI, Cxx, Cyy, Cxy, forcing

def poiseuille_oscil_flow():
    """
    Some flow variables must be calculated in realspace and then transformed
    spectral space, Cyy =1.0 so it is easy.
    """
    PSI = zeros(M*(2*N+1), dtype='complex')
    Cxx = zeros(M*(2*N+1), dtype='complex')
    Cxy = zeros(M*(2*N+1), dtype='complex')

    PSI[N*M]   += 2.0/3.0
    PSI[N*M+1] += 3.0/4.0
    PSI[N*M+2] += 0.0
    PSI[N*M+3] += -1.0/12.0
    
    Cxy[N*M + 1] = Wi * -2.0 
    Cxx[N*M] = Wi**2 * 2.0  + 1.0
    Cxx[N*M+2] = Wi**2 * 2.0 

    Cyy = zeros((2*N+1)*M, dtype='complex')
    Cyy[N*M] = 1

    forcing = zeros((M,2*N+1), dtype='complex')
    forcing[0,0] = 2./Re
    P = 1.

    return PSI, Cxx, Cyy, Cxy, forcing, P

def oscillatory_flow():
    """
    Some flow variables must be calculated in realspace and then transformed
    spectral space, Cyy =1.0 so it is easy.
    """

    y_points = cos(pi*arange(Mf)/(Mf-1))

    tmp = beta + (1-beta) / (1 + 1.j*De)
    print 'tmp', tmp
    alpha = sqrt( (1.j*pi*Re*De) / (2*Wi*tmp) )
    print 'alpha', alpha
    Chi = real( (1-1.j)*(1 - tanh(alpha) / alpha) )
    print 'Chi', Chi 

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

            cxx_cmplx = (1.0/(1.0+2.j*De))*(Wi/pi)*(cxy_cmplx*dyu_cmplx)
            cxx_cmplx += (1.0/(1.0-2.j*De))*(Wi/pi)*(conj(cxy_cmplx)*conj(dyu_cmplx)) 

            cxx_cmplx += 1. + (Wi/pi)*( cxy_cmplx*conj(dyu_cmplx) +
                                       conj(cxy_cmplx)*dyu_cmplx ) 
            Cxx[i,j] = real(cxx_cmplx)

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

    # VERRRYYY IMPORTANT! accidentally introducing tiny imaginary part to linear
    # code
    PSI[N*M:(N+1)*M] = real(PSI[N*M:(N+1)*M])
    Cxx[N*M:(N+1)*M] = real(Cxx[N*M:(N+1)*M])
    Cxy[N*M:(N+1)*M] = real(Cxy[N*M:(N+1)*M])
    Cyy[N*M:(N+1)*M] = real(Cyy[N*M:(N+1)*M])

    Cxx[(N+1)*M:] = 0.0
    Cxx[:N*M] = 0.0
    Cyy[(N+1)*M:] = 0.0
    Cyy[:N*M] = 0.0
    Cxy[(N+1)*M:] = 0.0
    Cxy[:N*M] = 0.0

    return PSI, Cxx, Cyy, Cxy, forcing, P

def easy_perturbation(PSI, Cxx, Cyy, Cxy, perAmp=1.0e-10):
    """
    perturb just the second Chebyshev of the xx conformation and the first of the
    xy conformation.

    perturbation used in pythonic test code.

    """
    Cxx[(N+1)*M + 2] = perAmp * (Wi*2./pi)
    Cxx[(N-1)*M + 2] = perAmp * (Wi*2./pi)

    Cxy[(N+1)*M + 1] = perAmp * (Wi*2./pi)
    Cxy[(N-1)*M + 1] = perAmp * (Wi*2./pi)

    print log(abs(perAmp))

    return PSI,Cxx,Cyy,Cxy

def simple_perturbation(PSI, Cxx, Cyy, Cxy, perAmp=1.0e-10):
    """ 
    Slightly better perturbation to catch more errors.
    """

    Cxx[(N+1)*M:(N+2)*M-2] = perAmp * (Wi*2./pi)
    Cxx[(N-1)*M:N*M-2] = perAmp * (Wi*2./pi)

    Cxy[(N+1)*M:(N+2)*M-2] = perAmp * (Wi*2./pi)
    Cxy[(N-1)*M:N*M-2] = perAmp * (Wi*2./pi)

    return PSI, Cxx, Cyy, Cxy

def BC_safe_perturbation(PSI, Cxx, Cyy, Cxy, perAmp=1.0e-10):
    """
    Perturbation which satisfies the bc's
    """

    #rn = (10.0**(-1))*(0.5-rand(5))
    rn = ones(5)

    rSpace = zeros(Mf, dtype='complex')
    y = 2.0*arange(Mf)/(Mf-1.0) -1.0

    ## sinusoidal
    rSpace =  perAmp*sin(1.0 * pi * y) * rn[0]
    rSpace += perAmp*sin(2.0 * pi * y) * rn[1]
    rSpace += perAmp*sin(3.0 * pi * y) * rn[2]

    ## cosinusoidal 
    rSpace += perAmp*cos(1.0 * 0.5*pi * y) * rn[3]
    rSpace += perAmp*cos(3.0 * 0.5*pi * y) * rn[4]


    PSI[(N+1)*M:(N+2)*M] = f2d.forward_cheb_transform(rSpace, CNSTS)
    PSI[(N-1)*M:(N)*M] = conj(PSI[(N+1)*M:(N+2)*M])
    Cxx[(N+1)*M:(N+2)*M] = f2d.forward_cheb_transform(rSpace, CNSTS)
    Cxx[(N-1)*M:(N)*M] = conj(Cxx[(N+1)*M:(N+2)*M])

    return PSI, Cxx, Cyy, Cxy

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

print"=====================================\n"
print "Settings:"
print """------------------------------------
N \t\t= {N}
M \t\t= {M}              
Re \t\t= {Re}         
beta \t\t= {beta}         
Wi \t\t= {Wi}         
kx \t\t= {kx}
dt\t\t= {dt}
De\t\t={De}
delta\t\t={delta}
totTime\t\t= {t}
NumTimeSteps\t= {NT}
------------------------------------
        """.format(N=N, M=M, kx=kx, Re=Re, beta=beta, Wi=Wi,
                   De=De, delta=delta,
                   dt=dt, NT=numTimeSteps, t=totTime)

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

if args.flow_type==0:
    # --------------- POISEUILLE -----------------
    PSI, Cxx, Cyy, Cxy, forcing = poiseuille_flow()

elif args.flow_type==1:
    # --------------- SHEAR LAYER -----------------
    PSI, Cxx, Cyy, Cxy, forcing = shear_layer_flow()
    # set BC
    CNSTS['U0'] = 1.0

elif args.flow_type==2:
    # --------------- OSCILLATORY FLOW -----------------
    PSI, Cxx, Cyy, Cxy, forcing, CNSTS['P'] = oscillatory_flow()
    #PSI, Cxx, Cyy, Cxy, forcing, CNSTS['P'] = poiseuille_oscil_flow()
    #Ubase, Cxybase, Cxxbase = give_base_profile(Wi,0)

    #Cxx[N*M:(N+1)*M] = Cxxbase
    #Cxy[N*M:(N+1)*M] = Cxybase
    

else:
    print "flow type unspecified"
    exit(1)


# ---------------------PERTURBATION-----------------------------------------

psiLam = copy(PSI)

#PSI, Cxx, Cyy, Cxy = easy_perturbation(PSI, Cxx, Cyy, Cxy, perAmp=1.0e-10)
#PSI, Cxx, Cyy, Cxy = simple_perturbation(PSI, Cxx, Cyy, Cxy, perAmp=1.0e-10)
PSI, Cxx, Cyy, Cxy = BC_safe_perturbation(PSI, Cxx, Cyy, Cxy, perAmp=1.0e-10)

# ----------------------------------------------------------------------------
print type(psiLam)
print shape(psiLam)

##  output forcing and the streamfunction corresponding to the initial stress
f = h5py.File("laminar.h5", "w")
dset = f.create_dataset("psi", ((2*N+1)*M,), dtype='complex')
psiLam = psiLam.reshape(2*N+1, M).T
psiLam = ifftshift(psiLam, axes=1)
dset[...] = array(psiLam).T.flatten()
f.close()

f = h5py.File("forcing.h5", "w")
dset = f.create_dataset("psi", ((2*N+1)*M,), dtype='complex')
dset[...] = forcing.T.flatten()
f.close()

# Form the operators
if args.flow_type==2:
    PsiOpInvList = form_oscil_operators(dt)
    PsiOpInvListHalf = form_oscil_operators(dt/2.0)
else:
    PsiOpInvList = form_operators(dt)
    PsiOpInvListHalf = form_operators(dt/2.0)

#### SAVE THE OPERATORS AND INITIAL STATE FOR THE C CODE

for i in range(N+1):
    # operator order in list is 0->N
    n = i
    opFn = "./operators/op{0}.h5".format(n)
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
cargs = ["./DNS_2D_linear_Visco", "-N", "{0:d}".format(CNSTS["N"]), "-M",
         "{0:d}".format(CNSTS["M"]),"-U", "{0:.6e}".format(CNSTS["U0"]), "-k",
         "{0:.6e}".format(CNSTS["kx"]), "-R", "{0:.6e}".format(CNSTS["Re"]),
         "-W", "{0:.6e}".format(CNSTS["Wi"]), "-b",
         "{0:.6e}".format(CNSTS["beta"]), "-D",
         "{0:.6e}".format(CNSTS["De"]),
         "-P", "{0:.6e}".format(CNSTS["P"]),
         "-t", "{0:.6e}".format(CNSTS["dt"]),
         "-s", "{0:d}".format(stepsPerFrame), "-T",
         "{0:d}".format(numTimeSteps), "-i", "{0:.6e}".format(initTime)]

if dealiasing:
    cargs.append("-d")

    print "./DNS_2D_linear_Visco", "-N", "{0:d}".format(CNSTS["N"]), "-M",\
             "{0:d}".format(CNSTS["M"]),"-U", "{0:.6e}".format(CNSTS["U0"]), "-k",\
             "{0:.6e}".format(CNSTS["kx"]), "-R", "{0:.6e}".format(CNSTS["Re"]),\
             "-W", "{0:.6e}".format(CNSTS["Wi"]), "-b",\
             "{0:.6e}".format(CNSTS["beta"]), "-D", "{0:.6e}".format(CNSTS["De"]),\
             "-P", "{0:.6e}".format(CNSTS["P"]), \
             "-t", "{0:.6e}".format(CNSTS["dt"]),\
             "-s", "{0:d}".format(stepsPerFrame), "-T",\
             "{0:d}".format(numTimeSteps), "-i", "{0:.6e}".format(initTime), "-d"

else:

    print "./DNS_2D_linear_Visco", "-N", "{0:d}".format(CNSTS["N"]), "-M",\
             "{0:d}".format(CNSTS["M"]),"-U", "{0:.6e}".format(CNSTS["U0"]), "-k",\
             "{0:.6e}".format(CNSTS["kx"]), "-R", "{0:.6e}".format(CNSTS["Re"]),\
             "-W", "{0:.6e}".format(CNSTS["Wi"]), "-b",\
             "{0:.6e}".format(CNSTS["beta"]), "-De",\
            "{0:.6e}".format(CNSTS[""]),\
             "-P", "{0:.6e}".format(CNSTS["P"]), \
             "-t", "{0:.6e}".format(CNSTS["dt"]),\
             "-s", "{0:d}".format(stepsPerFrame), "-T",\
             "{0:d}".format(numTimeSteps), "-i", "{0:.6e}".format(initTime)

if args.flow_type==2:
    cargs.append("-O")

subprocess.call(cargs)

# Read in data from the C code
print 'done'
