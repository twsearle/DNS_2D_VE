from scipy import *
from scipy import linalg
from scipy import fftpack
from numpy.fft import fftshift, ifftshift
from scipy.fftpack import dct as dct
from pylab import *

import cPickle as pickle

import ConfigParser
import argparse 
import h5py

argparser = argparse.ArgumentParser()

argparser.add_argument("-p", "--path", type=str, default="./", 
                help='specify the directory containing the data')
argparser.add_argument("-N", type=int, default=None, 
                help='Override Number of Fourier modes given in the config file')
argparser.add_argument("-M", type=int, default=None, 
                help='Override Number of Chebyshev modes in the config file')
argparser.add_argument("-Re", type=float, default=None, 
                help="Override Reynold's number in the config file") 
argparser.add_argument("-b", type=float, default=None, 
                help='Override beta of the config file')
argparser.add_argument("-Wi", type=float, default=None, 
                help='Override Weissenberg number of the config file')
argparser.add_argument("-kx", type=float, default=None, 
                help='Override wavenumber of the config file')

argparser.add_argument("-Newt", action='store_true', 
                help='Newtonian data flag')

args = argparser.parse_args()
N = args.N 
M = args.M
Re = args.Re
beta = args.b
Wi = args.Wi
kx = args.kx

config = ConfigParser.RawConfigParser()
fp = open(args.path + '/config.cfg')
config.readfp(fp)
if N == None : N = config.getint('General', 'N')
if M == None : M = config.getint('General', 'M')
if kx == None : kx = config.getfloat('General', 'kx')
if Re == None : Re = config.getfloat('General', 'Re')
if Wi == None : Wi = config.getfloat('General', 'Wi')
if beta == None : beta = config.getfloat('General', 'beta')

dt = config.getfloat('Time Iteration', 'dt')
totTime = config.getfloat('Time Iteration', 'totTime')
numFrames = config.getint('Time Iteration', 'numFrames')
dealiasing = config.getboolean('Time Iteration', 'Dealiasing')

Nf = 50
Mf = 100

fp.close()

numTimeSteps = int(totTime / dt)

kwargs = {'N': N, 'M': M, 'Nf': Nf, 'Mf':Mf, 
          'Re': Re,'Wi': Wi, 'beta': beta, 'kx': kx,'time':
          totTime, 'dt':dt, 'dealiasing':dealiasing }

if args.Newt:
    #inFileName = args.path + "/traj_psi.h5".format()
    inFileName = args.path + "/final.h5".format()
    twsFileName = args.path + "/pf-N{N}-M{M}-kx{kx}-Re{Re}.pickle".format(**kwargs)
else:
    inFileName = args.path + "/traj.h5".format()
    #inFileName = args.path + "/final.h5".format()
    twsFileName = args.path + "/pf-N{N}-M{M}-kx{kx}-Re{Re}-b{beta}-Wi{Wi}.pickle".format(**kwargs)


CNSTS = kwargs

def load_hdf5_snapshot_visco(fp, time):

    dataset_id = "/t{0:f}".format(time)
    print dataset_id

    psi = array(f[dataset_id+"/psi"])
    cxx = array(f[dataset_id+"/cxx"])
    cyy = array(f[dataset_id+"/cyy"])
    cxy = array(f[dataset_id+"/cxy"])

    return psi, cxx, cyy, cxy

def load_hdf5_snapshot(fp, time):

    dataset_id = "/t{0:f}".format(time)
    print dataset_id

    inarr = array(f[dataset_id])

    return inarr

def forward_cheb_transform(GLreal, CNSTS):
    """
    Use a real FFT to transform a single array from the Gauss-Labatto grid to
    Chebyshev polynomials.

    Note, this uses a real FFT therefore you must apply the transformations in
    the other directions before this one, otherwise you will loose the data from
    the imaginary parts.
    """

    M = CNSTS['M']
    Mf = CNSTS['Mf']

    Ly = CNSTS['Ly']


    tmp = dct(real(GLreal), type=1)

    out = zeros(M, dtype='complex')
    # Renormalise and divide by c_k to convert to Chebyshev polynomials
    out[0] = (0.5/(Mf-1.0)) * tmp[0]
    out[1:M-1] = (1.0/(Mf-1.0)) * tmp[1:M-1]
    if dealiasing:
        out[M-1] = (1.0/(Mf-1.0)) * tmp[M-1]
    else:
        out[M-1] = (0.5/(Mf-1.0)) * tmp[M-1]

    return out

def backward_cheb_transform(cSpec, CNSTS):
    """
    Use a real FFT to transform a single array of Chebyshev polynomials to the
    Gauss-Labatto grid.
    """

    M = CNSTS['M']

    # Define the temporary vector for the transformation
    tmp = zeros(Mf)

    # The first half contains the vector on the Gauss-Labatto points * c_k
    tmp[0] = real(cSpec[0])
    tmp[1:M] = 0.5*real(cSpec[1:M])
    tmp[Mf-1] = 2*tmp[Mf-1]

    out = dct(tmp, type=1)

    tmp[0] = imag(cSpec[0])
    tmp[1:M] = 0.5*imag(cSpec[1:M])
    tmp[Mf-1] = 2*tmp[Mf-1]

    out += dct(tmp, type=1) * 1.j

    return out[0:Mf]

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

def plot_modes(arr_ti, arr_true, phase_factor, arrname, time, CNSTS):

    fig=gcf()
    fig.suptitle(
        '${0}$ time iteration red, TWS in green at {1}'.format(arrname, time))
    subplot_indices= {0:231, 1:232, 2:233, 3:234, 4:235, 5:236}
    for n in range(3):

        ti_mode = stupid_transform_i(arr_ti[:, n], CNSTS)
        tws_mode = stupid_transform_i(arr_true[:, n]*(phase_factor**n), CNSTS)

        # plot graph
        splt =subplot(subplot_indices[n])
        plot(y, real(ti_mode), 'r', linewidth=2.0)
        plot(y, real(tws_mode), 'g')
        title('{0}'.format(n))
        xlabel('y')
        ylabel('$Re[{0}_{1}]$'.format(arrname, n))

    for n in range(3):

        ti_mode = stupid_transform_i(arr_ti[:, n], CNSTS)
        tws_mode = stupid_transform_i(arr_true[:, n]*(phase_factor**n), CNSTS)

        # plot graph
        splt =subplot(subplot_indices[(3+n)])
        plot(y, imag(ti_mode), 'r', linewidth=2.0)
        plot(y, imag(tws_mode), 'g')
        title('{0}'.format(n))
        xlabel('y')
        ylabel('$Im[{0}_{1}]$'.format(arrname, n))

    fig.tight_layout()
    if arrname == "\psi" :
        savefig(args.path + "/modal_comparison_{0}.pdf".format("psi"))
    else:
        savefig(args.path + "/modal_comparison_{0}.pdf".format(arrname))

    fig.clf()



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

#time = totTime
time = 9960

f = h5py.File(inFileName, "r")

if args.Newt:
    psi = array(f["/psi"])
    psi_ti = psi_ti.reshape((2*N+1, M)).T

else:
    psi_ti, cxx_ti, cyy_ti, cxy_ti = load_hdf5_snapshot_visco(f, time)

    #psi = array(f["/psi"])
    #cxx_ti = array(f["/cxx"])
    #cyy_ti = array(f["/cyy"])
    #cxy_ti = array(f["/cxy"])

    psi_ti = psi_ti.reshape((N+1, M)).T
    psi_ti = hstack((psi_ti, conj(psi_ti[:, N:0:-1])))
    cxx_ti = cxx_ti.reshape((N+1, M)).T
    cxx_ti = hstack((cxx_ti, conj(cxx_ti[:, N:0:-1])))
    cyy_ti = cyy_ti.reshape((N+1, M)).T
    cyy_ti = hstack((cyy_ti, conj(cyy_ti[:, N:0:-1])))
    cxy_ti = cxy_ti.reshape((N+1, M)).T
    cxy_ti = hstack((cxy_ti, conj(cxy_ti[:, N:0:-1])))

    psi_true, cxx_true, cyy_true, cxy_true, Nu = pickle.load(open(twsFileName, 'r'))

f.close()

silly_test = copy(psi_true)
psi_true = psi_true.reshape(2*N+1, M).T
psi_true = ifftshift(psi_true, axes=1)
print psi_true[:,1] - silly_test[(N+1)*M:(N+2)*M]

cxx_true = cxx_true.reshape(2*N+1, M).T
cxx_true = ifftshift(cxx_true, axes=1)
cyy_true = cyy_true.reshape(2*N+1, M).T
cyy_true = ifftshift(cyy_true, axes=1)
cxy_true = cxy_true.reshape(2*N+1, M).T
cxy_true = ifftshift(cxy_true, axes=1)

phase_factor = conj(psi_true[2, 1] / psi_ti[2, 1])
print phase_factor

#y = cos(pi*arange(Mf)/(Mf-1))
y = 2.0*arange(Mf)/(Mf-1.) -1

# Compare mode by mode

plot_modes(psi_ti, psi_true, phase_factor, "\psi", time, CNSTS)
plot_modes(cxx_ti, cxx_true, phase_factor, "cxx", time, CNSTS)
plot_modes(cyy_ti, cyy_true, phase_factor, "cyy", time, CNSTS)
plot_modes(cxy_ti, cxy_true, phase_factor, "cxy", time, CNSTS)

