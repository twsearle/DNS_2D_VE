from scipy import *
from scipy import linalg
from scipy import fftpack
from numpy.fft import fftshift, ifftshift
from scipy.fftpack import dct as dct
import matplotlib
from matplotlib import pyplot
matplotlib.use('tkAgg')
import matplotlib.animation

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
Wi = config.getfloat('General', 'Wi')
beta = config.getfloat('General', 'beta')
kx = config.getfloat('General', 'kx')
Nf = 2*N
Mf = 2*M

n = 0 #None
varName = 'psi'

dt = config.getfloat('Time Iteration', 'dt')
totTime = config.getfloat('Time Iteration', 'totTime')
numFrames = config.getint('Time Iteration', 'numFrames')

dealiasing = config.getboolean('Time Iteration', 'Dealiasing')

fp.close()

numTimeSteps = int(totTime / dt)

kwargs = {'N': N, 'M': M, 'Nf': Nf, 'Mf':Mf, 
          'Re': Re,'Wi': Wi, 'beta': beta, 'kx': kx,'time':
          totTime, 'dt':dt, 'dealiasing':dealiasing }

inFileName = "./output/traj.h5".format()
twsFileName = "pf-N{N}-M{M}-kx{kx}-Re{Re}.pickle".format(**kwargs)

CNSTS = kwargs

def load_hdf5_snapshot(fp, time, varName):
    dataset_id = "/t{0:f}/".format(time) + varName
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
    #tmp = zeros(2*Mf-2)
    tmp = zeros(Mf)

    # The first half contains the vector on the Gauss-Labatto points * c_k
    tmp[0] = real(cSpec[0])
    tmp[1:M] = 0.5*real(cSpec[1:M])
    tmp[Mf-1] = 2*tmp[Mf-1]

    # The second half contains the vector on the Gauss-Labatto points excluding
    # the first and last elements and in reverse order
    #tmp[2*Mf-M:] = real(0.5*cSpec[M-2:0:-1])

    # Perform the transformation and divide the result by 2
    #out = real(fftpack.rfft(tmp))
    out = real(dct(tmp, type=1))

    #return out[0:Mf]
    return out

def stupid_transform_i(GLspec, CNSTS):
    """
    apply the Chebyshev transform the stupid way.
    """

    M = CNSTS['M']
    Mf = CNSTS['Mf']
    Ly = CNSTS['Ly']

    out = zeros(Mf, dtype='complex')

    for i in range(Mf):
        out[i] += GLspec[0]
        for j in range(1,M-1):
            out[i] += GLspec[j]*cos(pi*i*j/(Mf-1))
        out[i] += GLspec[M-1]*cos(pi*i)
    del i,j

    return out

def convert_series(f,n):
    time = 0.0

    psi_ti = load_hdf5_snapshot(f, time, varName)
    psi_ti = psi_ti.reshape((N+1, M)).T
    psi_ti = hstack((psi_ti, conj(psi_ti[:, N:0:-1])))


    u_ti =  f2d.dy(psi_ti, CNSTS)

    ti_mode_r = backward_cheb_transform(real(u_ti[:,n]), CNSTS)
    ti_mode_i = backward_cheb_transform(imag(u_ti[:,n]), CNSTS)
    u_data = [ti_mode_r, ti_mode_i]

    for i in range(1, numFrames):
        time = i*(totTime/numFrames)
        psi_ti = load_hdf5_snapshot(f, time, varName)
        psi_ti = psi_ti.reshape((N+1, M)).T
        psi_ti = hstack((psi_ti, conj(psi_ti[:, N:0:-1])))


        u_ti =  f2d.dy(psi_ti, CNSTS)

        ti_mode_r = backward_cheb_transform(real(u_ti[:,n]), CNSTS)
        ti_mode_i = backward_cheb_transform(imag(u_ti[:,n]), CNSTS)
        u_data.append([ti_mode_r, ti_mode_i])

    return u_data

def init():

    # plot graph
    line1.set_data([], [])
    line2.set_data([], [])

    return line1, line2,

def animate(i):
    ti_mode_r = psi_data[i][0]
    ti_mode_i = psi_data[i][1]

    # plot graph
    time = i*(totTime/numFrames)
    matplotlib.pyplot.title('$\psi_{0}$ red real part, green imaginary, t ={1}'.format(n, time))
    matplotlib.pyplot.draw()

    line1.set_data(y, ti_mode_r)
    line2.set_data(y, ti_mode_i)

    return line1, line2,

def init_all():

    # plot graph
    timetext.set_text('')
    line0r.set_data([], [])
    line0i.set_data([], [])
    line1r.set_data([], [])
    line1i.set_data([], [])
    line2r.set_data([], [])
    line2i.set_data([], [])

    return line0r, line0i, line1r, line1i, line2r, line2i, timetext

def animate_all(i):

    timetext.set_text('time = {0}'.format(i*(totTime/numFrames)) )
    line0r.set_data(y, u_data0[i][0])
    line1r.set_data(y, u_data1[i][0])
    line2r.set_data(y, u_data2[i][0])

    line0i.set_data(y, u_data0[i][1])
    line1i.set_data(y, u_data1[i][1])
    line2i.set_data(y, u_data2[i][1])

    # plot graph
    #time = i*(totTime/numFrames)
    #matplotlib.pyplot.title('$\psi_{0}$ red real part, green imaginary, t ={1}'.format(n, time))
    #matplotlib.pyplot.draw()

    return line0r, line0i, line1r, line1i, line2r, line2i, timetext

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

time = 0.0
y = cos(pi*arange(Mf)/(Mf-1))

# Compare mode by mode
if n != None:
    fig = matplotlib.pyplot.figure()
    matplotlib.pyplot.title('$u_{0}$ red real part, green imaginary, t ={1}'.format(n, time))
    matplotlib.pyplot.xlabel('y')
    matplotlib.pyplot.ylabel('$u_{0}$'.format(n))

    subplot_indices= {0:321, 1:322, 2:323, 3:324, 4:325, 5:326}

    ax = matplotlib.pyplot.axes(xlim=(-1., 1.), ylim=(-5,5))

    line1, = ax.plot([], [], 'r', lw=1 )
    line2, = ax.plot([], [], 'g', lw=1 )


    tmp = load_hdf5_snapshot(f, time, varName).reshape((N+1, M)).T
    var_ti = zeros((M,2*N+1), dtype='complex')
    var_ti[:, :N+1] = tmp
    var_ti[:, N+1:] = conj(fliplr(tmp[:,1:]))


    # Match phase and convert to real space
    ti_mode_r = backward_cheb_transform(real(var_ti[:, n]), CNSTS)
    ti_mode_i = backward_cheb_transform(imag(var_ti[:, n]), CNSTS)
    psi_data = [ti_mode_r, ti_mode_i]

    for i in range(1, numFrames):
        time = i*(totTime/numFrames)
        tmp = load_hdf5_snapshot(f, time, varName).reshape((N+1, M)).T
        var_ti = zeros((M,2*N+1), dtype='complex')
        var_ti[:, :N+1] = tmp
        var_ti[:, N+1:] = conj(fliplr(tmp[:,1:]))

        #u_ti =  f2d.dy(var_ti, CNSTS)

        # Match phase and convert to real space
        ti_mode_r = backward_cheb_transform(real(var_ti[:,n]), CNSTS)
        ti_mode_i = backward_cheb_transform(imag(var_ti[:,n]), CNSTS)
        psi_data.append([ti_mode_r, ti_mode_i])

    anim = matplotlib.animation.FuncAnimation(fig, animate, init_func=init,
                                              frames=numFrames,
                                  interval=1, blit=False)

else:
    fig = matplotlib.pyplot.figure(figsize=(15,4.5), tight_layout=True)

    # Set up the plot

    # mode 0
    ax0 = fig.add_subplot(131)
    ax0.set_xlim([-1., 1.])
    ax0.set_ylim([-1.1, 1.1])
    ax0.set_xlabel('y')
    ax0.set_ylabel('$u_{n}$'.format(n=0))

    line0r, = ax0.plot([], [], 'r', lw=1 )
    line0i, = ax0.plot([], [], 'g', lw=1 )

    # mode 1
    ax1 = fig.add_subplot(132)
    ax1.set_xlim([-1., 1.])
    ax1.set_ylim([-0.01, 0.01])
    ax1.set_xlabel('y')
    ax1.set_ylabel('$u_{n}$'.format(n=1))
    timetext = ax1.text(0.0,0.05,'')

    line1r, = ax1.plot([], [], 'r', lw=1 )
    line1i, = ax1.plot([], [], 'g', lw=1 )

    # mode 2
    ax2 = fig.add_subplot(133)
    ax2.set_xlim([-1., 1.])
    ax2.set_ylim([-0.01, 0.01])
    ax2.set_xlabel('y')
    ax2.set_ylabel('$u_{n}$'.format(n=2))

    line2r, = ax2.plot([], [], 'r', lw=1 )
    line2i, = ax2.plot([], [], 'g', lw=1 )

    matplotlib.pyplot.suptitle(
        '$u$ red real part, green, imaginary'.format(time))

    #### Convert the time series to real space

    u_data0 = convert_series(f, 0)
    u_data1 = convert_series(f, 1)
    u_data2 = convert_series(f, 2)

    # animate
    anim = matplotlib.animation.FuncAnimation(fig, animate_all,
                                              init_func=init_all,
                                              frames=numFrames,
                                  interval=1, blit=False)

f.close()

matplotlib.pyplot.show(block='True')

