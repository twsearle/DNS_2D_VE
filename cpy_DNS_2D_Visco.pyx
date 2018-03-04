
cimport cpy_DNS_2D_Visco
import numpy as np
cimport numpy as np
# We now need to fix a datatype for our arrays. I've used the variable
# DTYPE for this, which is assigned to the usual NumPy runtime
# type info object.
DTYPE_DOUB = np.double
DTYPE_CMPLX = np.complex128

def run_full_simulation(np.ndarray[DTYPE_CMPLX_t, ndim=1, mode="c"] psi,
                   np.ndarray[DTYPE_CMPLX_t, ndim=1, mode="c"] cxx,
                   np.ndarray[DTYPE_CMPLX_t, ndim=1, mode="c"] cyy,
                   np.ndarray[DTYPE_CMPLX_t, ndim=1, mode="c"] cxy,
                   np.ndarray[DTYPE_CMPLX_t, ndim=1, mode="c"] forcing,
                   np.ndarray[DTYPE_CMPLX_t, ndim=1, mode="c"] psi_lam,
                   np.ndarray[DTYPE_CMPLX_t, ndim=1, mode="c"] opsList,
                   np.ndarray[DTYPE_CMPLX_t, ndim=1, mode="c"] hopsList,
                   flowConsts):

    cdef cpy_DNS_2D_Visco.flow_params params

    M = flowConsts['M']
    N = flowConsts['N']
    params.M = flowConsts['M']
    params.N = flowConsts['N']
    params.Mf = flowConsts['Mf']
    params.Nf = flowConsts['Nf']
    params.dealiasing = flowConsts['dealiasing'];
    params.oscillatory_flow = flowConsts['oscillatory_flow'];
    params.kx = flowConsts['kx'];
    params.U0 = flowConsts['U0'];
    params.Re = flowConsts['Re'];
    params.Wi = flowConsts['Wi'];
    params.beta = flowConsts['beta'];
    params.De = flowConsts['De'];
    params.P = flowConsts['P'];
    params.dt = flowConsts['dt'];
    params.stepsPerFrame = flowConsts['stepsPerFrame'];
    params.numTimeSteps = flowConsts['numTimeSteps'];
    params.initTime = flowConsts['initTime'];

    cdef np.ndarray[DTYPE_CMPLX_t, ndim=1, mode="c"] cij = np.zeros(
        [3*(N+1)*M], dtype=DTYPE_CMPLX)
    cij[:(N+1)*M] = cxx[:]
    cij[(N+1)*M:2*(N+1)*M] = cyy[:]
    cij[2*(N+1)*M:3*(N+1)*M] = cxy[:]

    paramsString=("PARAMETERS: "
                  "\nN                   \t {N:d} "
                  "\nM                   \t {M:d} "
                  "\nU0                  \t {U0:f} "
                  "\nkx                  \t {kx:f} "
                  "\nRe                  \t {Re:e} "
                  "\nWi                  \t {Wi:e} "
                  "\nbeta                \t {beta:e} "
                  "\nDe                  \t {De:e} "
                  "\nTime Step           \t {dt:e} "
                  "\nNumber of Time Steps\t {numTimeSteps} "
                  "\nTime Steps per frame\t {stepsPerFrame:d} "
                  "\nInitial Time        \t {initTime:f} \n")
    print paramsString.format(**flowConsts)

    cpy_DNS_2D_Visco.DNS_2D_Visco(&psi[0], &cij[0], &forcing[0], &psi_lam[0],
                                  &opsList[0], &hopsList[0], params)


