
cimport cpy_DNS_2D_Visco
import numpy as np
cimport numpy as np

def run_simulation(flowConsts):
    # additional arguments: psi, cij, forcing, laminar, ops, hops

    cdef cpy_DNS_2D_Visco.flow_params params

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
    #params.Omega = flowConsts['Omega'];
    params.De = flowConsts['De'];
    params.P = flowConsts['P'];
    params.dt = flowConsts['dt'];
    params.stepsPerFrame = flowConsts['stepsPerFrame'];
    params.numTimeSteps = flowConsts['numTimeSteps'];
    params.initTime = flowConsts['initTime'];
    

    cpy_DNS_2D_Visco.DNS_2D_Visco(params)


