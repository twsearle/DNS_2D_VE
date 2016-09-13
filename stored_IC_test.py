#-----------------------------------------------------------------------------
#   Test for 2D DNS codes using stored initial conditions
#
#   Last modified: Tue 13 Sep 14:29:39 2016
#
#-----------------------------------------------------------------------------

# MODULES
from scipy import *
from numpy.fft import fftshift, ifftshift
from numpy.linalg import norm
import subprocess
import h5py

### FUNCTIONS ###

def load_hdf5_visco(filename):
    f = h5py.File(filename, 'r')

    psi = array(f["psi"])
    cxx = array(f["cxx"])
    cyy = array(f["cyy"])
    cxy = array(f["cxy"])

    f.close()

    return psi, cxx, cyy, cxy

### MAIN ###

if __name__ == "__main__":

    # set constants 
    # copy the test config file over the config file
    subprocess.call(["cp", "TEST/PPF.cfg", "config.cfg"])

    # run the DNS code using the test initial conditions
    subprocess.call(["cp", "TEST/PPF_IC.h5", "input.h5"])

    # use subprocess to run the python program 
    subprocess.call(["python", "DNS_2D_Visco.py", "-flow_type", "0", "-test", "1"])

    # read in the result and compare to test final conditions
    psi, cxx, cyy, cxy  = load_hdf5_visco("output/final.h5") 
    psiTrue, cxxTrue, cyyTrue, cxyTrue  = load_hdf5_visco("TEST/PPF_FC.h5") 

    print "------------------------------"
    print "Test Plane Poiseuille flow"
    print "------------------------------"

    print "Identical psi? ", allclose(psi, psiTrue), "norm", norm(psi-psiTrue) 
    print "Identical cxx? ", allclose(cxx, cxxTrue), "norm", norm(cxx-cxxTrue)
    print "Identical cxy? ", allclose(cxy, cxyTrue), "norm", norm(cxy-cxyTrue)
    print "Identical cyy? ", allclose(cyy, cyyTrue), "norm", norm(cyy-cyyTrue)

