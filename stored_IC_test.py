#-----------------------------------------------------------------------------
#   Test for 2D DNS codes using stored initial conditions
#
#   Last modified: Mon 26 Sep 17:50:12 2016
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

def run_flow_test(cfgFilename, ICFilename, FCFilename, flowType):

    # set constants 
    # copy the test config file over the config file
    subprocess.call(["cp", cfgFilename, "config.cfg"])

    # run the DNS code using the test initial conditions
    subprocess.call(["cp", ICFilename, "input.h5"])

    # use subprocess to run the python program 
    subprocess.call(["time", "python", "DNS_2D_Visco.py", "-flow_type", flowType, "-test", "1"])

    # read in the result and compare to test final conditions
    psi, cxx, cyy, cxy  = load_hdf5_visco("output/final.h5") 
    psiTrue, cxxTrue, cyyTrue, cxyTrue  = load_hdf5_visco(FCFilename) 

    print "------------------------------"
    print "Test"
    if flowType=="0": print "Poiseuille Flow"
    if flowType=="1": print "Shear Layer Flow"
    if flowType=="2": print "Oscillatory Flow"
    print "------------------------------"

    psiTest = allclose(psi, psiTrue)
    cxxTest = allclose(cxx, cxxTrue)
    cxyTest = allclose(cxy, cxyTrue)
    cyyTest = allclose(cyy, cyyTrue)

    print "Identical psi? ", psiTest, "norm ", norm(psi-psiTrue)
    print "Identical cxx? ", cxxTest, "norm", norm(cxx-cxxTrue)
    print "Identical cxy? ", cxyTest, "norm", norm(cxy-cxyTrue)
    print "Identical cyy? ", cyyTest, "norm", norm(cyy-cyyTrue)

    if not psiTest:
        print "failed to match psi"
        exit(1)
    if not cxxTest:
        print "failed to match cxx"
        exit(1)
    if not cxyTest:
        print "failed to match cxy"
        exit(1)
    if not cyyTest:
        print "failed to match cyy"
        exit(1)

def test_Poiseuille_flow():
    # A mean time over 5 runs: 21.31
    #(21.03 + 21.25 + 21.54 + 21.19 + 21.53) / 5

    flowType = "0" # Poiseuille flow setting 
    run_flow_test("TEST/PPF.cfg", "TEST/PPF_IC.h5", "TEST/PPF_FC.h5", flowType)

def test_shear_layer_flow():
    # A mean time over 5 runs: 23.17
    #(22.95 + 22.92 + 23.08 + 23.33 + 23.56) / 5

    flowType = "1" # shear layer flow setting 
    run_flow_test("TEST/SLF.cfg", "TEST/SLF_IC.h5", "TEST/SLF_FC.h5", flowType)

def test_oscillatory_flow():
    # A mean time over 5 runs: 32.47
    #(32.91 + 32.69 + 31.85 + 31.90 + 33.02) / 5

    flowType = "2" # oscillatory flow setting 
    run_flow_test("TEST/OSCF.cfg", "TEST/OSCF_IC.h5", "TEST/OSCF_FC.h5", flowType)

### MAIN ###

if __name__ == "__main__":

    test_Poiseuille_flow()

    test_shear_layer_flow()

    test_oscillatory_flow()


