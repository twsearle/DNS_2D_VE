DNS_2D_VE
---------
*********


A Direct Numerical Simulator for 2D Viscoelastic flow of Oldroyd-B model fluids. This code uses a pseudo-spectral Chebychev-Fourier scheme in the spatial dimensions and a 4th order in time finite differencing scheme. The details of how the code works can be found in my thesis here http://www.tobysearle.com/pages/thesis. As I make changes to the technical details I will try to keep this document up-to-date.

The code is written in a combination of C and python, my intention is to rewrite the C into cython as far as possible.

Requirements
============

C packages
----------

hdf5 --- tested with version 1.10.0_0

fftw --- tested with version 3.3.5 


Python packages
---------------

h5py --- tested with version 2.6.0

cython --- tested with version 0.25.1_0

numpy --- tested with version 1.11.2_1

scipy --- tested with version 0.17.0_0


Build and Run Instructions
==========================

* First setup the cython files:
`python setup.py build_ext --inplace`

Then test using the test script
`python stored_IC.py`

If all goes well the script will run 3 simulations, before passing. If it doesn't pass, I am sorry about that :(

Now you can edit the DNS_2D_Visco.py script to change the initial conditions, and the config.cfg file to change the parameters of the simulation.

Then you can run the simulation on your own data `python DNS_2D_Visco.py`


Methods
=======

I used to use C code that was called from python using the subprocess module. This is clunky (the makefile from this scheme may still be kicking around). Now I use cython to wrap my C code in a nice interface. My long term objective is to move all of this C code into Cython.

There are a few different layers to the code::

    python          -->     cython              -->     C               -->  C                  --> C
    stored_IC.py            cpy_DNS_2D_Visco.pyx        DNS_2D_Visco.c       time_steppers.c        fields_2D.c
    DNS_2D_Visco.py         cpy_DNS_2D_Visco.pxd        

The idea is to roll-up the C layers into this new cython layer, but in the meantime it is going to look rather messy!

TODO
====

* Move all the functionality from c code into pyx and remove the c code
* Tidy up the python that wraps the C code so that there is only one set of methods to do field operations
* Create new tests for all of these methods as I go, build a test suite

For any piece of C there are a couple of steps:
#. Wrap whatever is calling the c code in a layer of pyx code
#. Move the C functionality into the pyx wrapper
#. Remove the underlying C code and interface directly with the layer underneath that

At the moment I need to:

* Move the rest of DNS_2D_Visco.c into cpy_DNS_2D_Visco.pyx
    - split off the functions called by DNS_2D_Visco so that the code has to run each function in turn, called by the pyx code.
    - move code above the DNS_2D_Visco function call into the .pyx file.
    - call the function from within the pyx file instead of the c file
    - repeat above for next function call


