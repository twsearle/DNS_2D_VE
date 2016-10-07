# /* -------------------------------------------------------------------------- *
#  *									        *
#  *  DNS_2D_Visco.h							        *
#  *                                                                            *
#  *  Time stepping DNS simulation cython header                                *
#  *                                                                            *
#  *                                                                            *
#  * -------------------------------------------------------------------------- */

# Last modified: Thu  6 Oct 17:10:19 2016
cdef extern from "include/DNS_2D_Visco.h":

    ctypedef struct flow_params:
        int N;
        int M;
        int dealiasing;
        int oscillatory_flow;
        int Nf;
        int Mf;
        double kx;
        double U0;
        double Re;
        double Wi;
        double beta;
        double Omega;
        double De;
        double P;
        double dt;
        int stepsPerFrame;
        int numTimeSteps;
        double initTime;

cdef extern from "include/DNS_2D_Visco.h":
    int DNS_2D_Visco(flow_params params);
