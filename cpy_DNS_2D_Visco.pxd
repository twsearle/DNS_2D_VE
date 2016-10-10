# /* -------------------------------------------------------------------------- *
#  *									        *
#  *  DNS_2D_Visco.h							        *
#  *                                                                            *
#  *  Time stepping DNS simulation cython header                                *
#  *                                                                            *
#  *                                                                            *
#  * -------------------------------------------------------------------------- */

# Last modified: Mon 10 Oct 14:31:50 2016
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
    int DNS_2D_Visco(double complex *psi, double complex *cij, double complex
                     *forcing, double complex *psi_lam, double complex *opsList,
                     double complex *hopsList, flow_params params);
