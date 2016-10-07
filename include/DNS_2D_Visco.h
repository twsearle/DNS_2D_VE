/* -------------------------------------------------------------------------- *
 *									      *
 *  DNS_2D_Visco.h							      *
 *                                                                            *
 *  Time stepping DNS simulation routines				      *
 *                                                                            *
 *                                                                            *
 * -------------------------------------------------------------------------- */
#ifndef DNS_2D_VISCO_H
#define DNS_2D_VISCO_H

// Last modified: Mon  3 Oct 16:29:55 2016


// Headers

#include"fields_IO.h"
#include"fields_2D.h"
#include"time_steppers.h"

void read_cline_args(int argc, char **argv, flow_params *params);

void setup_scratch_space(flow_scratch *scr, flow_params params);

void output_macro_state(complex_d *psi, complex_d *cij,  complex_d *trC, double phase, double time,
	FILE *traceKE, FILE *tracePSI, FILE *trace1mode, FILE *traceStressfp, flow_scratch scr, flow_params params);

void debug_output_halfstep_variables(complex_d *psiNL, complex_d *cijNL, flow_scratch scr, flow_params params);

void debug_output_fullstep_variables(complex_d *psi, complex_d *cij, flow_scratch scr, flow_params params);

//int DNS_2D_Visco(int argc, char **argv);
int DNS_2D_Visco(flow_params params);

#endif // DNS_2D_VISCO_H
