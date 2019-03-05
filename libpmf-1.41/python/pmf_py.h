#ifndef _PMF_PY_H_
#define _PMF_PY_H_

#include "../util.h"
#include "../pmf.h"

extern "C" {
const char* training_option();
parameter parse_training_command_line(const char *input);
void pmf_train(int m, int n, long nnz, int* i, int *j, double *v, parameter *param, double *ret_W, double *ret_H);
void pmf_weighted_train(int m, int n, long nnz, int* i, int *j, double *v, double *weight, parameter *param, double *ret_W, double *ret_H);
}


#endif
