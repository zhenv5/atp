#ifndef _PMF_R_H_
#define _PMF_R_H_

#include "../util.h"
#include "../pmf.h"

#include <R.h>
#include <Rinternals.h>
#include <Rdefines.h>

extern "C" {
SEXP print_training_option();
SEXP get_rank_from_param(SEXP Rstr_param);
SEXP pmf_train(SEXP R_m, SEXP R_n, SEXP R_nnz, SEXP R_i, SEXP R_j, SEXP R_v, 
		SEXP R_param_str, SEXP R_ret_W, SEXP R_ret_H);
SEXP pmf_weighted_train(SEXP R_m, SEXP R_n, SEXP R_nnz, SEXP R_i, SEXP R_j, SEXP R_v, SEXP R_w,
		SEXP R_param_str, SEXP R_ret_W, SEXP R_ret_H);
}


#endif
