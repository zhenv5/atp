#!/usr/bin/env python

from ctypes import *
from os import path

libpmf = CDLL(path.join(path.dirname(__file__),'pmf_py.so.1'))

def genFields(names, types): 
	return list(zip(names, types))

def fillprototype(f, restype, argtypes): 
	f.restype = restype
	f.argtypes = argtypes

class parameter(Structure):
	_names = ["solver_type", "k", "threads", "maxiter", "maxinneriter",
			"lambda", "rho", "eps", "eta0", "betaup", "betadown", "lrate_method",
			"num_blocks", "do_predict", "verbose", "do_nmf"]
	_types = [c_int, c_int, c_int, c_int, c_int, 
			c_double, c_double, c_double, c_double, c_double, c_double, c_int,
			c_int, c_int, c_int, c_int]
	_fields_ = genFields(_names, _types)

fillprototype(libpmf.training_option, c_char_p, [])
fillprototype(libpmf.parse_training_command_line, parameter, [c_char_p])
fillprototype(libpmf.pmf_train, None, [c_int, c_int, c_long, POINTER(c_int), POINTER(c_int), POINTER(c_double),
	POINTER(parameter), POINTER(c_double), POINTER(c_double)])

import numpy as np
import scipy.sparse as sparse


def train(A=None, param_str='', zero_as_missing = True):
	if A is None:
		print('train(A, param_str="", zero_as_missing = True)\n'
				'  A: a numpy array or a scipy.sparse.matrix\n'
				'  zero_as_missing: whether treat zero as missing (Default True)\n'
				'  param_str:\n'
				'%s' %(libpmf.training_option().split('\n',1)[-1]))
		return None
	m, n = A.shape
	if not zero_as_missing and isinstance(A, sparse.spmatrix):
		A = A.toarray()
	if isinstance(A, sparse.spmatrix):
		coo = sparse.coo_matrix(A)
		return train_coo(row_idx=coo.row, col_idx=coo.col, obs_val=coo.data, m=m, n=n, param_str=param_str)
	elif isinstance(A, np.ndarray):
		if zero_as_missing:
			row_idx, col_idx = np.nonzero(np.isfinite(A) & (A !=0))
		else :
			row_idx, col_idx = np.nonzero(np.isfinite(A))
		val = A[(row_idx, col_idx)]
		return train_coo(row_idx=row_idx, col_idx=col_idx, obs_val=val, m=m, n=n, param_str=param_str)
	else :
		print('type(A) = %s is not supported' % (type(A)))
		return None


def train_coo(row_idx=None, col_idx=None, obs_val=None, obs_weight=None, m=None, n=None, param_str = ''):
	'''
	if None in [row_idx, col_idx, obs_val, m, n]:
		print ( 
		'train_coo(row_idx, col_idx, obs_val, obs_weight, m, n, param_str="")\n'
		'  row_idx : a numpy.ndarray with dtype = numpy.int32\n'
		'  col_idx : a numpy.ndarray with dtype = numpy.int32\n'
		'  obs_val : a numpy.ndarray with dtype = numpy.float64\n'
		'  obs_weight : a numpy.ndarray with dtype = numpy.float64 (optional)\n'
		'  m, n    : # of rows, # of cols\n'
		'  param_str: \n'
		'%s' % (libpmf.training_option().split('\n',1)[-1]))
		return None
	'''	
	param = libpmf.parse_training_command_line(param_str)
	row_idx = np.array(row_idx, dtype=np.int32, copy=False)
	col_idx = np.array(col_idx, dtype=np.int32, copy=False)
	obs_val = np.array(obs_val, dtype=np.float64, copy=False)
	if not (obs_weight is None):
		obs_weight = np.array(obs_weight, dtype=np.float64, copy=False)
	if row_idx.max() >= m or col_idx.max() >= n:
		print row_idx.max(), col_idx.max(), m, n
		raise ValueError('row_idx or col_idx contains an index in wrong range')
	W = np.zeros((param.k, m), dtype=np.float64)
	H = np.zeros((param.k, n), dtype=np.float64)
	nnz = len(row_idx)
	libpmf.pmf_train(m, n, nnz, 
			row_idx.ctypes.data_as(POINTER(c_int)),
			col_idx.ctypes.data_as(POINTER(c_int)),
			obs_val.ctypes.data_as(POINTER(c_double)),
			param, 
			W.ctypes.data_as(POINTER(c_double)),
			H.ctypes.data_as(POINTER(c_double)))
	return {'W':W.T, 'H':H.T}
