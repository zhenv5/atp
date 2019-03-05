import nimfa
import numpy as np
import scipy.sparse as sp


def __fact_factor(X):
    """
    Return dense factorization factor, so that output is printed nice if factor is sparse.
    
    :param X: Factorization factor.
    :type X: :class:`scipy.sparse` of format csr, csc, coo, bsr, dok, lil, dia or :class:`numpy.matrix`
    """
    return X.todense() if sp.isspmatrix(X) else X

def print_info(fit, idx=None):
    """
    Print to stdout info about the factorization.
    
    :param fit: Fitted factorization model.
    :type fit: :class:`nimfa.models.mf_fit.Mf_fit`
    :param idx: Name of the matrix (coefficient) matrix. Used only in the multiple NMF model. Therefore in factorizations 
                that follow standard or nonsmooth model, this parameter can be omitted. Currently, SNMNMF implements 
                multiple NMF model.
    :type idx: `str` with values 'coef' or 'coef1' (`int` value of 0 or 1, respectively) 
    """
    print("=================================================================================================")
    print("Factorization method:", fit.fit)
    print("Initialization method:", fit.fit.seed)
    #print("Basis matrix W: ")
    #print(__fact_factor(fit.basis()))
    print("Mixture (Coefficient) matrix H%d: " % (idx if idx != None else 0))
    #print(__fact_factor(fit.coef(idx)))
    #print("Matrix Reconstruction...\n")
    #print(__fact_factor(np.matmul(fit.basis(),fit.coef(idx))))
    print("Distance (Euclidean): ", fit.distance(metric='euclidean', idx=idx))
    # We can access actual number of iteration directly through fitted model.
    # fit.fit.n_iter
    print("Actual number of iterations: ", fit.summary(idx)['n_iter'])
    # We can access sparseness measure directly through fitted model.
    # fit.fit.sparseness()
    print("Sparseness basis: %7.4f, Sparseness mixture: %7.4f" % (fit.summary(idx)['sparseness'][0], fit.summary(idx)['sparseness'][1]))
    # We can access explained variance directly through fitted model.
    # fit.fit.evar()
    print("Explained variance: ", fit.summary(idx)['evar'])
    # We can access residual sum of squares directly through fitted model.
    # fit.fit.rss()
    print("Residual sum of squares: ", fit.summary(idx)['rss'])
    # There are many more ... but just cannot print out everything =] and some measures need additional data or more runs
    # e.g. entropy, predict, purity, coph_cor, consensus, select_features, score_features, connectivity
    print("=================================================================================================")
    return fit.basis(),fit.coef(idx)

def run_lsnmf(V,rank = 12, max_iter = 5000):
    """
    Run least squares nonnegative matrix factorization.
    
    :param V: Target matrix to estimate.
    :type V: :class:`numpy.matrix`
    """
    rank = rank
    lsnmf = nimfa.Lsnmf(V, seed="random_vcol", rank=rank, max_iter=max_iter, sub_iter=10,
                        inner_sub_iter=10, beta=0.1, min_residuals=1e-5)
    fit = lsnmf()
    return print_info(fit)


def run_nmf(V,rank = 12, max_iter = 5000):
    """
    Run standard nonnegative matrix factorization.
    
    :param V: Target matrix to estimate.
    :type V: :class:`numpy.matrix`
    """
    # Euclidean
    
    nmf = nimfa.Nmf(V, seed="random_vcol", rank=rank, max_iter=max_iter, update='euclidean',
                      objective='fro')
    fit = nmf()
    print_info(fit)
    # divergence
    nmf = nimfa.Nmf(V, seed="random_vcol", rank=rank, max_iter=max_iter, initialize_only=True,
                    update='divergence', objective='div')
    fit = nmf()
    return print_info(fit)