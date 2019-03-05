#import os.path
#from os.path import dirname, abspath
#from os.path import join

#import numpy as np
#import pandas as pd
#import nimfa
#from scipy.sparse import csr_matrix

import libpmf
#import os.path
try:
    import cPickle as pickle
except Exception as e:
    import pickle 
import numpy as np
'''
def dir_tail_name(file_name):
    dir_name = os.path.dirname(file_name)
    head, tail = os.path.split(original_graph)
    print("dir name: %s, file_name: %s" % (dir_name,tail))
    return dir_name,tail
'''

def load_matrix(matrix_file_name):
    with open(matrix_file_name,"rb") as f:
        matrix = pickle.load(f)
        print("load matrix from: %s" % matrix_file_name)
        return matrix

def pmf(V, k = 30):
    model = libpmf.train(V, '-k ' + str(k) + ' -l 0.1 -t 100 -T 5 -n 12')
    '''
    [Usage]: omp-pmf-train [options] data_dir [model_filename]
options:
    -s type : set type of solver (default 0)
       0 -- CCDR1 with a fundec stopping condition
    -k rank : set the rank (default 10)
    -n threads : set the number of threads (default 4)
    -l lambda : set the regularization parameter lambda (default 0.1)
    -t max_iter: set the number of iterations (default 5)
    -T max_iter: set the number of inner iterations used in CCDR1 (default 5)
    -e epsilon : set inner termination criterion epsilon of CCDR1 (default 1e-3)
    -p do_predict: do prediction or not (default 0)
    -q verbose: show information or not (default 0)
    -N do_nmf: do nmf (default 0)
    '''
    print model['W'].shape
    print model["H"].shape  
    #print np.dot(model['W'], model['H'].T)
    return model 
def save_model(model,output_file):
    with open(output_file,"wb") as f:
        pickle.dump(model,f)

def run_pmf(matrix_file_name,save_model_file,k = 30):
    #dir_name,tail = dir_tail_name(matrix_file_name)
   # save_model_file = os.path.join(dir_name,tail.split(".")[0]+"_WH.pkl")
    V = load_matrix(matrix_file_name)
    model = pmf(V, k = k)
    save_model(model,save_model_file)


import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix",default= " ", help = "input matrix file name (pikcle)")
    parser.add_argument("--model",default = "WH.pkl",help = "pickle file to save WH")
    parser.add_argument("-k","--rank",type = int, default = 30, help = "# latent dimensions (k)")

    args = parser.parse_args()
    #run_pmf(args.matrix,args.model,k = args.rank)
    #V = [[1,2,3],[4,5,6],[7,8,9]]
    #pmf(np.array(V))
    #run(args.input)
    #pmf(np.array(V))
    run_pmf(args.matrix,args.model,args.rank)

