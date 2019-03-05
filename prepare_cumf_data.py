
# coding: utf-8


import os
import os.path
import pandas as pd
from six.moves import urllib
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
from scipy.sparse import coo_matrix
from scipy import sparse
from sklearn.cross_validation import train_test_split

def load_data(input_filename,unpack = True,value_type = np.float32,test_size = 1):
    user,item,rating = np.loadtxt(input_filename, dtype = value_type,unpack = unpack)
    user = np.int32(user)
    item = np.int32(item)

    assert len(user) == len(item) and len(item) == len(rating)
    
    min_user_id = np.min(user)
    max_user_id = np.max(user)
    min_item_id = np.min(item)
    max_item_id = np.max(item)
    min_value = np.min(rating)
    max_value = np.max(rating)

    print("min user id: %d, max user id: %d" % (min_user_id,max_user_id))
    print("min item id: %d, max item id: %d" % (min_item_id,max_item_id))
    print("min value: %0.6f, max value: %0.6f" % (min_value,max_value))
    print("# instances: %d" % len(user))

    user_item = np.vstack((user, item))

    start_index = min(min_user_id,min_item_id)

    m = max(max_user_id,max_item_id) - start_index + 1

    n = m

    print("m,n = %d" % m)

    user_item_train, user_item_test, rating_train, rating_test = train_test_split(user_item.T, rating, test_size=test_size, random_state=42)
    if test_size >=1:
        nnz_train = len(rating) - test_size
        nnz_test = test_size
    else:
        nnz_test = int(test_size * len(rating))
        nnz_train = len(rating) - test_size
        assert len(rating) == (nnz_train + nnz_test)
    print("# training instances: %d" % len(user_item_train))
    print("# testing instances: %d" % len(rating_test))

    import os.path
    dir_name,_ = dir_tail_name(input_filename)
    with open(os.path.join(dir_name,"meta_modified_all"),"w") as f:
        f.write(str(m) + " " + str(n) + " \n")
        f.write(str(nnz_train) + " \n")
        names = ["R_train_coo.data.bin","R_train_coo.row.bin","R_train_coo.col.bin",
        "R_train_csr.indptr.bin","R_train_csr.indices.bin","R_train_csr.data.bin",
        "R_train_csc.indptr.bin","R_train_csc.indices.bin","R_train_csc.data.bin"]
        for name in names:
            f.write(name + " \n")
        f.write(str(nnz_test) + " test.ratings")
    return user_item_train, user_item_test, rating_train, rating_test,nnz_train,nnz_test,start_index,dir_name



def prepare_data(user_item_train, user_item_test, rating_train, rating_test,nnz_train,nnz_test,dir_name,start_index = 0):
    #for test data, we need COO format to calculate test RMSE
    #1-based to 0-based
    R_test_coo = coo_matrix((rating_test,(user_item_test[:,0] - start_index,user_item_test[:,1] - start_index)))
    assert R_test_coo.nnz == nnz_test
    R_test_coo.data.astype(np.float32).tofile(os.path.join(dir_name,'R_test_coo.data.bin'))
    R_test_coo.row.tofile(os.path.join(dir_name,'R_test_coo.row.bin'))
    R_test_coo.col.tofile(os.path.join(dir_name,'R_test_coo.col.bin'))

    with open(os.path.join(dir_name,"test.ratings"),"w") as f:
        for u,v,r in zip(user_item_test[:,0],user_item_test[:,1],rating_test):
            f.write(str(u-start_index)+" " + str(v-start_index) + " " + str(r)+" \n")


    print np.max(R_test_coo.data)
    print np.max(R_test_coo.row)
    print np.max(R_test_coo.col)
    print R_test_coo.data
    print R_test_coo.row
    print R_test_coo.col


    '''
    test_data = np.fromfile(os.path.join(dir_name,'R_test_coo.data.bin'),dtype=np.float32)
    test_row = np.fromfile(os.path.join(dir_name,'R_test_coo.row.bin'), dtype=np.int32)
    test_col = np.fromfile(os.path.join(dir_name,'R_test_coo.col.bin'),dtype=np.int32)
    print test_data[0:10]
    print test_row[0:10]
    print test_col[0:10]
    '''


    #1-based to 0-based
    R_train_coo = coo_matrix((rating_train,(user_item_train[:,0] - start_index,user_item_train[:,1] - start_index)))



    print R_train_coo.data
    print R_train_coo.row
    print R_train_coo.col
    print np.max(R_train_coo.data)
    print np.max(R_train_coo.row)
    print np.max(R_train_coo.col)



    #print np.unique(user).size
    print np.unique(R_train_coo.row + 1).size
    #print np.unique(item).size
    print np.unique(R_train_coo.col + 1).size

    print np.unique(R_test_coo.row + 1).size
    print np.unique(R_test_coo.col + 1).size



    np.min(R_test_coo.col)



    #for training data, we need COO format to calculate training RMSE
    #we need CSR format R when calculate X from \Theta
    #we need CSC format of R when calculating \Theta from X
    assert R_train_coo.nnz == nnz_train
    R_train_coo.row.tofile(os.path.join(dir_name,'R_train_coo.row.bin'))
    R_train_coo.data.astype(np.float32).tofile(os.path.join(dir_name,'R_train_coo.data.bin'))
    R_train_coo.col.tofile(os.path.join(dir_name,'R_train_coo.col.bin'))



    R_train_csr = R_train_coo.tocsr()
    R_train_csc = R_train_coo.tocsc()
    R_train_csr.data.astype(np.float32).tofile(os.path.join(dir_name,'R_train_csr.data.bin'))
    R_train_csr.indices.tofile(os.path.join(dir_name,'R_train_csr.indices.bin'))
    R_train_csr.indptr.tofile(os.path.join(dir_name,'R_train_csr.indptr.bin'))
    R_train_csc.data.astype(np.float32).tofile(os.path.join(dir_name,'R_train_csc.data.bin'))
    R_train_csc.indices.tofile(os.path.join(dir_name,'R_train_csc.indices.bin'))
    R_train_csc.indptr.tofile(os.path.join(dir_name,'R_train_csc.indptr.bin'))



    print R_train_csr.data
    print R_train_csr.indptr
    print R_train_csr.indices


def dir_tail_name(file_name):
    
    dir_name = os.path.dirname(file_name)
    head, tail = os.path.split(file_name)
    print("dir name: %s, file_name: %s" % (dir_name,tail))
    return dir_name,tail

def generate_cumf_input_files(input_filename):
    user_item_train, user_item_test, rating_train, rating_test,nnz_train,nnz_test,start_index,dir_name= load_data(input_filename)
    prepare_data(user_item_train, user_item_test, rating_train, rating_test,nnz_train,nnz_test,dir_name,start_index = start_index)
    

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",default= "cumf_ccd/dataset/test/demo.rating", help = "input file format: from_node source_node weight/value; output files will be saved in corresponding directory")
    args = parser.parse_args()
    generate_cumf_input_files(args.input)

