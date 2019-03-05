try:
	import cPickle as pickle
except Exception as e:
	import pickle 
import numpy as np 
import argparse
import os.path
from tools import run_command

def embedding(original_graph,deleted_edges_file,rank = 30, is_dense_matrix = False,using_GPU = True, using_svd = False, strategy = "ln"):
	from hierarchical_matrix import  build_hierarchical_matrix
	matrix,ranking_difference_file = build_hierarchical_matrix(original_graph,deleted_edges_file,is_dense_matrix = is_dense_matrix, strategy = strategy)
	from tools import dir_tail_name
	dir_name,tail = dir_tail_name(original_graph)
	if not using_GPU:
		if is_dense_matrix:
			print("matrix shape: %s" % str(matrix.shape))
			if using_svd:
				W, s, V = np.linalg.svd(matrix, full_matrices=False)
				S = np.diag(s)
				H = np.dot(S,V)
				print("(SVD) W matrix shape: %s" % str(W.shape))
				print("(SVD) H matrix shape: %s" % str(H.shape))
				return W,H
			
			# from sklearn.decomposition import NMF
			# model = NMF(n_components= rank, init='random', random_state=0)
			# W = model.fit_transform(matrix)
			# H = model.components_
			
			from matrix_factorization import run_lsnmf,run_nmf
			W,H = run_nmf(matrix,rank = rank)
			print("(NMF) W matrix shape: %s" % str(W.shape))
			print("(NMF) H matrix shape: %s" % str(H.shape))
			# print np.matmul(W,H)
			return W,H
		else:
			if using_svd:
				import scipy
				W, s, V = scipy.sparse.linalg.svds(matrix,k = rank,)
				S = np.diag(s)
				H = np.dot(S,V)
				print("(SVDs) W matrix shape: %s" % str(W.shape))
				print("(SVDs) H matrix shape: %s" % str(H.shape))
				return W,H
			saved_matrix_file_name = os.path.join(dir_name,tail.split(".")[0]+"_HM.pkl")
			saved_WH_file_name = os.path.join(dir_name,tail.split(".")[0]+"_HM_WH.pkl")

			print("saved matrix file name: %s " % saved_matrix_file_name)
			with open(saved_matrix_file_name,"wb") as f:
				pickle.dump(matrix,f)
			command = "python libpmf-1.41/python/pmf_main.py --matrix " + saved_matrix_file_name + " --model " + saved_WH_file_name + " --rank " + str(rank)
			run_command(command)

			with open(saved_WH_file_name,"rb") as f:
				model = pickle.load(f)
				return model['W'],model['H'].T
	else:
		print("Preparing file for cumf_ccd...")
		from prepare_cumf_data import generate_cumf_input_files
		generate_cumf_input_files(ranking_difference_file)
		run_command("./cumf/cumf_ccd/ccdp_gpu -T 1 -t 100 -l 0.01 -k  " + str(rank) + " " + dir_name + " " + dir_name + "/test.ratings")
		from load_cumf_ccd_matrices import load_cumf_WH 
		W,H = load_cumf_WH(dir_name)
		return W,H

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--original_graph",default= " ", help = "original graph (with cycles)")
	parser.add_argument("--deleted_edges",default= " ", help = "edges removed to get a DAG")
	parser.add_argument("--output_file",default = "output/mf.emb", help = "store ranking difference between connected pairs")
	parser.add_argument("--rank",default = 64, help = "number of latent factors")
	args = parser.parse_args()

	embedding(args.original_graph,args.deleted_edges,rank = args.rank, output_file = args.output_file)