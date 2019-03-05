import os.path
from tools import run_command
try:
	import cPickle as pickle 
except Exception as e:
	import pickle

import pandas as pd
def hierarchical_graph_embedding(di_graph_file, deleted_edges_file = None, rank = 64, strategy = "harmonic", using_GPU = False, dense_M = True, using_SVD = False):
	abs_di_graph_file = os.path.abspath(di_graph_file)
	from graph_embedding import embedding
	# Using GPU: using_GPU = True
	# Using CPU: using_GPU = False, is_dense_matrix = True, using_svd = False/True
	W,H = embedding(abs_di_graph_file, deleted_edges_file = deleted_edges_file, rank = rank, is_dense_matrix = dense_M, using_GPU = using_GPU, using_svd = using_SVD, strategy = strategy)
	return W,H

def main(input_graph_name, rank = 64, nodeID_need_mapping = False, strategy = "Harmonic", using_GPU = False, dense_M = True, using_SVD = False):
	if nodeID_need_mapping:
		transformed_graph_file_name = os.path.abspath(input_graph_name).split(".")[0] + "_index0.edges"
		mapping_file = os.path.abspath(input_graph_name).split(".")[0] + "_id_mapping.pkl"
		if (not os.path.isfile(transformed_graph_file_name)) or (not os.path.isfile(mapping_file)):
			from nodeID_mapping import nodeID_mapping
			transformed_graph_file_name,mapping_file = nodeID_mapping(input_graph_name)
		with open(mapping_file,"rb") as f:
			mapping = pickle.load(f)
	else:
		transformed_graph_file_name = input_graph_name
	W,H = hierarchical_graph_embedding(transformed_graph_file_name, deleted_edges_file = None, rank = rank, strategy = strategy, using_GPU = using_GPU, dense_M = dense_M, using_SVD = using_SVD)
	transformed_W = {}
	transformed_H = {}
	for index,(w,h) in enumerate(zip(W,H.T)):
		if nodeID_need_mapping:
			transformed_W[mapping["i2s"][index]] = w
			transformed_H[mapping["i2s"][index]] = h
			# print mapping["i2s"][index], transformed_W[mapping["i2s"][index]]
			# print mapping["i2s"][index], transformed_H[mapping["i2s"][index]]
		else:
			transformed_W[index] = w 
			transformed_H[index] = h
	W_file = os.path.abspath(input_graph_name).split(".")[0] + "_r" + str(rank)  + "_" + strategy + "_W.pkl"
	H_file = os.path.abspath(input_graph_name).split(".")[0] + "_r" + str(rank)  + "_" + strategy + "_H.pkl"

	print("S is saved at: %s, T is saved at: %s" % (W_file,H_file))
	with open(W_file,"wb") as f:
		pickle.dump(transformed_W,f)
	with open(H_file,"wb") as f:
		pickle.dump(transformed_H,f)
	'''
	transformed_W: dict, key = nodeID, value = latent vector with k = rank dimensions
	transformed_H: dict, key = nodeID, value = latent vector with k = rank dimensions
	'''
	return transformed_W,transformed_H

import argparse
if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--dag",default= " ", help = "input directed acyclic graph (DAG) (format can be *.gpickle, *.edges)")
	parser.add_argument("--rank", type = int, default = 64, help = "number of latent factors")
	parser.add_argument("--strategy",default= "ln", help = "strategies to bulid hierarchical matrix: constant, linear, harmonic/ln")	
	parser.add_argument('--id_mapping', help='Making Node ID start with 0', action='store_true')
	parser.add_argument('--using_GPU', help='Using GPU to do the matrix factorization (cumf/cumf_ccd)', action='store_true')
	parser.add_argument('--dense_M', help='Dense representation of M', action='store_true')
	parser.add_argument('--using_SVD', help='Using SVD to generate embeddings from M', action='store_true')
	args = parser.parse_args()
	main(args.dag, rank = args.rank, strategy = args.strategy, nodeID_need_mapping = args.id_mapping, using_GPU = args.using_GPU, dense_M = args.dense_M, using_SVD = args.using_SVD)


