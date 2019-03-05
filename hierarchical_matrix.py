import networkx as nx
import argparse
import os.path
from infer_DAG_hierarchy import graph_hierarchy
import pandas as pd
import numpy as np 
import math 
Euler_Constant = 0.5772156649

def read_dict_from_file(file_name,key_type = int, value_type = int):
	input_file = open(file_name,"r")
	d = {}
	for line in input_file.readlines():
		k,v = line.split()
		if (key_type is not None) and (value_type is not None):
			try:
				k=key_type(k)
				v=value_type(v)
				d[k] = v
			except Exception as e:
				print e 
	return d

def Riemann_zeta_function(n,discount = 1):
	if n > 0:
		value = sum([1.0/((i+1.0)**discount) for i in xrange(n)])
		return value
	else:
		m = -n
		value = sum([1.0/((i+1.0)**discount) for i in xrange(m)])
		return -value

def build_hierarchical_matrix(original_graph,deleted_edges = None,connectivity_from_dag = True,is_dense_matrix = False,strategy = "harmonic"):
	rankings = graph_hierarchy(original_graph,deleted_edges = deleted_edges)
	pairs_ranking_difference = []

	if connectivity_from_dag:
		g=nx.read_edgelist(original_graph,nodetype = int, create_using=nx.DiGraph())
		number_of_nodes = g.number_of_nodes()
		print("number of nodes: %d" % number_of_nodes)
		if not nx.is_directed_acyclic_graph(g):
			small_g = nx.read_edgelist(deleted_edges,nodetype = int, create_using=nx.DiGraph())
			g.remove_edges_from(small_g.edges())
		assert nx.is_directed_acyclic_graph(g)

	discount = 1
	for node in g.nodes():
		if strategy == "constant" or strategy == "c":
			pairs_ranking_difference += [(node,des,1) for des in nx.descendants(g,node)]
		elif strategy == "linear" or strategy == "l":
			pairs_ranking_difference += [(node,des,rankings[des] - rankings[node]) for des in nx.descendants(g,node)]
		elif strategy == "harmonic" or strategy == "h":
			pairs_ranking_difference += [(node,des,Riemann_zeta_function(rankings[des] - rankings[node],discount = discount)) for des in nx.descendants(g,node)]
		elif strategy == "ln" or strategy == "log":
			pairs_ranking_difference += [(node,des, math.log(math.e + rankings[des] - rankings[node])) for des in nx.descendants(g,node)]

	sources = [s for (s,_,_) in pairs_ranking_difference]
	targets = [t for (_,t,_) in pairs_ranking_difference]
	scores = [score for (_,_,score) in pairs_ranking_difference]
	print("min: %0.8f, max: %0.8f" % (min(scores),max(scores)))
	max_node_id = max(max(sources),max(targets))
	number_of_nodes = max(max_node_id+1,number_of_nodes)
	print("# nodes in matrix: %d" % number_of_nodes)

	from tools import dir_tail_name
	dir_name,tail = dir_tail_name(original_graph)
	output_file = os.path.join(dir_name,"train_ranking_differences.dat")
	d = {"source_node":sources,"target_node":targets,"score":scores}
	df = pd.DataFrame.from_dict(d)
	df.to_csv(output_file,columns = ["source_node","target_node","score"],index = False,header = None,sep = " ")

	from scipy.sparse import coo_matrix
	from scipy.sparse import csr_matrix
	import scipy.sparse as sparse

	sparse_matrix = csr_matrix((scores,(sources,targets)),shape=(number_of_nodes,number_of_nodes))
	if is_dense_matrix:
		dense_matrix = sparse_matrix.toarray()
		print("is Sparse Matrix: %s" % isinstance(dense_matrix, sparse.spmatrix))
		return dense_matrix,output_file
	else:
		print("is Sparse Matrix: %s" % isinstance(sparse_matrix, sparse.spmatrix))
		return sparse_matrix,output_file
	
if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--original_graph",default= " ", help = "original graph (with cycles)")
	parser.add_argument("--deleted_edges",default= " ", help = "edges removed to get a DAG")
	args = parser.parse_args()
	build_hierarchical_matrix(args.original_graph,args.deleted_edges)

