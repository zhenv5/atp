import networkx as nx
import argparse
import os.path
try:
	import cPickle as pickle
except Exception as e:
	import pickle
from tools import dir_tail_name

def infer_dag_hierarchy(g):
	# input g: networkx, directed acyclic graph
	assert nx.is_directed_acyclic_graph(g)
	rankings = {}
	order = 0
	while(len(list(g.nodes())) > 0):	
		delted_nodes = []
		for n in g.nodes():
			pred = list(g.predecessors(n))
			if len(pred) == 0:
				delted_nodes.append(n)
				rankings[n] = order
		g.remove_nodes_from(delted_nodes)
		order += 1
	#print rankings
	return rankings

def graph_hierarchy(original_graph,deleted_edges = None):
	g = nx.read_edgelist(original_graph, nodetype = int, create_using = nx.DiGraph())
	if not nx.is_directed_acyclic_graph(g):
		small_g = nx.read_edgelist(deleted_edges,nodetype = int, create_using=nx.DiGraph())
		g.remove_edges_from(small_g.edges())
	rankings = infer_dag_hierarchy(g)
	dir_name,tail = dir_tail_name(original_graph)
	result_file = os.path.join(dir_name,tail.split(".")[0]+"_HGE_ranking" + ".pkl")	
	with open(result_file,"wb") as f:
		pickle.dump(rankings,f)
	return rankings
	
if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--original_graph",default= " ", help = "original graph (with cycles)")
	parser.add_argument("--deleted_edges",default= " ", help = "edges removed to get a DAG")
	parser.add_argument("--dag",default = " ", help = "DAG")
	
	args = parser.parse_args()

	if args.dag != " ":
		g=nx.read_edgelist(args.dag,nodetype = int, create_using=nx.DiGraph())	
		rankings = infer_dag_hierarchy(g)
	elif args.original_graph != " " and args.deleted_edges != " ":
		rankings = graph_hierarchy(args.original_graph,args.deleted_edges)

