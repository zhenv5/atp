import networkx as nx
def break_cycles_to_DAG(input_graph_name, deleted_edges_file):
	# remove cycle edges to reduce the input directed graph to a DAG
	if input_graph_name.endswith(".edges") or input_graph_name.endswith(".txt"):
		g = nx.read_edgelist(input_graph_name, nodetype = int, create_using = nx.DiGraph())
	elif input_graph_name.endswith(".gpickle"):
		g = nx.read_gpickle(input_graph_name)
	if deleted_edges_file:
		small_g = nx.read_edgelist(deleted_edges_file,nodetype = int, create_using=nx.DiGraph())
		g.remove_edges_from(small_g.edges())
		assert nx.is_directed_acyclic_graph(g)
		output_file_name = input_graph_name.split(".")[0] + "_DAG.edges"
		nx.write_edgelist(g,output_file_name,data = False)
	else:
		output_file_name = input_graph_name
	print("Corresponding DAG is saved at: %s" % output_file_name)
	return output_file_name

import argparse
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--original_graph",default= " ", help = "input directed graph (with cycles) (format can be *.gpickle, *.edges)")
	parser.add_argument("--deleted_edges",default= None, help = "edges to be deleted to reduce a directed graph to a DAG (format can be *.edges/txt)")
	args = parser.parse_args()
	break_cycles_to_DAG(args.original_graph,args.deleted_edges)


