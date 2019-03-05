import networkx as nx
import os.path
try:
	import cPikcle as pickle 
except Exception as e:
	import pickle

def nodeID_mapping(input_file_name,output_file_name = " ",reverse = False):
	if input_file_name.endswith(".edges") or input_file_name.endswith(".txt"):
		f = open(input_file_name,"r")
		g = nx.read_edgelist(f,create_using=nx.DiGraph(),nodetype = str,data = False)
		# print g.edges()[:10]
		f.close()
	elif input_file_name.endswith(".gpickle"):
		g = nx.read_gpickle(input_file_name)
	
	if output_file_name == " " or output_file_name == None:
		output_file_name = os.path.abspath(input_file_name).split(".")[0] + "_index0.edges"
	
	print("write graph edges list to: %s" % output_file_name)
	print("Original graph: # nodes: %d, # edges: %d" % (g.number_of_nodes(),g.number_of_edges()))

	id_mapping = {}
	i2s_mapping = {}

	index = 0

	for (u,v) in g.edges():

		if u not in id_mapping:
			id_mapping[u] = index
			i2s_mapping[index] = u
			index += 1

		if v not in id_mapping:
			id_mapping[v] = index
			i2s_mapping[index] = v 
			index += 1

	new_edges = [(id_mapping[u],id_mapping[v]) for (u,v) in g.edges()]

	new_g = nx.DiGraph()
	new_g.add_edges_from(new_edges)
	
	if reverse:
		print("edge reversed...")
		new_g.reverse(copy = False)
	
	print("New graph: # nodes: %d, # edges: %d" % (new_g.number_of_nodes(),new_g.number_of_edges()))
	nodes = list(new_g.nodes())
	print("New graph: min(node id): %d, max(node id):%d" % (min(nodes),max(nodes)))
	print("is Directed Acyclic Graph: %s " % nx.is_directed_acyclic_graph(new_g))
	
	nx.write_edgelist(new_g,output_file_name,data = False)
	
	print("# instances in mapping: %d (%d)" % (len(id_mapping),len(i2s_mapping)))
	mapping = {"s2i":id_mapping,"i2s":i2s_mapping}

	mapping_file = os.path.abspath(input_file_name).split(".")[0] + "_id_mapping.pkl"
	print("id mapping file is saved: %s" % mapping_file)
	print("mappged graph file is saved at: %s" % output_file_name)
	with open(mapping_file,"wb") as f:
		pickle.dump(mapping,f)
	return output_file_name,mapping_file

import argparse

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("-g","--graph",default= " ", help = "input graph file name")
	parser.add_argument("-o","--output",default = " ", help = "output graph file name")
	parser.add_argument("-r","--reverse",default = 0,type = int, help = "reverse edges direction in the output graph")
	args = parser.parse_args()
	if args.reverse == 0:
		reverse = False
	else:
		reverse = True

	nodeID_mapping(args.graph,args.output,reverse = reverse)
