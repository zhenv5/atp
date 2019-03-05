import numpy as np
import argparse
import os.path

def load_cumf_WH(dir_name):
	W = np.loadtxt(os.path.join(dir_name,"W.txt"),dtype = np.float32)
	H = np.loadtxt(os.path.join(dir_name,"H.txt"),dtype =  np.float32)
	return W.T,H
	

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--dir",default= "cumf/cumf_ccd/dataset/test", help = "fold directory to store W.txt and H.txt")
	args = parser.parse_args()
	load_cumf_WH(args.dir)
