# ATP: Directed Graph Embedding with Asymmetric Transitivity Preservation

## Required Packages


* networkx 
* numpy
* scipy
* pandas

## Optional Packages

Some packages for matrix factorization are optional. And you can use your own package to do the matrix factorization or simply use SVD (supported by numpy and scipy) to generate the embeddings.

* sklearn
* nimfa
* cumf_ccd (already included, please compile the code if you use it)
* libpmf (already included, please compile the code if you use it)

Check section ```Using other Matrix Factorization Algorithms``` for more details.

## Introduction

> Directed graphs have been widely used in Community Question Answering services (CQAs) to model asymmetric relationships among different types of nodes in CQA graphs, e.g., question, answer, user. Asymmetric transitivity is an essential property of directed graphs, since it can play an important role in downstream graph inference and analysis. Question difficulty and user expertise follow the characteristic of asymmetric transitivity. Maintaining such properties, while reducing the graph to a lower dimensional vector embedding space, has been the focus of much recent research. In this paper, we tackle the challenge of directed graph embedding with asymmetric transitivity preservation and then leverage the proposed embedding method to solve a fundamental task in CQAs: how to appropriately route and assign newly posted questions to users with the suitable expertise and interest in CQAs. The technique incorporates graph hierarchy and reachability information naturally by relying on a non-linear transformation that operates on the core reachability and implicit hierarchy within such graphs. Subsequently, the methodology levers a factorization-based approach to generate two embedding vectors for each node within the graph, to capture the asymmetric transitivity. Extensive experiments show that our framework consistently and significantly outperforms the state-of-the-art baselines on two diverse real-world tasks: link prediction, and question difficulty estimation and expert finding in online forums like Stack Exchange. Particularly, our framework can support inductive embedding learning for newly posted questions (unseen nodes during training), and therefore can properly route and assign these kinds of questions to experts in CQAs.
* The Thirty-Third AAAI Conference on Artificial Intelligence (AAAI 2019), acceptance rate: 1150/7095 = 16.2%
* [arXiv](https://arxiv.org/abs/1811.00839)
* [Slides for AAAI 2019 Presentation](https://www.dropbox.com/s/jk6auc7bvuw1dvb/Slides_AAAI_2019_ATP.pdf?dl=0)


## How does it work 


### First Step: Break Cycles

If the input directed graph is not a directed acyclic graph (DAG), we should delete some cycle edges to make it be a DAG.

Corresponding code is availabe at: [breaking_cycles_in_noisy_hierarchies](https://github.com/zhenv5/breaking_cycles_in_noisy_hierarchies)

Then use ```remove_cycle_edges_to_DAGs.py``` to save the corresponding DAG to a file.

For example, run:

* ```python remove_cycle_edges_to_DAGs.py --original_graph dataset/demo.edges --deleted_edges dataset/demo_deleted_edges.edges```

Corresponding DAG is saved at:

* ```dataset/demo_DAG.edges```

The DAG file (```dataset/demo_DAG.edges```) will be our input for generating embeddings. 

### Generate Embeddings

Given a DAG, we can run ```main_atp.py``` to generate the required embeddings.

Parameters of ```main_atp.py```:

* ```--dag```: input directed acyclic graph (DAG) (format can be *.gpickle, *.edges)
* ```--rank```: number of latent factors
* ```--strategy```: strategies to bulid hierarchical matrix: constant, linear, harmonic, ln (log)	
* ```--id_mapping```: 'Making Node ID start with 0', action='store_true'
* ```--using_GPU```: 'Using GPU to do the matrix factorization (cumf/cumf_ccd)', action='store_true'
* ```--dense_M```: 'Dense representation of M', action='store_true'
* ```--using_SVD```: 'Using SVD to generate embeddings from M', action='store_true'

Output:

* ```S``` is saved at: ```*_W.pkl```
* ```T``` is saved at: ```*_H.pkl```

Some examples:

Suppose we use CPU based matrix factorization to generate corresponding embeddigns, and nodes of ```demo_DAG.edges``` start with index 0, we run:

* ```python main_atp.py --dag dataset/demo_DAG.edges --rank 2 --strategy ln --dense_M```

We specify ```--id_mapping```, if nodes' id are not integers or do not start with index 0, For example, we run:

* ```python main_atp.py --dag dataset/demo_DAG_String.edges --rank 2 --strategy ln --id_mapping --dense_M```

If we would like to do some GPU based matrix factorization, we have to specify ```--using_GPU```:

* ```python main_atp.py --dag dataset/demo_DAG_String.edges --rank 2 --strategy ln --id_mapping --using_GPU```

Check [cumf_ccd](https://github.com/zhenv5/atp/tree/master/cumf/cumf_ccd) for more details about matrix factorization on GPU. We use ```prepare_cumf_data.py``` and ```load_cumf_ccd_matrices.py``` to prepare the inputs for ```cumf_ccd``` and process the outputs of ```cumf_ccd``` respectively.

## Using other Matrix Factorization Algorithms

```ATP``` can use other matrix factorization based methods easily. 

Simply modify ```graph_embedding.py``` to add new matrix factorization based methods.

For example:

We can use ```NMF``` from ```sklearn``` to generate corresponding embeddings:

```
from sklearn.decomposition import NMF
model = NMF(n_components= rank, init='random', random_state=0)
W = model.fit_transform(matrix)
H = model.components_
```

The input ```matrix``` is a dense matrix, so we have to specify ```--dense_M``` when we run ```main_atp.py```.

However, when we use ```libpmf``` to do the matrix factorization, the input matrix should use a sparse representation, it's not necessary for us to specify ```--dense_M```.


## Using SVD to generate embeddings

* Using dense representation of ```M```: ```python main_atp.py --dag dataset/demo_DAG.edges --rank 2 --using_SVD --dense_M```
* Using sparse representation of ```M```: ```python main_atp.py --dag dataset/demo_DAG.edges --rank 2 --using_SVD```

## Using ```M``` as an input for other applications

```M``` is the matrix which incorporates graph hierarchy and reachability. It's saved as ```train_ranking_differences.dat```.
You can use ```M``` as an input for other applications. 



## Datasets

There are three different types of datasets used in our paper:

* Synthetic datasets (randomly generated): See [breaking_cycles_in_noisy_hierarchies](https://github.com/zhenv5/breaking_cycles_in_noisy_hierarchies) for details
* Data from Stack Exchange sites: See [PyStack](https://github.com/zhenv5/PyStack) for more details
* Other datasets: Check our paper to access the download links

## Citation

If you use this code, please consider to cite ATP:

```
@article{DBLP:journals/corr/abs-1811-00839,
  author    = {Jiankai Sun and
               Bortik Bandyopadhyay and
               Armin Bashizade and
               Jiongqian Liang and
               P. Sadayappan and
               Srinivasan Parthasarathy},
  title     = {{ATP:} Directed Graph Embedding with Asymmetric Transitivity Preservation},
  journal   = {CoRR},
  volume    = {abs/1811.00839},
  year      = {2018},
  url       = {http://arxiv.org/abs/1811.00839},
  archivePrefix = {arXiv},
  eprint    = {1811.00839},
  timestamp = {Thu, 22 Nov 2018 17:58:30 +0100},
  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1811-00839},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

Download as ```bib``` file: [https://dblp.uni-trier.de/rec/bibtex/journals/corr/abs-1811-00839](https://dblp.uni-trier.de/rec/bibtex/journals/corr/abs-1811-00839)


