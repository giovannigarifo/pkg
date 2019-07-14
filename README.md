# RGCN

Implementation of an R-GCN based model for node embedding using [Deep Graph Library](https://www.dgl.ai/).

`data` folder containes in XML format the RDF description of the knowledge graphs used.


# Relational Graph Convolutional Network (RGCN)

RGCNs are a kind of GraphConvNet that operates on [knowledge graphs](https://en.wikipedia.org/wiki/Knowledge_Graph). 

The main difference with respect to classical GCNs that operates on graphs is that RGCNs operates on multigraphs with labeled edges.

## Link prediction

To run `link-predict.py` on 16GB of system memory with training and evaluation done on CPU:

```
python3 rgcn-linkpredict.py --eval-batch-size="200" --rdf-dataset-path="../data/serialized.xml" 
```

on 32GB of system memory and GPU with 8GB of video memory, with training on GPU and evaluation on CPU:

```
python3 rgcn-linkpredict.py --gpu=0 --eval-batch-size=400 --graph-batch-size=20000 --rdf-dataset-path="../data/serialized.xml" 
```