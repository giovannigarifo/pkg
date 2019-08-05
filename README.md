# RGCN

Implementation of an R-GCN based model for node embedding using [Deep Graph Library](https://www.dgl.ai/).

`data` folder containes in XML format the RDF description of the knowledge graphs used.


# Relational Graph Convolutional Network (RGCN)

RGCNs are a kind of GraphConvNet that operates on [knowledge graphs](https://en.wikipedia.org/wiki/Knowledge_Graph). 

The main difference with respect to classical GCNs that operates on graphs is that RGCNs operates on multigraphs with labeled edges.

## Link prediction

To run `link-predict.py` on 16GB of system memory with training and evaluation done on CPU with 8 threads available:

```
python3 -u rgcn-linkpredict.py --num-threads=7 --graph-perc="1.0" --eval-batch-size="40" --rdf-dataset-path="../data/anni2013-2017_with_img.xml" 2>&1 | tee output/out.log
```

on 32GB of system memory and GPU with 8GB of video memory, with training on GPU and evaluation on CPU with 8 threads available:

```
python3 -u rgcn-linkpredict.py --gpu=0 --num-threads=7 --graph-perc="1.0" --eval-batch-size="80" --graph-batch-size=20000 --rdf-dataset-path="../data/anni2013-2017_with_img.xml" 2>&1 | tee output/out.log
```
