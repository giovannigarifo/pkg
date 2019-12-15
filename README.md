# Deep Learning on Academic Knowledge Graphs

**Predicting new facts in a novel semantic graph built on top of the Politecnico di Torino scholarly data**

Implementation of an R-GCN based model for node embedding using [Deep Graph Library](https://www.dgl.ai/).

`data` folder contains in XML format the RDF description of the knowledge graphs used.


## Relational Graph Convolutional Network (R-GCN)

R-GCN is a kind of GraphConvNet that operates on [knowledge graphs](https://en.wikipedia.org/wiki/Knowledge_Graph). 

The main difference with respect to classical GCNs that operates on graphs is that R-GCN operates on multigraphs with labeled edges.


# How to run

On a Workstation with:

* 32GB of system memory
* 8GB of video memory

### Link predictor training

First train the link predictor.

* Training on GPU and evaluation on CPU:

```
python3 -u rgcn-linkpredict.py --job="train" --gpu=0 --num-threads=8 --graph-perc=1.0 --train-perc=0.9 --valid-perc=0.05 --test-perc=0.05 --eval-batch-size=80 --graph-batch-size=20000 --n-epochs=6000 --lr=0.001 --regularization=0.5 --evaluate-every=100 --rdf-graph-path="../data/anni2013-2017_no_img_7topics.xml" --load-dataset="input/pkg_dataset.pth" 2>&1 | tee output/output.log
```

### New links evaluation

Use the trained predictor to export candidate links to be added to the graph.

* Evaluation of candidate links to be used for the recommendation system:

```
python3 -u rgcn-linkpredict.py --job="eval" --gpu=-1 --num-threads=8 --graph-perc="1.0" --eval-batch-size="30" --graph-batch-size="10000" --load-model-state="model_state.pth" --load-data="input/pkg_dataset.pth" --num-scored-triples="30" 2>&1 | tee output/output.log
```
