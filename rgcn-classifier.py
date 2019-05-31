import rdftodata as rtd
from dgl import DGLGraph
import dgl.function as fn
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.contrib.data import load_data
import numpy as np
import torch
import logging, sys

class RGCNLayer(nn.Module):
    '''
    For each node, an R-GCN layer performs the following steps:
    * Compute outgoing message using node representation and weight matrix associated with the edge type (message function)
    * Aggregate incoming messages and generate new node representations (reduce and apply function)
    '''
    def __init__(self, in_feat, out_feat, num_rels, num_bases=-1, bias=None, activation=None, is_input_layer=False):
        
        super(RGCNLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.bias = bias
        self.activation = activation
        self.is_input_layer = is_input_layer

        # sanity check
        if self.num_bases <= 0 or self.num_bases > self.num_rels:
            self.num_bases = self.num_rels

        # weight bases in equation (3)
        self.weight = nn.Parameter(torch.Tensor(self.num_bases, self.in_feat,
                                                self.out_feat))
        if self.num_bases < self.num_rels:
            # linear combination coefficients in equation (3)
            self.w_comp = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases))

        # add bias
        if self.bias:
            self.bias = nn.Parameter(torch.Tensor(out_feat))

        # init trainable parameters
        nn.init.xavier_uniform_(self.weight,
                                gain=nn.init.calculate_gain('relu'))
        if self.num_bases < self.num_rels:
            nn.init.xavier_uniform_(self.w_comp,
                                    gain=nn.init.calculate_gain('relu'))
        if self.bias:
            nn.init.xavier_uniform_(self.bias,
                                    gain=nn.init.calculate_gain('relu'))

    def forward(self, g):
        if self.num_bases < self.num_rels:
            # generate all weights from bases
            weight = self.weight.view(self.in_feat, self.num_bases, self.out_feat)
            weight = torch.matmul(self.w_comp, weight).view(self.num_rels,
                                                        self.in_feat, self.out_feat)
        else:
            weight = self.weight

        if self.is_input_layer:
            def message_func(edges):
                # for input layer, matrix multiply can be converted to be
                # an embedding lookup using source node id
                embed = weight.view(-1, self.out_feat)
                index = edges.data['rel_type'] * self.in_feat
                index = index.float() + edges.src['id'].float()
                index = torch.Tensor(index).long()
                return {'msg': embed[index] * edges.data['norm']}
        else:
            def message_func(edges):
                w = weight[(edges.data['rel_type']).long()]
                msg = torch.bmm(edges.src['h'].unsqueeze(1), w).squeeze()
                msg = msg * edges.data['norm']
                return {'msg': msg}

        def apply_func(nodes):
            h = nodes.data['h']
            if self.bias:
                h = h + self.bias
            if self.activation:
                h = self.activation(h)
            return {'h': h}

        g.update_all(message_func, fn.sum(msg='msg', out='h'), apply_func)


class Model(nn.Module):
    def __init__(self, num_nodes, h_dim, out_dim, num_rels,
                 num_bases=-1, num_hidden_layers=1):
        super(Model, self).__init__()
        self.num_nodes = num_nodes
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.num_hidden_layers = num_hidden_layers

        # create rgcn layers
        self.build_model()

        # create initial features
        self.features = self.create_features()

    def build_model(self):
        self.layers = nn.ModuleList()
        # input to hidden
        i2h = self.build_input_layer()
        self.layers.append(i2h)
        # hidden to hidden
        for _ in range(self.num_hidden_layers):
            h2h = self.build_hidden_layer()
            self.layers.append(h2h)
        # hidden to output
        h2o = self.build_output_layer()
        self.layers.append(h2o)

    # initialize feature for each node
    def create_features(self):
        features = torch.arange(self.num_nodes)
        return features

    def build_input_layer(self):
        return RGCNLayer(self.num_nodes, self.h_dim, self.num_rels, self.num_bases,
                         activation=F.relu, is_input_layer=True)

    def build_hidden_layer(self):
        return RGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                         activation=F.relu)

    def build_output_layer(self):
        return RGCNLayer(self.h_dim, self.out_dim, self.num_rels, self.num_bases,
                         activation=partial(F.softmax, dim=1))

    def forward(self, g):
        if self.features is not None:
            g.ndata['id'] = self.features
        for layer in self.layers:
            layer(g)
        return g.ndata.pop('h')


def main():

    # setup logging
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

    #load data structures
    pubData = rtd.rdfToData() # use default filepath
    num_nodes = pubData.num_nodes
    num_relations = pubData.num_relations
    num_labels = pubData.num_labels
    labels = pubData.labels
    edges_sources = pubData.edges_sources
    edges_destinations = pubData.edges_destinations
    edges_relations = pubData.edges_relations
    edges_norms = pubData.edges_norms
    
    # split nodes intro train and test sets
    nodes_indexes = np.array(list(range(num_nodes)))
    train_nodes = nodes_indexes
    assert len(nodes_indexes) == (len(train_nodes))
    print("\ntrain_nodes:")
    print("- type:", type(train_nodes))
    print("- shape:", train_nodes.shape)

    # build edge and label tensors
    edge_type = (torch.as_tensor(np.array(edges_relations, dtype=np.float32), dtype=torch.float32)) \
        .float()
    edge_norm = (torch.as_tensor(np.array(edges_norms, dtype=np.float32), dtype=torch.float32)) \
        .float() \
        .unsqueeze(1)
    labels = (torch.as_tensor(np.array(labels, dtype=np.float32), dtype=torch.float32)) \
        .long()
    print("\nedge_type:")
    print("- type:", type(edge_type))
    print("- shape:", edge_type.shape)
    print("\nedge_norm:")
    print("- type:", type(edge_norm))
    print("- shape:", edge_norm.shape)
    print("\nlabels:")
    print("- type:", type(labels))
    print("- shape:", labels.shape)

    # configuration
    n_hidden = 16
    n_bases = -1 # don't use basis decomposition
    n_hidden_layers = 0
    n_epochs = 45 # number of epochs to train
    lr = 0.01 # learning rate
    l2norm = 0 # L2 norm coefficient

    # create graph
    graph = DGLGraph()
    graph.add_nodes(num_nodes)
    graph.add_edges(edges_sources, edges_destinations)
    graph.edata.update({'rel_type':edge_type, 'norm': edge_norm})

    # create model
    model = Model(num_nodes, n_hidden, num_labels, num_relations, num_bases = n_bases, num_hidden_layers=n_hidden_layers)

    # training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2norm)
    logging.debug("\n---------------------\n- Starting Training -\n---------------------\n")
    
    embeddings = 0 # save the last logits obtained by the training and use them as nodes embeddings
    model.train() # explicitly set training mode
    for epoch in range(n_epochs):
       
        optimizer.zero_grad() # zero the gradients before calculating the new one
        logits = model.forward(graph) # perform forward pass, get raw output
        loss = F.cross_entropy(logits, labels) # compute loss (only training nodes)
        loss.backward() # compute gradients
        optimizer.step() # update the parameters

        train_accuracy = torch.sum(logits.argmax(dim=1) == labels)
        train_accuracy = train_accuracy.item() / len(train_nodes)

        print("Epoch {:05d} | ".format(epoch) +
          "Train Accuracy: {:.4f} | Train Loss: {:.4f} | ".format(
              train_accuracy, loss.item()))

        if(epoch+1 == n_epochs):
            embeddings = logits

    logging.debug("\n---------------------\n- Training Finished -\n---------------------\n")
    logging.debug("embeddings:")
    print(embeddings)

if __name__ == "__main__":
    main()