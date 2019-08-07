import argparse
import logging

import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# for data import
import sys
sys.path.append('..')
import rdftodata

# DGL
from layers import RGCNBlockLayer as RGCNLayer
from model import BaseRGCN
import utils


class EmbeddingLayer(nn.Module):
    '''
    Creates an embedding lookup table (it's a wrapper around torch.nn.Embedding)

    Lookup table is like: node_index -> node_embedding

    The `forward()` method allows to perform the table lookup for all sampled nodes. It actually calls
    the forward method of the Embedding module, passing as argument the list of nodes for the lookup.
    '''
    def __init__(self, num_nodes, h_dim):
        super(EmbeddingLayer, self).__init__()
        self.embedding = torch.nn.Embedding(num_nodes, h_dim)

    def forward(self, g):
        '''
        Perform lookup of embeddings for all sampled nodes
        '''
        node_id = g.ndata['id'].squeeze() # build list of nodes id [0, 1, 2...num_sampled_nodes]
        g.ndata['h'] = self.embedding(node_id) # build embeddings tensor [[1st_sampled_node_features],...]


class RGCN(BaseRGCN):
    '''
    Defines the RGCN encoder model. Extends BaseRGCN Module which defined the forward method.
    '''
    def build_input_layer(self):
        '''
        Input layer of RGCN encoder can be seen as a table lookup if one hot encoded vectors
        are used as input features.
        '''
        return EmbeddingLayer(self.num_nodes, self.h_dim)

    def build_hidden_layer(self, idx):
        act = F.relu if idx < self.num_hidden_layers - 1 else None
        return RGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                         activation=act, self_loop=True, dropout=self.dropout)


class LinkPredict(nn.Module):
    '''
    Module to perform link prediction task on unknown triplets (s,r,o)
    '''
    def __init__(self, in_dim, h_dim, num_rels, num_bases=-1,
                 num_hidden_layers=1, dropout=0, use_cuda=False, reg_param=0):
        '''
        Called with this parameters:
                in_dim=num_nodes, h_dim=args.n_hidden, num_rels=num_rels, num_bases=args.n_bases,
                num_hidden_layers=args.n_layers, dropout=args.dropout, use_cuda=use_cuda, reg_param=args.regularization
        '''
        super(LinkPredict, self).__init__()

        # build the entity encoder (with out_dim=h_dim)
        self.rgcn = RGCN(in_dim, h_dim, h_dim, num_rels * 2, num_bases,
                         num_hidden_layers, dropout, use_cuda)
        
        # regularization parameter
        self.reg_param = reg_param
        
        # build relations weights
        self.w_relation = nn.Parameter(torch.Tensor(num_rels, h_dim))
        
        # initialize relations weights
        nn.init.xavier_uniform_(self.w_relation,
                                gain=nn.init.calculate_gain('relu'))

    def calc_score(self, embedding, triplets):
        '''
        DistMult implementation to score triples
        '''
        s = embedding[triplets[:,0]]
        r = self.w_relation[triplets[:,1]]
        o = embedding[triplets[:,2]]
        score = torch.sum(s * r * o, dim=1)
        return score

    def forward(self, g):
        '''
        Perform entity encoding using RGCN to obtain embeddings
        '''
        return self.rgcn.forward(g)

    def evaluate(self, g):
        ''' 
        get embedding and relation weights
        '''
        embedding = self.forward(g)
        return embedding, self.w_relation

    def regularization_loss(self, embedding):
        '''
        Implementation of the regularization loss to be added to prediction loss
        '''
        return torch.mean(embedding.pow(2)) + torch.mean(self.w_relation.pow(2))

    def get_loss(self, g, triplets, labels):
        '''
        triplets is a list of data samples (positive and negative, they're 330'000)
        each row in the triplets is a 3-tuple of (source, relation, destination)

        The loss function compares the obtained score with the labels: the labels
        are equal to 1 for positive samples and to 0 for the negative ones. 

        The model (the parameters w_relation) will be adjusted to **score** a negative 
        sample near 0 and a positive sample near 1.
        '''        
        embedding = self.forward(g) # calc embedding from RGCN encoder
        print("- embedding: embedding.shape=", embedding.shape, " embedding.type=", type(embedding))

        score = self.calc_score(embedding, triplets) # DistMult
        print("- score: score.shape=", score.shape, " score.type=", type(score))

        print("- labels: labels.shape=", labels.shape, " labels.type=", type(labels))
        predict_loss = F.binary_cross_entropy_with_logits(score, labels)
        reg_loss = self.regularization_loss(embedding)
        
        return predict_loss + self.reg_param * reg_loss # regularization to avoid overfitting


#----------#
# - Main - #
#----------#
def main(args):

    # load data required for the prediction task
    if(args.rdf_dataset_path):
        publications_data = rdftodata.rdfToData(args.rdf_dataset_path, args.graph_perc, "link-prediction")
    else: 
        publications_data = rdftodata.rdfToData(job="link-prediction")
    
    num_nodes = publications_data.num_nodes
    train_data = publications_data.train_triples # triples used for training
    valid_data = publications_data.valid_triples # triples used for validation
    test_data = publications_data.test_triples # triples used for test
    num_rels = publications_data.num_relations

    # Convert T,V,T data to correct type
    train_data = np.array(publications_data.train_triples)
    valid_data = torch.as_tensor(valid_data, dtype=torch.long) #as_tensor doesn't crate a copy
    test_data = torch.as_tensor(test_data, dtype=torch.long)

    # set CUDA if requested and available
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(args.gpu)
        logging.debug("CUDA activated for GPU {}".format(args.gpu))
    else: logging.debug("CUDA not available.")

    # set CPU threads to be used
    torch.set_num_threads(args.num_threads)
    logging.debug("CPU threads that will be used: {t}".format(t=torch.get_num_threads()))

    # create model
    model = LinkPredict(num_nodes,
                        args.n_hidden,
                        num_rels,
                        num_bases=args.n_bases,
                        num_hidden_layers=args.n_layers,
                        dropout=args.dropout,
                        use_cuda=use_cuda,
                        reg_param=args.regularization)

    # debug prints
    print("\n----------------------------------------")
    print("- Input data before creating DGL graph -\n")
    print("num_nodes: ", num_nodes)
    print("num_rels: ", num_rels)
    print("train_data: shape=", train_data.shape, "type=", type(train_data))
    print("valid_data: shape=", valid_data.shape, "type=", type(valid_data))
    print("test_data: shape=", test_data.shape, "type=", type(test_data))
    print("----------------------------------------\n")

    # build test (full) graph: used for validation.
    # 
    # Starting from train_data, build a DGL graph, add inverse relations.
    # returns the DGL graph, an ndarray where indexes are edges ID and values are relation type
    test_graph, test_rel, test_norm = utils.build_test_graph(
        num_nodes, num_rels, train_data)

    # get in-degrees of all nodes in test_graph
    test_deg = test_graph.in_degrees(range(test_graph.number_of_nodes())) \
                .float() \
                .view(-1,1)

    test_node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)
    test_rel = torch.from_numpy(test_rel)
    test_norm = torch.from_numpy(test_norm).view(-1, 1)

    test_graph.ndata.update({'id': test_node_id, 'norm': test_norm}) # add test_norm for each node
    test_graph.edata['type'] = test_rel # add relation type for each edge

    # at this point the DGL graph contains all nodes, all edges, and all edges have the associated relation type

    # debug prints
    print("\n---------------------")
    print("- DGL graph created -\n")
    print("test_graph: number_of_nodes=",test_graph.number_of_nodes(), \
          "number_of_edges=", test_graph.number_of_edges())    
    print("test_node_id: shape=", test_node_id.shape," type=", type(test_node_id))
    print("test_rel: shape=", test_rel.shape," type=", type(test_rel))
    print("test_norm: shape=", test_norm.shape," type=", type(test_norm))
    print("---------------------\n")

    # set cuda
    if use_cuda:
        model.cuda()

    # build adj list and calculate degrees for sampling
    adj_list, degrees = utils.get_adj_and_degrees(num_nodes, train_data)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    model_state_file = 'model_state.pth'
    forward_time = []
    backward_time = []

    print("*************************")
    print("Starting training loop...")
    print("*************************\n\n")

    epoch = 0
    best_mrr = 0
    loss_list = []

    while True:
        model.train() # set training mode explicitly
        epoch += 1
        
        print("-----------")
        print("Epoch #{n}".format(n=epoch))
        print("-----------")

        # ----------------------------------
        # 1) Prepare sampled training graph
        # ----------------------------------

        # perform edge neighborhood sampling to generate training graph and data.
        # Also add negative samples.
        #`data` will contain the triplets (both positive and negatives), and `labels`
        # will contain the corresponding labels (1 for positive samples, 0 for negative samples)
        sampled_graph, node_id, edge_type, node_norm, data, labels = utils\
            .generate_sampled_graph_and_labels(
                train_data, args.graph_batch_size, args.graph_split_size,
                num_rels, adj_list, degrees, args.negative_sample)
        
        # to tensors
        node_id = torch.from_numpy(node_id).view(-1, 1) 
        edge_type = torch.from_numpy(edge_type)
        node_norm = torch.from_numpy(node_norm).view(-1, 1)
        data, labels = torch.from_numpy(data), torch.from_numpy(labels) 
        deg = sampled_graph.in_degrees(range(sampled_graph.number_of_nodes())).float().view(-1, 1)
        
        if use_cuda: # set cuda tensors if available
            node_id, deg = node_id.cuda(), deg.cuda()
            edge_type, node_norm = edge_type.cuda(), node_norm.cuda()
            data, labels = data.cuda(), labels.cuda()
        
        # set norm for each node and set relation type for each edge of the training graph
        sampled_graph.ndata.update({'id': node_id, 'norm': node_norm})
        sampled_graph.edata['type'] = edge_type

        # ----------------------------------
        # 2) train the model and get loss
        # ----------------------------------
        print("/#/ Perform forward propagation...")
        t0 = time.time()
        loss = model.get_loss(sampled_graph, data, labels) # actually calls model.forward()
        t1 = time.time()
        print("...done\n")

        print("/#/ Perform backpropagation...")
        loss.backward() # calc the gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm) # clip gradients
        optimizer.step() # update weights
        t2 = time.time()
        print("...done\n")

        forward_time.append(t1 - t0)
        backward_time.append(t2 - t1)
        print("- Stats: Epoch {:04d} | Loss {:.4f} | Best MRR {:.4f} | Forward {:.4f}s | Backward {:.4f}s \n".
              format(epoch, loss.item(), best_mrr, forward_time[-1], backward_time[-1]))

        # save loss behavior
        loss_list.append(loss.item())

        optimizer.zero_grad() # zeroes the gradients for next training iteration
        
        # validation: evaluate over the test graph
        if epoch % args.evaluate_every == 0:

            print("/#/ Perform validation...".format(e=epoch))
           
            if use_cuda:
                model.cpu() # perform validation on CPU because full graph is too large

            model.eval() # set evaluation mode explicitly
            mrr, _ = utils.evaluate(epoch,
                                test_graph,
                                model,
                                valid_data,
                                num_nodes,
                                hits=[1, 3, 10], 
                                eval_bz=args.eval_batch_size)
            
            # save best model
            if mrr < best_mrr:
                if epoch >= args.n_epochs:
                    break
            else:
                best_mrr = mrr
                torch.save({'state_dict': model.state_dict(), 'epoch': epoch},
                           model_state_file)
            if use_cuda:
                model.cuda() # activate again GPU


    print("**************")
    print("Training done!")
    print("**************\n")
    print("Mean forward time: {:4f}s".format(np.mean(forward_time)))
    print("Mean Backward time: {:4f}s\n".format(np.mean(backward_time)))

    print("/#/ Perform evaluation of the best model found on test data...")
    # use best model checkpoint
    checkpoint = torch.load(model_state_file)
    if use_cuda:
        model.cpu() # test on CPU
    model.eval()
    model.load_state_dict(checkpoint['state_dict'])
    print("Using best epoch on test data: {}".format(checkpoint['epoch']))

    mrr_test, score_list = utils.evaluate("best_on_test_data", test_graph, model, test_data, num_nodes, hits=[1, 3, 10],
                    eval_bz=args.eval_batch_size, 
                    id_to_node_uri_dict=publications_data.id_to_node_uri_dict, # export scores
                    id_to_rel_uri_dict=publications_data.id_to_rel_uri_dict)

    # retrieve score for triplets paper-subject-topic
    utils.analyze_test_results(mrr_test, test_data, score_list,
        id_to_node_uri_dict=publications_data.id_to_node_uri_dict,
        id_to_rel_uri_dict=publications_data.id_to_rel_uri_dict)

    # print loss_list for debug
    print("Loss values for each training epoch:")
    for index, l in enumerate(loss_list):
        print("Epoch {e} loss: {l}".format(e=index, l=l))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RGCN')
    
    parser.add_argument("--rdf-dataset-path", type=str, help="path to RDF dataset to use")
    parser.add_argument("--gpu", type=int, default=-1, help="gpu")
    parser.add_argument("--num-threads", type=int, default=4, help="number of threads to be used for CPU computation")

    parser.add_argument("--dropout", type=float, default=0.2, help="dropout probability")
    parser.add_argument("--n-hidden", type=int, default=500, help="number of hidden units")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--n-bases", type=int, default=100, help="number of weight blocks for each relation")
    parser.add_argument("--n-layers", type=int, default=2, help="number of propagation rounds")
    parser.add_argument("--n-epochs", type=int, default=6000, help="number of minimum training epochs")
    parser.add_argument("--eval-batch-size", type=int, default=500, help="batch size when evaluating")
    parser.add_argument("--regularization", type=float, default=0.01, help="regularization weight")
    parser.add_argument("--grad-norm", type=float, default=1.0, help="norm to clip gradient to")
    parser.add_argument("--graph-batch-size", type=int, default=30000, help="number of edges to sample in each training epoch")
    parser.add_argument("--graph-split-size", type=float, default=0.5, help="portion of sampled edges (see graph-batch-size) used as positive sample")
    parser.add_argument("--negative-sample", type=int, default=10, help="number of negative samples per positive sample")
    parser.add_argument("--graph-perc", type=float, default=1.0, help="percentage of the graph to be used, 1 means all graph, 0.5 means half the triples for each relation.")
    parser.add_argument("--evaluate-every", type=int, default=500, help="perform evaluation every n epochs")

    args = parser.parse_args()
    print(args)
    main(args)

