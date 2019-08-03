"""
Utility functions for link prediction
Most code is adapted from authors' implementation of RGCN link prediction:
https://github.com/MichSchli/RelationPrediction

"""

import numpy as np
import torch
import dgl
import os
import time
from multiprocessing.pool import ThreadPool
import json

#######################################################################
#
# Utility function for building training and testing graphs
#
#######################################################################

def get_adj_and_degrees(num_nodes, triplets):
    """ 
    Get adjacency list of the graph and degrees of nodes.

    For each node, the adjacency list saves [triplet_number, neighbour_node_id]
    """
    adj_list = [[] for _ in range(num_nodes)] # list of lists, number of lists = num_nodes
    
    # "i" is the triplet_number, used during edge neighbourhood sampling to retireve the selected triplet
    for i,triplet in enumerate(triplets):
        adj_list[triplet[0]].append([i, triplet[2]])
        adj_list[triplet[2]].append([i, triplet[0]])

    degrees = np.array([len(a) for a in adj_list])
    adj_list = [np.array(a) for a in adj_list]
    return adj_list, degrees

def sample_edge_neighborhood(adj_list, degrees, n_triplets, sample_size):
    """ 
    Edge neighborhood sampling to reduce training graph size
    
    Parameters
        adj_list: list of lists, indexed by node id, it allow to retrieve all the nodes connected (in/out) to each node
        degrees: numpy array, indexed by node id, contains the number of nodes connected to each node
        n_triplets: number of tripltes in the training graph
        sample_size: args.graph_batch_size
    """

    edges = np.zeros((sample_size), dtype=np.int32)

    #initialize
    sample_counts = np.array([d for d in degrees])
    picked = np.array([False for _ in range(n_triplets)])
    seen = np.array([False for _ in degrees])

    for i in range(0, sample_size):
        weights = sample_counts * seen

        if np.sum(weights) == 0:
            weights = np.ones_like(weights)
            weights[np.where(sample_counts == 0)] = 0

        probabilities = (weights) / np.sum(weights)
        chosen_vertex = np.random.choice(np.arange(degrees.shape[0]),
                                         p=probabilities)
        chosen_adj_list = adj_list[chosen_vertex]
        seen[chosen_vertex] = True

        chosen_edge = np.random.choice(np.arange(chosen_adj_list.shape[0]))
        chosen_edge = chosen_adj_list[chosen_edge]
        edge_number = chosen_edge[0]
        while picked[edge_number]:
            chosen_edge = np.random.choice(np.arange(chosen_adj_list.shape[0]))
            chosen_edge = chosen_adj_list[chosen_edge]
            edge_number = chosen_edge[0]

        edges[i] = edge_number
        other_vertex = chosen_edge[1]
        picked[edge_number] = True
        sample_counts[chosen_vertex] -= 1
        sample_counts[other_vertex] -= 1
        seen[other_vertex] = True

    return edges


def generate_sampled_graph_and_labels(triplets, sample_size, split_size,
                                      num_rels, adj_list, degrees,
                                      negative_rate):
    """
    First perform edge neighborhood sampling to obtain a training graph
    composed by a number of "sample_size" edges, then perform negative sampling to 
    generate negative samples.

    Parameters:

        triplets: ndarray, train_data is passed as argument
        sample_size: args.graph_batch_size
        split_size: args.graph_split_size,
        negative_rate: args.negative_sample, number of negative samples for each positive one
    """
    print("Sampling train graph for this epoch...")
    
    # perform edge neighbor sampling
    edges = sample_edge_neighborhood(adj_list, degrees, len(triplets), sample_size)

    # relabel nodes to have consecutive node ids
    edges = triplets[edges]
    src, rel, dst = edges.transpose()
    uniq_v, edges = np.unique((src, dst), return_inverse=True)
    src, dst = np.reshape(edges, (2, -1))
    relabeled_edges = np.stack((src, rel, dst)).transpose()

    # negative sampling
    samples, labels = negative_sampling(relabeled_edges, len(uniq_v),
                                        negative_rate)

    # further split graph, only half of the edges will be used as graph
    # structure, while the rest half are used as unseen positive samples
    split_size = int(sample_size * split_size)
    graph_split_ids = np.random.choice(np.arange(sample_size),
                                       size=split_size, replace=False)
    src = src[graph_split_ids]
    dst = dst[graph_split_ids]
    rel = rel[graph_split_ids]

    # build the sampled train graph
    g, rel, norm = build_graph_from_triplets(len(uniq_v), num_rels,
                                             (src, rel, dst))
    print("Train graph sampled.\n")
    return g, uniq_v, rel, norm, samples, labels

def comp_deg_norm(g):
    in_deg = g.in_degrees(range(g.number_of_nodes())).float().numpy()
    norm = 1.0 / in_deg 
    norm[np.isinf(norm)] = 0
    return norm

def build_graph_from_triplets(num_nodes, num_rels, triplets):
    """ Create a DGL graph. The graph is bidirectional because RGCN authors
        use reversed relations.
        This function also generates edge type and normalization factor
        (reciprocal of node incoming degree)
    """
    g = dgl.DGLGraph()
    g.add_nodes(num_nodes)
    src, rel, dst = triplets

    # add reverse relations, so for (s,r,o) add (o,inv(r),s)
    src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
    rel = np.concatenate((rel, rel + num_rels)) # inverse relations, just add to rel num_rels (inv of "rel 0" is "rel num_rels")
    edges = sorted(zip(dst, src, rel))
    dst, src, rel = np.array(edges).transpose()

    g.add_edges(src, dst)

    norm = comp_deg_norm(g)
    print("...DGL Graph built from triples. number of nodes: {}, number of edges: {}".format(num_nodes, len(src)))
    return g, rel, norm

def build_test_graph(num_nodes, num_rels, edges):
    src, rel, dst = edges.transpose()
    print("Building test graph...")
    return build_graph_from_triplets(num_nodes, num_rels, (src, rel, dst))

def negative_sampling(pos_samples, num_entity, negative_rate):
    size_of_batch = len(pos_samples)
    num_to_generate = size_of_batch * negative_rate

    # just tile up pos_samples for a negative_rate number of times to obtain
    # the array of negative samples, that will later corrupted
    neg_samples = np.tile(pos_samples, (negative_rate, 1))
    
    # create labels array for both positive and negative, the positive samples
    # have all 1 as label, while the negative samples have zero
    labels = np.zeros(size_of_batch * (negative_rate + 1), dtype=np.float32)
    labels[:size_of_batch] = 1 

    values = np.random.randint(num_entity, size=num_to_generate)
    choices = np.random.uniform(size=num_to_generate)
    subj = choices > 0.5
    obj = choices <= 0.5
    
    neg_samples[subj, 0] = values[subj] # corrupting the subject
    neg_samples[obj, 2] = values[obj] #corrupting the object

    # concatenate the pos and neg samples
    return np.concatenate((pos_samples, neg_samples)), labels

#######################################################################
#
# Utility function for evaluations
#
#######################################################################


def sort_and_rank(score, target):
    '''
    From the scores of each triplet (a,r,every_node_of_graph) obtain the rank of the valid triplet (a,r,target)
    1. sort the scores
    2. calculate where the (a,r,target) triplets (=validation triplets) are positioned in the sorted scores
    3. the position obtained is the rank of the correct triplet (a,r,target)
    '''

    # in indices I get the nodes_id of "every_nodes" for the triplets (a,r,every_nodes), sorted 
    # from the highest score to the lowest 
    _, indices = torch.sort(score, dim=1, descending=True) # indices.shape = batchsize x numNodes
    
    # target is transformed to a column vector, and each value is compared to the values in the columns of indices
    # there is only one "1" for each row in cmp_res, this is used to extract where is positioned the target value
    # so to calculate the rank for the correct triplet (a,r,target)
    cmp_res = indices == target.view(-1, 1)

    # get the index of the correct triplet (a,r,target) from cmp_res, **the index is the rank**, that goes from 0 to num_nodes, hopefully 
    # the rank for the validation triplet should be lowest possible, ideally 1, this would mean that the valid triplet got the highest score
    indices = torch.nonzero(cmp_res)

    # just got the ranks, first column is added by torch.nonzero() and is useless
    ranks = indices[:, 1].view(-1)

    return ranks


def perturb_and_get_rank(embedding, w, a, r, b, num_entity, perturb_str, epoch, batch_size=100):
    """ 
    num_entity := num_nodes (total)

    It's called by evaluate:

        ranks_s = perturb_and_get_rank(embedding, w, o, r, s, num_entity, "perturb_subject", epoch, eval_bz)
        ranks_o = perturb_and_get_rank(embedding, w, s, r, o, num_entity, "perturb_object", epoch, eval_bz)


    Calculate the rank of each validation triplet (a,r,b).
    
    1. perform distmult over (a,r,every_node_of_graph), equal to apply a perturbation to the object 
       of the triplets. This means that it will also calculate the distmult for the correct triplet, (a,r,b).
    2. sort the obtained distmult scores, and get the rank of the true triplets (a,r,b). 
       The "Rank" is the position of the true triplet in the sorted scores for all the triplets (a,r,every_node_of_graph):
       i.e. if distmult gives the highest score for a true triplet, it will be sorted at top, and it's rank will be 0
    """
    n_batch = (num_entity + batch_size - 1) // batch_size
    ranks = []
    score_list = []
    
    # for each batch, calculate validation triplet (a,r,b) rank
    for idx in range(n_batch):
        
        #print("batch {} / {}".format(idx, n_batch))
        
        batch_start = idx * batch_size
        batch_end = min(num_entity, (idx + 1) * batch_size)

        # get indexes of subject nodes for the validation triples of this batch
        batch_a = a[batch_start: batch_end]

        # get indexes of the relations for the validation triples of this batch
        batch_r = r[batch_start: batch_end] 

        # Do element-wise Hadamard product, **NOT** matrix mult, between the embeddings of the subject nodes
        # and w_rel of the relations for the validation triples of this batch.
        #
        # it's the first product of distmult: subject_embedding * w_relation
        emb_ar = embedding[batch_a] * w[batch_r] 
        
        # transpose: swap dim=0 and dim=1 => each **column** contains e*w_rel
        # unsqueeze: add a dimension of size 1 at position 2
        emb_ar = emb_ar.transpose(0, 1).unsqueeze(2) # size: D x E x 1 <=> EMBEDDING_SIZE x BATCH_SIZE x 1
       
        # get in emb_c the embeddings of ALL nodes in the test set
        emb_c = embedding.transpose(0, 1).unsqueeze(1) # size: D x 1 x V <=> EMBEDDING_SIZE x 1 x NUM_NODES
        
        # out-prod and reduce sum
        #
        # perform product between (s*w_rel) and the embeddings of **all nodes**
        #
        # it's the second product of distmult: (subject_embedding * w_relation) * object_embedding
        out_prod = torch.bmm(emb_ar, emb_c) # size D x E x V <=> EMBEDDING_SIZE x BATCH_SIZE x NUM_NODES

        # sum: (subject_embedding * w_relation) * object_embedding to obtain the final distmult score
        #
        # such score are the score of the triples (batch_subject, batch_relations, ALL_NODES)
        score = torch.sum(out_prod, dim=0) # size E x V <=> BATCH_SIZE x NUM_NODES
        
        # cap the score between 0 and 1
        score = torch.sigmoid(score) 

        # get the embeddings of the objects of the validation triples for this batch
        target = b[batch_start: batch_end] 
        
        # export scores for this batch
        score_list.extend(export_triples_score(batch_a, batch_r, target, embedding, w, score))
        
        # obtain the rank (as defined in the top comment) of each validation triplet (a,r,b).
        ranks.append(sort_and_rank(score, target))

    print_scores_as_json(score_list, perturb_str, epoch) # export all scores
    return torch.cat(ranks)


# TODO (lingfan): implement filtered metrics
# return MRR (raw), and Hits @ (1, 3, 10)
def evaluate(epoch,
            test_graph, 
            model, 
            test_triplets, 
            num_entity, 
            hits=[], 
            eval_bz=100):
    '''
    called by main as: 
        mrr = utils.evaluate(test_graph, 
                            model, 
                            valid_data, 
                            num_nodes,
                            hits=[1, 3, 10], 
                            eval_bz=args.eval_batch_size)

    In this function we evaluate the ability of the model to calculate an high score
    for the validation triplets.

    1. embeddings and w_rel for the model are calculated for ALL triplets (full graph)
    2. ranks are calculated for each validation triplet, the rank is the position where the correct triplet
       is found when sorting ALL the possible triplets (obtained by keeping "s,p" fo the validation triplet 
       and put as "o" every possible node) by score (calculated using distmult)
    3. calcualte MRR for this batch, a measure that gives us an insight on how in average the model
       is able to rank (=lowest position, 1== highest rank) a correct triplet
    '''
    with torch.no_grad(): # DON'T KEEP TRACK OF GRADIENTS

        # get the embeddings and the w_relation parameters (without grad update)
        # for the model under evaluation
        embedding, w = model.evaluate(test_graph)
        
        # get s,r,o from validation data
        s = test_triplets[:, 0]
        r = test_triplets[:, 1]
        o = test_triplets[:, 2]

        print("- s,r,o shapes: ", s.shape, r.shape, o.shape)
        print("Computing ranks...")

        # get ranks for the inverse of the validation triplet (o,r,s)
        ranks_s = perturb_and_get_rank(embedding, w, o, r, s, num_entity, "perturb_subject", epoch, eval_bz)
        print("...half way...")

        # get rank for the validation triplets (s,r,o)
        ranks_o = perturb_and_get_rank(embedding, w, s, r, o, num_entity, "perturb_object", epoch, eval_bz)
        print("...done.")

        ranks = torch.cat([ranks_s, ranks_o])
        ranks += 1 # change to 1-indexed, because the highest rank is 0

        # si riesce ad ottenere un MRR per il modello migliore di circa 0.150, vuol dire che in media
        # la tripla di validazione compare tra le prime 7 triple (ordinate per dist-mult score).

        # MRR definition: The mean of all reciprocal ranks for the true candidates over the test set (1/rank)
        mrr = torch.mean(1.0 / ranks.float()) 
        print("- MRR (raw): {:.6f}".format(mrr.item()))

        for hit in hits:
            avg_count = torch.mean((ranks <= hit).float())
            print("- Hits (raw) @ {}: {:.6f}".format(hit, avg_count.item()))

        print("\n")

    return mrr.item()




######################################################
# Utility functions to retrieve score for each triple
######################################################

def print_scores_as_json(score_list, perturb_str, epoch):
    dir_path = "./output/epoch_" + str(epoch) + "/"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    with open(dir_path + perturb_str + "_score.json", "w") as f:
        json.dump(score_list, f, ensure_ascii=False, indent=4)

def create_list_from_batch(batch, embedding):
    """ 
    Create list of dictionaries including the id of the node (or the relation)
    and its embedding value
    """
    batch_list = []
    for index, value in enumerate(batch.tolist()):
        new_dict = {"id": value, "emb": embedding[batch][index, :].tolist()}
        batch_list.append(new_dict)
    return batch_list


score_list = []
batch_s_list = []
batch_r_list = []
batch_o_list = []

def export_triples_score(s, r, o, emb_nodes, emb_rels, score, multithread = False):
    """ 
    Export score associated to each triple included in the validation dataset.
    This function is called for each evaluation batch.
    Exported scores could be useful for a deep analysis of the evaluation results

    Arguments:
        s -- tensor batch of subject ids
        r -- tensor batch of relation ids
        o -- tensor batch of object ids
        emb_nodes -- tensor with embeddings of all nodes
        emb_rels -- tensor embeddings of all relations
        score -- tensor of scores associated to eache triple, size(batch, num_of_nodes)
    Returns:
        score_list -- list of dictionaries including triple ids and the associated score
    """
    global batch_s_list 
    global batch_r_list
    global batch_o_list
    global score_list

    batch_s_list = create_list_from_batch(s, emb_nodes)
    batch_r_list = create_list_from_batch(r, emb_rels)
    batch_o_list = create_list_from_batch(o, emb_nodes)
    
    #t1 = time.time()
    
    if multithread is False:
        for row_index, row in enumerate(score):
            for col_index, col in enumerate(row):
                s_id = str(batch_s_list[row_index]["id"])
                r_id = str(batch_r_list[row_index]["id"])
                o_id = str(batch_o_list[row_index]["id"])
                # score tensor includes also perturbed triples, for such reason
                # I need to get data from the correct column
                if str(col_index) == str(o_id):
                    score_value = col
                    score_dict = {"s": s_id, "r": r_id, "o": o_id, "score": score_value.item()}
                    score_list.append(score_dict)
    else:
        row_idx_list = [(row_index, row) for row_index, row in enumerate(score)]
        thread_pool = ThreadPool()
        thread_pool.map(extractTripleScore, row_idx_list)
        thread_pool.close()
        thread_pool.join()

    #t2 = time.time()
    #print("time elapsed: ", t2-t1)
    return score_list

def extractTripleScore(row_idx_list):
    '''
    Receives a tuple that contains row_index and row tensor for the scores of a batch,
    extracts from the row 
    '''
    global score_list
    global batch_s_list 
    global batch_r_list
    global batch_o_list
    row_index = row_idx_list[0]
    row = row_idx_list[1] # score tensor
    local_store_list = []

    for col_index, col in enumerate(row):
            s_id = str(batch_s_list[row_index]["id"])
            r_id = str(batch_r_list[row_index]["id"])
            o_id = str(batch_o_list[row_index]["id"])
            # score tensor includes also perturbed triples, for such reason
            # I need to get data from the correct column
            if str(col_index) == str(o_id):
                score_value = col
                score_dict = {"s": s_id, "r": r_id, "o": o_id, "score": score_value.item()}
                local_store_list.append(score_dict)
    
    score_list.append(local_store_list)
