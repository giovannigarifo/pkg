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
import matplotlib.pyplot as plt

# for data import
import sys
sys.path.append('..')
import rdftodata

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

    Returns
        edges: np array that contains the index of the sampled edges, index will be used to retrieve the triple
    """
    print("...sampling {} edges...".format(sample_size))
    num_nodes = len(adj_list) # number of nodes in the graph from which to sample the edges

    #initialize
    edges = np.zeros((sample_size), dtype=np.int32) # number (index) of the sampled edges
    sample_counts = np.array([d for d in degrees])
    picked = np.array([False for _ in range(n_triplets)])
    seen = np.array([False for _ in degrees])
    assert num_nodes == degrees.shape[0] == sample_counts.shape[0] == seen.shape[0]

    for i in range(0, sample_size):

        weights = sample_counts * seen
        if np.sum(weights) == 0:
            weights = np.ones_like(weights)
            weights[np.where(sample_counts == 0)] = 0
        probabilities = (weights) / np.sum(weights)

        # randomly choose a vertex of the training graph
        chosen_vertex = np.random.choice(np.arange(num_nodes), p=probabilities)
        chosen_adj_list = adj_list[chosen_vertex]
        seen[chosen_vertex] = True

        # randomly choose an edge of the choosen vertex
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

    # perform edge neighbor sampling, returns indexes of sampled edges
    edges = sample_edge_neighborhood(adj_list, degrees, len(triplets), sample_size)

    # relabel nodes to have consecutive node ids
    edges = triplets[edges] # filter the sampled edges
    src, rel, dst = edges.transpose()
    uniq_v, edges = np.unique((src, dst), return_inverse=True)
    src, dst = np.reshape(edges, (2, -1))
    relabeled_edges = np.stack((src, rel, dst)).transpose()

    # negative sampling
    samples, labels = negative_sampling(relabeled_edges, len(uniq_v), negative_rate)

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
    """
    Create a DGL graph.
    This function also generates edge type and normalization factor (reciprocal of node incoming degree)
    """
    g = dgl.DGLGraph()
    g.add_nodes(num_nodes)
    src, rel, dst = triplets

    # add reverse relations, so for (s,r,o) add (o,inv(r),s)
    src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
    rel = np.concatenate((rel, rel + num_rels)) # inverse relations, just add to rel num_rels (inv of "rel 0" is "rel num_rels")

    # Create the edges array
    edges = sorted(zip(dst, src, rel))
    dst, src, rel = np.array(edges).transpose() # 3 x Num of nodes
    g.add_edges(src, dst)

    norm = comp_deg_norm(g)

    print("...DGL Graph built from triplets. number of nodes: {}, number of edges: {}".format(num_nodes, len(src)))
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
    score: Tensor of dimension "BATCH_SIZE x NUM_NODES", a row for each triple (a,r,b) in the batch,
            each row contains the scores for all the corrupted triples (a,r,all_nodes), among all
            the scores there will be also the score for the correct triple (a,r,b)

    target: contains the index of the object nodes of the batch triplets (a,r,b). Contains all indicies of "b" entities.

    From the scores obtain the rank of the valid triplet (a,r,target)
    1. sort the scores
    2. calculate where the (a,r,target) triplets (=validation triplets) are positioned in the sorted scores
    3. the position obtained is the rank of the correct triplet (a,r,target)

    Returns a tensor of shape "BATCH_SIZE x 1", where in each row is present the rank obtained by
    a triple (a,r,b) of the current batch
    '''
    # in indices I get the index of "every_node" for the triplets (a,r,every_nodes), sorted
    # from the highest score to the lowest. Will also contain the index of the true object "b".
    # Here is done the actual conversion from "score" to "rank"
    _, indices = torch.sort(score, dim=1, descending=True) # indices.shape = batchsize x numNodes

    # target is transformed to a column vector, and each value (index of true object "b") is compared to the
    # values in the columns of indices => search for the index of "o" in sorted.
    # In cmp_res there is only one "1" for each row, this is used to extract where is positioned the target value
    # so to calculate the rank for the correct triplet (a,r,target)
    cmp_res = indices == target.view(-1, 1) # cmp_res.shape = batchsize x numNodes

    # get the index of the correct triplet (a,r,target) in cmp_res, **the index is the rank**, that goes from 0 to num_nodes, hopefully
    # the rank for the validation triplet should be lowest possible, ideally 0, this would mean that the valid triplet got the highest score
    indices = torch.nonzero(cmp_res)

    # just got the ranks, first column is added by torch.nonzero() and is useless
    ranks = indices[:, 1].view(-1)

    return ranks


def perturb_and_get_rank(embedding, w, a, r, b, epoch, batch_size=100, \
    id_to_node_uri_dict: dict = {}, id_to_rel_uri_dict: dict = {}):
    """
    Calculate the rank of each validation triplet (a,r,b).

    Please note:
        1. rank and score are not the same, the rank is obtained from the score
        2. The triplets contains both direct and inverse relation (done in `rdftodata.buildDataFromGraph()`)
        3. num_entity is equal to num_nodes (total)

    1. perform distmult over (a,r,every_node_of_graph), equal to apply a perturbation to the object
       of the triplets. This means that it will also calculate the distmult for the correct triplet, (a,r,b).
    2. sort the obtained distmult scores, and get the rank of the true triplets (a,r,b).
       The "Rank" is the position of the true triplet in the sorted scores for all the triplets (a,r,every_node_of_graph):
       i.e. if distmult gives the highest score for a true triplet, it will be sorted at top, and it's rank will be 0
    """
    n_batch = (len(a) + batch_size - 1) // batch_size
    ranks = []
    score_list = []
    rank_list = []

    # for each batch, calculate validation triplet (a,r,b) rank
    for idx in range(n_batch):

        t1 = time.time()
        batch_start = idx * batch_size
        batch_end = min(len(a), (idx + 1) * batch_size)

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

        # get in emb_c the embeddings of ALL nodes
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

        # get the indices of the objects of the validation triples for this batch
        target = b[batch_start: batch_end]

        # export scores for this batch (if requested)
        # if id_to_node_uri_dict and id_to_rel_uri_dict: # false if empty
        #     score_list.extend(export_triples_score(\
        #         batch_a, batch_r, target, embedding, w, score, multithread=False,
        #         id_to_node_uri_dict=id_to_node_uri_dict, id_to_rel_uri_dict=id_to_rel_uri_dict))

        # obtain the rank (as defined in the top comment) of each validation triplet (a,r,b).
        batch_ranks = sort_and_rank(score, target)
        batch_ranks += 1 # change to 1-indexed, because the highest rank is 0
        ranks.append(batch_ranks)

        # save rank_list for this batch
        if id_to_node_uri_dict and id_to_rel_uri_dict:
            batch_b = b[batch_start: batch_end]
            for triple_rank in zip(batch_a, batch_r, batch_b, batch_ranks):
                rank_dict = {
                        "s_id": triple_rank[0].item(),
                        "s_uri": id_to_node_uri_dict.get(int(triple_rank[0].item())),
                        "r_id": triple_rank[1].item(),
                        "r_uri": id_to_rel_uri_dict.get(int(triple_rank[1].item())),
                        "o_id": triple_rank[2].item(),
                        "o_uri": id_to_node_uri_dict.get(int(triple_rank[2].item())),
                        "rank": triple_rank[3].item()
                    }
                rank_list.append(rank_dict)

        t2 = time.time()
        #print("Finished batch {} / {} in {} seconds ({} triples)".format(idx, n_batch, t2-t1, len(batch_a)))

    return torch.cat(ranks), score_list, rank_list # last two will be empty if URI dicts are empty


# TODO (lingfan): implement filtered metrics
# return MRR (raw), and Hits @ (1, 3, 10)
def evaluate(epoch,
            test_graph,
            model,
            test_triplets, # can be both valid_data during validation or test_data during evaluation
            num_entity,
            hits=[],
            eval_bz=100,
            id_to_node_uri_dict: dict = {},
            id_to_rel_uri_dict: dict = {}):
    '''
    called by main as:
        mrr = utils.evaluate(test_graph,
                            model,
                            valid_data,
                            num_nodes,
                            hits=[1, 3, 10],
                            eval_bz=args.eval_batch_size)

    In this function we evaluate the ability of the model to calculate an high score
    for the "test_triplets".

    1. embeddings and w_rel for the model are calculated for ALL triplets (full graph)
    2. ranks are calculated for each "test_triplets" triplet, the rank is the position where the correct triplet
       is found when sorting ALL the possible triplets (obtained by keeping "s,p" fo the "test_triplets" triplet
       and put as "o" every possible node) by score (calculated using distmult)
    3. calcualte MRR for this batch, a measure that gives us an insight on how in average the model
       is able to rank (=lowest position, 1== highest rank) a correct triplet
    '''
    with torch.no_grad(): # DON'T KEEP TRACK OF GRADIENTS

        # get the embeddings and the w_relation parameters (without grad update)
        # for the model under evaluation
        embedding, w = model.evaluate(test_graph)

        # get s,r,o from test_triplets data
        s = test_triplets[:, 0]
        r = test_triplets[:, 1]
        o = test_triplets[:, 2]

        print("Computing ranks...")

        # get ranks for the inverse of the validation triplet (o,r,s)
        ranks_s, score_list, rank_list = perturb_and_get_rank(
            embedding, w, o, r, s, epoch, eval_bz,
            id_to_node_uri_dict=id_to_node_uri_dict, id_to_rel_uri_dict=id_to_rel_uri_dict
        )
        print("...half way...")

        # get rank for the validation triplets (s,r,o)
        ranks_o, score_list_o, rank_list_o = perturb_and_get_rank(
            embedding, w, s, r, o, epoch, eval_bz,
            id_to_node_uri_dict=id_to_node_uri_dict, id_to_rel_uri_dict=id_to_rel_uri_dict
        )
        print("...done.")

        score_list.extend(score_list_o)
        rank_list.extend(rank_list_o)
        ranks = torch.cat([ranks_s, ranks_o])

        print("Saving JSONs...")
        if len(score_list) > 0:
            print("...exporting score_list, length=", len(score_list))
            save_list_as_json(score_list, "scores", epoch) # export all scores as json
        if len(rank_list) > 0:
            print("...exporting rank_list, length=", len(rank_list))
            save_list_as_json(rank_list, "ranks", epoch) # export all ranks as json

        print("...done.\nComputing statistics...")
        # si riesce ad ottenere un MRR per il modello migliore di circa 0.150, vuol dire che in media
        # la tripla di validazione compare tra le prime 7 triple (ordinate per dist-mult score).

        # MRR definition: The mean of all reciprocal ranks for the true candidates over the test set (1/rank)
        mrr = torch.mean(1.0 / ranks.float())
        print("- MRR (raw): {:.6f}".format(mrr.item()))

        for hit in hits:
            avg_count = torch.mean((ranks <= hit).float())
            print("- Hits (raw) @ {}: {:.6f}".format(hit, avg_count.item()))

        print("\n")

    return mrr.item(), score_list, rank_list


######################################################
# Utility functions for testing
######################################################

def save_list_as_json(list_to_print, file_name, epoch):
    '''
    Serialize and save as JSON a Python list passed as argument.
    '''
    dir_path = ""
    if type(epoch) is str:
        if epoch == "best_on_test_data":
            dir_path = "./output/epoch_best_on_test_data/"
        if epoch == "link_evaluation":
            dir_path = "./output/evaluation/"
    if type(epoch) is int:
        if epoch >= 0:
            dir_path = "./output/epoch_" + str(epoch) + "/"
    if dir_path == "":
        return

    # write the json to file
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    with open(dir_path + file_name + ".json", "w") as f:
        json.dump(list_to_print, f, ensure_ascii=False, indent=4)


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

def export_triples_score(s, r, o, emb_nodes, emb_rels, score, multithread=False,
                        id_to_node_uri_dict: dict = {}, id_to_rel_uri_dict: dict = {}):
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
        score_list -- list of dictionaries including triples IDs, triples URIs and the associated score
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
                    score_dict = {
                        "s_id": s_id,
                        "s_uri": id_to_node_uri_dict.get(int(s_id)),
                        "r_id": r_id,
                        "r_uri": id_to_rel_uri_dict.get(int(r_id)),
                        "o_id": o_id,
                        "o_uri": id_to_node_uri_dict.get(int(o_id)),
                        "score": score_value.item()
                    }
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
                score_dict = {
                        "s_id": s_id,
                        "r_id": r_id,
                        "o_id": o_id,
                        "score": score_value.item()
                    }
                local_store_list.append(score_dict)

    score_list.append(local_store_list)


#########################################################################
# Utility functions to analyze in depth the result of the testing phase
#########################################################################

def check_accuracy(mrr_test, test_data,
                        rank_list, # list of score for each test data triplet
                        id_to_node_uri_dict: dict = {},
                        id_to_rel_uri_dict: dict = {}):

    # analyze scores
    ranks_summary = [
        [0, 1], # number of triples with rank equal to 1
        [0, 3], # number of triples with rank between 2 and 3
        [0, 5], # number of triples with rank between 4 and 5
        [0, 7], # number of triples with rank between 6 and 7
        [0, 10], # number of triples with rank between 8 and 10
        [0, 15]  # number of triples with rank between 11 and 15
    ]

    for score_dict in rank_list:
        rank = score_dict.get("rank")
        if rank == ranks_summary[0][1]:
            ranks_summary[0][0] += 1
        if rank > ranks_summary[0][1] and rank <= ranks_summary[1][1]:
            ranks_summary[1][0] += 1
        if rank > ranks_summary[1][1] and rank <= ranks_summary[2][1]:
            ranks_summary[2][0] += 1
        if rank > ranks_summary[2][1] and rank <= ranks_summary[3][1]:
            ranks_summary[3][0] += 1
        if rank > ranks_summary[3][1] and rank <= ranks_summary[4][1]:
            ranks_summary[4][0] += 1
        if rank > ranks_summary[4][1] and rank <= ranks_summary[5][1]:
            ranks_summary[5][0] += 1

    print("Number of ranked triples: ", len(rank_list))
    for summary in ranks_summary:
        print("Number of triples with rank in range with upper bound {range}: {n}".format(range=summary[1], n=summary[0]))

    accuracy = (ranks_summary[0][0]/len(rank_list))*100
    print("Accuracy (percentage of triples with rank equal to 1) over test data: {a}%".format(a=accuracy))



def plot_loss_to_file(loss_list):
    loss_values = np.array(loss_list).T
    epochs_values = np.arange(len(loss_list))

    plt.plot(epochs_values, loss_values, label="Loss")
    plt.xticks(epochs_values)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss behavior over epochs")
    plt.savefig('output/loss_over_epochs.png')


def plot_rank_statistics_from_json(ranks_json_path: str = "output/epoch_best_on_test_data/ranks.json"):
    '''
    Plot a graph that shows the number of triples for each rank
    '''
    ranks = [] # list of dicts
    num_triples_per_rank = {} # dict with key = rank, value = number of triple with given rank

    try:
        with open(ranks_json_path, 'r') as f:
            ranks = json.load(f)
    except Exception as exc:
        print("Unable to read the ranks from json, cannot plot number of triples over rank")
        return

    for rank_dict in ranks:
        test_triple_rank = rank_dict.get("rank", -1)
        assert test_triple_rank != -1

        if num_triples_per_rank.get(test_triple_rank, -1) == -1:
            num_triples_per_rank[test_triple_rank] = 1
        else:
            num_triples_per_rank[test_triple_rank] = num_triples_per_rank[test_triple_rank] +1

    num_ranks_in_range = 0
    num_triples_list = []
    for i in range(1, len(num_triples_per_rank)):
        # if num_triples_per_rank.get(i, 0) != 0:
        #     print("Number of triples with rank {} : {}".format(i, num_triples_per_rank.get(i)))
        num_triples_list.append(num_triples_per_rank.get(i, 0))

    num_triples_values = np.array(num_triples_list).T
    num_ranks_values = np.arange(1, len(num_triples_per_rank))

    plt.rcParams["figure.figsize"] = [16, 9]
    plt.plot(num_ranks_values, num_triples_values)
    plt.xlabel("Rank value")
    plt.ylabel("Number of triples")
    plt.title("Number of triples per rank")
    plt.xscale('log')

    plt.savefig('output/num_triples_per_rank.png', dpi=100)




#########################################################################
# Utility functions for linkeval()
#########################################################################


def calc_score(
        embedding, w,
        s, r,
        batch_size = 30,
        num_scored_triples = 30,
        id_to_node_uri_dict: dict = {},
        id_to_rel_uri_dict: dict = {}
    ):
    """
    Calculate the rank of triples (subj, rel, all_possible_objects) using
    distmult = sum(s*r*o)
    """
    n_batch = (len(s) + batch_size - 1) // batch_size
    score_dict = dict()

    for idx in range(n_batch):

        t1 = time.time()
        batch_start = idx * batch_size
        batch_end = min(len(s), (idx + 1) * batch_size)

        batch_s = s[batch_start: batch_end] # subjects batch
        batch_r = r[batch_start: batch_end] # relation batch

        emb_sr = embedding[batch_s] # element wise subject_embedding * w_relation
        emb_sr = emb_sr * w[batch_r]
        emb_sr = emb_sr.transpose(0, 1).unsqueeze(2) # size: D x E x 1 <=> EMBEDDING_SIZE x BATCH_SIZE x 1

        # get in emb_c the embeddings of ALL nodes
        emb_all_o = embedding.transpose(0, 1).unsqueeze(1) # size: D x 1 x V <=> EMBEDDING_SIZE x 1 x NUM_NODES

        # (subject_embedding * w_relation) * object_embedding (all)
        out_prod = torch.bmm(emb_sr, emb_all_o) # size D x E x V <=> EMBEDDING_SIZE x BATCH_SIZE x NUM_NODES
        score = torch.sum(out_prod, dim=0) # size E x V <=> BATCH_SIZE x NUM_NODES
        score = torch.sigmoid(score) # cap the score between 0 and 1

        # sort descending the scores, also get objects indicies in order
        # to reconstruct the triple (sub,rel,obj)
        sorted_score, obj_indices = torch.sort(score, dim=1, descending=True)

        # trim to get only the best scores (look at hyperparameter "num_scored_triples")
        sorted_score = sorted_score[:, :num_scored_triples]
        obj_indices = obj_indices[:, :num_scored_triples]

        # save all the scores obtained for this batch
        for scores, objects in zip(sorted_score, obj_indices):
            for sub, rel, obj, triple_score in zip(batch_s, batch_r, objects, scores):
                # get URIs
                sub = id_to_node_uri_dict.get(int(sub))
                rel = id_to_rel_uri_dict.get(int(rel))
                obj = id_to_node_uri_dict.get(int(obj.item()))

                # test if the triple is semantically correct (domain)
                if check_domain_correctness(sub, rel, obj) is False:
                    continue

                # get dict relation->list of all objects with associated scores
                rel_dict = score_dict.get(sub, None)
                if rel_dict == None:
                    score_dict[sub] = dict()
                    rel_dict = score_dict[sub]

                # add to list the dict obj->score for (sub,rel,obj)
                obj_list = rel_dict.get(rel, None)
                if obj_list == None:
                    rel_dict[rel] = list()
                    obj_list = rel_dict[rel]
                obj_list.append({obj: triple_score.item()})

        t2 = time.time()
        print("...scores calculated for batch {} / {} in {}s".format(idx, n_batch, t2-t1))

    # save scores info to json
    save_list_as_json(score_dict, "predictions", "link_evaluation")


def check_domain_correctness(sub: str, rel: str, obj: str) -> bool:
    '''
    Checks if the triple is semantically correct in the domain
    of interest, in example, a triple (publication, subject, publication)
    doesn't make any sense.

    Returns true or false.
    '''
    check = False

    # check if correct triple (publication, subject, topic)
    if rel == rdftodata.PURLTerms.PURL_TERM_SUBJECT.value \
        and "http://dbpedia.org/resource/" in obj:
        check = True
    # check if correct triple (publication, publisher, journal)
    elif rel == rdftodata.PURLTerms.PURL_TERM_PUBLISHER.value \
        and rdftodata.GeraniumNamespace.GERANIUM_JOU.value in obj:
        check = True
    # check if correct triple (publication, creator, author)
    elif rel == rdftodata.PURLTerms.PURL_TERM_CREATOR.value \
        and rdftodata.GeraniumNamespace.GERANIUM_AUT.value in obj:
        check = True
    # check if correct triple (publication, contributor, author)
    elif rel == rdftodata.PURLTerms.PURL_TERM_CONTRIBUTOR.value \
        and rdftodata.GeraniumNamespace.GERANIUM_AUT.value in obj:
        check = True

    return check
