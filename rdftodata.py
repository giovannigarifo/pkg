from rdflib import Graph
from rdflib import util, URIRef
import logging, sys
from enum import Enum


class Label(Enum):
    '''
    Labels for the RGCN nodes
    '''
    PUBLICATION = 0
    AUTHOR = 1
    JOURNAL = 2
    RESOURCE = 3
    KEYWORD = 4

class PublicationsDataset:
    '''
    Encapsulate the informations needed by the RGCN to work with the publications dataset
    '''
    def __init__(self, num_nodes: int, num_relations: int, num_labels: int, labels: list, \
        edges_sources: list, edges_destinations: list, edges_relations: list, edges_norms: list):
        self.num_nodes = num_nodes # number of nodes in the graph
        self.num_relations = num_relations # number of relations in the graph
        self.num_labels = num_labels # number of labels of the nodes
        self.labels = labels # list, label for i-th node
        self.edges_sources = edges_sources # list, source node for i-th edge
        self.edges_destinations = edges_destinations # list, dest node for i-th edge
        self.edges_relations = edges_relations # list, relation type for i-th edge
        self.edges_norms = edges_norms # list, norm value for i-th edge
    

def readFileInGraph(filepath: str = "../../data/serialized.xml"):
    '''
    Given a /path/to/file, parse it in a rdflib Graph object
    '''
    g = Graph()
    g.parse(filepath, util.guess_format(filepath))
    logging.debug("Input file has been read in rdflib Graph object")
    return g


def buildDataFromGraph(g: Graph) -> PublicationsDataset:
    '''
    This functions will scrape from the rdflib.Graph "g" the data required for the classification.
    '''

    logging.debug("Start building the data structures from the rdflib Graph...")

    # 1. retrieve the set (no duplicates) of nodes for the RGCN task: only the listed entities
    #    are interpreted as nodes for the learning task
    nodes = set() 
    for (s,p,o) in g.triples((None,None,None)):
        if "http://geranium-project.org/publications/" in s or \
            "http://geranium-project.org/keywords/" in s or \
            "http://dbpedia.org/resource/" in s or \
            "http://geranium-project.org/authors/" in s or \
            "http://geranium-project.org/journals/" in s:
            nodes.add(s)

    num_nodes = len(nodes)
    nodes_dict = {uri: index for (index, uri) in enumerate(nodes)} # build nodes dictionary: key = node's URIref object, value = node index
    assert len(set(g.subjects())) == len(nodes), "Number of nodes and number of subjects are different!"

    # 2. Build the following dictionaries to gather labels and relations data:
    #    * labels_dict: key = node index, value = node label
    #    * relations_dict: key = relation URIref object, value = relation index
    #      Please note: index 0 is associated to the self-relation (self-loop), manually added after
    labels_dict = {} 
    num_labels = len(Label) # len of the enum
    relations_set = set() 

    for (s,p,o) in g.triples((None, None, None)):
        # save label of node in dictionary
        if "http://geranium-project.org/publications/" in s:
            labels_dict[nodes_dict.get(s)] = Label.PUBLICATION.value
        elif "http://geranium-project.org/keywords/" in s:
            labels_dict[nodes_dict.get(s)] = Label.KEYWORD.value
        elif "http://dbpedia.org/resource/" in s:
            labels_dict[nodes_dict.get(s)] = Label.RESOURCE.value
        elif "http://geranium-project.org/authors/" in s:
            labels_dict[nodes_dict.get(s)] = Label.AUTHOR.value
        elif "http://geranium-project.org/journals/" in s:
            labels_dict[nodes_dict.get(s)] = Label.JOURNAL.value    

        # if it's a triple between nodes, add predicate relation to set (no duplicates)
        if s in nodes and o in nodes: 
            relations_set.add(p)
    
    assert len(labels_dict) == num_nodes, "Some label is missing!"

    # build label list, element i-th of list correspond to the label of the i-th node (this is why dictionary is sorted for key=node_index)
    labels = list()
    for index, (node,label) in enumerate(sorted(labels_dict.items())):
        labels.append(label)   
        #print("node_index=", node, " node_label=", label, file=open("log_node_label.txt", "a"))


    relations_dict = {relation: index+1 for (index, relation) in enumerate(relations_set)} #+1 because of self relation that has 0 as index
    num_relations = len(relations_dict.keys())*2+1 # relations are in both ways, +1 for self relation

    # 2. build edge list (preserve order) with tuples (src_id, dst_id, rel_id)
    #   the predicates that will be used as relations are:
    #   - http://purl.org/dc/terms/subject     -> corresponding to triples: paper - subject - topic
    #   - http://purl.org/dc/terms/publisher   -> for triples: paper - publisher - journal
    #   - http://purl.org/dc/terms/creator     -> for triples: paper - creator - author
    #   - http://purl.org/dc/terms/contributor -> for triples: paper - contributor - author
    edge_list = []

    # add self loops (self relation)
    for i in range(num_nodes):
        edge_list.append((i, i, 0))

    for s,p,o in g.triples((None,None,None)):
        if(p in relations_dict):
            src_id = nodes_dict.get(s)
            dst_id = nodes_dict.get(o)
            rel_id = relations_dict.get(p)
            # add edge in both direction
            edge_list.append((src_id, dst_id, 2*rel_id - 1))
            edge_list.append((dst_id, src_id, 2*rel_id)) #reverse relation

    # edges lists used by RGCN
    edges_sources = [edge[0] for edge in edge_list]
    edges_destinations = [edge[1] for edge in edge_list]
    edges_relations = [edge[2] for edge in edge_list]
    edges_norms = [1 for edge in edge_list] 

    logging.debug("...finished:")
    logging.debug(" - Number of nodes: %d" % num_nodes)
    logging.debug(" - Number of relations: %d" % num_relations)
    logging.debug(" - Number of labels/classes: %d" % num_labels)
    logging.debug(" - Number of edges: %d" % len(edges_sources))

    return PublicationsDataset(num_nodes, num_relations, num_labels, labels, edges_sources, edges_destinations, edges_relations, edges_norms)


def rdfToData(filepath: str = "serialized.xml") -> PublicationsDataset:
    '''
    return a data tuple that contains all the required data sctructures
    to build an RGCN-based model
    '''
    return buildDataFromGraph(readFileInGraph(filepath))


def main(argv):
    '''
    Execute the following script if not used as library
    '''

    # setup logging
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

    # test arguments
    if(len(argv) == 1):
        filepath = "serialized.xml" #default path
    elif(len(argv) == 2):
        if(argv[1] == "--help" or argv[1] == "-h"): # print help
            print("Usage: `python3 script.py /path/to/file.(xml|nt|rdf)`")
            exit(0)
        filepath = argv[1] # path from user
    else:
        logging.error("wrong arguments: `python3 script.py /path/to/file.(xml|nt|rdf)`")
        exit(-1)

    # read file to rdflib's graph 
    g = readFileInGraph(filepath)

    # extract data needed for RGCN from rdflib's graph
    data = buildDataFromGraph(g)
    return 0


#----------#
# - Main - #
#----------#
if __name__ == "__main__":
    main(sys.argv)

