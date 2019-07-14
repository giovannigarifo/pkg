from rdflib import Graph
from rdflib import util, URIRef, Namespace
import logging, sys
from enum import Enum
from rdflib.namespace import RDF


class GeraniumNamespace(Enum):
    GERANIUM_PUB = Namespace("http://geranium-project.org/publications/")
    GERANIUM_AUT = Namespace("http://geranium-project.org/authors/")
    GERANIUM_JOU = Namespace("http://geranium-project.org/journals/")
    PURL = Namespace("http://purl.org/dc/terms/")

class GeraniumOntology(Enum):
    GERANIUM_ONTOLOGY_PUB = URIRef("http://geranium-project.org/ontology/Publication")
    GERANIUM_ONTOLOGY_AUT = URIRef("http://geranium-project.org/ontology/Author")
    GERANIUM_ONTOLOGY_JOU = URIRef("http://geranium-project.org/ontology/Journal")
    GERANIUM_ONTOLOGY_TMF = URIRef("http://geranium-project.org/ontology/TMFResource")

class Label(Enum):
    '''
    Labels for the RGCN nodes
    '''
    PUBLICATION = 0
    AUTHOR = 1
    JOURNAL = 2
    TOPIC = 3

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
        
        # used only for link-prediction
        self.train_triples = list()
        self.valid_triples = list()
        self.test_triples = list()

    def initTrainValidTestTriples(self, train_perc: float=0.7, valid_perc: float=0.15, test_perc: float=0.15):
        '''
        Splits the triples dataset into train, test and validation sets. The split
        is based on the relation, taking the given percentage for each relation type.
        '''
        assert valid_perc + test_perc + train_perc == 1

        logging.debug("Splitting triples in Train, Test and Validation sets...")

        triples = list(zip(self.edges_sources, self.edges_relations, self.edges_destinations))

        # divide triples by relations
        dict_rel_triples = {rel_index: list() for rel_index in range(self.num_relations)}
        for triple in triples:
            dict_rel_triples.get(triple[1]).append(triple)

        # for each relation (also inverse ones), excluded the self relation, split triples in training/valid/test sets.
        for rel in range(1, self.num_relations):
            rel_triples = dict_rel_triples.get(rel) # all triples for this relation
            num_triples = len(rel_triples)

            num_train = int(num_triples * train_perc)
            self.train_triples.extend(rel_triples[:num_train])

            num_valid = int(num_triples * valid_perc)
            self.valid_triples.extend(rel_triples[num_train:(num_train+num_valid)])

            self.test_triples.extend(rel_triples[num_train+num_valid:])

        # add relations between nodes and themselves for train, test, validation data,
        # the self relations are added to all sets (if they'll be merged later, duplicates will arise!)
        num_self_rel = 0
        for triples_subset in (self.train_triples, self.valid_triples, self.test_triples):
            nodes = set((triple[0] for triple in triples_subset))
            nodes = nodes|set((triple[2] for triple in triples_subset)) #union
            for node in nodes:
                triples_subset.append((node, 0, node)) # append to train, test or valid
                num_self_rel = num_self_rel + 1 
        
        assert len(triples) == (len(self.train_triples) \
                + len(self.valid_triples) \
                + len(self.test_triples) \
                - num_self_rel + self.num_nodes), "Wrong number of Train, Valid, Test triples" # all the self relations added

        logging.debug("...finished:")
        logging.debug(" - Number of training triples: {}".format(len(self.train_triples)))
        logging.debug(" - Number of validation triples: {}".format(len(self.valid_triples)))
        logging.debug(" - Number of test triples: {}".format(len(self.test_triples)))


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
    for (s,p,o) in g.triples((None, None, None)):
        if (s, RDF.type, GeraniumOntology.GERANIUM_ONTOLOGY_PUB.value) in g or \
            (s, RDF.type, GeraniumOntology.GERANIUM_ONTOLOGY_AUT.value) in g or \
            (s, RDF.type, GeraniumOntology.GERANIUM_ONTOLOGY_JOU.value) in g or \
            (s, RDF.type, GeraniumOntology.GERANIUM_ONTOLOGY_TMF.value) in g:
            nodes.add(s)
            
    num_nodes = len(nodes)
    nodes_dict = {uri: index for (index, uri) in enumerate(nodes)} # build nodes dictionary: key = node's URIref object, value = node index

    # 2. Build the following dictionaries to gather labels and relations data:
    #    * labels_dict: key = node index, value = node label
    #    * relations_dict: key = relation URIref object, value = relation index
    #
    #      Please note: index 0 is associated to the self-relation (self-loop), manually added after
    labels_dict = {} 
    num_labels = len(Label) # len of the enum
    relations_set = set() 

    for (s,p,o) in g.triples((None, None, None)):
        # save label of node in dictionary
        if (s, RDF.type, GeraniumOntology.GERANIUM_ONTOLOGY_PUB.value) in g: # s it's a publication
            labels_dict[nodes_dict.get(s)] = Label.PUBLICATION.value 
        elif (s, RDF.type, GeraniumOntology.GERANIUM_ONTOLOGY_AUT.value) in g:
            labels_dict[nodes_dict.get(s)] = Label.AUTHOR.value
        elif (s, RDF.type, GeraniumOntology.GERANIUM_ONTOLOGY_JOU.value) in g:
            labels_dict[nodes_dict.get(s)] = Label.JOURNAL.value    
        elif (s, RDF.type, GeraniumOntology.GERANIUM_ONTOLOGY_TMF.value) in g: #s it's a TMF topic
            labels_dict[nodes_dict.get(s)] = Label.TOPIC.value
            
        # if it's a triple between nodes, add predicate to relation set (=> no duplicates allowed)
        if s in nodes and o in nodes: 
            relations_set.add(p)    

    logging.debug("\nRelations found:")
    for relation in relations_set:
        logging.debug("- {r}".format(r=relation))
    
    assert len(labels_dict) == num_nodes, "Some labels are missing!"

    # build label list, element i-th of list correspond to the label of the i-th node (this is why dictionary is sorted for key=node_index)
    labels = list()
    for index, (node,label) in enumerate(sorted(labels_dict.items())):
        labels.append(label)   

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
        if(p in relations_dict 
            and (o, RDF.type, GeraniumOntology.GERANIUM_ONTOLOGY_TMF.value) in g): # only topics, no author keywords
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

    logging.debug("\n...finished, some stats:")
    logging.debug(" - Number of nodes: %d" % num_nodes)
    logging.debug(" - Number of relations: %d" % num_relations)
    logging.debug(" - Number of labels/classes: %d" % num_labels)
    logging.debug(" - Number of edges: %d" % len(edges_sources))

    return PublicationsDataset(num_nodes, num_relations, num_labels, labels, edges_sources, edges_destinations, edges_relations, edges_norms)


def rdfToData(filepath: str = "serialized.xml", job: str = "classification") -> PublicationsDataset:
    '''
    return a data tuple that contains all the required data sctructures
    to build an RGCN-based model
    '''
    # setup logging
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

    if job == "classification":
        return buildDataFromGraph(readFileInGraph(filepath))
    elif job == "link-prediction":
        data = buildDataFromGraph(readFileInGraph(filepath))
        data.initTrainValidTestTriples();
        return data
    else:
        logging.error("use as job \"classification\" or \"link-prediction\"")
        exit(0)


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
            print("Usage: `python3 s    \cript.py /path/to/file.(xml|nt|rdf)`")
            exit(0)
        filepath = argv[1] # path from user
    else:
        logging.error("wrong arguments: `python3 script.py /path/to/file.(xml|nt|rdf)`")
        exit(-1)

    # read file to rdflib's graph 
    g = readFileInGraph(filepath)

    # extract data needed for RGCN from rdflib's graph
    data = buildDataFromGraph(g)
    data.initTrainValidTestTriples()

    return 0


#----------#
# - Main - #
#----------#
if __name__ == "__main__":
    main(sys.argv)
