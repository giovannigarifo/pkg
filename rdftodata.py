from rdflib import Graph
from rdflib import util, URIRef, Namespace
import logging, sys
from enum import Enum
from rdflib.namespace import RDF
import random


PURL = Namespace("http://purl.org/dc/terms/")

class GeraniumNamespace(Enum):
    GERANIUM_PUB = Namespace("http://geranium-project.org/publications/")
    GERANIUM_AUT = Namespace("http://geranium-project.org/authors/")
    GERANIUM_JOU = Namespace("http://geranium-project.org/journals/")

class GeraniumOntology(Enum):
    GERANIUM_ONTOLOGY_PUB = URIRef("http://geranium-project.org/ontology/Publication")
    GERANIUM_ONTOLOGY_AUT = URIRef("http://geranium-project.org/ontology/Author")
    GERANIUM_ONTOLOGY_JOU = URIRef("http://geranium-project.org/ontology/Journal")
    GERANIUM_ONTOLOGY_TMF = URIRef("http://geranium-project.org/ontology/TMFResource")

class GeraniumTerms(Enum):
    GERANIUM_TERM_SELF = URIRef("http://geranium-project.org/terms/self") # self relation
    GERANIUM_TERM_IS_SUBJECT = URIRef("http://geranium-project.org/terms/is_subject") # inverse relation of PURL.subject
    GERANIUM_TERM_IS_PUBLISHER = URIRef("http://geranium-project.org/terms/is_publisher") # inverse relation of PURL.publisher
    GERANIUM_TERM_IS_CREATOR = URIRef("http://geranium-project.org/terms/is_creator") # inverse relation of PURL.creator
    GERANIUM_TERM_IS_CONTRIBUTOR = URIRef("http://geranium-project.org/terms/is_contributor") # inverse relation of PURL.contributor

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
        edges_sources: list, edges_destinations: list, edges_relations: list, edges_norms: list, \
        id_to_node_uri_dict: dict = {}, id_to_rel_uri_dict: dict = {}):
        self.num_nodes = num_nodes # number of nodes in the graph
        self.num_relations = num_relations # number of relations in the graph
        self.num_labels = num_labels # number of labels of the nodes
        self.labels = labels # list, label for i-th node
        self.edges_sources = edges_sources # list, source node for i-th edge
        self.edges_destinations = edges_destinations # list, dest node for i-th edge
        self.edges_relations = edges_relations # list, relation type for i-th edge
        self.edges_norms = edges_norms # list, norm value for i-th edge
        self.id_to_node_uri_dict = id_to_node_uri_dict # used to retrieve nodes URIs from IDs (during scoring)
        self.id_to_rel_uri_dict = id_to_rel_uri_dict # used for retrieve relations URIs from IDs (during scoring)

        # used only for link-prediction
        self.train_triples = list()
        self.valid_triples = list()
        self.test_triples = list()

    def initTrainValidTestTriples(self, train_perc: float=0.9, valid_perc: float=0.05, test_perc: float=0.05):
        '''
        Splits the triples dataset into train, test and validation sets. The split
        is based on the relation, taking the given percentage for each relation type.
        '''
        assert valid_perc + test_perc + train_perc == 1

        # setup logging
        logger = logging.getLogger('rdftodata_logger')
        logger.debug("Splitting triples in: {}% train, {}% validation, {}% test...".format(train_perc*100, valid_perc*100, test_perc*100))

        triples = list(zip(self.edges_sources, self.edges_relations, self.edges_destinations))

        # divide triples by relations
        dict_rel_triples = {rel_index: list() for rel_index in range(self.num_relations)}
        for triple in triples:
            dict_rel_triples.get(triple[1]).append(triple)

        # # calc relation ratio: how much triples for each relation
        # dict_rel_num_triples = {rel_index: (len(dict_rel_triples.get(rel_index))/len(triples))*100 for rel_index in range(self.num_relations)}

        # for each relation (also inverse ones), excluded the self relation, split triples in training/valid/test sets.
        for rel in range(1, self.num_relations):
            rel_triples = dict_rel_triples.get(rel) # all triples for this relation
            num_triples = len(rel_triples)

            num_train = int(num_triples * train_perc)
            self.train_triples.extend(rel_triples[:num_train])

            num_valid = int(num_triples * valid_perc)
            self.valid_triples.extend(rel_triples[num_train:(num_train+num_valid)])

            self.test_triples.extend(rel_triples[num_train+num_valid:])

        # add self relations only to train set
        train_nodes = set((triple[0] for triple in self.train_triples))
        train_nodes = train_nodes|set((triple[2] for triple in self.train_triples)) #union
        for node in train_nodes:
            self.train_triples.append((node, 0, node))

        logger.debug("...finished:")
        logger.debug(" - Number of training triples: {}".format(len(self.train_triples)))
        logger.debug(" - Number of validation triples: {}".format(len(self.valid_triples)))
        logger.debug(" - Number of test triples: {}".format(len(self.test_triples)))


def readFileInGraph(filepath: str = "../../data/serialized.xml"):
    '''
    Given a /path/to/file, parse it in a rdflib Graph object
    '''
    # setup logging
    logger = logging.getLogger('rdftodata_logger')

    g = Graph()
    g.parse(filepath, util.guess_format(filepath))

    logger.debug("Input file has been read in rdflib Graph object")
    return g


def buildDataFromGraph(g: Graph, graphperc: float = 1.0) -> PublicationsDataset:
    '''
    This functions will scrape from the rdflib.Graph "g" the data required for the classification.
    '''
    # setup logging
    logger = logging.getLogger('rdftodata_logger')
    logger.debug("Start building the data structures from the rdflib Graph...")

    # 1. retrieve the set (no duplicates) of nodes for the RGCN task: only the listed entities
    #    are interpreted as nodes for the learning task
    nodes = set()
    for (s,p,o) in g.triples((None, None, None)):
        if (s, RDF.type, GeraniumOntology.GERANIUM_ONTOLOGY_PUB.value) in g or \
            (s, RDF.type, GeraniumOntology.GERANIUM_ONTOLOGY_AUT.value) in g or \
            (s, RDF.type, GeraniumOntology.GERANIUM_ONTOLOGY_JOU.value) in g or \
            (s, RDF.type, GeraniumOntology.GERANIUM_ONTOLOGY_TMF.value) in g:
            nodes.add(s)

    # take only some nodes based on the percentage defined by graphperc
    num_nodes = int(len(nodes) * graphperc)
    while num_nodes != len(nodes):
        nodes.pop()

    nodes_dict = {uri: index for (index, uri) in enumerate(nodes)} # build nodes dictionary: key = node's URIref object, value = node index

    # 2. Build the following dictionaries to gather labels and relations data:
    #    * labels_dict: key = node index, value = node label
    #    * relations_dict: key = relation URIref object, value = relation index
    #
    #      Please note: index 0 is associated to the self-relation (self-loop), manually added after
    labels_dict = {}
    num_labels = len(Label) # len of the enum
    relations_set = set()
    relations_to_inverse_dict = {}

    for (s,p,o) in g.triples((None, None, None)):

        # if it's a triple between nodes, add predicate to relation set (=> no duplicates allowed)
        if s in nodes and o in nodes:
            relations_set.add(p)
            # save mapping from relation to it's inverse one
            if "http://purl.org/dc/terms/subject" in p:
                relations_to_inverse_dict[p] = GeraniumTerms.GERANIUM_TERM_IS_SUBJECT.value
            if "http://purl.org/dc/terms/publisher" in p:
                relations_to_inverse_dict[p] = GeraniumTerms.GERANIUM_TERM_IS_PUBLISHER.value
            if "http://purl.org/dc/terms/creator" in p:
                relations_to_inverse_dict[p] = GeraniumTerms.GERANIUM_TERM_IS_CREATOR.value
            if "http://purl.org/dc/terms/contributor" in p:
                relations_to_inverse_dict[p] = GeraniumTerms.GERANIUM_TERM_IS_CONTRIBUTOR.value

        # save label of node in dictionary
        if s in nodes:
            if (s, RDF.type, GeraniumOntology.GERANIUM_ONTOLOGY_PUB.value) in g: # s it's a publication
                labels_dict[nodes_dict.get(s)] = Label.PUBLICATION.value
            elif (s, RDF.type, GeraniumOntology.GERANIUM_ONTOLOGY_AUT.value) in g:
                labels_dict[nodes_dict.get(s)] = Label.AUTHOR.value
            elif (s, RDF.type, GeraniumOntology.GERANIUM_ONTOLOGY_JOU.value) in g:
                labels_dict[nodes_dict.get(s)] = Label.JOURNAL.value
            elif (s, RDF.type, GeraniumOntology.GERANIUM_ONTOLOGY_TMF.value) in g: #s it's a TMF topic
                labels_dict[nodes_dict.get(s)] = Label.TOPIC.value


    logger.debug("Relations found:")
    for relation in relations_set:
        logger.debug("- {r}".format(r=relation))
    logger.debug("Inverse relations added:")
    for relation, inverse_r in relations_to_inverse_dict.items():
        logger.debug("- {ir}".format(ir=inverse_r))

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
    id_to_node_uri_dict = {} # used to retrieve URIs from IDs later during evaluation
    id_to_rel_uri_dict = {}

    # add self loops (self relation)
    for i in range(num_nodes):
        edge_list.append((i, i, 0))
    id_to_rel_uri_dict[0] = GeraniumTerms.GERANIUM_TERM_SELF.value

    for s,p,o in g.triples((None,None,None)):
        if(p in relations_dict and s in nodes and o in nodes): # s and o have to be selected nodes (depends on graphperc)
            if p == PURL.subject and not (o, RDF.type, GeraniumOntology.GERANIUM_ONTOLOGY_TMF.value) in g:
                pass # only TMF topics, no author keywords
            else:
                src_id = nodes_dict.get(s)
                dst_id = nodes_dict.get(o)
                rel_id = relations_dict.get(p)

                # add edge in both direction
                edge_list.append((src_id, dst_id, 2*rel_id - 1))
                edge_list.append((dst_id, src_id, 2*rel_id)) #reverse relation

                # add node and relation to dictionaries used to retrieve URIs from IDs (during scoring)
                id_to_node_uri_dict[src_id] = s
                id_to_node_uri_dict[dst_id] = o
                id_to_rel_uri_dict[2*rel_id - 1] = p
                id_to_rel_uri_dict[2*rel_id] = relations_to_inverse_dict.get(p) # get corresponding inverse relation

    # shuffle the edge list
    random.shuffle(edge_list)

    # edges lists used by RGCN
    edges_sources = [edge[0] for edge in edge_list]
    edges_destinations = [edge[1] for edge in edge_list]
    edges_relations = [edge[2] for edge in edge_list]
    edges_norms = [1 for edge in edge_list] # TODO

    logger.debug("...finished, some stats:")
    logger.debug(" - Number of nodes: %d" % num_nodes)
    logger.debug(" - Number of relations: %d" % num_relations)
    logger.debug(" - Number of labels/classes: %d" % num_labels)
    logger.debug(" - Number of edges: %d" % len(edges_sources))

    return PublicationsDataset(num_nodes, num_relations, num_labels, labels, \
        edges_sources, edges_destinations, edges_relations, edges_norms, \
        id_to_node_uri_dict, id_to_rel_uri_dict)


def topicSampling(g: Graph, num_topics_to_remove: int):
    '''
    Removes from the graph a number of "num_topics_to_remove" topics for each triple (paper, subject, topic), the
    triple is removed only if the topic has been already seen at least one time.

    The removed triples are collected so that they can be later added in the evaluation graph to check
    if the model is able to give them an high score, thus to be able to correctly predict the topic of a paper.
    '''
    seen_topics = dict()
    seen_papers = dict()

    for s,p,o in g.triples((None, PURL.subject, None)):
        if (o, RDF.type, GeraniumOntology.GERANIUM_ONTOLOGY_TMF.value) in g: # only TMF topics, discard user keywords
            if not seen_papers.get(s):
                seen_papers[s] = True
                for _,_,topic in g.triples((s,PURL.subject, None)):
                    if (topic, RDF.type, GeraniumOntology.GERANIUM_ONTOLOGY_TMF.value) in g: # only TMF topics
                        pass # TODO




def rdfToData(filepath: str = "serialized.xml", graph_perc: float = 1.0, job: str = "classification",
                train_perc: float = 0.9, valid_perc: float = 0.9, test_perc: float = 0.9) -> PublicationsDataset:
    '''
    return a PublicationsDataset object that contains all the required data sctructures
    to build an RGCN-based model
    '''
    # setup logging
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='[%(name)s - %(levelname)s] %(message)s')
    logger = logging.getLogger('rdftodata_logger')

    # from RDF file to rdflib Graph
    g = readFileInGraph(filepath)

    if job == "classification":
        return buildDataFromGraph(g)

    elif job == "link-prediction":

        #topicSampling(g, 2) # remove validation topics from rdflib graph
        data = buildDataFromGraph(g, graph_perc)
        data.initTrainValidTestTriples(train_perc, valid_perc, test_perc)
        return data

    else:
        logger.error("use as job \"classification\" or \"link-prediction\"")
        exit(0)


def main(argv):
    '''
    Execute the following script if not used as library
    '''

    # setup logging
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='[%(name)s - %(levelname)s] %(message)s')
    logger = logging.getLogger('rdftodata_logger')

    # test arguments
    if(len(argv) == 1):
        filepath = "data/anni2013-2017_with_img.xml" #default path
    elif(len(argv) == 2):
        if(argv[1] == "--help" or argv[1] == "-h"): # print help
            print("Usage: `python3 script.py /path/to/file.(xml|nt|rdf)`")
            exit(0)
        filepath = argv[1] # path from user
    else:
        logger.error("wrong arguments: `python3 script.py /path/to/file.(xml|nt|rdf)`")
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
