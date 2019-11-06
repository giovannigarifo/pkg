from rdflib import Graph
from rdflib import util, URIRef, Namespace
import logging, sys
from enum import Enum
from rdflib.namespace import RDF
import random
import time

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
    GERANIUM_ONTOLOGY_KEY = URIRef("http://geranium-project.org/ontology/AuthorKeyword")

class GeraniumTerms(Enum):
    GERANIUM_TERM_SELF = URIRef("http://geranium-project.org/terms/self") # self relation
    GERANIUM_TERM_IS_SUBJECT = URIRef("http://geranium-project.org/terms/is_subject") # inverse relation of PURL.subject
    GERANIUM_TERM_IS_PUBLISHER = URIRef("http://geranium-project.org/terms/is_publisher") # inverse relation of PURL.publisher
    GERANIUM_TERM_IS_CREATOR = URIRef("http://geranium-project.org/terms/is_creator") # inverse relation of PURL.creator
    GERANIUM_TERM_IS_CONTRIBUTOR = URIRef("http://geranium-project.org/terms/is_contributor") # inverse relation of PURL.contributor

class PURLTerms(Enum):
    PURL_TERM_SUBJECT = PURL.subject
    PURL_TERM_PUBLISHER = PURL.publisher
    PURL_TERM_CREATOR = PURL.creator
    PURL_TERM_CONTRIBUTOR = PURL.contributor

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
        is done per relation, taking the given percentage of triples for each relation type.

        All the triples are of the form (publications,x,y), this means that the source
        node is always a publication.

        All the nodes must be present in the train set in order to obtain an
        embedding for all nodes. This is why there are two for loop, the first one is
        used to pick a train triple for each node present in the relation-triples dictionary.
        '''
        assert valid_perc + test_perc + train_perc == 1

        # setup logging
        logger = logging.getLogger('rdftodata_logger')
        logger.debug("Splitting {} triples in: {}% train, {}% validation, {}% test...".format(
                len(self.edges_sources), train_perc*100, valid_perc*100, test_perc*100))

        # set of all triples
        triples = set(zip(self.edges_sources, self.edges_relations, self.edges_destinations))
        assert len(triples) == len(self.edges_sources) == len(self.edges_destinations)

        # collect triples by relation in a dictionary
        dict_rel_triples = {rel_index: set() for rel_index in range(self.num_relations)}
        for triple in triples:
            dict_rel_triples.get(triple[1]).add(triple)

        # set of unseen nodes (all nodes at the beginning)
        unseen_nodes = set(self.edges_sources)
        unseen_nodes.update(set(self.edges_destinations))
        assert len(unseen_nodes) == self.num_nodes

        # set of seen nodes (empty at beginning)
        seen_nodes = set()

        # for each list of triples
        for rel, rel_triples in dict_rel_triples.items():

            # seen and unseen nodes for this relation
            unseen_nodes = [triple[0] for triple in rel_triples]
            unseen_nodes.extend([triple[2] for triple in rel_triples])
            unseen_nodes = set(unseen_nodes) # all nodes for this relation
            seen_nodes = set()

            # amount of triples to get for train, test, valid for this relation
            num_train = int(train_perc*len(rel_triples))
            num_valid = int(valid_perc*len(rel_triples))
            num_test = int(test_perc*len(rel_triples))

            for num_set in [(num_train, self.train_triples, "train"),
                            (num_valid, self.valid_triples, "valid"),
                            (num_test, self.test_triples, "test")]:

                num_to_pick = num_set[0] # number of train/valid/test triples to pick for this relation
                triples_set = num_set[1] # train/valid/test set
                num_picked = 0

                # be sure to have all nodes present in train set
                if num_set[2] == "train":
                    for triple in rel_triples.copy(): #iterate over copy and remove from original
                        if triple[0] in seen_nodes and triple[2] in seen_nodes:
                            continue
                        elif num_picked != num_to_pick:
                            triples_set.append(triple) # get the triple if one of the nodes or both not seen
                            rel_triples.discard(triple)
                            seen_nodes.add(triple[0])
                            seen_nodes.add(triple[2])
                            num_picked += 1

                for triple in rel_triples.copy():
                    if num_picked == num_to_pick and num_set[2] != "test": # for test set, get all the remaining
                        break
                    else:
                        triples_set.append(triple) # add triple to T/V/T set
                        rel_triples.discard(triple)
                        num_picked += 1

        logger.debug("...finished:")
        logger.debug(" - Number of training triples: {}".format(len(self.train_triples)))
        logger.debug(" - Number of validation triples: {}".format(len(self.valid_triples)))
        logger.debug(" - Number of test triples: {}".format(len(self.test_triples)))
        assert len(self.train_triples) + len(self.test_triples) + len(self.valid_triples) == len(triples)


    def checkCorrectness(self, g: Graph):
        '''
        Test if all the triples regarding the selected nodes (see GeraniumOntology)
        and relations (see PURLTerms) contained in the graph g were correctly selected
        as triples for the PublicationsDataset.
        If some triples haven't been added and were lost, the program will exit.
        '''
        # setup logging
        logger = logging.getLogger('rdftodata_logger')
        logger.debug("Checking correctness of the data scraped from the RDF graph...")

        # build an RDFlib graph from the edges in PublicationsDataset
        edges_graph = Graph()
        num_triples_in_edge_graph = 0

        triples_from_edges = set([(self.id_to_node_uri_dict.get(t[0]),
                            self.id_to_rel_uri_dict.get(t[1]),
                            self.id_to_node_uri_dict.get(t[2])) \
                            for t in zip(self.edges_sources, self.edges_relations, self.edges_destinations)])

        for triple in triples_from_edges:
            edges_graph.add(triple)
            num_triples_in_edge_graph += 1
        logger.debug("...number of triples to be checked: {}".format(num_triples_in_edge_graph))

        assert len(triples_from_edges) == num_triples_in_edge_graph

        # build graph of lost triples
        lost_graph = Graph()
        num_triples_in_lost_graph = 0
        for (s,p,o) in g:
            if (s,p,o) not in edges_graph and p in [term.value for term in PURLTerms] \
                and not (o, RDF.type, GeraniumOntology.GERANIUM_ONTOLOGY_KEY.value) in g: # exclude (auth,subj,authorkey)
                lost_graph.add((s,p,o))
                num_triples_in_lost_graph += 1

        if num_triples_in_lost_graph > 0:
            logger.error("...number of lost triples: {}".format(num_triples_in_lost_graph))
            exit(0)

        logger.debug("...finished, data ok.")


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

    # 1. retrieve the set (no duplicates) of nodes for the RGCN task, only the nodes listen in
    #    GeraniumOntology are added, with the exclusion of:
    #    - Authorkeyword nodes
    #    - External authors (which doesn't have a Polito ID, starting with "rp")
    nodes = set()
    for (s, p, o) in g:
        for geranium_ont_type in [t for t in GeraniumOntology if t != GeraniumOntology.GERANIUM_ONTOLOGY_KEY]:
            if (s, RDF.type, geranium_ont_type.value) in g:
                if (s, RDF.type, GeraniumOntology.GERANIUM_ONTOLOGY_AUT.value) in g and "rp" not in s: # no external authors
                    pass
                else:
                    nodes.add(s)
            if (o, RDF.type, geranium_ont_type.value) in g:
                if (o, RDF.type, GeraniumOntology.GERANIUM_ONTOLOGY_AUT.value) in g and "rp" not in o: # no external authors
                    pass
                else:
                    nodes.add(o)

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

    for (s, p, o) in g:
        # if it's a triple between nodes, add predicate to relation set (=> no duplicates allowed)
        if s in nodes and o in nodes:
            relations_set.add(p)
            # save mapping from relation to it's inverse one
            if p == PURLTerms.PURL_TERM_SUBJECT.value:
                relations_to_inverse_dict[p] = GeraniumTerms.GERANIUM_TERM_IS_SUBJECT.value
            if p == PURLTerms.PURL_TERM_PUBLISHER.value:
                relations_to_inverse_dict[p] = GeraniumTerms.GERANIUM_TERM_IS_PUBLISHER.value
            if p == PURLTerms.PURL_TERM_CREATOR.value:
                relations_to_inverse_dict[p] = GeraniumTerms.GERANIUM_TERM_IS_CREATOR.value
            if p == PURLTerms.PURL_TERM_CONTRIBUTOR.value:
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
        # save label of node in dictionary
        if o in nodes:
            if (o, RDF.type, GeraniumOntology.GERANIUM_ONTOLOGY_PUB.value) in g:
                labels_dict[nodes_dict.get(o)] = Label.PUBLICATION.value
            elif (o, RDF.type, GeraniumOntology.GERANIUM_ONTOLOGY_AUT.value) in g:
                labels_dict[nodes_dict.get(o)] = Label.AUTHOR.value
            elif (o, RDF.type, GeraniumOntology.GERANIUM_ONTOLOGY_JOU.value) in g:
                labels_dict[nodes_dict.get(o)] = Label.JOURNAL.value
            elif (o, RDF.type, GeraniumOntology.GERANIUM_ONTOLOGY_TMF.value) in g:
                labels_dict[nodes_dict.get(o)] = Label.TOPIC.value

    logger.debug("Relations found:")
    for relation in relations_set:
        logger.debug("- {r}".format(r=relation))
    logger.debug("Inverse relations are (they're not added!):")
    for relation, inverse_r in relations_to_inverse_dict.items():
        logger.debug("- {ir}".format(ir=inverse_r))

    assert len(labels_dict) == num_nodes, "Some labels are missing!"

    # build label list, element i-th of list correspond to the label of the i-th node (this is why dictionary is sorted for key=node_index)
    labels = list()
    for index, (node,label) in enumerate(sorted(labels_dict.items())):
        labels.append(label)

    relations_dict = {relation: index for (index, relation) in enumerate(relations_set)}
    num_relations = len(relations_dict.keys())

    # 2. build edge list (preserve order) with tuples (src_id, dst_id, rel_id)
    #   the predicates that will be used as relations are:
    #   - http://purl.org/dc/terms/subject     -> corresponding to triples: paper - subject - topic
    #   - http://purl.org/dc/terms/publisher   -> for triples: paper - publisher - journal
    #   - http://purl.org/dc/terms/creator     -> for triples: paper - creator - author
    #   - http://purl.org/dc/terms/contributor -> for triples: paper - contributor - author
    edge_list = []
    id_to_node_uri_dict = {} # used to retrieve URIs from IDs later during evaluation
    id_to_rel_uri_dict = {}

    for s, p, o in g:
        if(p in relations_dict and s in nodes and o in nodes): # s and o have to be selected nodes (depends on graphperc)
            if p == PURLTerms.PURL_TERM_SUBJECT.value and not (o, RDF.type, GeraniumOntology.GERANIUM_ONTOLOGY_TMF.value) in g:
                pass # only TMF topics, no author keywords
            else:
                src_id = nodes_dict.get(s)
                dst_id = nodes_dict.get(o)
                rel_id = relations_dict.get(p)
                edge_list.append((src_id, dst_id, rel_id)) # add edge

                # add node and relation to dictionaries used to retrieve URIs from IDs (during scoring)
                id_to_node_uri_dict[src_id] = s
                id_to_node_uri_dict[dst_id] = o
                id_to_rel_uri_dict[rel_id] = p
                id_to_rel_uri_dict[rel_id + num_relations] = relations_to_inverse_dict.get(p) # get corresponding inverse relation

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

def printRDFGraphStatistics(g: Graph):

    # Entities statistics
    # key = class, value = number of instances for such clas
    num_entities_dict = {key.value: 0 for key in GeraniumOntology}

    # Edges statistics
    # key = relation, value = number of edges for such relation
    num_edges_dict = {key.value: 0 for key in PURLTerms if key != PURLTerms.PURL_TERM_SUBJECT}

    keyword_rel = str(PURL.subject + "_keyword") # special case for keyword subject
    num_edges_dict[keyword_rel] = 0

    tmf_rel = str(PURL.subject + "_tmf") # special case for TMF subject
    num_edges_dict[tmf_rel] = 0

    # entity statistics
    nodes = set()
    for (s, p, o) in g:
        for geranium_ont_type in [t for t in GeraniumOntology]:
            if (s, RDF.type, geranium_ont_type.value) in g:
                nodes.add(s)
            if (o, RDF.type, geranium_ont_type.value) in g:
                nodes.add(o)

    for entity in nodes:
        for geranium_ont_type in [t for t in GeraniumOntology]:
            if (entity, RDF.type, geranium_ont_type.value) in g:
                num_entities_dict[geranium_ont_type.value] += 1

    # edges statistics
    for (s, p, o) in g:
        # if it's a triple between nodes
        if s in nodes and o in nodes:
            # first test if relation is special case (subject can be keyword or tmf)
            if p == PURLTerms.PURL_TERM_SUBJECT.value:
                if (o, RDF.type, GeraniumOntology.GERANIUM_ONTOLOGY_TMF.value) in g:
                    num_edges_dict[tmf_rel] += 1
                if (o, RDF.type, GeraniumOntology.GERANIUM_ONTOLOGY_KEY.value) in g:
                    num_edges_dict[keyword_rel] += 1
            else: num_edges_dict[p] += 1 # some other relation, add it

    print("---> Entities statistics:\n")
    print(num_entities_dict)
    print("\n\n---> Edges statistics:\n")
    print(num_edges_dict)


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
        #data.checkCorrectness(g)
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
