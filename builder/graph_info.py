import logging
import sys
import argparse
from rdflib import URIRef, Literal, Namespace, Graph, util
from rdflib.namespace import FOAF, XSD, RDF, RDFS
from urllib.parse import quote
import os.path
import sparql as sparqlQueries
sys.path.append('../pkg/')
from pkg_ns import *


# setup logging
logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format="[%(name)s - %(levelname)s] %(message)s",
)
logger = logging.getLogger("graph_info")


def print_graph_info(g: Graph, ont: Enum):
    # Entities statistics
    # key = class, value = number of instances for such clas
    num_entities_dict = {key.value: 0 for key in ont}

    # Edges statistics
    # key = relation, value = number of edges for such relation
    num_edges_dict = {key.value: 0 for key in PURLTerms if key != PURLTerms.PURL_TERM_SUBJECT}

    keyword_rel = str(PURL_TERMS.subject + "_keyword") # special case for keyword subject
    num_edges_dict[keyword_rel] = 0

    tmf_rel = str(PURL_TERMS.subject + "_tmf") # special case for TMF subject
    num_edges_dict[tmf_rel] = 0

    # entity statistics
    nodes = set()
    for (s, p, o) in g:
        for ont_type in [t for t in ont]:
            if (s, RDF.type, ont_type.value) in g:
                nodes.add(s)
            if (o, RDF.type, ont_type.value) in g:
                nodes.add(o)

    for entity in nodes:
        for ont_type in [t for t in ont]:
            if (entity, RDF.type, ont_type.value) in g:
                num_entities_dict[ont_type.value] += 1

    # edges statistics
    for (s, p, o) in g:
        # if it's a triple between nodes
        if s in nodes and o in nodes:
            # first test if relation is special case (subject can be keyword or tmf)
            if p == PURLTerms.PURL_TERM_SUBJECT.value:
                if (o, RDF.type, PkgOntology.PKG_TMFRESOURCE.value) in g:
                    num_edges_dict[tmf_rel] += 1
                if (o, RDF.type, PkgOntology.PKG_AUTHORKEYWORD.value) in g:
                    num_edges_dict[keyword_rel] += 1
            else: num_edges_dict[p] += 1 # some other relation, add it

    for statistics_dicts in [num_entities_dict, num_edges_dict]:
        for key, val in statistics_dicts.items():
            logger.debug("%s -> %s" % (key, val))
        logger.debug("\n")

# --------- #
# -  Main - #
# --------- #
if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description="Graph Info")
    parser.add_argument("-p", "--pkg", help="path to PKG RDF graph")
    args = parser.parse_args()

    if args.pkg:
        pkg = Graph()
        pkg.parse(args.pkg, util.guess_format(args.pkg))
        logger.debug("*** PKG info:***")
        print_graph_info(pkg, PkgOntology)