'''
Namespaces, ontologies, vocabularies used to build the PKG schema 
'''

from enum import Enum
from rdflib import Namespace, URIRef


# ------------------------------------------------------------------
# FOAF, XSD, RDF and RDF Schema
# ------------------------------------------------------------------

from rdflib.namespace import FOAF, XSD, RDF, RDFS


# ------------------------------------------------------------------
# DCMI
# ------------------------------------------------------------------

PURL_TERMS = Namespace("https://purl.org/dc/terms/")

class PURLTerms(Enum):
    PURL_TERM_SUBJECT = PURL_TERMS.subject
    PURL_TERM_PUBLISHER = PURL_TERMS.publisher
    PURL_TERM_CREATOR = PURL_TERMS.creator
    PURL_TERM_CONTRIBUTOR = PURL_TERMS.contributor


# ------------------------------------------------------------------
# POLITO KNOWLEDGE GRAPH
# ------------------------------------------------------------------

# namespaces
PKG = "https://pkg.polito.it/"
PKG_ONTOLOGY = Namespace(PKG + "ontology/")  # classes of the PKG ontology
PKG_TERMS = Namespace(PKG + "terms/")  # terms of the PKG vocaboulary
PKG_RESOURCE = Namespace(PKG + "resource/")  # entities (instances of the PKG classes)

# Classes
PKG_PUBLICATION = URIRef(PKG_ONTOLOGY + "Publication")
PKG_AUTHOR = URIRef(PKG_ONTOLOGY + "Author")
PKG_JOURNAL = URIRef(PKG_ONTOLOGY + "Journal")
PKG_TMFRESOURCE = URIRef(PKG_ONTOLOGY + "TMFResource")
PKG_AUTHORKEYWORD = URIRef(PKG_ONTOLOGY + "AuthorKeyword")

# predicates
PKG_SUG_TOPIC = URIRef(PKG_TERMS + "suggestedTopic")
PKG_SUG_JOURNAL = URIRef(PKG_TERMS + "suggestedJournal")
PKG_SUG_CREATOR = URIRef(PKG_TERMS + "suggestedCreator")
PKG_SUG_CONTRIBUTOR = URIRef(PKG_TERMS + "suggestedContributor")


class PkgNamespace(Enum):
    PKG_ONTOLOGY = Namespace(PKG + "ontology/")
    PKG_TERMS = Namespace(PKG + "terms/")
    PKG_RESOURCE = Namespace(PKG + "resource/")

class PkgOntology(Enum):
    PKG_PUBLICATION = URIRef(PKG_ONTOLOGY + "Publication")
    PKG_AUTHOR = URIRef(PKG_ONTOLOGY + "Author")
    PKG_JOURNAL = URIRef(PKG_ONTOLOGY + "Journal")
    PKG_TMFRESOURCE = URIRef(PKG_ONTOLOGY + "TMFResource")
    PKG_AUTHORKEYWORD = URIRef(PKG_ONTOLOGY + "AuthorKeyword")

class PkgPredicates(Enum):
    PKG_SUG_TOPIC = URIRef(PKG_TERMS + "suggestedTopic")
    PKG_SUG_JOURNAL = URIRef(PKG_TERMS + "suggestedJournal")
    PKG_SUG_CREATOR = URIRef(PKG_TERMS + "suggestedCreator")
    PKG_SUG_CONTRIBUTOR = URIRef(PKG_TERMS + "suggestedContributor")