import json
import logging
import sys
import argparse
import requests
import time
import threading
import _thread
from rdflib import URIRef, Literal, Namespace, Graph
from rdflib.exceptions import *
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor
import os.path
import sparql as sparqlQueries
from polyglot.detect import Detector
from polyglot.detect.base import UnknownLanguage
import sys
sys.path.append('../pkg/')
from pkg_ns import *


# setup logging
logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format="[%(name)s - %(levelname)s] %(message)s",
)
logger = logging.getLogger("graph_builder")


# global variables
num_topics = 7
pref_format = "xml"
num_records = 0
authors = set()
num_workers = 12
lock_graph = threading.Lock() # lock to handle concurrence


def assign_label_keyword(keywords, graph: Graph):
    for uri in keywords:
        graph.add(
            (
                uri, 
                RDFS.label, 
                Literal(keywords[uri])
            )
        )
        split_uri = uri.split("/")
        split_uri = [x for x in split_uri if x]
        graph.add(
            (
                uri, 
                PURL_TERMS.identifier,
                Literal(split_uri[-1])
            )
        )


def assign_label_tmf(topics, graph: Graph):
    for uri in topics:
        graph.add(
            (
                URIRef(uri), 
                RDFS.label,
                Literal(topics[uri])
            )
        )
        split_uri = uri.split("/")
        split_uri = [x for x in split_uri if x]
        graph.add(
            (
                URIRef(uri), 
                PURL_TERMS.identifier, 
                Literal(split_uri[-1])
            )
        )


def assign_type(topics, subject_type, graph: Graph):
    for topic in topics:
        graph.add(
            (
                topic, 
                RDF.type, 
                subject_type
            )
        )


def get_topics(text, lang="english"):
    """
    Sends a POST request to TellMeFirst and retrieves n topics (Where n is equal to num_topics).
    :return: List of strings containing the topic URIs extracted by TellMeFirst
    """
    global num_topics
    connection_attempts = 10
    topics = {}

    # TellMeFirst API interaction
    files = {"text": text, "numTopics": num_topics, "lang": lang}

    while connection_attempts > 0:
        try:
            r = requests.post(
                url="http://tellmefirst.polito.it:2222/rest/classify", files=files
            )

            if r.status_code != 200: 
                raise Exception("HTTP wrong response") 
            break
       
        except requests.exceptions.HTTPError as h:
            logger.error("HTTP invalid response error: %s, when retrieving topics from TMF API" % h)
            _thread.interrupt_main()
        
        except requests.exceptions.ConnectionError as c:
            connection_attempts -= 1
            if connection_attempts > 0:
                logger.debug("Unable to connect to remote endpoint: %s, remaining attempts: %d" % c, connection_attempts)
                continue
            else:
                logger.error("Unable to connect to remote endpoint: %s, no more remaining attempts when retrieving topics from TMF API" % c)
                _thread.interrupt_main()
        
        except requests.exceptions.Timeout as t:
            connection_attempts -= 1
            if connection_attempts > 0:
                logger.debug("Connection timeout: %s, remaining attempts: %d" % t, connection_attempts)
                continue
            else:
                logger.error("Connection timeout: %s, no more remaining attempts when retrieving topics from TMF API" % c)
                _thread.interrupt_main()
        
        except requests.exceptions.RequestException as e:
            logger.error("Requests generic error: %s, when retrieving topics from TMF API" % e)
            _thread.interrupt_main()
        
        except Exception as exc:
            logger.error("Generic error when retrieving topics from TMF API: %s" % e)
            _thread.interrupt_main()

    try: 
        for resource in r.json()["Resources"]:
            topics.update({resource["@uri"]: resource["@label"]})
    except ValueError as e:
        logging.error("Failed to decode the JSON from TMF: %s" % e)
        _thread.interrupt_main()

    return topics


def add_author(author, authors, graph: Graph):
    """
    Add author entity to the graph
    """
    if not author["authority"]:
        if author["author"]:
            author["authority"] = quote(author["author"])
        else:
            return
    if author["authority"] not in authors:
        # author type
        graph.add(
            (
                PKG_RESOURCE[author["authority"]], 
                RDF.type, 
                PKG_AUTHOR
            )
        )
        # add author name relationship
        graph.add(
            (
                PKG_RESOURCE[author["authority"]], 
                FOAF.name, 
                Literal(author["author"])
            )
        )
        # add identifier property
        graph.add(
            (
                PKG_RESOURCE[author["authority"]],
                PURL_TERMS.identifier,
                Literal(author["authority"]),
            )
        )
        # add label property
        graph.add(
            (
                PKG_RESOURCE[author["authority"]], 
                RDFS.label, 
                Literal(author["author"])
            )
        )

        authors.add(author["authority"])


def build_graph_from_iris_dump(publicationsDumpPath: str, graph=Graph()) -> Graph:

    # worker job
    def process_record(record):

        global lock_graph
        keywords = []
        keywords_clean = []
        tmf_topics = []
        pub_id = None
        pub_title = None
        pub_abstract = None
        pub_date = None
        journal_id = None
        journal_title = None

        lock_graph.acquire() # acquire graph LOCK

        try:

            ###############
            # Publication #
            ###############
            pub_id = str(record["handle"]).split("/")[1] # publication handle example: 11583/1653585
            graph.add(
                (
                    PKG_RESOURCE[pub_id], 
                    RDF.type, 
                    PKG_PUBLICATION
                )
            )

            graph.add(
                (
                    PKG_RESOURCE[pub_id],
                    PURL_TERMS.identifier,
                    Literal(pub_id)
                )
            )

            pub_title = record["metadata"]["dc.title"][0]["value"]
            graph.add(
                (
                    PKG_RESOURCE[pub_id], 
                    RDFS.label, 
                    Literal(pub_title)
                )
            )

            try:
                pub_abstract = record['metadata']['dc.description.abstract'][0]['value']
                graph.add(
                    (
                        PKG_RESOURCE[pub_id],
                        PURL_TERMS.abstract,
                        Literal(pub_abstract)
                    )
                )
            except KeyError as exc:
                logger.error("Missing abstract for publication %s" % pub_id)
                pass
            
            try:
                pub_date = str(record['metadata']['dc.date.issued'][0]['value'])
                graph.add(
                    (
                        PKG_RESOURCE[pub_id],
                        PURL_TERMS.dateSubmitted,
                        Literal(pub_date, datatype=XSD.date)
                    )
                )
            except KeyError as exc:
                logger.error("Missing date for publication %s" % pub_id)
                pass

            ###################
            # Author Keywords #
            ###################
            try:
                keywords = record["metadata"]["dc.subject.keywords"][0]["value"]
                if keywords:
                    keywords = (
                        keywords.replace("#", ";")
                                .replace("\t", ";")
                                .replace("\r\n", ";")
                                .replace(",", ";")
                                .replace("·", ";")
                                .split(";")
                    )
                    keywords_clean = [str(quote(t.strip())) for t in keywords]
                    keywords_clean = [
                        PKG_RESOURCE[t] for t in keywords_clean if len(t) > 0
                    ]
                    assign_label_keyword(dict(zip(keywords_clean, keywords)), graph)
                    assign_type(keywords_clean, PKG_AUTHORKEYWORD, graph)

                    for keyword in keywords_clean:
                        graph.add( # link publication and keyword
                            (
                                PKG_RESOURCE[pub_id],
                                PURL_TERMS.subject,
                                keyword,
                            )
                        )             
            except KeyError as exc:
                logger.error("Missing author keywords for publication %s" % pub_id)
                pass

            ##########
            # Topics #
            ##########
            if pub_abstract is not None and len(pub_abstract) > 0:
                try:                    
                    detector = Detector(pub_abstract)
                    
                    if detector.reliable is False:
                        logger.debug("Warning, polyglot unable to reliably detect the language for the following abstract: %s" % pub_abstract)
                    
                    if detector.language.code == "it":
                        lock_graph.release() # release graph LOCK
                        tmf_topics = get_topics(pub_abstract, "italian")
                        lock_graph.acquire() # acquire graph LOCK
                    elif detector.language.code == "en":
                        lock_graph.release() # release graph LOCK
                        tmf_topics = get_topics(pub_abstract, "english")
                        lock_graph.acquire() # acquire graph LOCK
                    else:
                        raise Exception("Language not recognized as IT or EN") 

                    if len(tmf_topics) > 0:
                        assign_label_tmf(tmf_topics, graph)
                        tmf_topics = [URIRef(uri) for uri in [*tmf_topics]]
                        assign_type(tmf_topics, PKG_TMFRESOURCE, graph)
                        for topic in tmf_topics:
                            graph.add( # link publication and topics
                                (
                                    PKG_RESOURCE[pub_id],
                                    PURL_TERMS.subject,
                                    topic,
                                )
                            )
                except UnknownLanguage as exc:
                    logger.error("Error, polyglot UnknownLanguage exception: %s, no topics added for this publication, continuing." % exc.args)
                    
                except Exception as exc:
                    logger.error("Error, exception thrown when adding topics: %s. no topics added for this publication, continuing." % exc.args)
            
            # test wether lock hasn't been acquired due to exception
            if not lock_graph.locked():
                lock_graph.acquire()

            ###########
            # JOURNAL #
            ###########
            try:
                journal_id = str(record['lookupValues']['jissn']).strip()
                graph.add(
                    (
                        PKG_RESOURCE[journal_id],
                        RDF.type,
                        PKG_JOURNAL
                    )
                )

                graph.add(
                    (
                        PKG_RESOURCE[journal_id],
                        PURL_TERMS.identifier,
                        Literal(journal_id)
                    )
                )

                graph.add( # link journal and publication
                    (
                        PKG_RESOURCE[pub_id],
                        PURL_TERMS.publisher,
                        PKG_RESOURCE[journal_id]
                    )
                )

                journal_title = str(record['lookupValues']['jtitle'])
                if journal_title is not None:
                    graph.add(
                        (
                            PKG_RESOURCE[journal_id],
                            RDFS.label,
                            Literal(journal_title)
                        )
                    )
                    graph.add(
                        (
                            PKG_RESOURCE[journal_id],
                            PURL_TERMS.title,
                            Literal(journal_title)
                        )
                    )
                else:
                    graph.add(
                        (
                            PKG_RESOURCE[journal_id],
                            RDFS.label,
                            Literal(journal_id)
                        )
                    )
                    
            except KeyError as exc:
                logger.error("Missing journal id or journal title for journal with id: %s" % journal_id)
                pass

            # link publication and author
            author = record['internalAuthors'][0]
            add_author(author, authors, graph)
            if author['authority']:
                graph.add(
                    (
                        PKG_RESOURCE[pub_id],
                        PURL_TERMS.creator,
                        PKG_RESOURCE[author['authority']]
                    )
                )

            # link publication and contributors
            for author in record['internalAuthors'][1:]:
                add_author(author, authors, graph)
                if author['authority']:
                    graph.add(
                        (
                            PKG_RESOURCE[pub_id],
                            PURL_TERMS.contributor,
                            PKG_RESOURCE[author['authority']]
                        )
                    )

            lock_graph.release() # release graph LOCK

        except TypeCheckError as exc:
            logger.error("type check error: %s" % exc.args)
            _thread.interrupt_main()         
        except SubjectTypeError as exc:
            logger.error("subject type error: %s" % exc.args)
            _thread.interrupt_main()    
        except PredicateTypeError as exc:
            logger.error("predicate type error: %s" % exc.args)
            _thread.interrupt_main()
        except ObjectTypeError as exc:
            logger.error("object type error: %s" % exc.args)
            _thread.interrupt_main()
        except ContextTypeError as exc:
            logger.error("context type error: %s" % exc.args)
            _thread.interrupt_main()
        except ParserError as exc:
            logger.error("parser error: %s" % exc.args)
            _thread.interrupt_main()
        except Error as exc:
            logger.error("generic rdflib error: %s" % exc.args)
            _thread.interrupt_main()
        except Exception as exc:
            logger.error("generic exception: %s" % exc.args)
            pass

    # read json file
    with open(publicationsDumpPath, "r") as file:
        content = file.read()

    # create records list, every element in records is a dictionary
    records = json.loads(content)["records"]
    global num_records
    num_records = len(records)

    # spawn workers and process the records
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        executor.map(process_record, records)

    return graph


def add_images_to_topics(g: Graph):
    """
    for each topic in the graph, adds the FOAF.image relation to the URIref of the retrived image.
    """
    for topic in g.subjects(RDF.type, PKG_TMFRESOURCE):
        # try getting the thumbnail from dbpediagraph
        topicImgURI = get_thumbnail_from_dbpedia(topic)
        if topicImgURI != "":
            g.add((topic, FOAF.img, URIRef(topicImgURI)))
            logger.debug("Added {uri}".format(uri=topicImgURI), end="\r")

    logger.debug("\nAdded images to topic entities.\n")


def get_thumbnail_from_dbpedia(topic: str) -> str:
    query = sparqlQueries.set_dbpedia_thumbnail_query(topic)
    result = requests.get(
        "https://dbpedia.org/sparql?default-graph-uri=http%3A%2F%2Fdbpedia.org&format=json&query="
        + quote(query)
    )

    if not result.ok:
        return ""  # something went wrong with the http request
    else:
        bindingsList = result.json()["results"]["bindings"]
        if len(bindingsList) == 0:
            return ""  # no thumbnail available from dbpedia
        else:
            return bindingsList[0]["thumbnail"]["value"].replace(
                "http://", "https://"
            )  # uri of thumbnail


def add_topic_abstracts_to_graph(file):
    graph = Graph()

    # Read static file of topics
    topics = ""
    with open(file) as f:
        topics = json.load(f)

    for topic in topics:
        url = topic["url"]
        abstract = get_abstract_from_dbpedia(url)
        if abstract != "":
            graph.add((URIRef(url), PURL_TERMS.abstract, Literal(abstract)))

    return graph


def get_abstract_from_dbpedia(topic):
    query = sparqlQueries.set_dbpedia_abstract_query(topic)
    result = requests.get(
        "https://dbpedia.org/sparql?default-graph-uri=http%3A%2F%2Fdbpedia.org&format=json&query="
        + quote(query)
    )
    if not result.ok:
        return ""  # something went wrong with the http request
    else:
        bindingsList = result.json()["results"]["bindings"]
        if len(bindingsList) == 0:
            return ""  # no abstract available from dbpedia
        else:
            return bindingsList[0]["abstract"]["value"]  # uri of thumbnail


def save_graph_to_file(graph, outputFilename):
    global pref_format
    serialized = graph.serialize(format=pref_format)
    with open(outputFilename, "wb") as file:
        file.write(serialized)


def update(dump, old_rdf, outputFilename):
    global pref_format
    old_graph = Graph()
    logger.debug("Parsing old graph...")
    old_graph.parse(old_rdf, pref_format)
    logger.debug("Old graph parsed!")
    new_graph = build_graph_from_iris_dump(dump)

    graph = old_graph + new_graph
    save_graph_to_file(graph, outputFilename)


def build(dump, outputFilename):
    graph = Graph()
    logger.debug("Loading dump: " + dump)
    graph = build_graph_from_iris_dump(dump)
    save_graph_to_file(graph, outputFilename)


def add_images(input_file, output_file):
    global pref_format
    graph = Graph()
    graph.parse(input_file, format=pref_format)
    logger.debug("Adding images to topics...\n")
    add_images_to_topics(graph)
    outputFilename = output_file
    save_graph_to_file(graph, outputFilename)


def add_abstracts(input_file, output_file):
    if os.path.isfile(input_file):
        graph = add_topic_abstracts_to_graph(input_file)
        outputFilename = output_file
        save_graph_to_file(graph, outputFilename)
    else:
        logger.debug(input_file + " not found")


def suggestions(suggestionsFile, old_rdf, outputFilename):
    '''
    Adds the suggestions to the PKG, the result is the Enhanced PKG.
    '''
    global pref_format
    graph = Graph()
    graph.parse(old_rdf, pref_format)

    # read json file
    with open(suggestionsFile, "r") as file:
        suggestions = file.read()

    # create records list, every element is a dictionary
    records = json.loads(suggestions)
    for publication in records:
        for field in records[publication]:
            if "creator" in field:
                sug_field = PKG_SUG_CREATOR
            elif "contributor" in field:
                sug_field = PKG_SUG_CONTRIBUTOR
            elif "publisher" in field:
                sug_field = PKG_SUG_JOURNAL
            elif "subject" in field:
                sug_field = PKG_SUG_TOPIC
            for value in records[publication][field]:
                for key in value:
                    graph.add(
                        (
                            URIRef(publication), 
                            URIRef(sug_field), 
                            URIRef(key)
                        )
                    )

    save_graph_to_file(graph, outputFilename)


# --------- #
# -  Main - #
# --------- #
if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description="Graph Builder")
    parser.add_argument(
        "-b",
        "--build",
        help="build rdf file starting from the json dump/suggestions json",
        type=str,
    )
    parser.add_argument("-i", "--images", help="get images for the rdf file")
    parser.add_argument(
        "-a", "--topics-abstracts", help="get abstracts for the topics' json file"
    )
    parser.add_argument(
        "-t",
        "--num-topics",
        help="specify number of topics to extract with TellMeFirst",
        default=7,
    )
    parser.add_argument(
        "-w",
        "--num-workers",
        help="specify number of workers to spawn when processing the records",
        default=12,
    )
    parser.add_argument(
        "-u",
        "--update",
        help="update previously generated rdf file (updatedGraph = oldGraph UNION newGraph)",
        type=str,
    )
    parser.add_argument(
        "-o",
        "--output",
        help="output file filename",
        default="../pkg/pkg_" + time.strftime("%Y%m%d_%H%M") + ".rdf",
        type=str,
    )
    parser.add_argument(
        "-s",
        "--suggestions",
        help="load suggestions on previously created rdf file",
        type=str,
    )
    parser.add_argument(
        "-f",
        "--format",
        help="specify rdf file format (xml by default)",
        default="xml",
        type=str,
    )
    args = parser.parse_args()

    num_topics = int(args.num_topics)
    pref_format = args.format
    num_workers = int(args.num_workers)

    # Perform required tasks
    if args.build and not args.update and not args.suggestions:
        build(args.build, args.output)

    if args.suggestions:
        suggestions(args.build, args.suggestions, args.output)

    if args.update:
        update(args.build, args.update, args.output)

    if args.images:
        add_images(args.images, args.output)

    if args.topics_abstracts:
        add_abstracts(args.topics_abstracts, args.output)

    # if no task required, exit
    if not args.build and not args.update and not args.suggestions and not args.images and not args.topics:
        logger.error("wrong usage, arguments given: %s" % args)
        exit(-1)