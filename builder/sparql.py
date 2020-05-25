def set_dbpedia_thumbnail_query(topicUri: str) -> str:
    query = """\
        prefix dbo: <http://dbpedia.org/ontology/>
        select ?thumbnail where {{
        <{t}> dbo:thumbnail ?thumbnail .
        \
        }}""".format(t=topicUri)
    return query


def set_dbpedia_abstract_query(topic_uri):
    query = """ \
    PREFIX dbo: <http://dbpedia.org/ontology/>
    SELECT DISTINCT ?abstract
    WHERE {{
        <{t}> dbo:abstract ?abstract .
        FILTER (langMatches(lang(?abstract),"en"))
    }}
    \
    """.format(t=topic_uri)
    return query