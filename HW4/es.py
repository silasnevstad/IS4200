from elasticsearch import Elasticsearch as ElasticSearchClient
from elasticsearch.helpers import scan
from constants import CLOUD_ID, HTTP_AUTH, INDEX_NAME

class ElasticSearch:
    def __init__(self):
        self.es = ElasticSearchClient(cloud_id=CLOUD_ID, request_timeout=10000, http_auth=HTTP_AUTH)

    def fetch_link_graph(self):
        query = {"query": {"match_all": {}}, "_source": ["url", "inlinks", "outlinks"]}
        link_graph = {}
        num_pages = 0

        for doc in scan(self.es, query=query, index=INDEX_NAME):
            num_pages += 1
            if num_pages % 1000 == 0:
                print(f"Fetched {num_pages} pages.")
            url = doc['_source']['url']
            inlinks = doc['_source'].get('inlinks', [])
            outlinks = doc['_source'].get('outlinks', [])
            link_graph[url] = {'inlinks': set(inlinks), 'outlinks': set(outlinks)}

        return link_graph

    def search_top_n(self, search_query, n=1000):
        es_query = {
            "query_string": {
                "query": search_query,
                "default_field": "content"
            }
        }

        response = self.es.search(index=INDEX_NAME, query=es_query, size=n)

        top_documents = []
        for hit in response['hits']['hits']:
            top_documents.append(hit['_source'])

        return top_documents

    # def fetch_authors_of_pages_with_no_or_empty_outlinks(self):
    #     query = {
    #         "query": {
    #             "bool": {
    #                 "should": [
    #                     {"bool": {"must_not": {"exists": {"field": "outlinks"}}}},
    #                 ],
    #             }
    #         },
    #         "_source": ["url", "authors"]
    #     }
    #
    #     num_pages = 0
    #
    #     for doc in scan(self.es, query=query, index=INDEX_NAME):
    #         num_pages += 1
    #         if num_pages % 1000 == 0:
    #             print(f"Processed {num_pages} pages.")
    #         url = doc['_source']['url']
    #         authors = doc['_source'].get('authors', 'Author unknown')
    #         print(f"URL: {url}, Authors: {authors}")
    #
    #     print(f"Total pages processed: {num_pages}")