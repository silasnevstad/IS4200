from elasticsearch import Elasticsearch as ElasticSearchClient
from elasticsearch.helpers import scan
from constants import CLOUD_ID, HTTP_AUTH, INDEX_NAME

class ElasticSearch:
    def __init__(self):
        self.es = ElasticSearchClient(cloud_id=CLOUD_ID, request_timeout=10000, http_auth=HTTP_AUTH)

    def get_1000_docs(self):
        return scan(self.es, index=INDEX_NAME, query={"query": {"match_all": {}}}, size=1000)

    def search(self, q, size):
        return self.es.search(index=INDEX_NAME, q=q, size=size)

