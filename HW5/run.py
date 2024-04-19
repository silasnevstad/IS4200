from es import ElasticSearch
from constants import GROUP_QUERIES

es = ElasticSearch()

with open("run_file_1000.txt", "w") as run_file:
    for query_id, query_text in GROUP_QUERIES.items():
        response = es.search(q=query_text, size=1000)
        rank = 1
        for doc in response['hits']['hits']:
            if ' ' in doc['_id']:
                continue

            run_file.write(f"{query_id} Q0 {doc['_id']} {rank} {doc['_score']} es_search\n")
            rank += 1