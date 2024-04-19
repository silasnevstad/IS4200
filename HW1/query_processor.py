import os
from elasticsearch import Elasticsearch
import math
from gensim.parsing import PorterStemmer

ES_HOST = "http://localhost:9200"
INDEX_NAME = "ap89_collection"

def get_stop_words():
    """
    Fetch the stop words from the stoplist.txt file.

    :return: A list of stop words.
    """
    with open('../config/stoplist.txt', 'r') as file:
        stop_words = file.read().splitlines()
    return stop_words

def calculate_vocabulary_size(es):
    """
    Dynamically calculates the vocabulary size (unique terms in the collection) for the given Elasticsearch index.

    :param es: Elasticsearch client instance.
    :return: The vocabulary size.
    """
    response = es.search(
        index=INDEX_NAME,
        aggs={
            "unique_terms": {
                "cardinality": {
                    "field": "text.keyword"
                }
            },
        },
        size=0
    )
    return response['aggregations']['unique_terms']['value']

def save_corpus_statistics(avg_doc_length, total_docs, vocab_size, doc_stats_dict):
    """
    Save the corpus statistics to file.

    :param avg_doc_length: The average document length.
    :param total_docs: The total number of documents.
    :param vocab_size: The vocabulary size.
    :param doc_stats_dict: A dictionary containing the document lengths.
    :return: None
    """
    with open("../IR_data/AP_DATA/doc_stats.txt", "w") as file:
        file.write(f"AVG_DOC_LENGTH = {avg_doc_length}\n")
        file.write(f"TOTAL_NUM_DOCS = {total_docs}\n")
        file.write(f"VOCABULARY_SIZE = {vocab_size}\n")
    with open("../IR_data/AP_DATA/doc_length_dict.txt", "w") as file:
        file.write(str(doc_stats_dict))

def load_corpus_statistics():
    """
    Load the corpus statistics from file.

    :return: The corpus statistics (average document length, total number of documents, vocabulary size
             and document length dictionary).
    """
    with open("../IR_data/AP_DATA/doc_stats.txt", "r") as file:
        stats = file.read()
        avg_doc_length = int(stats.split("\n")[0].split(" = ")[1])
        total_docs = int(stats.split("\n")[1].split(" = ")[1])
        vocab_size = int(stats.split("\n")[2].split(" = ")[1])
    with open("../IR_data/AP_DATA/doc_length_dict.txt", "r") as file:
        doc_stats_dict = eval(file.read())
    return doc_stats_dict, avg_doc_length, total_docs, vocab_size

def calculate_corpus_statistics(es):
    doc_stats_dict = dict()

    # Check if corpus statistics have already been calculated and saved to file
    if os.path.exists("../IR_data/AP_DATA/doc_stats.txt") and os.path.exists("../IR_data/AP_DATA/doc_length_dict.txt"):
        print("Loading corpus statistics from file...")
        return load_corpus_statistics()
    elif os.path.exists("../IR_data/AP_DATA/doc_length_dict.txt"):
        print("Loading document length dictionary from file...")
        with open("../IR_data/AP_DATA/doc_length_dict.txt", "r") as file:
            doc_stats_dict = eval(file.read())
    else:
        print("Calculating corpus statistics...")
        # Fetch the length of each document and store it in a dictionary
        with open("../IR_data/AP_DATA/doclist.txt", "r") as doc:
            for line in doc:
                if line != "":
                    if line.split(" ")[0] != "0":
                        doc_id = line.split(" ")[1].strip()
                        resp = es.termvectors(
                            index=INDEX_NAME,
                            id=doc_id,
                            fields=['text'],
                            positions=False,
                            term_statistics=True,
                            field_statistics=True,
                        )
                        try:
                            term_vectors = resp['term_vectors']
                        except KeyError:
                            term_vectors = {}
                        if len(term_vectors) > 0:
                            doc_length = 0
                            for term in term_vectors['text']['terms']:
                                doc_length += term_vectors['text']['terms'][term]['term_freq']
                            doc_stats_dict.update({doc_id: doc_length})
                        else:
                            doc_stats_dict.update({doc_id: 0})
                        print("Finished " + doc_id)

    total_docs = len(doc_stats_dict.keys())
    avg_doc_length = round(sum(doc_stats_dict.values())/total_docs)

    vocab_size = calculate_vocabulary_size(es)

    print(f"Average document length: {avg_doc_length}")
    print(f"Total number of documents: {total_docs}")
    print(f"Vocabulary size: {vocab_size}")

    # save statistics to file
    save_corpus_statistics(avg_doc_length, total_docs, vocab_size, doc_stats_dict)

    return doc_stats_dict, avg_doc_length, total_docs, vocab_size

def clean_text(text, stopwords):
    """
    Cleans the given text by removing unnecessary spaces, stop words, and non-ASCII/alphanumeric characters.

    :param text: The text to clean.
    :param stopwords: The list of stop words.
    :return: The cleaned text.
    """
    cleaned_text = ''.join([char for char in text if char.isalnum() or char.isspace()]).strip()
    cleaned_words = [word.lower() for word in cleaned_text.split() if word.lower() not in stopwords]
    return ' '.join(cleaned_words)

def parse_queries(stop_words):
    """
    Parse the queries from the given file and clean the text.
    :param stop_words: The list of stop words.
    :return: A dictionary containing the parsed and cleaned queries.
    """
    queries = {}
    with open('../IR_data/AP_DATA/query_desc.51-100.short.txt', 'r') as file:
        for line in file.readlines():
            parts = line.strip().split('.', 1)
            if len(parts) == 2:
                query_num, query_text = parts[0], parts[1].strip()
                queries[query_num] = clean_text(query_text, stop_words)
    return queries

def get_term_vectors(term):
    """
    Fetch the term vectors for a given term from Elasticsearch.
    :param term: The term to fetch the vectors for.
    :return: The term vectors for the given term.
    """
    es = Elasticsearch([ES_HOST], timeout=5)
    try:
        resp = es.search(index=INDEX_NAME, query={"match": {"text": term}}, size=1000)
    except Exception as e:
        print(f"Error fetching term vectors for term {term}: {e}")

    result_tf = []
    result_ttf = 0
    for hit in range(len(resp['hits']['hits'])):
        doc_id = resp['hits']['hits'][hit]['_id']
        docTF = resp["hits"]["hits"][hit]["_score"]
        result_tf.append((doc_id, docTF))
        result_ttf += docTF

    return result_tf, resp['hits']['total']['value'], result_ttf

def save_scores(scores, model_name, query_num):
    out = open("./results/results_" + model_name + ".txt", "a")
    rank = 1
    for doc_id, score in scores.items():
        out.write(f"{query_num} Q0 {doc_id} {rank} {score} Exp\n")
        rank += 1
    out.close()

def calculate_es_built_in_score(es, term_freqs, query_num):
    """
    Calculate the score of a document for a given query using Elasticsearch's built-in scoring function.

    :param es: Elasticsearch client instance.
    :param term_freqs: The term frequency in the document.
    :param query_num: The query number.
    """
    es_scores = {}

    # Calculate the scores for each document
    for doc_id in term_freqs.keys():
        es_score = 0

        # Calculate the Okapi TF score for each term in the document
        for term in term_freqs[doc_id].keys():
            response = es.search(
                index=INDEX_NAME,
                query= {
                    "bool": {
                        "must": {
                            "match": {
                                "text": term
                            },
                        },
                        "filter": {
                            "term": {
                                "_id": doc_id
                            },
                        },
                    },
                },
                _source=False,
                size=1
            )
            hits = response['hits']['hits']
            if len(hits) > 0:
                for hit in hits:
                    es_score += hit['_score']
        es_scores.update({doc_id: es_score})

    # sort and return top 1000
    sorted_es_scores = dict(sorted(es_scores.items(), key=lambda item: item[1], reverse=True)[:1000])
    save_scores(sorted_es_scores, "es", query_num)

def calculate_okapi_tf(term_freqs, avg_doc_length, doc_len_dict, query_num):
    """
    Calculate the Okapi TF score for a term in a document.

    :param term_freqs: The term frequency in the document.
    :param avg_doc_length: The average document length.
    :param doc_len_dict: The dictionary with the length of each document.
    :param query_num: The query number.
    """
    okapi_tf_scores = {}

    # Calculate the scores for each document
    for doc_id in term_freqs.keys():
        doc_len = int(doc_len_dict[doc_id])
        okapi_tf_wd = 0

        # Calculate the Okapi TF score for each term in the document
        for term in term_freqs[doc_id].keys():
            tf = term_freqs[doc_id][term]
            d = tf + 0.5 + 1.5 * (doc_len / float(avg_doc_length))
            okapi_tf_wd += (tf / d)
        okapi_tf_scores.update({doc_id: okapi_tf_wd})

    # sort and return top 1000
    sorted_okapi_tf_scores = dict(sorted(okapi_tf_scores.items(), key=lambda item: item[1], reverse=True)[:1000])
    save_scores(sorted_okapi_tf_scores, "okapi_tf", query_num)

def calculate_tfidf(term_freqs, term_dfs, total_docs, avg_doc_length, doc_len_dict, query_num):
    """
    Calculate the TF-IDF score for a term in a document.

    :param term_freqs: The term frequency in the document.
    :param term_dfs: The document frequency of the term.
    :param total_docs: The total number of documents.
    :param avg_doc_length: The average document length.
    :param doc_len_dict: The dictionary with the length of each document.
    :param query_num: The query number.
    """
    tfidf_scores = {}

    for doc_id in term_freqs.keys():
        doc_len = int(doc_len_dict[doc_id])
        tf_idf = 0
        for term in term_freqs[doc_id].keys():
            tf = term_freqs[doc_id][term]
            log = math.log(int(total_docs) / int(term_dfs[term]))
            d = tf + 0.5 + (1.5 * (doc_len / float(avg_doc_length)))
            tf_idf += ((tf / d) * log)

        tfidf_scores.update({doc_id: tf_idf})

    sorted_tfidf = dict(sorted(tfidf_scores.items(), key=lambda item: item[1], reverse=True)[:1000])
    save_scores(sorted_tfidf, "tfidf", query_num)

def calculate_bm25(
        term_freqs,
        term_dfs,
        total_docs,
        avg_doc_length,
        doc_len_dict,
        query_terms,
        query_num,
        k1=1.2,
        b=0.75,
        k2=100,
):
    """
    Calculate the BM25 score for a term in a document.

    :param term_freqs: The term frequency in the document.
    :param term_dfs: The document frequency of the term.
    :param total_docs: The total number of documents.
    :param avg_doc_length: The average length of a document.
    :param doc_len_dict: The dictionary with the length of each document.
    :param query_terms: The query text to calculate the score for.
    :param query_num: The query number.
    :param k1: The k1 parameter for BM25.
    :param b: The b parameter for BM25.
    :param k2: The k2 parameter for BM25.
    """
    okapi_bm25_scores = {}

    for doc_id in term_freqs.keys():
        doc_len = int(doc_len_dict[doc_id])
        okapiBM25 = 0
        for term in term_freqs[doc_id].keys():
            tf_wd = term_freqs[doc_id][term]

            log = math.log(int(total_docs) + 0.5 / term_dfs[term] + 0.5)

            lt_num = tf_wd + (k1 * tf_wd)
            lt_denom = tf_wd + (k1 * ((1 - b) + (b * (doc_len / float(avg_doc_length)))))
            large_term = lt_num / lt_denom

            tf_wq = query_terms.count(term)
            tf_num = tf_wq + (k2 * tf_wq)
            tf_denom = tf_wq + k2
            tf_term = tf_num / tf_denom

            okapiBM25 += (log * large_term * tf_term)
        okapi_bm25_scores.update({doc_id: okapiBM25})

    sorted_bm25 = dict(sorted(okapi_bm25_scores.items(), key=lambda item: item[1], reverse=True)[:1000])
    save_scores(sorted_bm25, "bm25", query_num)

def calculate_laplace_smoothing(term_freqs, stemmed_query, vocab_size, doc_len_dict, query_num):
    """
    Calculate the Laplace smoothing score for a term in a document.

    :param term_freqs: The term frequency in the document.
    :param stemmed_query: The query text to calculate the score for.
    :param vocab_size: The size of the vocabulary.
    :param doc_len_dict: The dictionary with the length of each document.
    :param query_num: The query number.
    """
    laplace_scores = {}

    for doc_id in term_freqs.keys():
        doc_len = float(doc_len_dict[doc_id])

        for word in stemmed_query:
            if word not in term_freqs[doc_id].keys():
                term_freqs[doc_id].update({word: 0})
        laplace_score = 0

        for term in term_freqs[doc_id].keys():
            tf = term_freqs[doc_id][term]
            new_tf = tf + 1.0
            new_doc_len = doc_len + float(vocab_size)
            p_laplace = new_tf / new_doc_len

            laplace_score += math.log(p_laplace)
        laplace_scores.update({doc_id: laplace_score})

    sorted_laplace = dict(sorted(laplace_scores.items(), key=lambda item: item[1], reverse=True)[:1000])
    save_scores(sorted_laplace, "laplace", query_num)

def calculate_jelinek_mercer(term_freqs, term_ttfs, stemmed_query, doc_len_dict, query_num):
    """
    Calculate the Jelinek-Mercer score for a term in a document.
    :param term_freqs: The term frequency in the document.
    :param term_ttfs: The term frequency in the entire collection.
    :param stemmed_query: The query text to calculate the score for.
    :param doc_len_dict: The dictionary with the length of each document.
    :param query_num: The query number.
    """
    jelinek_mercer_scores = {}
    corpus_prob = 0.21
    total_doc_len = sum(doc_len_dict.values())

    for doc_id in term_freqs.keys():
        doc_len = float(doc_len_dict[doc_id])

        for word in stemmed_query:
            if word not in term_freqs[doc_id].keys():
                term_freqs[doc_id].update({word: 0})
        jelinek_mercer_score = 0

        for term in term_freqs[doc_id].keys():
            tf = term_freqs[doc_id][term]
            ttf = term_ttfs.get(term, 0)

            fg = tf / doc_len
            bg1 = max(ttf - tf, 0.1)
            bg2 = max(total_doc_len - doc_len, 1.0)
            bg = bg1 / bg2
            p_jm = (corpus_prob * fg) + ((1.0 - corpus_prob) * bg)
            if p_jm <= 0:
                p_jm = 1e-10

            jelinek_mercer_score += math.log(p_jm)
        jelinek_mercer_scores.update({doc_id: jelinek_mercer_score})

    sorted_jelinek_mercer = dict(sorted(jelinek_mercer_scores.items(), key=lambda item: item[1], reverse=True)[:1000])
    save_scores(sorted_jelinek_mercer, "jelinek_mercer", query_num)

def get_stemmed_query(query_text, stopwords):
    """
    Cleans the given query line by removing stop words and stemming the words.
    :param query_text: The query text to clean.
    :param stopwords: The list of stop words.
    :return: Query number, query text, stemmed query text
    """
    raw_data = query_text.split()

    for i in stopwords:
        while i in raw_data:
            raw_data.remove(i)

    stemmedArray = raw_data
    p = PorterStemmer()

    for i in range(1, len(stemmedArray)):
        while stemmedArray[i] != p.stem(stemmedArray[i]):
            stemmedArray[i] = p.stem(stemmedArray[i])

    return stemmedArray

def calculate_scores(es, queries, doc_len_dict, avg_doc_length, total_docs, vocab_size, stop_words):
    for query_num, query_text in queries.items():
        query_terms = query_text.split()
        stemmed_query = get_stemmed_query(query_text, stop_words)
        print(f"Processing query {query_num}, with stemmed query: {stemmed_query}")
        term_freqs = {}
        term_dfs = {}
        term_ttfs = {}

        for term in stemmed_query:
            result_tf, result_df, result_ttf = get_term_vectors(term)

            if term not in term_ttfs.keys():
                term_ttfs.update({term: result_ttf})

            if term not in term_dfs.keys():
                term_dfs.update({term: result_df})

            for docId, docTermFreq in result_tf:
                if docId in term_freqs.keys():
                    term_freqs[docId].update({term: docTermFreq})
                else:
                    term_freqs.update({docId: {term: docTermFreq}})

        print(f"Calculating scores for query {query_num}")
        calculate_es_built_in_score(es, term_freqs, query_num)
        calculate_okapi_tf(term_freqs, avg_doc_length, doc_len_dict, query_num)
        calculate_tfidf(term_freqs, term_dfs, total_docs, avg_doc_length, doc_len_dict, query_num)
        calculate_bm25(term_freqs, term_dfs, total_docs, avg_doc_length, doc_len_dict, query_terms, query_num)
        calculate_laplace_smoothing(term_freqs, stemmed_query, vocab_size, doc_len_dict, query_num)
        calculate_jelinek_mercer(term_freqs, term_ttfs, stemmed_query, doc_len_dict, query_num)

def main():
    es = Elasticsearch([ES_HOST])

    doc_len_dict, avg_doc_length, total_docs, vocab_size = calculate_corpus_statistics(es)

    stop_words = get_stop_words()

    queries = parse_queries(stop_words)

    calculate_scores(es, queries, doc_len_dict, avg_doc_length, total_docs, vocab_size, stop_words)

if __name__ == "__main__":
    main()