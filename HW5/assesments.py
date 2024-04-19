import csv
import random
import re
import pickle
from es import ElasticSearch
from embeddings import get_similarity_from_texts
from constants import GROUP_QUERIES
from llm import get_relevance_grade

def parse_document_content(doc_content):
    docno_match = re.search(r'<DOCNO>(.*?)</DOCNO>', doc_content, re.DOTALL)
    title_match = re.search(r'<HEAD>(.*?)</HEAD>', doc_content, re.DOTALL)
    text_match = re.search(r'<TEXT>(.*?)</TEXT>', doc_content, re.DOTALL)

    docno = docno_match.group(1).strip() if docno_match else None
    title = title_match.group(1).strip() if title_match else None
    text = text_match.group(1).strip() if text_match else None
    return docno, title, text

def assess_document(query, title, text):
    similarity_title = get_similarity_from_texts(query, title)
    similarity_text = get_similarity_from_texts(query, text)
    overall_similarity = (similarity_title + similarity_text) / 2

    if overall_similarity > 0.65:
        return 2  # Very relevant
    elif overall_similarity > 0.35:
        return 1  # Relevant
    else:
        return 0  # Non-relevant

# es = ElasticSearch()
# all_documents = list(es.get_1000_docs())

# with open('all_documents.pkl', 'wb') as f:
#     pickle.dump(all_documents, f)

# with open('all_documents.pkl', 'rb') as f:
#     all_documents = pickle.load(f)

# print(f"Total documents: {len(all_documents)}")

# filename = "results/assessments_gpt.csv"
# with open(filename, mode='w', newline='', encoding='utf-8') as file:
#     writer = csv.writer(file)
#     writer.writerow(['QueryID', 'AssessorID', 'DocID', 'Auto Grade', 'Grade'])
#
#     for query_id, query in GROUP_QUERIES.items():
#         print(f"Assessing documents for query: {query}")
#         shuffled_docs = all_documents[:]
#         random.shuffle(shuffled_docs)
#         documents_to_assess = shuffled_docs[:210]
#         i = 0
#         for doc in documents_to_assess:
#             print(f"Assessing document {i + 1} of {len(documents_to_assess)}")
#             doc_content = doc['_source']['content']
#             doc_id, title, content = parse_document_content(doc_content)
#             if title and content:
#                 # auto_grade = assess_document(query, title, content)
#                 gpt_grade = get_relevance_grade(query, doc_id, title, content)
#                 writer.writerow([query_id, "Silas_Nevstad", doc_id, gpt_grade, ''])
#                 i += 1

def convert_csv_to_qrels(input_csv_path, output_qrels_path):
    with open(input_csv_path, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        with open(output_qrels_path, mode='w', newline='', encoding='utf-8') as qrelsfile:
            for row in reader:
                query_id = row['QueryID']
                assessor_id = row['AssessorID']
                document_id = row['DocID']
                grade = row['Grade'].strip()

                # QueryID AssessorID DocID Grade
                qrelsfile.write(f"{query_id} {assessor_id} {document_id} {grade}\n")

convert_csv_to_qrels('results/assessments_gpt.csv', 'results/qrels.txt')

# convert txt to csv
# def convert_qrels_to_csv(input_qrels_path, output_csv_path):
#     with open(input_qrels_path, mode='r', newline='', encoding='utf-8') as qrelsfile:
#         with open(output_csv_path, mode='w', newline='', encoding='utf-8') as csvfile:
#             writer = csv.writer(csvfile)
#             writer.writerow(['QueryID', 'AssessorID', 'DocID', 'Grade'])
#
#             for line in qrelsfile:
#                 query_id, assessor_id, document_id, grade = line.strip().split()
#                 writer.writerow([query_id, assessor_id, document_id, grade])
#
# convert_qrels_to_csv('results/final_qrels_random.txt', 'results/final_qrels_random.csv')