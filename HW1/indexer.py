import os
from elasticsearch import Elasticsearch

def get_stop_words():
    """
    Reads the stop words from the stoplist.txt file and returns them as a list.
    :return: A list of stop words.
    """
    with open('../config/stoplist.txt', 'r') as file:
        stop_words = file.read().splitlines()
    return stop_words


def clean_text(text):
    """
    Cleans the given text by removing non-ASCII characters and non-alphanumeric words.

    :param text: The text to clean.
    :return: The cleaned text.
    """
    cleaned_text = ' '.join([word for word in text.split() if word.isalnum() and word.isascii()])
    return cleaned_text

def parse_document(doc_content):
    """
    Parses a single document and extracts the DOCNO and TEXT content.

    :param doc_content: A string containing the content of a document.
    :return: A dictionary with DOCNO and TEXT content.
    """
    docno = None
    text = []
    in_text = False  # Flag to track if we're inside <TEXT> tags
    for line in doc_content.split('\n'):
        if '<DOCNO>' in line:
            docno = line.replace('<DOCNO>', '').replace('</DOCNO>', '').strip()
        elif '<TEXT>' in line:
            in_text = True
        elif '</TEXT>' in line:
            in_text = False
        elif in_text:
            text.append(line)
    return {'docno': docno, 'text': ' '.join(text)}

def process_corpus(directory_path):
    """
    Processes all files in the given directory, parsing each document.

    :param directory_path: Path to the directory containing the corpus files.
    :return: A generator yielding parsed documents.
    """
    for filename in os.listdir(directory_path):
        if filename.startswith('.'):  # Skip hidden/system files
            continue
        file_path = os.path.join(directory_path, filename)
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            file_content = file.read()
            docs = file_content.split('</DOC>')
            for doc in docs:
                if '<DOC>' in doc:
                    yield parse_document(doc + '</DOC>')  # Ensure closing tag is included for parsing

def index_document(es, index_name, document):
    """
    Indexes a document in Elasticsearch, ensuring there's text content before indexing.

    :param es: Elasticsearch client instance.
    :param index_name: The name of the index.
    :param document: The document to index must contain 'docno' and 'text'.
    """
    # Check if the 'text' field is present and not empty
    if 'text' in document and document['text'].strip():
        es.index(index=index_name, id=document['docno'], document=document)
        print(f"Indexed document: {document['docno']}")
    else:
        print(f"Skipped document {document['docno']} due to missing or empty 'text' field.")

def main():
    es = Elasticsearch(["http://localhost:9200"])
    index_name = 'ap89_collection'
    directory_path = '../IR_data/AP_DATA/ap89_collection'

    for document in process_corpus(directory_path):
        index_document(es, index_name, document)
        print(f"Indexed document: {document['docno']}")


if __name__ == "__main__":
    main()
