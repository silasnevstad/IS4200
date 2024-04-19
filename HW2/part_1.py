import os
import re
from nltk.stem.snowball import SnowballStemmer
import json
import math
import gzip

DIRECTORY_PATH = "./ap89_collection/"
STOP_WORDS_FILE = "./stoplist.txt"
MEMORY_DOCUMENT_LIMIT = 1000
NUMBER_OF_DOCUMENTS = 84660

def file_parser(directory=DIRECTORY_PATH):
    """
    Parses the files in the given directory and returns a dictionary of documents.

    :param directory: The directory containing the files to be parsed.
    :return: A dictionary of documents, where the key is the document ID, and the value is the document content.
    """
    documents = {}
    doc_id = None
    collecting_text = False
    document_content = []

    # Regex patterns from the old code, assuming correctness
    doc_start_pattern = re.compile(r"<DOC>")
    doc_end_pattern = re.compile(r"</DOC>")
    doc_no_pattern = re.compile(r"<DOCNO>[\s]*(.+?)[\s]*</DOCNO>")
    text_start_pattern = re.compile(r"<TEXT>")
    text_end_pattern = re.compile(r"</TEXT>")

    for filename in os.listdir(directory):
        with open(os.path.join(directory, filename), 'r', encoding='utf-8', errors='ignore') as file:
            for line in file:
                if doc_start_pattern.search(line):
                    document_content = []
                elif doc_no_pattern.search(line):
                    doc_id = doc_no_pattern.search(line).group(1).strip()
                elif text_start_pattern.search(line):
                    collecting_text = True
                elif text_end_pattern.search(line):
                    collecting_text = False
                    documents[doc_id] = ' '.join(document_content).replace('\n', ' ')
                elif collecting_text:
                    document_content.append(line.strip())
                elif doc_end_pattern.search(line):
                    if doc_id and document_content:
                        documents[doc_id] = ' '.join(document_content).replace('\n', ' ')
                    doc_id = None
                    collecting_text = False
                    document_content = []
    return documents

# Load stop words
def load_stop_words(file_path=STOP_WORDS_FILE):
    with open(file_path, 'r') as file:
        return set(file.read().split())

# Parsing files to create document instances
files = file_parser()
instances = [{"id": doc_id, "content": content} for doc_id, content in files.items()]

# Load stop words
stop_words = load_stop_words()

class MyIndex:
    """A class representing an inverted index."""

    def __init__(self, name, stop_words, is_stem, is_compress):
        """
        Initializes the MyIndex instance.

        :param name: The name of the index.
        :param stop_words: A set of stop words.
        :param is_stem: Boolean indicating if stemming is applied.
        :param is_compress: Boolean indicating whether compression is used.
        """
        self.name = name
        self.stop_words = stop_words
        self.is_stem = is_stem
        self.is_compress = is_compress
        self.folder_name = self._determine_folder_name()
        self.terms_map = {}
        self.terms_map_rev = {}
        self.terms_id = 0
        self.terms_df = {}
        self.docs_map = {}
        self.docs_map_rev = {}
        self.docs_id = 0
        self.doc_lengths = {}
        self.terms_ttf = {}
        self.terms_tf = {}
        self.field_info = {'D': 0, 'V': 0, 'total_ttf': 0}
        self.doc_count = 0
        self.partial = 0
        self.catalog = {}
        self.new_catalog = {}
        self.tracker = 0
        self.analyzer = Analyzer(stop_words, is_stem)  # Assumes existence of Analyzer class
        print("Index Created")

    def _determine_folder_name(self):
        """
        Determines the folder name based on stemming and compression options.

        :return: A string representing the folder name.
        """
        if self.is_stem and self.is_compress:
            return "with_stem_compress"
        elif self.is_stem and not self.is_compress:
            return "with_stem_no_compress"
        elif not self.is_stem and self.is_compress:
            return "no_stem_compress"
        else:
            return "no_stem_no_compress"

    def insert_document(self, instance):
        """Inserts a document into the index, updating all relevant metrics."""
        document = Document(instance, self.analyzer)
        self.doc_count += 1
        if not document.tokens:
            return
        self._update_terms_map(document)
        self._update_document_info(document)
        self._update_document_frequency(document)
        self._update_total_term_frequency(document)
        self._update_field_info(document)
        self._calculate_term_position(document)
        self._check_and_reset()
        self._finalize_insertion()
        print(self.field_info['D'])

    def _update_terms_map(self, document):
        """Updates the mapping of terms to their unique IDs."""
        for term in {token[0] for token in document.tokens}:
            if term not in self.terms_map:
                self.terms_map[term] = self.terms_id
                self.terms_id += 1

    def _update_document_info(self, document):
        """Updates the mapping of documents to their unique IDs."""
        if document.id not in self.docs_map:
            self.docs_map[document.id] = self.docs_id
            self.doc_lengths[str(self.docs_id)] = document.length
            self.docs_id += 1

    def _update_field_info(self, document):
        """Updates the index's field information."""
        self.field_info['D'] += 1
        self.field_info['V'] = len(self.terms_map)
        self.field_info['total_ttf'] += len(document.tokens)

    def _update_document_frequency(self, document):
        """Updates the document frequency for each term."""
        for token in {token[0] for token in document.tokens}:
            self.terms_df[token] = self.terms_df.get(token, 0) + 1

    def _update_total_term_frequency(self, document):
        """Updates the total term frequency for each term."""
        for token in (token[0] for token in document.tokens):
            self.terms_ttf[token] = self.terms_ttf.get(token, 0) + 1

    def _calculate_term_position(self, document):
        """Calculates and stores term positions within documents."""
        for token in document.tokens:
            term_id, doc_id, positions = self.terms_map[token[0]], self.docs_map[token[1]], token[2]
            if term_id not in self.terms_tf:
                self.terms_tf[term_id] = {}
                self.terms_tf[term_id][doc_id] = [positions]
            else:
                self.terms_tf[term_id][doc_id] = self.terms_tf[term_id].get(doc_id, []) + [positions]

    def _serialize_index(self, is_final=False):
        """Serializes index components to disk, with optional compression."""
        # Create the folder if it doesn't exist
        folder_path = f"./{self.name}/{self.folder_name}/merge"
        os.makedirs(folder_path, exist_ok=True)

        # Determine file mode based on whether compression is used
        file_mode = "ab" if self.is_compress else "a"
        file_path = os.path.join(folder_path, f"partial_{self.partial}")

        catalog = {}
        start = 0

        with open(file_path, file_mode) as file:
            for key, inv_list in self.terms_tf.items():
                sorted_inv_list = self.sort_documents(inv_list)
                if is_final:
                    sorted_inv_list = {idx: inv_list[idx] for idx in sorted_inv_list}
                dump_line = self.format_output(sorted_inv_list)

                # compress and write to file, otherwise just write to file
                if self.is_compress:
                    compressed_line = self.gzip_compress(dump_line)
                    file.write(compressed_line)
                else:
                    file.write(dump_line)

                length = file.tell() - start
                catalog[str(key)] = [start, length]
                start += length

        self.catalog[str(self.partial)] = catalog
        self.partial += 1
        if is_final:
            self._dump_metadata()

    def _check_and_reset(self):
        """Resets the document-related structures after reaching a document limit."""
        if self.doc_count == MEMORY_DOCUMENT_LIMIT:
            self._serialize_index()
            self.terms_tf = {}
            self.doc_count = 0

    def _finalize_insertion(self):
        """Finalizes the insertion process, triggering serialization if needed."""
        if self.field_info['D'] == NUMBER_OF_DOCUMENTS:
            self._serialize_index(is_final=True)

    def _dump_metadata(self):
        """Dumps index metadata to JSON files."""
        metadata_items = {
            "term_df": self.terms_df,
            "term_ttf": self.terms_ttf,
            "field_info": self.field_info,
            "catalog": self.catalog,
            "terms_map": self.terms_map,
            "docs_map": self.docs_map,
            "doc_lengths": self.doc_lengths,
            "docs_map_rev": {value: key for key, value in self.docs_map.items()}
        }
        for filename, data in metadata_items.items():
            with open(f"./{self.name}/{self.folder_name}/{filename}.json", "w") as f:
                json.dump(data, f)

    @staticmethod
    def gzip_compress(string):
        """Compresses a string using gzip."""
        return gzip.compress(string.encode())

    @staticmethod
    def gzip_decompress(compressed):
        """Decompresses a gzip-compressed string."""
        return gzip.decompress(compressed).decode()

    @staticmethod
    def sort_documents(document_terms):
        """Sorts documents by term frequency."""
        return {k: document_terms[k] for k in sorted(document_terms, key=lambda k: len(document_terms[k]), reverse=True)}

    @staticmethod
    def format_output(document_terms):
        """Formats the document terms for output."""
        if not isinstance(document_terms, dict):
            raise ValueError("document_terms must be a dictionary")
        final_line = ' '.join(f"{doc_id},{','.join(map(str, positions))}" for doc_id, positions in document_terms.items())
        return final_line

    def merge_control(self):
        """Controls the merging process of inverted lists."""
        current_left = '0'
        current_right = '1'
        merge_function = self.merge_compress if self.is_compress else self.merge

        self.new_catalog = merge_function(
            current_left, current_right, self.catalog[current_left], self.catalog[current_right])

        for n in range(2, 85):
            print(n)
            current_left = str(int(current_left) + 100)
            current_right = str(n)
            is_final = n == 84
            if n > 2:
                os.remove(f"./{self.name}/{self.folder_name}/merge/partial_{str(int(current_left) - 100)}")
            self.new_catalog = merge_function(
                current_left, current_right, self.new_catalog, self.catalog[current_right], is_final=is_final)

        os.remove(f"./{self.name}/{self.folder_name}/merge/partial_{current_left}")
        with open(f"./{self.name}/{self.folder_name}/merge/new_catalog.json", "w") as f:
            json.dump(self.new_catalog, f)

    def merge(self, left, right, left_catalog, right_catalog, is_final=False):
        """Merges two uncompressed inverted lists."""
        new_catalog = self._merge_generic(left, right, left_catalog, right_catalog, mode="text", decompress=False, is_final=is_final)
        return new_catalog

    def merge_compress(self, left, right, left_catalog, right_catalog, is_final=False):
        """Merges two compressed inverted lists."""
        new_catalog = self._merge_generic(left, right, left_catalog, right_catalog, mode="binary", decompress=True, is_final=is_final)
        return new_catalog

    def _merge_generic(self, left, right, left_catalog, right_catalog, mode="text", decompress=False, is_final=False):
        """Generic merge function for both compressed and uncompressed lists."""
        new_catalog = {}
        file_mode = "rb" if mode == "binary" else "r"
        update_method = self.gzip_decompress if decompress else lambda x: x

        # Open the output file once for all write operations
        output_file_path = f"./{self.name}/{self.folder_name}/merge/{'final' if is_final else f'partial_{str(int(left) + 100)}'}"
        write_mode = "ab" if mode == "binary" else "a"
        with open(output_file_path, write_mode) as f_new:
            new_start = 0

            all_terms = set(left_catalog) | set(right_catalog)
            for idx, term_id in enumerate(all_terms):
                left_data = self._read_term_data(left, term_id, left_catalog, file_mode, update_method) if term_id in left_catalog else {}
                right_data = self._read_term_data(right, term_id, right_catalog, file_mode, update_method) if term_id in right_catalog else {}
                merged_data = self._update_partial_term(left_data, right_data) if term_id in right_catalog else left_data
                new_start = self._write_term_data(term_id, merged_data, new_catalog, new_start, f_new, mode)

                # if idx % 1000 == 0:
                #     print(f"{idx / len(all_terms) * 100:.2f}%")

        return new_catalog

    def _write_term_data(self, term_id, term_data, new_catalog, new_start, f_new, mode):
        """Writes term data to the file and updates the catalog, file passed as argument."""
        term_data = term_data if isinstance(term_data, dict) else self.convert_to_dict(term_data)
        new_line = self.format_output(term_data) if mode == "text" else self.gzip_compress(self.format_output(term_data))
        f_new.write(new_line)
        new_length = f_new.tell() - new_start
        new_catalog[term_id] = [new_start, new_length]
        return new_start + new_length

    def _read_term_data(self, side, term_id, catalog, mode, update_method):
        """Reads and optionally decompresses term data from a file."""
        if term_id not in catalog:
            return {}
        offset, length = catalog[term_id]
        with open(f"./{self.name}/{self.folder_name}/merge/partial_{side}", mode) as file:
            file.seek(offset)
            data = file.read(length)
        processed_data = update_method(data) if mode == "rb" else data
        return processed_data

    @staticmethod
    def convert_to_dict(data_str):
        """Parses the custom data format into a dictionary, handling empty or malformed entries."""
        result = {}
        for entry in data_str.strip().split(' '):
            if not entry:
                continue

            parts = entry.split(',')
            parts = [part for part in parts if part]
            doc_id = int(parts[0])
            positions = [int(pos) for pos in parts[1:]]
            result[doc_id] = positions
        return result

    def _update_partial_term(self, left, right):
        """Merges two partial term lists into a single sorted list."""
        left = self.convert_to_dict(left) if not isinstance(left, dict) else left
        right = self.convert_to_dict(right) if not isinstance(right, dict) else right
        result = {}
        left_keys = set(left.keys())
        right_keys = set(right.keys())

        # Add unique keys from both sides directly
        unique_to_left = left_keys - right_keys
        unique_to_right = right_keys - left_keys

        for key in unique_to_left:
            result[key] = left[key]

        for key in unique_to_right:
            result[key] = right[key]

        # Keys present in both add the smaller list
        common_keys = left_keys & right_keys
        for key in common_keys:
            if len(left[key]) < len(right[key]):
                result[key] = left[key]
            else:
                result[key] = right[key]

        return result

    def search(self, q, model):
        """Executes a search query against the index using the specified model and proximity settings."""
        query = Query(q, self.analyzer)
        sorted_tokens = sorted(query.tokens, key=lambda t: self.terms_df.get(t, math.inf))
        tokens_id = [self.terms_map[t] for t in sorted_tokens if t in self.terms_map]
        doc_scores = {}

        if model in ["BM", "TFIDF"]:
            doc_scores = self._score_vsm(sorted_tokens, tokens_id, model)
        elif model == "LML":
            doc_scores = self._score_lml(sorted_tokens, tokens_id)

        top_docs = self._get_top_n(doc_scores, 1000)
        self._write_scores(model, top_docs, query.id)

    def _score_vsm(self, tokens, tokens_id, model):
        """Scores documents using Vector Space Model approaches (TFIDF, BM25)."""
        doc_scores = {}

        for idx, token_id in enumerate(tokens_id):
            postings = self.read_postings(token_id)
            for doc_id in postings:
                score = self._compute_score(tokens[idx], postings, doc_id, model)
                doc_scores[doc_id] = doc_scores.get(doc_id, 0) + score

        return doc_scores

    def _score_lml(self, tokens, tokens_id):
        """Scores documents using the Language Modeling approach."""
        all_doc_ids, token_docs = self.docs_to_run(tokens_id)
        doc_scores = {}

        for idx, token_id in enumerate(tokens_id):
            postings = self.read_postings(token_id)
            for doc_id in postings:
                score = self._compute_score(tokens[idx], postings, doc_id, "LML")
                doc_scores[doc_id] = doc_scores.get(doc_id, 0) + score

        for doc_id in all_doc_ids:
            for idx, token_id in enumerate(tokens_id):
                if doc_id not in token_docs[token_id]:
                    score = self._compute_score(tokens[idx], {}, doc_id, "LML")
                    doc_scores[doc_id] = doc_scores.get(doc_id, 0) + score

        return doc_scores

    def docs_to_run(self, tokens_id):
        """Prepare document IDs and token-to-document mappings for LML scoring."""
        token_docs = {}
        all_doc_ids = set()

        for token_id in tokens_id:
            postings = self.read_postings(token_id)
            token_docs[token_id] = list(postings.keys())
            all_doc_ids.update(postings.keys())

        return all_doc_ids, token_docs

    def _compute_score(self, token, postings, doc_id, model):
        """Computes the score for a given token and document."""
        if model == "BM":
            return self._compute_bm25(token, postings, doc_id)
        elif model == "TFIDF":
            return self._compute_tfidf(token, postings, doc_id)
        elif model == "LML":
            return self._compute_lml(token, postings, doc_id)

    def _compute_bm25(self, token, postings, doc_id, k1=1.2, b=0.75):
        """Computes the BM25 score for a given token and document."""
        tf = len(postings[doc_id]) if doc_id in postings else 0
        df = self.terms_df[token]
        doc_length = self.doc_lengths[str(doc_id)]
        avg_doc_length = self.field_info['total_ttf'] / self.field_info['D']
        idf = math.log((self.field_info['D'] + 0.5) / (df + 0.5))
        score = idf * ((tf + k1 * tf) / (tf + k1 * ((1 - b) + b * (doc_length / avg_doc_length))))
        return score

    def _compute_tfidf(self, token, postings, doc_id):
        """Computes the TFIDF score for a given token and document."""
        tf = len(postings[doc_id]) if doc_id in postings else 0
        df = self.terms_df[token]
        doc_length = self.doc_lengths[str(doc_id)]
        avg_doc_length = self.field_info['total_ttf'] / self.field_info['D']
        score = (tf / (tf + 0.5 + 1.5 * (doc_length / avg_doc_length))) * math.log(self.field_info['D'] / df)
        return score

    def _compute_lml(self, token, postings, doc_id):
        """Computes the LML score for a given token and document."""
        tf = len(postings[doc_id]) if doc_id in postings else 0
        doc_length = self.doc_lengths[str(doc_id)]
        score = math.log((tf + 1) / (doc_length + self.field_info['V']))
        return score

    def _get_top_n(self, doc_scores, n):
        """Retrieves top N documents based on score."""
        return sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[:n]

    def _write_scores(self, model, top_docs, query_id):
        """Writes the scores of the top documents to a file."""
        for rank, (doc_id, score) in enumerate(top_docs, start=1):
            self.write_score_file(model, doc_id, score, query_id, rank)

    def write_score_file(self, model, doc_id, score, query_id, rank):
        """Writes the search results with scores to a file."""
        doc_id = self.docs_map_rev[doc_id]
        line = f"{query_id} Q0 {doc_id} {rank} {score} Exp\n"
        folder_path = f"./my_index_scores/{self.folder_name}"
        os.makedirs(folder_path, exist_ok=True)
        with open(f"{folder_path}/{model}.txt", "a+") as f:
            f.write(line)

    def read_postings(self, token_id):
        """Retrieves postings for a given token ID, handling compression if necessary."""
        offset, length = self.new_catalog.get(str(token_id), (0, 0))
        file_path = f"./{self.name}/{self.folder_name}/merge/final"
        file_mode = "rb" if self.is_compress else "r"

        with open(file_path, file_mode) as f:
            f.seek(offset)
            line = f.read(length)
            if self.is_compress:
                line = self.gzip_decompress(line)

        return self.convert_to_dict(line)

class Analyzer:
    """A class for analyzing and processing text."""

    def __init__(self, stop_words, is_stem):
        self.stop_words = set(stop_words)
        self.is_stem = is_stem
        self.stemmer = SnowballStemmer('english') if is_stem else None

    def is_stop(self, token):
        """Checks if a token is a stop word."""
        return token in self.stop_words

    def stem(self, token):
        """Stems a token using the Snowball stemmer."""
        if self.is_stem and self.stemmer:
            return self.stemmer.stem(token)
        return token

class Document:
    """A class representing a document."""

    def __init__(self, file, analyzer):
        self.id = file['id']
        self.analyzer = analyzer
        self.length = 0
        self.tokens = self.tokenize(file['content'])

    def tokenize(self, string):
        """Tokenizes a string and filters out stop words."""
        if len(string) == 0:
            return []
        tokens = re.findall(r'\b\w+(?:\.\w+)*\b', string.lower())
        stopped_tokens = [self.analyzer.stem(token) for token in tokens if not self.analyzer.is_stop(token)]
        filtered_tokens = [(token, self.id, idx + 1) for idx, token in enumerate(stopped_tokens)]
        self.length = len(filtered_tokens)
        return filtered_tokens

class Query:
    """A class representing a query."""

    def __init__(self, query, analyzer):
        self.analyzer = analyzer
        self.length = 0
        self.id = 0
        self.tokens = self.tokenize(query)

    def tokenize(self, string):
        tokens = re.findall(r'\b\w+(?:\.\w+)*\b', string.lower())
        filtered_tokens = [self.analyzer.stem(token) for token in tokens if not self.analyzer.is_stop(token)]
        self.length = len(filtered_tokens)
        if filtered_tokens:
            self.id = filtered_tokens[0]
        return filtered_tokens[1:] if len(filtered_tokens) > 1 else []

def read_my_index(name, folder_name, is_stem, is_compress, has_merged=False):
    """Loads an index from disk."""
    index_path = f"./my_index/{folder_name}"
    my_index = MyIndex(name=name, stop_words=stop_words, is_stem=is_stem, is_compress=is_compress)

    metadata_files = {
        "catalog": "catalog.json",
        "terms_df": "term_df.json",
        "terms_ttf": "term_ttf.json",
        "field_info": "field_info.json",
        "terms_map": "terms_map.json",
        "docs_map": "docs_map.json",
        "doc_lengths": "doc_lengths.json",
    }
    if has_merged:
        metadata_files["new_catalog"] = "merge/new_catalog.json"

    for attribute, filename in metadata_files.items():
        with open(os.path.join(index_path, filename), "r") as file:
            setattr(my_index, attribute, json.load(file))

    my_index.docs_map_rev = {value: key for key, value in my_index.docs_map.items()}
    return my_index

def insert_docs(index, docs):
    """Inserts multiple documents into the index."""
    for doc in docs:
        index.insert_document(doc)

def read_queries(file_path):
    """Reads queries from a file."""
    with open(file_path, "r") as file:
        return [line.strip() for line in file]


# Initialization and document insertion
# stemmingCompress = MyIndex("my_index", stop_words, True, True)
# stemmingNoCompress = MyIndex("my_index", stop_words, True, False)
# noStemCompress = MyIndex("my_index", stop_words, False, True)
# noStemNoCompress = MyIndex("my_index", stop_words, False, False)
# insert_docs(stemmingCompress, instances)
# insert_docs(stemmingNoCompress, instances)
# insert_docs(noStemCompress, instances)
# insert_docs(noStemNoCompress, instances)

# Index reading example
# stemmingCompress = read_my_index("my_index", "with_stem_compress", True, True, True)
# stemmingNoCompress = read_my_index("my_index", "with_stem_no_compress", True, False)
# noStemCompress = read_my_index("my_index", "no_stem_compress", False, True)
# noStemNoCompress = read_my_index("my_index", "no_stem_no_compress", False, False)

# Merge control
# stemmingCompress.merge_control()
# stemmingNoCompress.merge_control()
# noStemCompress.merge_control()
# noStemNoCompress.merge_control()

# # Reading and executing queries
# queries = read_queries("./queries.txt")
# models = ["BM", "TFIDF", "LML"]
# for query in queries:
#     for model in models:
#         stemmingCompress.search(query, model)
#         stemmingNoCompress.search(query, model)
#         noStemCompress.search(query, model)
#         noStemNoCompress.search(query, model)

test = Document(instances[0], Analyzer(stop_words, True))

print(test.tokenize("For instance, bob and 376 and 98.6 and 192.160.0.1 are all tokens. 123,456 and aunt's are not tokens"))
