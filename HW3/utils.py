import hashlib
import urllib.robotparser
from threading import Lock
from urllib.parse import urlparse, urljoin, parse_qs, urlunparse
import urllib.robotparser
from bs4 import BeautifulSoup
import re
from elasticsearch import Elasticsearch as ElasticSearchClient
import socket

socket.setdefaulttimeout(5)

GROUP_TOPIC = "Sports League and Championship"
INDIVIDUAL_TOPIC = "Baseball leagues and championships"

COMMON_SEED_URLS = [
    'https://en.wikipedia.org/wiki/Sports_league',
    'https://en.wikipedia.org/wiki/List_of_world_sports_championships'
]
INDIVIDUAL_SEED_URLS = [
    'https://en.wikipedia.org/wiki/Baseball',
    'https://www.mlb.com/',
    'https://www.wbsc.org',
    # 'https://www.espn.com/mlb/',
    # 'https://www.nbcsports.com/mlb',
    # 'https://www.si.com/mlb',
    # 'https://www.cbssports.com/mlb/',
    # 'https://www.baseball-reference.com/',
]
ALL_SEED_URLS = INDIVIDUAL_SEED_URLS + COMMON_SEED_URLS

CRAWL_DELAY = 1
PREFERRED_DOMAINS = ['en.wikipedia.org', 'mlb.com', 'wbsc.org', 'espn.com']
PREFERRED_KEYWORDS = ['sports', 'championships', 'league', 'baseball', 'mlb', 'wbsc', 'pitch', 'bat', 'base', 'run']

def efficient_merge_deduplicate(original, new_items):
    original_set = set(original)
    original_len = len(original_set)

    original_set.update(new_items)

    if len(original_set) > original_len:
        return list(original_set)
    else:
        return original

class ElasticSearch:
    def __init__(self, index_name):
        # self.es = ElasticSearchClient(["http://localhost:9200"])
        cloud_id = 'HW3_Merge:dXMtY2VudHJhbDEuZ2NwLmNsb3VkLmVzLmlvOjQ0MyRhMGRlNTA5NDg0Y2Y0N2YyYmZhODVmOTBkNjIzMzBmZiQ2ZDExNWE5ZDA4OWY0ZWRhYTJiNmU5MjFmNDAxMDlkYg=='
        self.es = ElasticSearchClient(cloud_id = cloud_id, request_timeout = 10000, http_auth = ('elastic','LXuKgJCq2N1vnNyB5J7pXvy6'))
        self.index_name = index_name
        self.lock = Lock()

    def index_document(self, doc_id, document):
        try:
            # with self.lock:
            if self.es.exists(index=self.index_name, id=doc_id):
                source = self.es.get(index=self.index_name, id=doc_id)['_source']
                # if 'inlinks' in source:
                #     document['inlinks'] = list(set(source['inlinks'] + document['inlinks']))
                document['inlinks'] = efficient_merge_deduplicate(source.get('inlinks', []), document['inlinks'])
                document['outlinks'] = efficient_merge_deduplicate(source.get('outlinks', []), document['outlinks'])
                document['authors'] = efficient_merge_deduplicate(source.get('authors', []), document['authors'])
                result = self.es.update(
                    index=self.index_name,
                    id=doc_id,
                    doc=document
                ).get('result')
            else:
                result = self.es.index(
                    index=self.index_name,
                    id=doc_id,
                    document=document,
                ).get('result')
            if result not in ['created', 'updated', 'noop']:
                print(f"Error indexing document {doc_id}: {result if result else 'Unknown error'}")
        except Exception as e:
            print(f"Error indexing document {doc_id}: {e}")

    def update_document_in_links(self, doc_id, in_links):
        try:
            with self.lock:
                if self.es.exists(index=self.index_name, id=doc_id):
                    combined_in_links = efficient_merge_deduplicate(self.es.get(index=self.index_name, id=doc_id)['_source'].get('inlinks', []), in_links)
                    result = self.es.update(
                        index=self.index_name,
                        id=doc_id,
                        doc={"inlinks":combined_in_links}
                    ).get('result')
                else:
                    result = self.es.index(
                        index=self.index_name,
                        id=doc_id,
                        document={"inlinks":in_links},
                    ).get('result')
            if result not in ['created', 'updated', 'noop']:
                print(f"Error updating document {doc_id}: {result if result else 'Unknown error'}")
        except Exception as e:
            print(f"Error updating document {doc_id}: {e}")

    def get_document(self, doc_id):
        try:
            with self.lock:
                return self.es.get(index=self.index_name, id=doc_id)
        except Exception as e:
            print(f"Error getting document {doc_id}: {e}")
            return None

    def create_index(self):
        configurations = {
            "settings" : {
                "number_of_shards": 1,
                "number_of_replicas": 1,
                "analysis": {
                    "analyzer": {
                        "stopped": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": [
                                "lowercase",
                            ]
                        }
                    }
                }
            },
            "mappings": {
                "total_fields": {
                    "limit": "10000"
                },
                "properties": {
                    "url": {
                        "type": "keyword"
                    },
                    "content": {
                        "type": "text",
                        "fielddata": True,
                        "analyzer": "stopped",
                        "index_options": "positions"
                    },
                    "inlinks": {
                        "type": "keyword"
                    },
                    "outlinks": {
                        "type": "keyword"
                    },
                    "authors": {
                        "type": "keyword"
                    }
                }
            }
        }
        try:
            with self.lock:
                self.es.indices.create(index=self.index_name, body=configurations)
        except Exception as e:
            print(f"Error creating index: {e}")

    def delete_index(self):
        try:
            with self.lock:
                self.es.indices.delete(index=self.index_name)
        except Exception as e:
            print(f"Error deleting index: {e}")

    def clean_index(self):
        # removes all documents from the index that don't have content
        try:
            with self.lock:
                self.es.delete_by_query(
                    index=self.index_name,
                    body={
                        "query": {
                            "bool": {
                                "must_not": {
                                    "exists": {
                                        "field": "content"
                                    }
                                }
                            }
                        }
                    }
                )
        except Exception as e:
            print(f"Error cleaning index: {e}")

class RobotsPolicy:
    def __init__(self,):
        self.parsers = {}

    def can_fetch(self, url, domain):
        if domain not in self.parsers:
            self.parse_robots_txt(domain)
        parser = self.parsers.get(domain)
        return parser.can_fetch("*", url) if parser else False

    def get_crawl_delay(self, domain):
        if domain not in self.parsers:
            self.parse_robots_txt(domain)
        parser = self.parsers.get(domain)
        crawl_delay = parser.crawl_delay("*") if parser else None
        return crawl_delay if crawl_delay else CRAWL_DELAY

    def parse_robots_txt(self, domain):
        robots_url = f"http://{domain}/robots.txt"
        parser = urllib.robotparser.RobotFileParser()
        try:
            parser.set_url(robots_url)
            # BUG: Can hang here, if the robots.txt file is too large or the server is slow, need to set a timeout
            parser.read()
            self.parsers[domain] = parser
        except Exception as e:
            print(f"Error parsing robots.txt for domain {domain}: {e}")
            self.parsers[domain] = None

class DocumentProcessor:
    @staticmethod
    def clean_html(soup):
        for elem in soup(["script", "style", "iframe", "noscript", "footer", "nav", "header"]):
            elem.decompose()

        body = soup.find('body')

        if body:
            text = body.get_text(separator=' ', strip=True)
            clean_text = re.sub(r'[^\w\s]', '', text)
            return clean_text
        else:
            text = soup.get_text(separator=' ', strip=True)
            clean_text = re.sub(r'[^\w\s]', '', text)
            return clean_text

    @staticmethod
    def extract_title_and_clean_text(html_content):
        soup = BeautifulSoup(html_content, 'html.parser')
        title = soup.title.string if soup.title else ''
        clean_text = DocumentProcessor.clean_html(soup)
        return title, clean_text

    @staticmethod
    def extract_links(html_content, base_url):
        soup = BeautifulSoup(html_content, 'html.parser')
        links = set()
        for link in soup.find_all('a', href=True):
            href = link.get('href')
            if not href.startswith('#'):
                absolute_url = urljoin(base_url, href)
                if URLUtils.is_valid_url(absolute_url):
                    links.add((absolute_url, link.text.strip()))
        return links

    @staticmethod
    def extract_title_tags(html_content):
        soup = BeautifulSoup(html_content, 'html.parser')
        return [tag.text for tag in soup.find_all('title')]

class URLUtils:
    @staticmethod
    def canonicalize_url(url):
        try:
            parsed_url = urlparse(url)

            netloc = parsed_url.netloc.lower()

            if parsed_url.port == 80 or parsed_url.port == 443:
                netloc = netloc.split(':')[0]

            path = re.sub(r'//+', '/', parsed_url.path)

            return urlunparse(("http", netloc, path, parsed_url.params, parsed_url.query, ''))
        except Exception as e:
            print(f"Error canonicalizing URL {url}: {e}")
            return url

    @staticmethod
    def is_valid_url(url):
        parsed = urlparse(url)
        is_valid = bool(parsed.scheme) and bool(parsed.netloc)
        is_short_enough = len(url) <= 512
        return is_valid and is_short_enough

    @staticmethod
    def extract_domain(url):
        parsed = urlparse(url)
        return parsed.netloc

    @staticmethod
    def extract_hash(url):
        return hashlib.md5(url.encode()).hexdigest()

    @staticmethod
    def extract_keywords(url):
        parsed = urlparse(url)
        query = parse_qs(parsed.query)
        return query.get('q')

def test_canonicalize_url():
    assert URLUtils.canonicalize_url("HTTP://www.Example.com/SomeFile.html") == "http://www.example.com/somefile.html"
    assert URLUtils.canonicalize_url("http://www.example.com:80") == "http://www.example.com"
    assert URLUtils.canonicalize_url("http://www.example.com/a.html#anything") == "http://www.example.com/a.html"
    assert URLUtils.canonicalize_url("http://www.example.com//a.html") == "http://www.example.com/a.html"