import concurrent
import json
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import requests
import time
from frontier import Frontier
from utils import (
    RobotsPolicy, DocumentProcessor, URLUtils, ElasticSearch,
    PREFERRED_KEYWORDS, INDIVIDUAL_SEED_URLS, COMMON_SEED_URLS, ALL_SEED_URLS, test_canonicalize_url
)
import logging

logging.basicConfig(level=logging.DEBUG, filename='crawler.log', filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')

def save_index_with_pickle(index, filename='index.pkl'):
    with open(filename, 'wb') as file:
        pickle.dump(index, file)

def load_index_with_pickle(filename='index.pkl'):
    with open(filename, 'rb') as file:
        return pickle.load(file)

class WebCrawler:
    def __init__(self, seed_urls, num_threads=5, max_pages=10000, index_name='individual-crawl-hw3'):
        self.frontier = Frontier(seed_urls)
        self.elastic_search = ElasticSearch(index_name)
        self.robots_policy = RobotsPolicy()
        self.total_crawled = 0
        self.lock = Lock()
        self.num_threads = num_threads
        self.max_pages = max_pages
        self.start_time = time.time()
        self.session = requests.Session()
        self.index = {}

    def create_index(self):
        self.elastic_search.delete_index()
        self.elastic_search.create_index()

    def clean_local_index(self, filename='index.pkl'):
        self.index = load_index_with_pickle(filename)
        new_index = {url: doc for url, doc in self.index.items() if 'content' in doc}
        print(f"Old index length: {len(self.index)}")
        print(f"New index length: {len(new_index)}")
        save_index_with_pickle(new_index, 'cleaned_index.pkl')

    def load_local_index(self, filename='cleaned_index.pkl'):
        self.index = load_index_with_pickle(filename)
        print(f"Loaded index with {len(self.index)} documents")

        # split into 12 batches
        batch_size = len(self.index) // 12
        batches = [list(self.index.items())[i:i+batch_size] for i in range(0, len(self.index), batch_size)]
        with ThreadPoolExecutor(max_workers=12) as executor:
            futures = [executor.submit(self.upload_document_batch, batch) for batch in batches]
            for future in futures:
                try:
                    future.result(timeout=60)
                except concurrent.futures.TimeoutError:
                    logging.error(f"Thread {threading.current_thread().name}: Operation timed out")

    def upload_document_batch(self, batch):
        i = 0
        for url, document in batch:
            try:
                if i % 100 == 0:
                    print(f"Uploading document {i}")
                self.elastic_search.index_document(url, document)
                i += 1
            except Exception as e:
                print(f"Failed to upload document {url}: {e}")

    def start_crawling(self):
        while not self.frontier.is_empty() and self.total_crawled < self.max_pages:
            with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                futures = [executor.submit(self.crawl_pages) for _ in range(self.num_threads)]
                for future in futures:
                    try:
                        future.result(timeout=60)
                    except concurrent.futures.TimeoutError:
                        logging.error(f"Thread {threading.current_thread().name}: Operation timed out")
        else:
            total_time = time.time() - self.start_time
            print("Crawling completed")
            print(f"Total time taken: {total_time:.2f} seconds")
            print(f"Total pages crawled: {self.total_crawled}")


        # save index to pickle file
        # save_index_with_pickle(self.index, 'index.pkl')
        #
        # # clean the index
        # new_index = {url: doc for url, doc in self.index.items() if 'content' in doc}
        #
        # # upload index to elastic search
        # for url, document in new_index.items():
        #     self.elastic_search.index_document(url, document)

        # self.save_link_graph('link_graph.json')
        # with open('index.json', 'w') as file:
        #     json.dump(self.index, file, indent=4)

    def crawl_pages(self):
        while not self.frontier.is_empty() and self.total_crawled < self.max_pages:
            url, domain, next_crawl_time = self.frontier.get_next_url()
            if url is None:
                break

            try:
                if not self.robots_policy.can_fetch(url, domain):
                    # print(f"Skipping {url} as per robots.txt policy")
                    continue
            except Exception as e:
                print(f"Error fetching robots.txt for {url}: {e}")
                continue

            # if not self.is_html_content(url, self.session):
            #     continue

            robot_delay = self.robots_policy.get_crawl_delay(domain)
            # delay = self.frontier.get_domain_crawl_delay(domain)
            # if delay > 0:
            #     # print(f"Sleeping for {delay} seconds")
            #     time.sleep(delay)

            content = self.fetch_page(url, self.session)
            self.frontier.update_domain_crawl_time(domain, robot_delay)
            if content:
                self.parse_and_process_page(url, domain, content)
                with self.lock:
                    self.total_crawled += 1
                total_time = time.time() - self.start_time
                print(f"Total pages crawled: {self.total_crawled}, in {total_time:.2f} seconds")
                if self.total_crawled >= self.max_pages:
                    print("Max pages limit reached")
                    break
        else:
            print("Frontier is empty")

    @staticmethod
    def fetch_page(url, session=None):
        try:
            # logging.debug(f"Thread {threading.current_thread().name}: Starting request to {url}")
            response = session.get(url, timeout=6, allow_redirects=True) if session else requests.get(url, timeout=6, allow_redirects=True)
            # logging.debug(f"Thread {threading.current_thread().name}: Completed request to {url}")
            if response.status_code == 200 and 'text/html' in response.headers.get('Content-Type', ''):
                return response.text
            else:
                logging.error(f"Request failed for {url}: {response.status_code}")
        except requests.RequestException as e:
            print(f"Request failed for {url}: {e}")
            logging.error(f"Request failed for {url}: {e}")
        return None

    @staticmethod
    def is_html_content(url, session=None):
        try:
            response = session.head(url, timeout=6, allow_redirects=True) if session else requests.head(url, timeout=5, allow_redirects=True)
            return 'text/html' in response.headers.get('Content-Type', '')
        except requests.RequestException as e:
            print(f"HEAD request failed for {url}: {e}")
            return False

    def parse_and_process_page(self, url, domain, content):
        title, text = DocumentProcessor.extract_title_and_clean_text(content)
        links_and_texts = DocumentProcessor.extract_links(content, url)

        # with self.link_graph_lock:
        #     if url not in self.link_graph:
        #         self.link_graph[url] = {'in': set(), 'out': set()}

        out_links = set()

        for link, link_text in links_and_texts:
            canonical_link = URLUtils.canonicalize_url(link)

            out_links.add(canonical_link)

            # with self.link_graph_lock:
            #     self.link_graph[url]['out'].add(canonical_link)
            #
            #     if canonical_link not in self.link_graph:
            #         self.link_graph[canonical_link] = {'in': {url}, 'out': set()}
            #     else:
            #         self.link_graph[canonical_link]['in'].add(url)

            if self.topical_filter(canonical_link, link_text):
                self.frontier.add_url(canonical_link, domain, link_text, in_links=[url], elastic_search=self.elastic_search, index=self.index)

        # an id, the URL, the HTTP headers, the page contents cleaned (with term positions), the raw html, and a list of all in-links (known) and out-links for the page.
        # doc_id = URLUtils.extract_hash(url)
        document = {
            'url': url,
            # 'headers': {key: value for key, value in headers.items()},
            'content': f'<DOC>\n<DOCNO>{url}</DOCNO>\n<HEAD>{title}</HEAD>\n<TEXT>{text}</TEXT>\n</DOC>',
            'inlinks': [],
            'outlinks': list(out_links),
            'authors': ['Silas Nevstad'],
        }
        self.elastic_search.index_document(url, document)
        # if url not in self.index:
        #     self.index[url] = document
        # else:
        #     document['inlinks'] = self.index[url]['inlinks']
        #     self.index[url] = document

    @staticmethod
    def topical_filter(url, url_text):
        url_keywords = URLUtils.extract_keywords(url)
        url_relevance = any(keyword in url_keywords for keyword in PREFERRED_KEYWORDS) if url_keywords else False
        url_text_relevance = any(keyword in url_text.lower() for keyword in PREFERRED_KEYWORDS)
        return url_relevance or url_text_relevance

    # def save_link_graph(self, file_path):
    #     with open(file_path, 'w') as file:
    #         serializable_link_graph = {url: {'in': list(links['in']), 'out': list(links['out'])} for url, links in self.link_graph.items()}
    #         json.dump(serializable_link_graph, file, indent=4)


if __name__ == "__main__":
    crawler = WebCrawler(ALL_SEED_URLS, num_threads=12, index_name='sports')
    # crawler.create_index()
    # crawler.start_crawling()
    crawler.load_local_index()
    # test_canonicalize_url()