import logging
import threading
from datetime import datetime, timedelta
from queue import PriorityQueue
from threading import Lock
from utils import URLUtils, CRAWL_DELAY, PREFERRED_KEYWORDS, PREFERRED_DOMAINS

class Frontier:
    def __init__(self, seed_urls):
        self.queue = PriorityQueue()
        self.temp_queue = PriorityQueue()
        self.visited = set()
        self.domain_last_crawled = {}
        self.current_wave_number = 0
        self.seed_urls = set(seed_urls)
        self.lock = Lock()
        self._initialize_with_seed_urls(seed_urls)

    def _initialize_with_seed_urls(self, seed_urls):
        for url in seed_urls:
            # canonicalize url
            canonic_url = URLUtils.canonicalize_url(url)
            domain = URLUtils.extract_domain(canonic_url)
            self.add_url(canonic_url, domain, is_seed=True)

    def add_url(self, url, domain, link_text='', last_modified=None, is_seed=False, in_links=None, elastic_search=None, index=None):
        if url in self.visited:
            if elastic_search:
                elastic_search.update_document_in_links(url, in_links)
                # if url not in index:
                #     index[url] = {'inlinks': in_links}
                # else:
                #     index[url]['inlinks'] += in_links

            # if still in queue, update priority
            # if url in self.in_queue:
            #     wave_num = self.url_info[url]['wave_number']
            #     old_priority = self.url_info[url]['priority']
            #     new_priority = self.calculate_priority(url, link_text, last_modified, self.url_info[url]['in_link_count'])
            #     self.url_info[url]['priority'] = new_priority
            #     self.in_queue[url] = (wave_num, new_priority)
            #     self.queue.queue.remove((wave_num, old_priority, url))
            #     self.queue.put((wave_num, new_priority, url))
            return

        delay = self.get_domain_crawl_delay(domain)
        next_crawl_time = datetime.now() + timedelta(seconds=delay)
        wave_number = 0 if is_seed else self.current_wave_number + 1

        priority = self.calculate_priority(url, link_text, last_modified, domain)

        self.queue.put((wave_number, priority, next_crawl_time, url))
        with self.lock:
            self.visited.add(url)

    @staticmethod
    def calculate_priority(url, anchor_text, last_modified, domain, in_links=0):
        priority = 1000

        # prefer certain keywords and domains
        total_keyword_matches = sum(1 for keyword in PREFERRED_KEYWORDS if keyword in anchor_text.lower())
        total_keyword_matches += sum(1 for keyword in PREFERRED_KEYWORDS if keyword in url.lower())
        priority -= total_keyword_matches * 40

        if domain in PREFERRED_DOMAINS:
            priority -= 100

        # prefer recent urls
        if last_modified:
            last_modified_date = datetime.strptime(last_modified, '%a, %d %b %Y %H:%M:%S %Z')
            elapsed_time = datetime.now() - last_modified_date
            priority += elapsed_time.days * 10

        # priority += wave_number * 1000 # wave number is already included in the tuple (so earlier waves are prioritized)

        # the more in-links the more important an url is
        priority -= in_links * 25

        return priority

    def update_domain_crawl_time(self, domain, robot_delay):
        with self.lock:
            self.domain_last_crawled[domain] = [datetime.now(), robot_delay]

    def get_domain_crawl_delay(self, domain):
        if domain in self.domain_last_crawled:
            last_crawled, robot_delay = self.domain_last_crawled[domain]
            elapsed_time = datetime.now() - last_crawled
            delay = max(0, robot_delay - elapsed_time.total_seconds())
            return delay
        return 0

    def get_next_url(self):
        with self.lock:
            while not self.temp_queue.empty():
                wave_number, priority, next_crawl_time, url = self.temp_queue.get()
                self.queue.put((wave_number, priority, next_crawl_time, url))

            while not self.queue.empty():
                wave_number, priority, next_crawl_time, url = self.queue.queue[0]
                if datetime.now() >= next_crawl_time:
                    self.current_wave_number = wave_number
                    domain = URLUtils.extract_domain(url)
                    self.queue.get()
                    return url, domain, next_crawl_time
                else:
                    self.temp_queue.put(self.queue.get())
            else:
                print("Frontier is empty")

        return None, None, None

    def is_empty(self):
        return self.queue.empty()