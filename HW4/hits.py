import random
import numpy as np
from es import ElasticSearch
from utils import get_link_graph, save_hub_authority_scores
from constants import TOP_HUBS_FILENAME, TOP_AUTHORITIES_FILENAME

def normalize_scores(scores):
    norm = np.sqrt(sum(score ** 2 for score in scores.values()))
    return {url: score / norm for url, score in scores.items()} if norm != 0 else scores

def update_authority_scores(base_set, link_graph, hub_scores):
    authority_scores = {url: 0 for url in base_set}
    for url in base_set:
        for inlink in link_graph[url]['inlinks']:
            if inlink in hub_scores:
                authority_scores[url] += hub_scores[inlink]

    return normalize_scores(authority_scores)

def update_hub_scores(base_set, link_graph, authority_scores):
    hub_scores = {url: 0 for url in base_set}
    for url in base_set:
        for outlink in link_graph[url]['outlinks']:
            if outlink in authority_scores:
                hub_scores[url] += authority_scores[outlink]

    return normalize_scores(hub_scores)

def has_converged_hubs_authorities(old_hub_scores, new_hub_scores, old_authority_scores, new_authority_scores, convergence_threshold):
    hub_change = np.linalg.norm(np.array(list(old_hub_scores.values())) - np.array(list(new_hub_scores.values())))
    authority_change = np.linalg.norm(np.array(list(old_authority_scores.values())) - np.array(list(new_authority_scores.values())))
    print(f"Hub change: {hub_change} | Authority change: {authority_change}")
    return hub_change < convergence_threshold and authority_change < convergence_threshold

def expand_root_set(root_set, link_graph, target_size=10000, d=200):
    base_set = set(root_set)
    current_set = set(root_set)

    while len(base_set) < target_size:
        next_set = set()

        for url in current_set:
            if url in link_graph:
                outlinks = link_graph[url]['outlinks']
                inlinks = link_graph[url]['inlinks']

                next_set.update(outlinks)

                if len(inlinks) <= d:
                    next_set.update(inlinks)
                else:
                    next_set.update(random.sample(sorted(inlinks), d))

        base_set.update(next_set)

        current_set = next_set

        if len(base_set) > target_size:
            base_set = set(list(base_set)[:target_size])
            break

    return base_set

def hits_crawl(query='sports'):
    link_graph = get_link_graph()

    # Obtain root set of 1000 pages
    es = ElasticSearch()
    root_documents = es.search_top_n(query, n=1000)
    root_set = {doc['url'] for doc in root_documents}

    # Expand to base set
    base_set = expand_root_set(root_set, link_graph, d=200)

    print(f"Root set size: {len(root_set)}")
    print(f"Base set size: {len(base_set)}")

    # Initialize hub and authority scores for base set
    hub_scores = {url: 1 for url in base_set}
    authority_scores = {url: 1 for url in base_set}

    for iteration in range(100):
        new_hub_scores = update_hub_scores(base_set, link_graph, authority_scores)
        new_authority_scores = update_authority_scores(base_set, link_graph, hub_scores)

        if has_converged_hubs_authorities(hub_scores, new_hub_scores, authority_scores, new_authority_scores, convergence_threshold=0.0001):
            print(f"Converged after {iteration + 1} iterations.")
            break

        hub_scores, authority_scores = new_hub_scores, new_authority_scores

    # Saving the top 500 hub and authority scores
    top_hubs = sorted(hub_scores.items(), key=lambda x: x[1], reverse=True)[:500]
    top_authorities = sorted(authority_scores.items(), key=lambda x: x[1], reverse=True)[:500]

    save_hub_authority_scores(top_hubs, TOP_HUBS_FILENAME)
    save_hub_authority_scores(top_authorities, TOP_AUTHORITIES_FILENAME)
    print("HITS computation done.")