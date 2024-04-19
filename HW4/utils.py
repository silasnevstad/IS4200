import pickle
from constants import LINK_GRAPH_FILENAME
from es import ElasticSearch

def preprocess_link_graph(link_graph):
    """
    Remove invalid URLs from inlinks and outlinks (ones that are not in the link graph).
    Removes self-votes.

    :param link_graph: Uncleaned link graph
    :return: Cleaned link graph
    """
    cleaned_link_graph = {}
    valid_urls = set(link_graph.keys())
    for url, data in link_graph.items():
        data['inlinks'].discard(url)
        data['outlinks'].discard(url)

        valid_inlinks = set(data['inlinks'] & valid_urls)
        valid_outlinks = set(data['outlinks'] & valid_urls)

        cleaned_link_graph[url] = {
            'inlinks': valid_inlinks,
            'outlinks': valid_outlinks
        }

    return cleaned_link_graph

def get_link_graph():
    """
    Load link graph from pickle file if it exists, otherwise fetch from ElasticSearch.

    :return: A cleaned link graph
    """
    link_graph = load_pickle(LINK_GRAPH_FILENAME)
    link_graph = preprocess_link_graph(link_graph)
    if not link_graph:
        print("Fetching link graph from ElasticSearch.")
        es = ElasticSearch()
        link_graph = es.fetch_link_graph()
        link_graph = preprocess_link_graph(link_graph)
        save_pickle(link_graph, LINK_GRAPH_FILENAME)
    return link_graph

def extract_wt2g_graph():
    """
    Extracts the link graph from the WT2G text file.
    First extracts inlinks, then outlinks since every inlink is an outlink for another page.
    :return: Link graph
    """
    inlinks_graph = {}
    with open('../Resources/wt2g_inlinks.txt', 'r') as file:
        for line in file:
            parts = line.strip().split()
            url, inlinks = parts[0], parts[1:]
            inlinks_graph[url] = inlinks

    link_graph = {}
    for url, inlinks in inlinks_graph.items():
        if url not in link_graph:
            link_graph[url] = {"inlinks": set(inlinks), "outlinks": set()}
        else:
            link_graph[url]["inlinks"].update(inlinks)

        for inlink in inlinks:
            if inlink not in link_graph:
                link_graph[inlink] = {"inlinks": set(), "outlinks": set([url])}
            else:
                link_graph[inlink]["outlinks"].add(url)

    return link_graph

def format_output(pagerank, link_graph):
    enhanced_output = sorted(
        [(url, pr, len(link_graph[url]['inlinks']), len(link_graph[url]['outlinks'])) for url, pr in pagerank.items()],
        key=lambda x: -x[1])
    return enhanced_output[:500]

# [webpageurl][tab][hub/authority score]\n
def save_hub_authority_scores(scores, filename):
    with open(filename, 'w') as f:
        for url, score in scores:
            f.write(f"{url}\t{score}\n")
    print(f"Hub/authority scores saved to {filename}.")

def save_pickle(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)
    print(f"Saved pickle file to {filename}.")

def load_pickle(filename):
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"No saved pickle file found at {filename}.")
        return None

