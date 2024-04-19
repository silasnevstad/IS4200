import numpy as np
from utils import get_link_graph, save_pickle, preprocess_link_graph, extract_wt2g_graph, format_output
from constants import PAGE_RANK_CRAWL_FILENAME, PAGE_RANK_GRAPH_FILENAME

def initialize_pagerank(link_graph):
    """
    Initialize the PR values for each page in the link graph to 1/N, where N is the number of pages in the graph.
    """
    num_pages = len(link_graph)
    page_rank = {url: 1.0 / num_pages for url in link_graph}
    return page_rank

def calculate_perplexity(pagerank):
    entropy = -sum(pr * np.log2(pr) for pr in pagerank.values())
    return 2 ** entropy

def has_converged(old_pagerank, new_pagerank, convergence_threshold=0.0001):
    old_perplexity = calculate_perplexity(old_pagerank)
    new_perplexity = calculate_perplexity(new_pagerank)
    change = abs(new_perplexity - old_perplexity)
    print(f"Perplexity change: {change}")
    return change < convergence_threshold

def normalize_pagerank(pagerank):
    norm = sum(pagerank.values())
    return {url: pr / norm for url, pr in pagerank.items()}


def compute_pagerank(link_graph, d=0.85, max_iterations=100, convergence_threshold=0.0001):
    pagerank = initialize_pagerank(link_graph)

    num_pages = len(link_graph)
    sink_node_urls = [url for url, data in link_graph.items() if not data['outlinks']]

    iteration = 0
    while iteration < max_iterations:
        new_pagerank = {}
        sink_pagerank = sum(pagerank[url] for url in sink_node_urls)

        for url, data in link_graph.items():
            # teleportation
            new_pagerank[url] = (1 - d) / num_pages
            # spread remaining sink PR evenly to all pages
            new_pagerank[url] += d * (sink_pagerank / num_pages)

            # a "share" of the PageRank of every page that links to it
            inlinks_pagerank = sum(pagerank[inlink] / len(link_graph[inlink]['outlinks']) for inlink in data['inlinks'] if  len(link_graph[inlink]['outlinks']) > 0)
            new_pagerank[url] += d * inlinks_pagerank
            # for inlink in data['inlinks']:
            #     if inlink not in link_graph:
            #         continue
            #     if len(link_graph[inlink]['outlinks']) == 0:
            #         continue
            #     new_pagerank[url] += d * pagerank[inlink] / len(link_graph[inlink]['outlinks'])

        if has_converged(pagerank, new_pagerank, convergence_threshold):
            print(f"Converged after {iteration + 1} iterations.")
            break

        new_pagerank = normalize_pagerank(new_pagerank)

        pagerank = new_pagerank
        iteration += 1
    else:
        print(f"Did not converge after {max_iterations} iterations.")

    return pagerank

def page_rank_crawl():
    # get the link graph
    link_graph = get_link_graph()

    print(f"Number of pages in the link graph: {len(link_graph)}")

    # compute the pagerank
    pagerank = compute_pagerank(link_graph)

    # save the pagerank
    save_pickle(pagerank, PAGE_RANK_CRAWL_FILENAME)

    # display top pages
    top_pages = format_output(pagerank, link_graph)

    for url, pr, inlinks_count, outlinks_count in top_pages:
        print(f"URL: {url}, PageRank: {pr}, Inlinks: {inlinks_count}, Outlinks: {outlinks_count}")


def page_rank_graph():
    # get the link graph
    link_graph = extract_wt2g_graph()
    link_graph = preprocess_link_graph(link_graph)

    print(f"Number of pages in the link graph: {len(link_graph)}")

    # compute the pagerank
    pagerank = compute_pagerank(link_graph)

    # save the pagerank
    save_pickle(pagerank, PAGE_RANK_GRAPH_FILENAME)

    # display top pages
    top_pages = format_output(pagerank, link_graph)

    for url, pr, inlinks_count, outlinks_count in top_pages:
        print(f"URL: {url}, PageRank: {pr}, Inlinks: {inlinks_count}, Outlinks: {outlinks_count}")

    """
    The reason some pages rank higher but have smaller inlink counts than others is because the PR is not only
    based on the number of inlinks, but also on the quality of the inlinks. So a page with fewer inlinks from high-quality
    pages can have a higher PR than a page with many inlinks from low-quality pages.
    
    If we follow WT23-B39-340 (3rd), which doesn't have as many inlinks as WT01-B18-225 (7th), we can see it has inlinks    
    from pages like WT23-B39-340, WT23-B39-341, etc which it then also outlinks to. This means that the page is
    part of a small community of pages that link to each other, which can boost its PR.
    Then if we look at WT01-B18-225 (7th), which has a lot more inlinks than pages ranking 3rd-6th but still ranks lower, 
    we can see its inlinks, such as WT01-B29-71 and WT01-B29-72, are not as high-quality as the inlinks of the other pages.
    Both WT01-B29-71 and WT01-B29-72 only have 1 inlink each.
    """
