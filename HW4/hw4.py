import argparse
from pagerank import page_rank_crawl, page_rank_graph
from hits import hits_crawl

def main():
    parser = argparse.ArgumentParser(description="IS4200 HW4: PageRank and HITS algorithms.")
    parser.add_argument('function', choices=['page_rank_crawl', 'page_rank_graph', 'hits_crawl'],
                        help="The function to run")
    parser.add_argument('--query', type=str, default='sports',
                        help="The query for HITS crawl (optional, default='sports')")

    args = parser.parse_args()

    if args.function == 'page_rank_crawl':
        page_rank_crawl()
    elif args.function == 'page_rank_graph':
        page_rank_graph()
    elif args.function == 'hits_crawl':
        hits_crawl(args.query)
    else:
        print("Invalid function.")
        return

    print("Code completed.")


if __name__ == "__main__":
    main()