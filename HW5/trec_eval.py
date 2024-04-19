import argparse
import numpy as np

def parse_qrels(filepath):
    qrels = {}
    num_rel = {}
    with open(filepath, 'r') as file:
        for line in file:
            parts = line.split()
            if len(parts) >= 4:
                query_id, _, doc_id, relevance = parts[:4]
                if query_id not in qrels:
                    qrels[query_id] = {}
                    num_rel[query_id] = 0

                relevance = int(relevance)
                qrels[query_id][doc_id] = relevance

                num_rel[query_id] += relevance
    return qrels, num_rel

def parse_trec(filepath):
    trec = {}
    with open(filepath, 'r') as file:
        for line in file:
            parts = line.split()
            if len(parts) >= 6:
                query_id, _, doc_id, _, score, _ = parts[:6]
                if query_id not in trec:
                    trec[query_id] = []
                trec[query_id].append((doc_id, float(score)))
    for query in trec.keys():
        trec[query] = sorted(trec[query], key=lambda x: (-x[1], x[0]))
    return trec

def dcg(relevances, k=None):
    relevances = np.asfarray(relevances)[:k]
    n = relevances.size
    if n > 0:
        return np.sum((2**relevances - 1) / np.log2(np.arange(2, n + 2)))
    return 0

def ndcg(retrieved_docs, relevant_docs, k):
    retrieved_relevances = [relevant_docs.get(doc_id, 0) for doc_id, _ in retrieved_docs[:k]]
    ideal_relevances = sorted(relevant_docs.values(), reverse=True)[:k]
    actual_dcg = dcg(retrieved_relevances, k)
    ideal_dcg = dcg(ideal_relevances, k)
    if ideal_dcg == 0:
        return 0
    else:
        return actual_dcg / ideal_dcg

def calculate_metrics(qrels, num_rel, trec):
    ks = [5, 10, 20, 50, 100]
    results = {}
    for query_id in trec:
        retrieved_docs = trec[query_id]
        relevant_docs = qrels.get(query_id, {})
        num_relevant = num_rel.get(query_id, 0)
        if num_relevant == 0:
            continue

        metrics = {}
        num_retrieved = 0
        num_retrieved_relevant = 0
        sum_precision = 0.0
        precisions = {}
        recalls = {}

        for k, (doc_id, _) in enumerate(retrieved_docs):
            num_retrieved += 1
            doc_relevance = relevant_docs.get(doc_id, 0)

            if doc_relevance > 0:
                sum_precision += (num_retrieved_relevant + 1) / num_retrieved
                num_retrieved_relevant += doc_relevance
            precisions[num_retrieved] = num_retrieved_relevant / num_retrieved
            recalls[num_retrieved] = num_retrieved_relevant / num_relevant

        average_precision = sum_precision / num_relevant

        # Fill out remainder of precision and recall values
        for k in range(num_retrieved + 1, max(ks) + 1):
            precisions[k] = num_retrieved_relevant / k
            recalls[k] = recalls[num_retrieved]

        if num_relevant > num_retrieved:
            r_precision = num_retrieved_relevant / num_relevant
        else:
            int_num_relevant = int(num_relevant)
            frac_num_relevant = num_relevant - int_num_relevant
            r_precision = (1 - frac_num_relevant) * precisions[int_num_relevant] + frac_num_relevant * precisions[int_num_relevant + 1] if frac_num_relevant > 0 else precisions[num_relevant]

        metrics['R-precision'] = r_precision
        metrics['Average Precision'] = average_precision
        metrics['Precision@k'] = [(k, precisions[k] if k in precisions else precisions[num_retrieved]) for k in ks]
        metrics['Recall@k'] = [(k, recalls[k] if k in recalls else recalls[num_retrieved]) for k in ks]
        metrics['F1@k'] = [(k, 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0)
                           for (k, prec), (_, rec) in zip(metrics['Precision@k'], metrics['Recall@k'])]

        # Interpolate precision values for all k
        # max_prec = 0
        # for k in range(num_retrieved, 0, -1):
        #     if precisions[k] > max_prec:
        #         max_prec = precisions[k]
        #     else:
        #         precisions[k] = max_prec

        # Calculate nDCG for specified k values
        ndcgs = {k: ndcg(retrieved_docs, relevant_docs, k) for k in ks}

        metrics['nDCG@k'] = list(ndcgs.items())

        results[query_id] = metrics

    return results

def main():
    parser = argparse.ArgumentParser(description="Homework 5 TREC Evaluation Script")
    parser.add_argument("qrels_path", type=str, help="Path to the QREL file", default='hw1/qrels.adhoc.51-100.AP89.txt', nargs='?')
    parser.add_argument("run_path", type=str, help="Path to the run file", default='hw1/results_es.txt', nargs='?')
    parser.add_argument("-q", "--query", action="store_true", help="Display metrics per query")

    args = parser.parse_args()

    qrels, num_rel = parse_qrels(args.qrels_path)
    trec = parse_trec(args.run_path)
    metrics = calculate_metrics(qrels, num_rel, trec)

    if args.query:
        for query_id, query_metrics in metrics.items():
            print(f"Metrics for Query ID {query_id}:")
            for metric, value in query_metrics.items():
                if isinstance(value, list):
                    print(f"{metric}:")
                    for k, v in value:
                        print(f"  @{k}: {v}")
                else:
                    print(f"{metric}: {value}")
            print()

    # Calculate average metrics across all queries
    print("Average Metrics Across All Queries:")
    avg_metrics = {}
    for metric in metrics[next(iter(metrics))]:
        if isinstance(metrics[next(iter(metrics))][metric], list):
            k_values = [k for k, _ in metrics[next(iter(metrics))][metric]]
            avg_metrics[metric] = [(k, np.mean([m[metric][i][1] for m in metrics.values()])) for i, k in enumerate(k_values)]
        else:
            avg_metrics[metric] = np.mean([m[metric] for m in metrics.values()])

    for metric, value in avg_metrics.items():
        if isinstance(value, list):
            print(f"{metric}:")
            for k, v in value:
                print(f"  @{k}: {v}")
        else:
            print(f"{metric}: {value}")

if __name__ == "__main__":
    main()
