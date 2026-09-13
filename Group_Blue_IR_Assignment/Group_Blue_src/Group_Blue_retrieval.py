import argparse
import time
from pathlib import Path

import pandas as pd
import pyterrier as pt

from Group_Blue_config import (
    INDEX_DIR, NUM_RESULTS, RESULTS_DIR,
    BM25_RESULTS_FILE, TFIDF_RESULTS_FILE, PL2_RESULTS_FILE,
    FINAL_RESULTS_FILE, create_directories,
)
from Group_Blue_dataset import read_queries
from Group_Blue_preprocessing import preprocess


def initialize_pyterrier():
    if not pt.started():
        pt.init()


def load_index():
    initialize_pyterrier()
    if not INDEX_DIR.exists():
        raise FileNotFoundError(f"Index not found: {INDEX_DIR}. Run Group_Blue_index.py first.")
    return pt.IndexFactory.of(str(INDEX_DIR))


def prepare_queries(query_file=None):
    queries = read_queries(query_file) if query_file else read_queries()
    queries["query"] = queries["query"].fillna("").apply(preprocess)
    queries = queries[queries["query"].str.strip() != ""].copy()
    return queries


def create_retriever(model, k1=None, b=None, num_results=NUM_RESULTS):
    index = load_index()
    controls = {}
    if model == "BM25":
        if k1 is not None:
            controls["bm25.k_1"] = str(k1)
        if b is not None:
            controls["bm25.b"] = str(b)
    elif model not in {"TF_IDF", "PL2"}:
        raise ValueError("Supported models: TF_IDF, BM25, PL2")
    return pt.terrier.Retriever(index, wmodel=model, num_results=num_results, controls=controls)


def run_retrieval(model, k1=None, b=None, num_results=NUM_RESULTS, query_file=None):
    queries = prepare_queries(query_file)
    retriever = create_retriever(model, k1, b, num_results)
    start = time.perf_counter()
    results = retriever.transform(queries)
    total_time = time.perf_counter() - start
    nqueries = len(queries)
    avg_time = total_time / nqueries if nqueries else 0.0
    if "rank" not in results.columns:
        results = results.sort_values(["qid", "score"], ascending=[True, False], kind="mergesort").copy()
        results["rank"] = results.groupby("qid").cumcount() + 1
    results["qid"] = results["qid"].astype(str)
    results["docno"] = results["docno"].astype(str)
    return results, total_time, avg_time, nqueries


def save_results(results, output_file, run_name="Group_Blue"):
    create_directories()
    output_file = Path(output_file)
    with output_file.open("w", encoding="utf-8") as file:
        for _, row in results.iterrows():
            file.write(
                f"{row['qid']} Q0 {row['docno']} {int(row['rank'])} "
                f"{float(row['score']):.8f} {run_name}\n"
            )
    print(f"Saved results to: {output_file}")


def run_baseline_models(query_file=None):
    outputs = {"TF_IDF": TFIDF_RESULTS_FILE, "BM25": BM25_RESULTS_FILE, "PL2": PL2_RESULTS_FILE}
    rows = []
    for model, output in outputs.items():
        results, total, avg, nqueries = run_retrieval(model, query_file=query_file)
        save_results(results, output, f"Group_Blue_{model}")
        rows.append({"model": model, "k1": None, "b": None, "total_search_time": total, "average_query_time": avg, "num_queries": nqueries})
    return pd.DataFrame(rows)

def retrieve(model="BM25", k1=None, b=None, query_file=None, num_results=NUM_RESULTS):
    """
    Compatibility wrapper for the experiment runner.

    Returns only the retrieval DataFrame.
    Timing information is handled separately by the experiment runner.
    """
    results, _, _, _ = run_retrieval(
        model=model,
        k1=k1,
        b=b,
        num_results=num_results,
        query_file=query_file,
    )
    return results
    
def main():
    parser = argparse.ArgumentParser(description="Group Blue sparse Cranfield retrieval")
    parser.add_argument("--model", choices=["TF_IDF", "BM25", "PL2"], default="BM25")
    parser.add_argument("--k1", type=float)
    parser.add_argument("--b", type=float)
    parser.add_argument("--queries", default=None)
    parser.add_argument("--output", default=str(FINAL_RESULTS_FILE))
    parser.add_argument("--run-name", default="Group_Blue_FINAL")
    parser.add_argument("--num-results", type=int, default=NUM_RESULTS)
    args = parser.parse_args()
    results, total, avg, nqueries = run_retrieval(
        args.model, args.k1, args.b, args.num_results, args.queries
    )
    save_results(results, args.output, args.run_name)
    print(f"Queries: {nqueries}\nTotal search time: {total:.6f} sec\nAverage/query: {avg:.6f} sec")


if __name__ == "__main__":
    main()
