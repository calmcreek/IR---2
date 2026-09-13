import time
from pathlib import Path

import pandas as pd
import pyterrier as pt

from Group_Blue_config import (
    INDEX_DIR,
    RESULTS_DIR,
    NUM_RESULTS,
    BM25_RESULTS_FILE,
    TFIDF_RESULTS_FILE,
    PL2_RESULTS_FILE,
    create_directories,
)

from Group_Blue_dataset import read_queries

from Group_Blue_preprocessing import preprocess


# ============================================================
# PYTERRIER INITIALIZATION
# ============================================================

def initialize_pyterrier():

    try:
        if not pt.started():
            pt.init()
    except AttributeError:
        pt.init()


# ============================================================
# LOAD INDEX
# ============================================================

def load_index():

    initialize_pyterrier()

    index = pt.IndexFactory.of(str(INDEX_DIR))

    return index


# ============================================================
# PREPARE QUERIES
# ============================================================

def prepare_queries():

    queries = read_queries()

    queries["query"] = queries["query"].fillna("")

    queries["query"] = queries["query"].apply(preprocess)

    return queries


# ============================================================
# CREATE RETRIEVER
# ============================================================

def create_retriever(
    model,
    k1=None,
    b=None,
    num_results=NUM_RESULTS
):
    """
    Create Terrier sparse retrieval model.

    Supported models:

        TF_IDF
        BM25
        PL2
    """

    index = load_index()

    controls = {}

    if model == "BM25":

        if k1 is not None:
            controls["bm25.k_1"] = str(k1)

        if b is not None:
            controls["bm25.b"] = str(b)

    retriever = pt.terrier.Retriever(
        index,
        wmodel=model,
        num_results=num_results,
        controls=controls
    )

    return retriever


# ============================================================
# RUN RETRIEVAL
# ============================================================

def run_retrieval(
    model,
    k1=None,
    b=None,
    num_results=NUM_RESULTS
):
    """
    Run a retrieval model over all Cranfield queries.

    Returns:

        results
        retrieval_time
        average_query_time
    """

    queries = prepare_queries()

    retriever = create_retriever(
        model=model,
        k1=k1,
        b=b,
        num_results=num_results
    )

    print("\n" + "=" * 60)

    if model == "BM25":
        print(
            f"Running {model} "
            f"(k1={k1}, b={b})"
        )
    else:
        print(f"Running {model}")

    print("=" * 60)

    start = time.perf_counter()

    results = retriever.transform(queries)

    total_time = time.perf_counter() - start

    number_of_queries = len(queries)

    average_query_time = (
        total_time / number_of_queries
        if number_of_queries > 0
        else 0
    )

    results = results.sort_values(
        ["qid", "score"],
        ascending=[True, False]
    )

    results["rank"] = (
        results
        .groupby("qid")
        .cumcount()
        + 1
    )

    print(f"Queries          : {number_of_queries}")
    print(f"Total time       : {total_time:.4f} sec")
    print(f"Average/query    : {average_query_time:.6f} sec")
    print(f"Retrieved rows   : {len(results)}")

    return (
        results,
        total_time,
        average_query_time
    )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    results,
    output_file,
    run_name="Group_Blue"
):
    """
    Save results in TREC format:

        qid Q0 docno rank score runname
    """

    create_directories()

    output_file = Path(output_file)

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        for _, row in results.iterrows():

            file.write(
                f"{row['qid']} "
                f"Q0 "
                f"{row['docno']} "
                f"{int(row['rank'])} "
                f"{float(row['score']):.8f} "
                f"{run_name}\n"
            )

    print(f"Saved results to: {output_file}")


# ============================================================
# BASELINE EXPERIMENT
# ============================================================

def run_baseline_models():

    create_directories()

    models = [
        ("TF_IDF", TFIDF_RESULTS_FILE),
        ("BM25", BM25_RESULTS_FILE),
        ("PL2", PL2_RESULTS_FILE)
    ]

    timings = []

    for model, output_file in models:

        results, total_time, average_time = run_retrieval(
            model=model
        )

        save_results(
            results,
            output_file,
            run_name=f"Group_Blue_{model}"
        )

        timings.append({
            "model": model,
            "k1": None,
            "b": None,
            "total_time": total_time,
            "average_query_time": average_time
        })

    return pd.DataFrame(timings)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    timing_results = run_baseline_models()

    print("\nTiming summary:")
    print(timing_results.to_string(index=False))