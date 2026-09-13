
import json
import os
import time
import pandas as pd

from Group_Blue_config import (
    BASELINE_MODELS,
    BM25_K1_VALUES,
    BM25_B_VALUES,
    EVALUATION_METRICS,
    RESULTS_DIR,
    EXPERIMENT_RESULTS_FILE,
    TIMING_RESULTS_FILE,
    FINAL_RESULTS_FILE,
    BEST_MODEL_FILE,
)

from Group_Blue_retrieval import retrieve
from Group_Blue_evaluate import (
    evaluate_results,
    evaluation_to_dataframe,
)


def run_experiment(model, k1=None, b=None, save_run=False):
    """
    Run one retrieval experiment and evaluate it.
    """

    start_time = time.perf_counter()

    results = retrieve(
        model=model,
        k1=k1,
        b=b
    )

    search_time = time.perf_counter() - start_time

    # Average query time
    number_of_queries = results["qid"].nunique()

    if number_of_queries > 0:
        avg_query_time = search_time / number_of_queries
    else:
        avg_query_time = 0.0

    evaluation = evaluate_results(
        results,
        metrics=EVALUATION_METRICS
    )

    row = evaluation_to_dataframe(
        evaluation,
        model,
        k1,
        b,
        search_time,
        avg_query_time
    )

    if save_run:

        if model == "TF_IDF":
            filename = "Group_Blue_TFIDF_results.txt"

        elif model == "PL2":
            filename = "Group_Blue_PL2_results.txt"

        elif model == "BM25":
            if k1 is not None and b is not None:
                filename = (
                    f"Group_Blue_BM25_k1_{k1}_b_{b}_results.txt"
                )
            else:
                filename = "Group_Blue_BM25_results.txt"

        else:
            filename = f"Group_Blue_{model}_results.txt"

        output_path = os.path.join(
            RESULTS_DIR,
            filename
        )

        results.to_csv(
            output_path,
            sep=" ",
            index=False,
            header=False
        )

        print(f"Saved results to: {output_path}")

    return row


def run_all_experiments():

    os.makedirs(RESULTS_DIR, exist_ok=True)

    rows = []

    # -----------------------------------------------------
    # TF-IDF baseline
    # -----------------------------------------------------

    print("Running TF-IDF baseline...")

    rows.append(
        run_experiment(
            "TF_IDF",
            save_run=True
        )
    )

    # -----------------------------------------------------
    # BM25 baseline
    # -----------------------------------------------------

    print("Running BM25 baseline...")

    rows.append(
        run_experiment(
            "BM25",
            save_run=True
        )
    )

    # -----------------------------------------------------
    # BM25 parameter tuning
    # -----------------------------------------------------

    total_configs = (
        len(BM25_K1_VALUES)
        * len(BM25_B_VALUES)
    )

    config_number = 0

    for k1 in BM25_K1_VALUES:

        for b in BM25_B_VALUES:

            config_number += 1

            print(
                f"Running BM25 {config_number}/{total_configs}: "
                f"k1={k1}, b={b}"
            )

            rows.append(
                run_experiment(
                    "BM25",
                    k1=k1,
                    b=b,
                    save_run=False
                )
            )

    # -----------------------------------------------------
    # PL2 baseline
    # -----------------------------------------------------

    print("Running PL2 baseline...")

    rows.append(
        run_experiment(
            "PL2",
            save_run=True
        )
    )

    # -----------------------------------------------------
    # Combine experiment results
    # -----------------------------------------------------

    experiment_results = pd.concat(
        rows,
        ignore_index=True
    )

    # -----------------------------------------------------
    # Select best configuration
    # Primary metric = MAP
    # Secondary metric = nDCG@10
    # -----------------------------------------------------

    sort_columns = []

    if "map" in experiment_results.columns:
        sort_columns.append("map")

    if "ndcg_cut_10" in experiment_results.columns:
        sort_columns.append("ndcg_cut_10")

    if sort_columns:

        experiment_results = experiment_results.sort_values(
            by=sort_columns,
            ascending=False
        )

    experiment_results.to_csv(
        EXPERIMENT_RESULTS_FILE,
        index=False
    )

    # -----------------------------------------------------
    # Best configuration
    # -----------------------------------------------------

    best = experiment_results.iloc[0]

    best_model = str(best["model"])

    best_k1 = (
        None
        if pd.isna(best["bm25_k1"])
        else float(best["bm25_k1"])
    )

    best_b = (
        None
        if pd.isna(best["bm25_b"])
        else float(best["bm25_b"])
    )

    # -----------------------------------------------------
    # Run best configuration again
    # -----------------------------------------------------

    print("\nBest configuration:")
    print(f"Model : {best_model}")
    print(f"k1    : {best_k1}")
    print(f"b     : {best_b}")

    final_results = retrieve(
        model=best_model,
        k1=best_k1,
        b=best_b
    )

    final_results.to_csv(
        FINAL_RESULTS_FILE,
        sep=" ",
        index=False,
        header=False
    )

    print(
        f"Saved results to: {FINAL_RESULTS_FILE}"
    )

    # -----------------------------------------------------
    # Save best model information
    # -----------------------------------------------------

    best_model_info = {
        "model": best_model,
        "k1": best_k1,
        "b": best_b,
        "map": (
            float(best["map"])
            if "map" in best and not pd.isna(best["map"])
            else None
        ),
        "ndcg_cut_10": (
            float(best["ndcg_cut_10"])
            if "ndcg_cut_10" in best
            and not pd.isna(best["ndcg_cut_10"])
            else None
        )
    }

    with open(
        BEST_MODEL_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            best_model_info,
            f,
            indent=4
        )

    # -----------------------------------------------------
    # Timing results
    # -----------------------------------------------------

    timing_columns = [
        "model",
        "bm25_k1",
        "bm25_b",
        "search_time_sec",
        "avg_query_time_sec"
    ]

    available_timing_columns = [
        column
        for column in timing_columns
        if column in experiment_results.columns
    ]

    experiment_results[
        available_timing_columns
    ].to_csv(
        TIMING_RESULTS_FILE,
        index=False
    )

    print(
        f"Saved experiment results to: "
        f"{EXPERIMENT_RESULTS_FILE}"
    )

    print(
        f"Saved timing results to: "
        f"{TIMING_RESULTS_FILE}"
    )

    print(
        f"Saved best model to: "
        f"{BEST_MODEL_FILE}"
    )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("EXPERIMENTS COMPLETE")
    print("=" * 70)

    print(f"Best model : {best_model}")
    print(f"Best k1    : {best_k1}")
    print(f"Best b     : {best_b}")

    if "map" in best:
        print(f"Best MAP   : {best['map']}")

    if "ndcg_cut_10" in best:
        print(f"Best nDCG@10: {best['ndcg_cut_10']}")

    print("=" * 70)


if __name__ == "__main__":
    run_all_experiments()

