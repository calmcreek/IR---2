import pandas as pd
import pyterrier as pt

from Group_Blue_config import (
    QREL_FILE,
    EVALUATION_METRICS,
)


# ============================================================
# INITIALIZATION
# ============================================================

def initialize_pyterrier():

    try:
        if not pt.started():
            pt.init()
    except AttributeError:
        pt.init()


# ============================================================
# LOAD QRELS
# ============================================================

def load_qrels():

    qrels = pd.read_csv(
        QREL_FILE,
        sep=r"\s+",
        header=None,
        names=[
            "qid",
            "docno",
            "label"
        ],
        usecols=[0, 1, 2],
        dtype={
            "qid": str,
            "docno": str
        }
    )

    qrels["label"] = qrels["label"].astype(int)

    return qrels


# ============================================================
# EVALUATE
# ============================================================

def evaluate_results(
    results,
    qrels=None,
    metrics=None
):
    """
    Evaluate retrieval results.

    Metrics:

        MAP
        MRR
        Precision@5
        Precision@10
        Recall@100
        nDCG@10
    """

    initialize_pyterrier()

    if qrels is None:
        qrels = load_qrels()

    if metrics is None:
        metrics = EVALUATION_METRICS

    evaluation = pt.Utils.evaluate(
        results,
        qrels,
        metrics=metrics
    )

    return evaluation


# ============================================================
# CONVERT TO DATAFRAME
# ============================================================

def evaluation_to_dataframe(
    evaluation,
    model_name,
    k1=None,
    b=None
):
    """
    Convert PyTerrier evaluation dictionary
    into a one-row DataFrame.
    """

    row = {
        "model": model_name,
        "k1": k1,
        "b": b
    }

    for metric, value in evaluation.items():
        row[metric] = value

    return pd.DataFrame([row])


# ============================================================
# PRINT EVALUATION
# ============================================================

def print_evaluation(evaluation):

    print("\n" + "=" * 60)
    print("RETRIEVAL EVALUATION")
    print("=" * 60)

    for metric, value in evaluation.items():

        if isinstance(value, float):
            print(f"{metric:20s}: {value:.6f}")
        else:
            print(f"{metric:20s}: {value}")

    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    from Group_Blue_retrieval import run_retrieval

    for model in [
        "TF_IDF",
        "BM25",
        "PL2"
    ]:

        results, total_time, avg_time = run_retrieval(
            model=model
        )

        evaluation = evaluate_results(results)

        print(f"\nMODEL: {model}")

        print_evaluation(evaluation)