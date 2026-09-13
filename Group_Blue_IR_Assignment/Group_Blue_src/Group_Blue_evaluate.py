
import os
import pandas as pd
import pyterrier as pt


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ASSIGNMENT_DIR = os.path.dirname(SRC_DIR)

DATA_DIR = os.path.join(
    ASSIGNMENT_DIR,
    "Group_Blue_data",
    "cran"
)

QUERY_PATH = os.path.join(DATA_DIR, "cran.qry")
QREL_PATH = os.path.join(DATA_DIR, "cranqrel")


# ---------------------------------------------------------
# Load Cranfield queries
# ---------------------------------------------------------

def load_queries():
    """
    Load queries from the Cranfield cran.qry file.
    """

    queries = []

    current_qid = None
    current_text = []
    reading_query = False

    with open(QUERY_PATH, "r", encoding="utf-8") as f:

        for line in f:
            line = line.rstrip("\n")

            if line.startswith(".I"):

                if current_qid is not None:
                    queries.append({
                        "qid": current_qid,
                        "query": " ".join(current_text).strip()
                    })

                current_qid = line[2:].strip()
                current_text = []
                reading_query = False

            elif line.startswith(".W"):
                reading_query = True

            elif line.startswith(".T"):
                reading_query = False

            elif line.startswith(".A") or line.startswith(".B"):
                reading_query = False

            elif reading_query:
                current_text.append(line.strip())

    # Add final query
    if current_qid is not None:
        queries.append({
            "qid": current_qid,
            "query": " ".join(current_text).strip()
        })

    return pd.DataFrame(queries)


# ---------------------------------------------------------
# Load Cranfield qrels
# ---------------------------------------------------------

def load_qrels():
    """
    Load Cranfield relevance judgments.

    Format:
        qid docno relevance
    """

    qrels = []

    with open(QREL_PATH, "r", encoding="utf-8") as f:

        for line in f:

            parts = line.strip().split()

            if len(parts) < 3:
                continue

            qid = parts[0]
            docno = parts[1]
            relevance = int(parts[2])

            qrels.append({
                "qid": qid,
                "docno": docno,
                "label": relevance
            })

    return pd.DataFrame(qrels)


# ---------------------------------------------------------
# Evaluate retrieval results
# ---------------------------------------------------------

def evaluate_results(results, metrics=None):
    """
    Evaluate a retrieval run against the Cranfield qrels.
    """

    if metrics is None:
        metrics = [
            "map",
            "recip_rank",
            "P.5",
            "P.10",
            "recall.100",
            "ndcg_cut.10"
        ]

    topics = load_queries()
    qrels = load_qrels()

    evaluation = pt.Experiment(
        [results],
        topics,
        qrels,
        eval_metrics=metrics,
        names=["Group_Blue_Run"],
        filter_by_qrels=False
    )

    return evaluation.iloc[0].to_dict()


# ---------------------------------------------------------
# Convert evaluation to experiment result row
# ---------------------------------------------------------

def evaluation_to_dataframe(
    evaluation,
    model,
    k1=None,
    b=None,
    search_time=None,
    avg_query_time=None
):
    """
    Convert evaluation results and experiment metadata
    into a single DataFrame row.
    """

    row = {
        "model": model,
        "bm25_k1": k1,
        "bm25_b": b,
        "search_time_sec": search_time,
        "avg_query_time_sec": avg_query_time
    }

    # Add evaluation metrics
    row.update(evaluation)

    return pd.DataFrame([row])


# ---------------------------------------------------------
# Standalone test
# ---------------------------------------------------------

if __name__ == "__main__":

    queries = load_queries()
    qrels = load_qrels()

    print("Queries loaded :", len(queries))
    print("Qrels loaded   :", len(qrels))
