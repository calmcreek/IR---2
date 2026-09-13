import time
import shutil

import pyterrier as pt

from Group_Blue_config import (
    INDEX_DIR,
    create_directories,
)

from Group_Blue_dataset import read_documents

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
# PREPARE DOCUMENTS
# ============================================================

def prepare_documents():

    documents = read_documents()

    print(f"Loaded {len(documents)} documents.")

    documents["text"] = documents["text"].fillna("")

    print("Preprocessing documents...")

    start = time.perf_counter()

    documents["text"] = documents["text"].apply(preprocess)

    preprocessing_time = time.perf_counter() - start

    print(
        f"Document preprocessing time: "
        f"{preprocessing_time:.4f} seconds"
    )

    return documents


# ============================================================
# BUILD INDEX
# ============================================================

def build_index():

    create_directories()

    initialize_pyterrier()

    documents = prepare_documents()

    # Remove an existing index so experiments start cleanly.
    if INDEX_DIR.exists():
        shutil.rmtree(INDEX_DIR)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    print("\nBuilding Terrier index...")

    start = time.perf_counter()

    indexer = pt.terrier.IterDictIndexer(
        str(INDEX_DIR),
        meta={
            "docno": 32
        },
        type=pt.index.IndexingType.CLASSIC
    )

    indexref = indexer.index(
        documents[
            ["docno", "text"]
        ].to_dict("records")
    )

    indexing_time = time.perf_counter() - start

    print("\n" + "=" * 60)
    print("INDEXING COMPLETE")
    print("=" * 60)

    print(f"Index location : {INDEX_DIR}")
    print(f"Indexing time  : {indexing_time:.4f} seconds")

    print("=" * 60)

    return indexref, indexing_time


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    build_index()