import time
import shutil

import pyterrier as pt

from Group_Blue_config import INDEX_DIR, INDEX_TIMING_FILE, create_directories
from Group_Blue_dataset import read_documents
from Group_Blue_preprocessing import preprocess


def initialize_pyterrier():
    if not pt.started():
        pt.init()


def prepare_documents():
    documents = read_documents()
    if len(documents) != 1400:
        print(f"WARNING: expected 1400 documents, found {len(documents)}")
    documents["text"] = documents["text"].fillna("")
    start = time.perf_counter()
    documents["text"] = documents["text"].apply(preprocess)
    preprocessing_time = time.perf_counter() - start
    documents = documents[documents["text"].str.strip() != ""].copy()
    print(f"Documents loaded : {len(documents)}")
    print(f"Preprocessing    : {preprocessing_time:.6f} sec")
    return documents, preprocessing_time


def build_index():
    create_directories()
    initialize_pyterrier()
    documents, preprocessing_time = prepare_documents()
    if INDEX_DIR.exists():
        shutil.rmtree(INDEX_DIR)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    indexer = pt.terrier.IterDictIndexer(
        str(INDEX_DIR),
        meta={"docno": 32},
        type=pt.index.IndexingType.CLASSIC,
    )
    indexref = indexer.index(documents[["docno", "text"]].to_dict("records"))
    indexing_time = time.perf_counter() - start
    INDEX_TIMING_FILE.write_text(
        f"documents_indexed={len(documents)}\n"
        f"preprocessing_time_seconds={preprocessing_time:.6f}\n"
        f"indexing_time_seconds={indexing_time:.6f}\n"
        f"total_build_time_seconds={preprocessing_time + indexing_time:.6f}\n"
        f"index_directory={INDEX_DIR}\n",
        encoding="utf-8",
    )
    print("=" * 70)
    print("INDEXING COMPLETE")
    print("=" * 70)
    print(f"Index       : {INDEX_DIR}")
    print(f"Preprocess  : {preprocessing_time:.6f} sec")
    print(f"Index       : {indexing_time:.6f} sec")
    return indexref, indexing_time, preprocessing_time


if __name__ == "__main__":
    build_index()
