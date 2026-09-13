# Group Blue - Information Retrieval Programming Assignment II Report

## 1. Group Details

Add member names and roll numbers.

## 2. Problem Statement

Implement sparse ranked retrieval on Cranfield using an open-source search engine, compare retrieval models and parameter settings, evaluate against the supplied relevance judgments, and produce a final ranked run.

## 3. Implementation Details

### Dataset

Cranfield: 1400 documents, 225 queries, supplied qrels.

### Preprocessing

Title + abstract only, followed by tokenization, normalization, stopword removal and Porter stemming. The same preprocessing function is used for documents and queries.

### Indexing

PyTerrier/Terrier inverted index.

### Retrieval

TF-IDF, BM25, and PL2. BM25 is tuned over the configured k1/b grid.

## 4. Plan of Experiments

Run TF-IDF and BM25/PL2 baselines, evaluate 20 BM25 parameter combinations, record indexing/search time, and select the best configuration using MAP.

## 5. Results

Insert the actual contents of `Group_Blue_experiment_results.csv` and discuss the measured timing files.

## 6. Discussion

Discuss the best model, best BM25 parameters, metric differences, timing, limitations of tuning on the supplied qrels, and expected performance on unknown queries.
