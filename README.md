# Group Blue - Information Retrieval Programming Assignment II

## Run
From IR---2/python Group_Blue_IR_Assignment/Group_Blue_run_all.py
From `Group_Blue_src`:

```bash
pip install -r ../Group_Blue_requirements.txt
python Group_Blue_index.py
python Group_Blue_experiments.py
```

The experiment script runs TF-IDF, BM25, 20 BM25 `(k1,b)` combinations, and PL2, evaluates them, records timings, selects the best configuration by MAP, and writes the final TREC run.

## Unknown queries

```bash
python Group_Blue_retrieval.py --model BM25 --k1 1.2 --b 0.75 --queries unknown_queries.txt --output unknown_results.txt
```

Query files use Cranfield `.I` / `.W` format.

## Outputs

`Group_Blue_results/` receives experiment CSVs, timing data, baseline runs, the selected best-model JSON, and `Group_Blue_final_results.txt`.

The report must use the measured values from these generated files; no scores should be fabricated.
