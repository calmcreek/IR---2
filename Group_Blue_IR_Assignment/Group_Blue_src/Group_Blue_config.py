from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "Group_Blue_src"
DATA_DIR = PROJECT_ROOT / "Group_Blue_data" / "cran"
DOCUMENT_FILE = DATA_DIR / "cran.all.1400"
QUERY_FILE = DATA_DIR / "cran.qry"
QREL_FILE = DATA_DIR / "cranqrel"
INDEX_DIR = PROJECT_ROOT / "Group_Blue_index"
RESULTS_DIR = PROJECT_ROOT / "Group_Blue_results"
REPORTS_DIR = PROJECT_ROOT / "Group_Blue_reports"

EXPERIMENT_RESULTS_FILE = RESULTS_DIR / "Group_Blue_experiment_results.csv"
TIMING_RESULTS_FILE = RESULTS_DIR / "Group_Blue_timing_results.csv"
BEST_MODEL_FILE = RESULTS_DIR / "Group_Blue_best_model.txt"
FINAL_RESULTS_FILE = RESULTS_DIR / "Group_Blue_final_results.txt"
BM25_RESULTS_FILE = RESULTS_DIR / "Group_Blue_BM25_results.txt"
TFIDF_RESULTS_FILE = RESULTS_DIR / "Group_Blue_TFIDF_results.txt"
PL2_RESULTS_FILE = RESULTS_DIR / "Group_Blue_PL2_results.txt"
INDEX_TIMING_FILE = RESULTS_DIR / "Group_Blue_index_timing.txt"

NUM_RESULTS = 1000
BASELINE_MODELS = ["TF_IDF", "BM25", "PL2"]
BM25_K1_VALUES = [0.6, 0.9, 1.2, 1.5, 2.0]
BM25_B_VALUES = [0.3, 0.5, 0.75, 1.0]
EVALUATION_METRICS = ["map", "recip_rank", "P.5", "P.10", "recall.100", "ndcg_cut.10"]
PRIMARY_METRIC = "map"


def create_directories():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
