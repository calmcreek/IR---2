from pathlib import Path
import pandas as pd

from Group_Blue_config import (
    DOCUMENT_FILE,
    QUERY_FILE,
    QREL_FILE,
)


# ============================================================
# DOCUMENT PARSER
# ============================================================

def read_documents(path=DOCUMENT_FILE):
    """
    Read Cranfield documents.

    Returns a pandas DataFrame with:

        docno
        title
        author
        text
    """

    documents = []

    current_docno = None
    current_field = None

    title = []
    author = []
    body = []
    keywords = []

    def save_document():
        if current_docno is None:
            return

        searchable_text = " ".join(
            title + author + body + keywords
        ).strip()

        documents.append({
            "docno": str(current_docno),
            "title": " ".join(title).strip(),
            "author": " ".join(author).strip(),
            "text": searchable_text
        })

    with open(path, "r", encoding="utf-8", errors="ignore") as file:

        for raw_line in file:

            line = raw_line.rstrip("\n")

            if line.startswith(".I"):
                save_document()

                current_docno = line[2:].strip()
                current_field = None

                title = []
                author = []
                body = []
                keywords = []

            elif line.startswith(".T"):
                current_field = "title"

            elif line.startswith(".A"):
                current_field = "author"

            elif line.startswith(".B"):
                current_field = "bibliography"

            elif line.startswith(".W"):
                current_field = "body"

            elif line.startswith(".K"):
                current_field = "keywords"

            elif current_field == "title":
                title.append(line.strip())

            elif current_field == "author":
                author.append(line.strip())

            elif current_field == "body":
                body.append(line.strip())

            elif current_field == "keywords":
                keywords.append(line.strip())

    save_document()

    return pd.DataFrame(documents)


# ============================================================
# QUERY PARSER
# ============================================================

def read_queries(path=QUERY_FILE):
    """
    Read Cranfield queries.

    Returns DataFrame:

        qid
        query
    """

    queries = []

    current_qid = None
    current_field = None
    query_text = []

    def save_query():
        if current_qid is None:
            return

        queries.append({
            "qid": str(current_qid),
            "query": " ".join(query_text).strip()
        })

    with open(path, "r", encoding="utf-8", errors="ignore") as file:

        for raw_line in file:

            line = raw_line.rstrip("\n")

            if line.startswith(".I"):
                save_query()

                current_qid = line[2:].strip()
                current_field = None
                query_text = []

            elif line.startswith(".W"):
                current_field = "query"

            elif current_field == "query":
                query_text.append(line.strip())

    save_query()

    return pd.DataFrame(queries)


# ============================================================
# QREL PARSER
# ============================================================

def read_qrels(path=QREL_FILE):
    """
    Read Cranfield relevance judgments.

    Expected format:

        qid docno relevance

    Returns DataFrame:

        qid
        docno
        label
    """

    qrels = []

    with open(path, "r", encoding="utf-8", errors="ignore") as file:

        for raw_line in file:

            line = raw_line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) < 3:
                continue

            qid = parts[0]
            docno = parts[1]

            try:
                relevance = int(parts[2])
            except ValueError:
                continue

            qrels.append({
                "qid": str(qid),
                "docno": str(docno),
                "label": relevance
            })

    return pd.DataFrame(qrels)


# ============================================================
# DATASET SUMMARY
# ============================================================

def print_dataset_summary():

    documents = read_documents()
    queries = read_queries()
    qrels = read_qrels()

    print("=" * 60)
    print("CRANFIELD DATASET SUMMARY")
    print("=" * 60)

    print(f"Documents : {len(documents)}")
    print(f"Queries   : {len(queries)}")
    print(f"Qrels     : {len(qrels)}")

    print("\nFirst document:")
    print(documents.iloc[0].to_dict())

    print("\nFirst query:")
    print(queries.iloc[0].to_dict())

    print("\nFirst 10 qrels:")
    print(qrels.head(10).to_string(index=False))

    print("=" * 60)


if __name__ == "__main__":
    print_dataset_summary()