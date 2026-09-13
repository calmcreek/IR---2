from pathlib import Path
import pandas as pd

from Group_Blue_config import DOCUMENT_FILE, QUERY_FILE, QREL_FILE


def read_documents(path=DOCUMENT_FILE):
    """Parse Cranfield documents; searchable content is title + abstract only."""
    documents = []
    current_docno = None
    current_field = None
    title, author, abstract = [], [], []

    def save_document():
        if current_docno is None:
            return
        title_text = " ".join(title).strip()
        author_text = " ".join(author).strip()
        abstract_text = " ".join(abstract).strip()
        documents.append({
            "docno": str(current_docno),
            "title": title_text,
            "author": author_text,
            "abstract": abstract_text,
            "text": f"{title_text} {abstract_text}".strip(),
        })

    with Path(path).open("r", encoding="utf-8", errors="ignore") as file:
        for raw_line in file:
            line = raw_line.rstrip("\n")
            if line.startswith(".I"):
                save_document()
                current_docno = line[2:].strip()
                current_field = None
                title, author, abstract = [], [], []
            elif line.startswith(".T"):
                current_field = "title"
            elif line.startswith(".A"):
                current_field = "author"
            elif line.startswith(".B"):
                current_field = "bibliography"
            elif line.startswith(".W"):
                current_field = "abstract"
            elif line.startswith(".X"):
                current_field = "references"
            elif current_field == "title":
                title.append(line.strip())
            elif current_field == "author":
                author.append(line.strip())
            elif current_field == "abstract":
                abstract.append(line.strip())
    save_document()
    return pd.DataFrame(documents)


def read_queries(path=QUERY_FILE):
    """Parse Cranfield .qry files into qid/query."""
    queries = []
    current_qid = None
    current_field = None
    query_text = []

    def save_query():
        if current_qid is not None:
            queries.append({"qid": str(current_qid), "query": " ".join(query_text).strip()})

    with Path(path).open("r", encoding="utf-8", errors="ignore") as file:
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


def read_qrels(path=QREL_FILE):
    rows = []
    with Path(path).open("r", encoding="utf-8", errors="ignore") as file:
        for raw_line in file:
            parts = raw_line.strip().split()
            if len(parts) < 3:
                continue
            try:
                label = int(parts[2])
            except ValueError:
                continue
            rows.append({"qid": str(parts[0]), "docno": str(parts[1]), "label": label})
    return pd.DataFrame(rows)


def print_dataset_summary():
    documents = read_documents()
    queries = read_queries()
    qrels = read_qrels()
    print("=" * 70)
    print("CRANFIELD DATASET SUMMARY")
    print("=" * 70)
    print(f"Documents : {len(documents)}")
    print(f"Queries   : {len(queries)}")
    print(f"Qrels     : {len(qrels)}")
    print("=" * 70)


if __name__ == "__main__":
    print_dataset_summary()
