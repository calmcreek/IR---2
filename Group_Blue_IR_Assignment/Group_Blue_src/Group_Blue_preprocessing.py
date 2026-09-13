from pathlib import Path
import re

from Group_Blue_porter_stemmer import PorterStemmer

STOPWORD_FILE = Path(__file__).resolve().parent / "Group_Blue_stopwords.txt"


class Preprocessor:
    """Shared preprocessing for documents and queries."""

    TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")

    def __init__(self, stopword_file=STOPWORD_FILE):
        self.stopwords = self.load_stopwords(stopword_file)
        self.stemmer = PorterStemmer()

    def tokenize(self, text):
        if text is None:
            return []
        return self.TOKEN_PATTERN.findall(str(text))

    def normalization(self, tokens):
        result = []
        for token in tokens:
            token = token.lower()
            token = re.sub(r"[^a-z0-9]", "", token)
            if token:
                result.append(token)
        return result

    def load_stopwords(self, filename):
        stopwords = set()
        with Path(filename).open("r", encoding="utf-8") as file:
            for line in file:
                word = line.strip().lower()
                if word:
                    stopwords.add(word)
        return stopwords

    def stopword_removal(self, tokens):
        return [t for t in tokens if t not in self.stopwords]

    def stemming(self, tokens):
        return [s for s in (self.stemmer.stem(t) for t in tokens) if s]

    def process(self, text):
        tokens = self.tokenize(text)
        tokens = self.normalization(tokens)
        tokens = self.stopword_removal(tokens)
        tokens = self.stemming(tokens)
        return tokens

    def process_to_string(self, text):
        return " ".join(self.process(text))

    def process_document(self, title, abstract):
        return self.process(title + " " + abstract)

    def process_collection(self, input_file, output_file):
        from Group_Blue_dataset import read_documents
        documents = read_documents(input_file)
        with open(output_file, "w", encoding="utf-8") as out:
            for _, row in documents.iterrows():
                tokens = self.process_document(row["title"], row["abstract"])
                out.write(f".I {row['docno']}\n.S\n{' '.join(tokens)}\n")
        print(f"Documents processed: {len(documents)}")
        print(f"Output file: {output_file}")


_DEFAULT_PREPROCESSOR = Preprocessor()


def preprocess(text):
    """Return the preprocessed text as whitespace-separated tokens."""
    return _DEFAULT_PREPROCESSOR.process_to_string(text)


def preprocess_tokens(text):
    return _DEFAULT_PREPROCESSOR.process(text)


if __name__ == "__main__":
    print(preprocess("The aerodynamic analysis of aircraft wings."))
