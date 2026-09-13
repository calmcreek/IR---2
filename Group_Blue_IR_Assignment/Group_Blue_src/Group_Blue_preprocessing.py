import re

from Group_Blue_porter_stemmer import PorterStemmer


INPUT_FILE = "cran.all.1400"
OUTPUT_FILE = "Group_Blue_processed.all"
STOPWORD_FILE = "stopwords.txt"


class Preprocessor:

    def __init__(self, stopword_file):

        self.stopwords = self.load_stopwords(
            stopword_file
        )

        self.stemmer = PorterStemmer()

    # =========================================================
    # FUNCTION 1: TOKENIZATION
    # =========================================================
    def tokenize(self, text):
        """
        Convert text into a collection of token strings.

        Punctuation is treated as a separator.
        """

        tokens = re.findall(
            r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?",
            text
        )

        return tokens

    # =========================================================
    # FUNCTION 2: STEMMING
    # =========================================================
    def stemming(self, tokens):
        """
        Convert tokens to their word stems using
        the Porter stemming algorithm.
        """

        stemmed_tokens = []

        for token in tokens:

            stemmed = self.stemmer.stem(
                token.lower()
            )

            if stemmed:
                stemmed_tokens.append(stemmed)

        return stemmed_tokens

    # =========================================================
    # FUNCTION 3: STOP WORD REMOVAL
    # =========================================================
    def stopword_removal(self, tokens):
        """
        Remove English stop words.
        """

        result = []

        for token in tokens:

            if token.lower() not in self.stopwords:
                result.append(token)

        return result

    # =========================================================
    # FUNCTION 4: NORMALIZATION
    # =========================================================
    def normalization(self, tokens):
        """
        Normalize tokens.

        Normalization includes:
        - lowercase conversion
        - removal of non-alphanumeric characters
        """

        normalized = []

        for token in tokens:

            token = token.lower()

            token = re.sub(
                r"[^a-z0-9]",
                "",
                token
            )

            if token:
                normalized.append(token)

        return normalized

    # =========================================================
    # STOP WORD LOADER
    # =========================================================
    def load_stopwords(self, filename):

        stopwords = set()

        with open(
            filename,
            "r",
            encoding="utf-8"
        ) as f:

            for line in f:

                word = line.strip().lower()

                if word:
                    stopwords.add(word)

        return stopwords

    # =========================================================
    # COMPLETE DOCUMENT PROCESSING
    # =========================================================
    def process_document(self, title, abstract):

        # Combine ONLY title and abstract.
        text = title + " " + abstract

        # 1. Tokenization
        tokens = self.tokenize(text)

        # 2. Stemming
        tokens = self.stemming(tokens)

        # 3. Stop word removal
        tokens = self.stopword_removal(tokens)

        # 4. Normalization
        tokens = self.normalization(tokens)

        return tokens

    # =========================================================
    # PROCESS CRANFIELD COLLECTION
    # =========================================================
    def process_collection(
        self,
        input_file,
        output_file
    ):

        documents = []

        current_docid = None
        current_section = None

        title = []
        abstract = []

        def save_document():

            nonlocal current_docid
            nonlocal title
            nonlocal abstract

            if current_docid is None:
                return

            tokens = self.process_document(
                " ".join(title),
                " ".join(abstract)
            )

            documents.append(
                (current_docid, tokens)
            )

            title = []
            abstract = []

        with open(
            input_file,
            "r",
            encoding="utf-8",
            errors="ignore"
        ) as f:

            for line in f:

                line = line.rstrip("\n")

                # ---------------------------------------------
                # Document ID
                # ---------------------------------------------
                if line.startswith(".I"):

                    save_document()

                    parts = line.split()

                    if len(parts) >= 2:

                        try:
                            current_docid = int(parts[1])

                        except ValueError:
                            current_docid = None

                    else:
                        current_docid = None

                    current_section = "I"

                # ---------------------------------------------
                # Title
                # ---------------------------------------------
                elif line.startswith(".T"):

                    current_section = "T"

                # ---------------------------------------------
                # Authors
                # ---------------------------------------------
                elif line.startswith(".A"):

                    current_section = "A"

                # ---------------------------------------------
                # Author affiliation
                # ---------------------------------------------
                elif line.startswith(".B"):

                    current_section = "B"

                # ---------------------------------------------
                # Abstract
                # ---------------------------------------------
                elif line.startswith(".W"):

                    current_section = "W"

                # ---------------------------------------------
                # Other fields
                # ---------------------------------------------
                elif line.startswith(".X"):

                    current_section = "X"

                # ---------------------------------------------
                # Normal content
                # ---------------------------------------------
                else:

                    if current_section == "T":

                        title.append(line)

                    elif current_section == "W":

                        abstract.append(line)

        # Save final document
        save_document()

        # =====================================================
        # WRITE OUTPUT
        # =====================================================
        with open(
            output_file,
            "w",
            encoding="utf-8"
        ) as out:

            for docid, tokens in documents:

                out.write(
                    f".I {docid}\n"
                )

                out.write(".S\n")

                out.write(
                    " ".join(tokens)
                    + "\n"
                )

        print("========================================")
        print("PREPROCESSING COMPLETE")
        print("========================================")
        print(
            "Documents processed:",
            len(documents)
        )
        print(
            "Output file:",
            output_file
        )


def main():

    processor = Preprocessor(
        STOPWORD_FILE
    )

    processor.process_collection(
        INPUT_FILE,
        OUTPUT_FILE
    )


if __name__ == "__main__":
    main()