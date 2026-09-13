"""
Group Blue - Porter Stemmer

Porter Stemming Algorithm
Based on the Python implementation of the Porter stemming
algorithm by Vivake Gupta, based on Martin Porter's ANSI C
implementation.

The Porter algorithm was originally described in:

M.F. Porter, "An algorithm for suffix stripping",
Program, Vol. 14, No. 3, 1980, pp. 130-137.

The assignment permits the use of a Porter stemmer from
an external source.

This file is used by Group_Blue_preprocess.py.
"""


class PorterStemmer:

    def __init__(self):
        self.b = ""
        self.k = 0
        self.k0 = 0
        self.j = 0

    # ---------------------------------------------------------
    # Determine whether character at position i is consonant
    # ---------------------------------------------------------
    def cons(self, i):

        if self.b[i] in "aeiou":
            return False

        if self.b[i] == "y":

            if i == self.k0:
                return True

            return not self.cons(i - 1)

        return True

    # ---------------------------------------------------------
    # Measure the number of consonant sequences
    # ---------------------------------------------------------
    def m(self):

        n = 0
        i = self.k0

        while True:

            if i > self.j:
                return n

            if not self.cons(i):
                break

            i += 1

        i += 1

        while True:

            while True:

                if i > self.j:
                    return n

                if self.cons(i):
                    break

                i += 1

            i += 1
            n += 1

            while True:

                if i > self.j:
                    return n

                if not self.cons(i):
                    break

                i += 1

            i += 1

    # ---------------------------------------------------------
    # Check whether stem contains a vowel
    # ---------------------------------------------------------
    def vowelinstem(self):

        for i in range(self.k0, self.j + 1):

            if not self.cons(i):
                return True

        return False

    # ---------------------------------------------------------
    # Check for double consonant
    # ---------------------------------------------------------
    def doublec(self, j):

        if j < self.k0 + 1:
            return False

        if self.b[j] != self.b[j - 1]:
            return False

        return self.cons(j)

    # ---------------------------------------------------------
    # Check consonant-vowel-consonant ending
    # ---------------------------------------------------------
    def cvc(self, i):

        if i < self.k0 + 2:
            return False

        if not self.cons(i):
            return False

        if self.cons(i - 1):
            return False

        if not self.cons(i - 2):
            return False

        if self.b[i] in "wxy":
            return False

        return True

    # ---------------------------------------------------------
    # Check whether current word ends with string s
    # ---------------------------------------------------------
    def ends(self, s):

        length = len(s)

        if length > (self.k - self.k0 + 1):
            return False

        if self.b[self.k - length + 1:self.k + 1] != s:
            return False

        self.j = self.k - length

        return True

    # ---------------------------------------------------------
    # Replace suffix
    # ---------------------------------------------------------
    def setto(self, s):

        length = len(s)

        self.b = (
            self.b[:self.j + 1]
            + s
            + self.b[self.j + length + 1:]
        )

        self.k = self.j + length

    # ---------------------------------------------------------
    # Replace suffix if measure > 0
    # ---------------------------------------------------------
    def r(self, s):

        if self.m() > 0:
            self.setto(s)

    # ---------------------------------------------------------
    # Step 1a and 1b
    # ---------------------------------------------------------
    def step1ab(self):

        if self.b[self.k] == "s":

            if self.ends("sses"):
                self.k -= 2

            elif self.ends("ies"):
                self.setto("i")

            elif self.b[self.k - 1] != "s":
                self.k -= 1

        if self.ends("eed"):

            if self.m() > 0:
                self.k -= 1

        elif (
            self.ends("ed")
            or self.ends("ing")
        ):

            if self.vowelinstem():

                self.k = self.j

                if self.ends("at"):
                    self.setto("ate")

                elif self.ends("bl"):
                    self.setto("ble")

                elif self.ends("iz"):
                    self.setto("ize")

                elif self.doublec(self.k):

                    self.k -= 1

                    if self.b[self.k] in "lsz":
                        self.k += 1

                elif (
                    self.m() == 1
                    and self.cvc(self.k)
                ):
                    self.setto("e")

    # ---------------------------------------------------------
    # Step 1c
    # ---------------------------------------------------------
    def step1c(self):

        if self.ends("y") and self.vowelinstem():

            self.b = (
                self.b[:self.k]
                + "i"
                + self.b[self.k + 1:]
            )

    # ---------------------------------------------------------
    # Step 2
    # ---------------------------------------------------------
    def step2(self):

        if self.b[self.k - 1] == "a":

            if self.ends("ational"):
                self.r("ate")

            elif self.ends("tional"):
                self.r("tion")

        elif self.b[self.k - 1] == "c":

            if self.ends("enci"):
                self.r("ence")

            elif self.ends("anci"):
                self.r("ance")

        elif self.b[self.k - 1] == "e":

            if self.ends("izer"):
                self.r("ize")

        elif self.b[self.k - 1] == "l":

            if self.ends("bli"):
                self.r("ble")

            elif self.ends("alli"):
                self.r("al")

            elif self.ends("entli"):
                self.r("ent")

            elif self.ends("eli"):
                self.r("e")

            elif self.ends("ousli"):
                self.r("ous")

        elif self.b[self.k - 1] == "o":

            if self.ends("ization"):
                self.r("ize")

            elif self.ends("ation"):
                self.r("ate")

            elif self.ends("ator"):
                self.r("ate")

        elif self.b[self.k - 1] == "s":

            if self.ends("alism"):
                self.r("al")

            elif self.ends("iveness"):
                self.r("ive")

            elif self.ends("fulness"):
                self.r("ful")

            elif self.ends("ousness"):
                self.r("ous")

        elif self.b[self.k - 1] == "t":

            if self.ends("aliti"):
                self.r("al")

            elif self.ends("iviti"):
                self.r("ive")

            elif self.ends("biliti"):
                self.r("ble")

        elif self.b[self.k - 1] == "g":

            if self.ends("logi"):
                self.r("log")

    # ---------------------------------------------------------
    # Step 3
    # ---------------------------------------------------------
    def step3(self):

        if self.b[self.k] == "e":

            if self.ends("icate"):
                self.r("ic")

            elif self.ends("ative"):
                self.r("")

            elif self.ends("alize"):
                self.r("al")

        elif self.b[self.k] == "i":

            if self.ends("iciti"):
                self.r("ic")

        elif self.b[self.k] == "l":

            if self.ends("ical"):
                self.r("ic")

            elif self.ends("ful"):
                self.r("")

        elif self.b[self.k] == "s":

            if self.ends("ness"):
                self.r("")

    # ---------------------------------------------------------
    # Step 4
    # ---------------------------------------------------------
    def step4(self):

        if self.b[self.k - 1] == "a":

            if not self.ends("al"):
                return

        elif self.b[self.k - 1] == "c":

            if self.ends("ance"):
                pass

            elif self.ends("ence"):
                pass

            else:
                return

        elif self.b[self.k - 1] == "e":

            if not self.ends("er"):
                return

        elif self.b[self.k - 1] == "i":

            if not self.ends("ic"):
                return

        elif self.b[self.k - 1] == "l":

            if self.ends("able"):
                pass

            elif self.ends("ible"):
                pass

            else:
                return

        elif self.b[self.k - 1] == "n":

            if self.ends("ant"):
                pass

            elif self.ends("ement"):
                pass

            elif self.ends("ment"):
                pass

            elif self.ends("ent"):
                pass

            else:
                return

        elif self.b[self.k - 1] == "o":

            if self.ends("ion"):

                if self.b[self.j] == "s":
                    pass

                elif self.b[self.j] == "t":
                    pass

                else:
                    return

            elif self.ends("ou"):
                pass

            else:
                return

        elif self.b[self.k - 1] == "s":

            if not self.ends("ism"):
                return

        elif self.b[self.k - 1] == "t":

            if self.ends("ate"):
                pass

            elif self.ends("iti"):
                pass

            else:
                return

        elif self.b[self.k - 1] == "u":

            if not self.ends("ous"):
                return

        elif self.b[self.k - 1] == "v":

            if not self.ends("ive"):
                return

        elif self.b[self.k - 1] == "z":

            if not self.ends("ize"):
                return

        else:
            return

        if self.m() > 1:
            self.k = self.j

    # ---------------------------------------------------------
    # Step 5
    # ---------------------------------------------------------
    def step5(self):

        self.j = self.k

        if self.b[self.k] == "e":

            a = self.m()

            if (
                a > 1
                or (
                    a == 1
                    and not self.cvc(self.k - 1)
                )
            ):
                self.k -= 1

        if (
            self.b[self.k] == "l"
            and self.doublec(self.k)
            and self.m() > 1
        ):
            self.k -= 1

    # ---------------------------------------------------------
    # Main stemming function
    # ---------------------------------------------------------
    def stem(self, word):

        if not word:
            return ""

        word = word.lower()

        # Only process alphabetic words
        if not word.isalpha():
            return word

        self.b = word
        self.k = len(word) - 1
        self.k0 = 0
        self.j = 0

        # The canonical implementation does not stem
        # one- or two-character words.
        if self.k <= self.k0 + 1:
            return self.b

        self.step1ab()
        self.step1c()
        self.step2()
        self.step3()
        self.step4()
        self.step5()

        return self.b[self.k0:self.k + 1]


# -------------------------------------------------------------
# Simple test
# -------------------------------------------------------------
if __name__ == "__main__":

    stemmer = PorterStemmer()

    test_words = [
        "caresses",
        "ponies",
        "ties",
        "caress",
        "cats",
        "feed",
        "agreed",
        "disabled",
        "matting",
        "mating",
        "meeting",
        "milling",
        "messing",
        "meetings",
        "relational",
        "conditional",
        "rational",
        "valenci",
        "hesitanci",
        "digitizer",
        "conformabli",
        "radicalli",
        "differentli",
        "vileli",
        "analogousli",
        "vietnamization",
        "predication",
        "operator",
        "feudalism",
        "decisiveness",
        "hopefulness",
        "callousness",
        "formaliti",
        "sensitiviti",
        "sensibiliti"
    ]

    for word in test_words:
        print(
            f"{word:20s} -> {stemmer.stem(word)}"
        )