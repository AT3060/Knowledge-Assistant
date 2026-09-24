"""
Step 4 - BM25, written from scratch.

BM25 scores a chunk d for a query q as a sum over the query words t:

    score(q, d) = Σ_t  IDF(t) · tf(t,d) · (k1 + 1) / ( tf(t,d) + k1 · (1 - b + b · |d| / avgdl) )

    tf(t,d)  how often word t occurs in chunk d
    IDF(t)   log(1 + (N - n_t + 0.5) / (n_t + 0.5))   - rare words count more than common ones
             N = number of chunks, n_t = number of chunks containing t
    |d|      length of chunk d in words, avgdl = average chunk length
    k1       saturation: the 2nd occurrence of a word adds less than the 1st (typical 1.2-2.0)
    b        length normalisation: long chunks are penalised a bit (0 = off, 1 = full; typical 0.75)
"""
import math
import re
from collections import Counter

# A short German stopword list: very frequent words that carry almost no meaning for search.
STOPWORDS = set("""
aber alle als am an auch auf aus bei bin bis bzw da damit dann das dass dem den der des die dies
diese dieser du durch ein eine einem einen einer eines er es für hat habe haben ich ihr im in ist
ja kann man mein meine mich mir mit muss nach nicht noch nur oder sich sie sind so über um und uns
von vor war was wenn werden wer wie wird wir zu zum zur
""".split())


class Tokenizer:
    """Turns text into a list of search terms.
    Each option is a separate switch, so we can MEASURE what each one contributes."""

    def __init__(self, stopwords=False, stemming=False):
        self.stopwords = stopwords
        self.stemmer = None
        if stemming:
            import snowballstemmer
            self.stemmer = snowballstemmer.stemmer("german")

    def __call__(self, text):
        tokens = re.findall(r"\w+", text.lower())        # 'PACS-Zugang' -> ['pacs', 'zugang']
        if self.stopwords:
            tokens = [t for t in tokens if t not in STOPWORDS]
        if self.stemmer:
            tokens = self.stemmer.stemWords(tokens)      # 'berechtigungen' -> 'berecht'
        return tokens


class BM25:
    def __init__(self, texts, tokenizer, k1=1.5, b=0.75):
        self.tokenizer, self.k1, self.b = tokenizer, k1, b
        self.docs = [Counter(tokenizer(t)) for t in texts]           # term frequencies per chunk
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avgdl = sum(self.lengths) / len(self.lengths)
        n_docs = len(self.docs)
        df = Counter(term for d in self.docs for term in d)          # in how many chunks is each term?
        self.idf = {t: math.log(1 + (n_docs - n + 0.5) / (n + 0.5)) for t, n in df.items()}

    def scores(self, query):
        """One BM25 score per chunk (0 if the chunk shares no word with the query)."""
        q_terms = self.tokenizer(query)
        out = []
        for d, length in zip(self.docs, self.lengths):
            s = 0.0
            for t in q_terms:
                tf = d.get(t, 0)
                if tf:
                    norm = self.k1 * (1 - self.b + self.b * length / self.avgdl)
                    s += self.idf[t] * tf * (self.k1 + 1) / (tf + norm)
            out.append(s)
        return out

    def search(self, query, k=50):
        """Indices of the k best chunks, best first."""
        s = self.scores(query)
        return sorted(range(len(s)), key=lambda i: -s[i])[:k]
