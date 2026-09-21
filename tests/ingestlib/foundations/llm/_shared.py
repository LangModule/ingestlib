"""Corpus shared across AWS and Jina rerank tests so results are directly comparable."""

NLP_QUERY = "best python natural language processing library"

STANDARD_DOCS: tuple[str, ...] = (
    "A recipe for chocolate cake with buttercream frosting.",             # 0 — unrelated
    "spaCy is an industrial-strength NLP library for Python.",            # 1 — NLP
    "The stock market saw major gains today after the tech rally.",       # 2 — unrelated
    "NLTK is a Python natural language toolkit with tokenizers.",         # 3 — NLP
    "How to plant tomatoes in your garden during summer.",                # 4 — unrelated
    "Hugging Face transformers offers state-of-the-art NLP models.",      # 5 — NLP
)

NLP_INDICES: frozenset[int] = frozenset({1, 3, 5})
UNRELATED_INDICES: frozenset[int] = frozenset({0, 2, 4})
