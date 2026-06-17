"""POS tagging with the glass-box lattice-CRF tagger (English, UD UPOS tags)."""
from interpretable_corenlp import load_tagger

tagger = load_tagger("models/pos/model.npz", w_dir="models/embedding")
tokens = "The Stanford CoreNLP server was running on port 9000 .".split()
print("tokens:", tokens)
print("tags:  ", tagger.tag(tokens))
print("NP chunks:", tagger.np_chunks(tokens))   # (start, end) half-open spans
