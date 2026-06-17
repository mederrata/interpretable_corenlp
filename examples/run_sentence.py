"""Sentence boundary detection. segment_text() takes raw text and returns char-span sentences."""
from interpretable_corenlp import load_tagger, load_sbd

tagger = load_tagger("models/pos/model.npz", w_dir="models/embedding")
sbd = load_sbd("models/sbd/sbd.npz", engram_npz="models/sbd/engram.npz")

text = ("Dr. Smith joined in 2009. The model, e.g. ours, beats CoreNLP. "
        "It scored 0.926 on GUM. Numbers like 3.14 don't split.")
for lo, hi in sbd.segment_text(text, tagger):
    print("|", text[lo:hi])
