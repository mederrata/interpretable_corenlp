"""Intrinsic interpretability: every prediction is an EXACT sum of NAMED contributions — read off WHY."""
from interpretable_corenlp import load_tagger, load_sbd

tagger = load_tagger("models/pos/model.npz", w_dir="models/embedding")
toks = "Apple unveiled the new iPhone today .".split()
print("=== POS explain() — each tag = sum of named feature contributions ===")
for w, tag, contribs in tagger.explain(toks):
    top = ", ".join(f"{n}={v:+.2f}" for n, v in contribs[:5])
    print(f"  {w:10s} -> {tag:6s}  | {top}")

print("\n=== SBD explain() — each boundary = sum of named contributions ===")
sbd = load_sbd("models/sbd/sbd.npz")
tks = "He left . She stayed .".split()
from interpretable_corenlp.posfeats import TAG2I
pos_ids = [TAG2I[t] for t in tagger.tag(tks)]
for w, p, contribs in sbd.explain(tks, pos_ids):
    top = ", ".join(f"{n}={v:+.2f}" for n, v in contribs[:4])
    print(f"  after {w:8s} P(boundary)={p:.2f} | {top}")
