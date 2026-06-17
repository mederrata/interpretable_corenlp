"""Clean SBD eval on REAL (un-augmented) UD sentences — the production-relevant number, vs the deliberately-hard
augmented held-out used in training. Concatenates consecutive real UD sentences into running-text documents
(full punctuation, real case, no injected word-wrap), segments, and reports:
  - SENTENCE-LEVEL EXACT MATCH: fraction of gold sentences whose (start,end) span is recovered exactly.
  - boundary P/R/F1 (gap-level) for comparison.

  uv run python -m bq_embedding.pos.eval_sbd --sbd NLP/sbd-corenlp-v3-*/sbd.npz --engram NLP/sbd-corenlp-v3-*/engram.npz \
     --pos NLP/pos-robust-ctxwin-20260617-1355/model.npz --conllu data/en_ewt-ud-test.conllu
"""
from __future__ import annotations
import argparse
from interpretable_corenlp.sbd import load_sbd
from interpretable_corenlp.tagger import load_tagger
from interpretable_corenlp.posfeats import TAG2I


def read_conllu(path):
    sents = []; toks = []
    for line in open(path):
        line = line.rstrip("\n")
        if not line or line.startswith("#"):
            if toks:
                sents.append(toks); toks = []
            continue
        c = line.split("\t")
        if "-" in c[0] or "." in c[0]:
            continue
        toks.append(c[1])
    if toks:
        sents.append(toks)
    return sents


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sbd", required=True)
    ap.add_argument("--engram", default="")
    ap.add_argument("--pos", required=True)
    ap.add_argument("--w", default="data/W_11M_K512_signed")
    ap.add_argument("--conllu", default="data/en_ewt-ud-test.conllu")
    ap.add_argument("--group", type=int, default=4, help="real sentences concatenated per document")
    a = ap.parse_args()

    sents = read_conllu(a.conllu)
    tagger = load_tagger(a.pos, w_dir=a.w)
    sbd = load_sbd(a.sbd, engram_npz=(a.engram or None))
    TP = FP = FN = 0; sent_total = sent_exact = 0
    for start in range(0, len(sents), a.group):
        chunk = sents[start:start + a.group]
        toks = []; gold_bnd = []; gold_spans = []; off = 0
        for s in chunk:
            for j, w in enumerate(s):
                toks.append(w); gold_bnd.append(1 if j == len(s) - 1 else 0)
            gold_spans.append((off, off + len(s))); off += len(s)
        if len(toks) < 2:
            continue
        pos_ids = [TAG2I[t] for t in tagger.tag(toks)]
        spans = sbd.segment(toks, pos_ids)
        pred_bnd = [0] * len(toks)
        for (x, y) in spans:
            if 0 <= y - 1 < len(toks):
                pred_bnd[y - 1] = 1
        for i in range(len(toks) - 1):                         # exclude forced last-token boundary
            if pred_bnd[i] and gold_bnd[i]:
                TP += 1
            elif pred_bnd[i] and not gold_bnd[i]:
                FP += 1
            elif not pred_bnd[i] and gold_bnd[i]:
                FN += 1
        pset = set(spans)
        for sp in gold_spans:
            sent_total += 1; sent_exact += int(sp in pset)
    P = TP / (TP + FP + 1e-9); R = TP / (TP + FN + 1e-9); F1 = 2 * P * R / (P + R + 1e-9)
    print(f"[eval-sbd] {len(sents)} real sentences, group={a.group}")
    print(f"[eval-sbd] boundary  P={P:.3f}  R={R:.3f}  F1={F1:.3f}")
    print(f"[eval-sbd] SENTENCE exact-match = {sent_exact/max(sent_total,1):.3f}  ({sent_exact}/{sent_total})")


if __name__ == "__main__":
    main()
