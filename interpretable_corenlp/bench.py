"""Speed + accuracy benchmark of our glass-box POS tagger and SBD vs baseline models.
  POS  : ours vs spaCy en_core_web_sm (CNN) vs en_core_web_trf (RoBERTa transformer)  on UD-EWT (UPOS, tok/s).
  SBD  : ours vs NLTK punkt vs spaCy sentencizer  on GUM/EWT (sentence exact-match, boundary F1, docs|chars/s).

  uv run python -m bq_embedding.pos.bench_compare --what both --conllu data/ud/en_gum-train.conllu
"""
from __future__ import annotations
import argparse, time
import numpy as np
from interpretable_corenlp.eval_sbd import read_conllu


def _reconstruct(tokens):
    """raw text = space-joined tokens; return (raw, tok_end_char[]) so a char offset maps back to a token."""
    raw = ""; ends = []
    for i, w in enumerate(tokens):
        if i:
            raw += " "
        raw += w; ends.append(len(raw))
    return raw, ends


def _char_bounds_to_tokens(sent_char_ends, tok_end):
    """predicted boundary tokens from a segmenter's sentence END char offsets."""
    pred = set()
    j = 0
    for ce in sent_char_ends:
        while j < len(tok_end) and tok_end[j] < ce:
            j += 1
        # boundary is after the last token fully inside this sentence
        bt = j if (j < len(tok_end) and tok_end[j] <= ce) else j - 1
        if 0 <= bt < len(tok_end):
            pred.add(bt)
    return pred


def bench_sbd(conllu, sbd_path, engram, pos_model, w, group=4):
    from interpretable_corenlp.sbd import load_sbd
    from interpretable_corenlp.tagger import load_tagger
    from interpretable_corenlp.posfeats import TAG2I
    import nltk, spacy
    sents = read_conllu(conllu)
    tagger = load_tagger(pos_model, w_dir=w)
    sbd = load_sbd(sbd_path, engram_npz=(engram or None))
    spc = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer", "tagger", "attribute_ruler"])
    docs = []                                                  # (tokens, raw, tok_end, gold_boundaries, gold_spans)
    for st in range(0, len(sents), group):
        toks = []; gb = set(); gs = []; off = 0
        for s in sents[st:st + group]:
            for j in range(len(s)):
                toks.append(s[j])
            gb.add(off + len(s) - 1); gs.append((off, off + len(s))); off += len(s)
        if len(toks) < 2:
            continue
        raw, ends = _reconstruct(toks)
        docs.append((toks, raw, ends, gb, gs))
    tot_chars = sum(len(d[1]) for d in docs); ndoc = len(docs)
    results = {}

    def run(predict_boundaries, name):
        TP = FP = FN = 0; em = emtot = 0
        t0 = time.time()
        for toks, raw, ends, gb, gs in docs:
            pb = predict_boundaries(toks, raw, ends)
            pb = set(pb); pb.add(len(toks) - 1)                # last token always a boundary
            for i in range(len(toks) - 1):
                if i in pb and i in gb:
                    TP += 1
                elif i in pb:
                    FP += 1
                elif i in gb:
                    FN += 1
            # sentence exact-match from predicted boundary set
            starts = [0] + [i + 1 for i in sorted(pb)]
            pspans = set()
            s = 0
            for i in sorted(pb):
                pspans.add((s, i + 1)); s = i + 1
            for sp in gs:
                emtot += 1; em += int(sp in pspans)
        dt = time.time() - t0
        P = TP / (TP + FP + 1e-9); R = TP / (TP + FN + 1e-9); F1 = 2 * P * R / (P + R + 1e-9)
        results[name] = (em / emtot, F1, P, R, tot_chars / dt, ndoc / dt)

    # ours
    def ours(toks, raw, ends):
        pos_ids = [TAG2I[t] for t in tagger.tag(toks)]
        spans = sbd.segment(toks, pos_ids)
        return {b - 1 for (a, b) in spans if b - 1 < len(toks)}
    run(ours, "ours (glass-box SBD)")

    # NLTK punkt
    def punkt(toks, raw, ends):
        spans = list(nltk.tokenize.PunktSentenceTokenizer().span_tokenize(raw))
        return _char_bounds_to_tokens([hi for _, hi in spans], ends)
    run(punkt, "NLTK punkt")

    # spaCy sentencizer (rule-based, fast)
    spc.add_pipe("sentencizer") if "sentencizer" not in spc.pipe_names else None
    def spacy_sent(toks, raw, ends):
        d = spc(raw); return _char_bounds_to_tokens([s.end_char for s in d.sents], ends)
    run(spacy_sent, "spaCy sentencizer")

    print(f"\n=== SBD: ours vs baselines on {conllu.split('/')[-1]} ({ndoc} docs, {tot_chars:,} chars) ===")
    print(f"{'model':24s} {'sent-EM':>8s} {'bnd-F1':>7s} {'P':>6s} {'R':>6s} {'chars/s':>10s} {'docs/s':>8s} {'speedup':>8s}")
    base = results["ours (glass-box SBD)"][4]
    for name, (em, f1, p, r, cps, dps) in results.items():
        print(f"{name:24s} {em:8.3f} {f1:7.3f} {p:6.3f} {r:6.3f} {cps:10.0f} {dps:8.1f} {cps/base:8.2f}x")


def bench_pos(conllu, pos_model, w, max_sents=2077):
    from interpretable_corenlp.tagger import load_tagger
    import spacy
    sents = []; toks = []; gold = []
    for line in open(conllu):
        line = line.rstrip("\n")
        if not line or line.startswith("#"):
            if toks:
                sents.append((toks, gold)); toks = []; gold = []
            continue
        c = line.split("\t")
        if "-" in c[0] or "." in c[0]:
            continue
        toks.append(c[1]); gold.append(c[3])
    if toks:
        sents.append((toks, gold))
    sents = sents[:max_sents]; ntok = sum(len(t) for t, _ in sents)
    rows = {}
    tg = load_tagger(pos_model, w_dir=w)
    t0 = time.time(); cor = 0
    for tk, gd in sents:
        for p, g in zip(tg.tag(tk), gd):
            cor += p == g
    dt = time.time() - t0; rows["ours (glass-box CRF)"] = (cor / ntok, ntok / dt)
    for model in ["en_core_web_sm", "en_core_web_trf"]:
        try:
            nlp = spacy.load(model, disable=["ner", "lemmatizer", "parser"])
        except Exception:
            continue
        t0 = time.time(); cor = 0
        for tk, gd in sents:
            doc = spacy.tokens.Doc(nlp.vocab, words=tk)
            for name, proc in nlp.pipeline:
                doc = proc(doc)
            for tkn, g in zip(doc, gd):
                cor += tkn.pos_ == g
        dt = time.time() - t0; rows[f"spaCy {model}"] = (cor / ntok, ntok / dt)
    print(f"\n=== POS: ours vs baselines on {conllu.split('/')[-1]} ({len(sents)} sents, {ntok:,} tok) ===")
    base = rows["ours (glass-box CRF)"][1]
    print(f"{'model':26s} {'UPOS':>7s} {'tok/s':>9s} {'speedup':>8s}")
    for name, (acc, tps) in rows.items():
        print(f"{name:26s} {acc:7.4f} {tps:9.0f} {tps/base:8.2f}x")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", choices=["pos", "sbd", "both"], default="both")
    ap.add_argument("--pos", default="/home/josh/workspace/NLP/pos-robust-ctxwin-20260617-1355/model.npz")
    ap.add_argument("--sbd", default="/home/josh/workspace/NLP/sbd-corenlp-v4-20260617-1635/sbd.npz")
    ap.add_argument("--engram", default="/home/josh/workspace/NLP/sbd-corenlp-v4-20260617-1635/engram.npz")
    ap.add_argument("--w", default="data/W_11M_K512_signed")
    ap.add_argument("--conllu", default="data/en_ewt-ud-test.conllu")
    a = ap.parse_args()
    if a.what in ("pos", "both"):
        bench_pos(a.conllu, a.pos, a.w)
    if a.what in ("sbd", "both"):
        bench_sbd(a.conllu, a.sbd, a.engram, a.pos, a.w)


if __name__ == "__main__":
    main()
