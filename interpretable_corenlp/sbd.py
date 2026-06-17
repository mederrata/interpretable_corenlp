"""Glass-box SENTENCE BOUNDARY DETECTOR (SBD) — the segmentation primitive that the POS tagger co-trains with.

WHY it exists: the POS tagger assumes the tokens it is handed are ONE sentence (its ±W context window and
position/BOS features all depend on that). On running text — especially uncased / unpunctuated / WORD-WRAPPED
text — we cannot trust punctuation+capitalization to segment. So we learn a boundary detector from MANY weak
cues, the strongest of which is SYNTACTIC (POS): a sentence boundary is likely after a clause-final finite VERB
followed by a capitalized PRONOUN/DETERMINER, and unlikely between DET->NOUN. That POS dependency is the
coupling: SBD takes the tagger's tags as features; the tagger takes SBD's segmentation to bound its context.
They optimize iteratively together (see cotrain.py), exactly like the POS<->W bootstrap.

Model (per inter-token gap i, "does a sentence end AFTER token i?"):
  logit(i) = Θ_lattice(κ_i)·[1]  +  w_fine · fine_i  +  b
    κ = (endpunct_i × nextcap × POS_i × POS_{i+1} × newline)   -> the cell IS the boundary log-odds, readable
    fine = lexical enders/starters (word_i, word_{i+1}), length-since-last-boundary bucket
  p(boundary) = σ(logit).  Independent per gap (boundaries are local); class-imbalance-weighted BCE.
Glass-box: every gap decomposes EXACTLY into named additive contributions (Sbd.explain()).

ROBUSTNESS: a newline (word-wrap) is NOT a boundary by itself — its lattice weight is FITTED and gated by
nextcap/endpunct, so a mid-sentence wrap (lowercase continuation, no end punct) scores LOW. Augmentation
(cotrain.build_documents) injects word-wrap newlines, lowercases, and de-punctuates so these cues are learned.
"""
from __future__ import annotations
import json
import numpy as np

# (training-only projection import removed for inference package)
from interpretable_corenlp.posfeats import UD_TAGS, NT, shape6

# ---- named categorical axes (low-card -> lattice) -------------------------------------------------------
SENT_FINAL = {".", "!", "?", "…", "．", "。", "！", "？"}
COMMA_CLASS = {",", ";", ":", "-"}
EM_DASH = {"—", "–", "--", "---", "––"}                          # em/en dash — often a sentence join, distinct from comma
CLOSE_BRK = {'"', "'", "”", "’", ")", "]", "}", "»"}
# CoreNLP-style ABBREVIATION list: a period after these is usually NOT a sentence boundary (period ambiguity,
# the central SBD hard case). Titles, business, latin, months, units, etc. — a fitted feature, not a hard rule.
ABBREV = {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "rev", "hon", "gen", "col", "capt", "lt", "sgt",
          "gov", "sen", "rep", "pres", "supt", "inc", "ltd", "co", "corp", "llc", "plc", "dept", "univ",
          "assn", "bros", "etc", "vs", "viz", "al", "no", "vol", "pp", "fig", "eq", "ref", "approx", "est",
          "e.g", "i.e", "a.m", "p.m", "u.s", "u.k", "u.n", "ph.d", "m.d", "b.a", "m.a", "jan", "feb", "mar",
          "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec", "mon", "tue", "wed", "thu", "fri",
          "sat", "sun", "ave", "blvd", "rd", "mt", "ft", "kg", "km", "cm", "mm", "lb", "oz", "hr", "min", "sec"}
N_ENDPUNCT = 7    # 0 none, 1 sentence-final terminal, 2 comma-class, 3 close-quote/bracket, 4 ABBREV-period,
                  # 5 number-internal period (decimal/version: digit . digit), 6 EM-DASH (sentence join vs clause)
N_NEXTCAP = 5     # 0 lower, 1 init-cap (Xxx), 2 ALL-CAPS (XXX, headline/acronym — weak), 3 digit/other, 4 none
N_NL = 3          # 0 none, 1 single newline (word-wrap, weak), 2 double newline (paragraph, strong) — CoreNLP "two"
NT1 = NT + 1      # POS tag of next token, with an extra "none" state for end-of-stream
# common sentence-STARTER words (weak cue, fitted) — used as a fine lexical feature, not a hard rule
STARTERS = {"the", "a", "an", "he", "she", "it", "they", "we", "i", "you", "this", "that", "these",
            "those", "in", "on", "at", "but", "and", "however", "meanwhile", "then", "so", "after",
            "when", "while", "if", "there", "his", "her", "their", "our", "my"}


_PRED_POS = {i for i, t in enumerate(UD_TAGS) if t in ("VERB", "AUX")}   # finite-predicate POS (clause-completeness)


# Punkt-LLR abbreviations learned UNSUPERVISED from corpora, SPLIT BY DOMAIN. General ones are always active;
# domain-specific (medical) ones fire ONLY when the input is detected to be in that domain (the in-domain gate) —
# else `i.v`/`p.o`/`b.i.d` would mis-suppress boundaries on general text.
GENERAL_LEARNED_ABBREV = set()
MEDICAL_LEARNED_ABBREV = set()
LEARNED_ABBREV = set()   # the ACTIVE set for the current document (set per-call by domain gating)

# glass-box MEDICAL-DOMAIN detector: medical morphology (suffixes) + a small seed lexicon -> term density
_MED_SUFFIX = ("itis", "osis", "emia", "aemia", "pathy", "ectomy", "otomy", "plasty", "scopy", "megaly",
               "penia", "cytosis", "oma", "ostomy", "gram", "rrhea", "rrhoea", "plegia")
_MED_SEED = {"patient", "patients", "dose", "dosage", "mg", "ml", "clinical", "treatment", "therapy", "therapeutic",
             "diagnosis", "diagnostic", "symptoms", "disease", "blood", "cells", "protein", "syndrome", "infection",
             "tumor", "tumour", "plasma", "serum", "renal", "hepatic", "cardiac", "chronic", "acute", "lesion",
             "carcinoma", "malignant", "benign", "intravenous", "oral", "administered", "mortality", "morbidity"}


def medical_domain(tokens, frac=0.035):
    """Is this text in the medical domain? Fraction of tokens matching medical morphology/seed terms >= frac."""
    if not tokens:
        return False
    hits = sum(1 for t in tokens if t.lower() in _MED_SEED or t.lower().endswith(_MED_SUFFIX))
    return hits / len(tokens) >= frac


def _activate_domain(tokens):
    """Set LEARNED_ABBREV = general (+ medical IFF medical domain detected) for the current document."""
    LEARNED_ABBREV.clear(); LEARNED_ABBREV.update(GENERAL_LEARNED_ABBREV)
    if MEDICAL_LEARNED_ABBREV and medical_domain(tokens):
        LEARNED_ABBREV.update(MEDICAL_LEARNED_ABBREV)
    return LEARNED_ABBREV


def is_abbrev(w: str) -> bool:
    """CoreNLP-style: token is a known abbreviation or an initial (single capital letter), with/without period.
    Consults the hand-curated ABBREV list + any LEARNED_ABBREV (Punkt-style, domain-adaptive — clinical/legal)."""
    if not w:
        return False
    s = w.lower().rstrip(".")
    if s in ABBREV or s in LEARNED_ABBREV:
        return True
    return len(w.rstrip(".")) == 1 and w[0].isupper()        # initial, e.g. "J." in "J. Smith"


def learn_abbreviations(corpus_jsonl, max_docs=0, strict=True, text_key="text", min_freq=30):
    """Punkt-style UNSUPERVISED abbreviation detection: train NLTK's PunktTrainer (the actual abbreviation
    log-likelihood-ratio of Kiss & Strunk 2006) on a corpus and return the learned abbreviation types. Makes
    the SBD domain-adaptive (auto-discovers clinical `i.v`/`p.o`, legal `et seq.` etc.) instead of hand-listing.
    Route the result into LEARNED_ABBREV (the trained endpunct=4 abbrev class then suppresses those periods at
    inference, no retrain needed).

    strict=True applies a STRICT filter — Punkt's raw output is dominated by web/biomedical noise (numbers,
    units `125cm`, versions `2.1v`, file-ext `.gpg`, gene symbols). Keep only ALPHABETIC types that are either
    DOTTED (`i.v`, `u.s.a` — almost always real) or very short (≤3, `dr`/`vs`/`st`). Removes the digit/unit noise."""
    import json
    from nltk.tokenize.punkt import PunktTrainer
    tr = PunktTrainer(); n = 0
    for line in open(corpus_jsonl):
        try:
            txt = json.loads(line).get(text_key, "")
        except Exception:
            continue
        if txt:
            tr.train(txt, finalize=False); n += 1
        if max_docs and n >= max_docs:
            break
    tr.finalize_training()
    return _filter_abbrev(set(tr._params.abbrev_types), tr._type_fdist, strict, min_freq)


def _filter_abbrev(ab, fdist, strict=True, min_freq=30):
    """Strict alphabetic + FREQUENCY filter. Genuine abbreviations (i.v, e.g, dr) recur often; spurious
    dotted-acronym/web noise (b.m.w, a.b) is rare. Frequency is the key discriminator Punkt's type-string isn't."""
    if not strict:
        return ab
    return {a for a in ab
            if a.replace(".", "").isalpha() and (("." in a) or len(a) <= 3)
            and 2 <= len(a.replace(".", "")) <= 6 and fdist[a + "."] >= min_freq}


def nextcap_class(w: str) -> int:
    if not w:
        return 4                                                # end of stream
    if w[0].isdigit():
        return 3
    if not w[0].isupper():
        return 0                                                # lowercase
    if w.isupper() and len(w) > 1:
        return 2                                                # ALL-CAPS (headline/acronym) — weak boundary cue
    return 1                                                    # init-cap Xxx — sentence-start cue


def endpunct_class(w: str, prev: str = "", gap=None, wnext: str = "") -> int:
    """End-punctuation class at the boundary AFTER token w.
    Two modes:
      - gap is None (raw-text tokenization): w itself may BE the punctuation token (`.`, `"`, ...).
      - gap is a string (IE side-channel): w is a content token and `gap` is the inter-token punctuation that
        sat between w and the next token (e.g. `.`, `,`, `.\"`). This is the alphanumeric-stream regime.
    Returns: 0 none, 1 sentence-final terminal, 2 comma-class, 3 close-quote/bracket follower, 4 ABBREV-period."""
    if gap is not None:                                          # IE side-channel: classify the gap punctuation
        g = gap.strip()
        if not g:
            return 0
        if "." in g and prev[-1:].isdigit() and wnext[:1].isdigit():
            return 5                                             # decimal/version — NEVER a boundary
        if any(c in SENT_FINAL for c in g):
            return 4 if is_abbrev(w) else 1                      # "Dr" + "." -> abbrev-period; else terminal
        if g in EM_DASH or any(c in "—–" for c in g):
            return 6                                             # em/en dash join
        if any(c in COMMA_CLASS for c in g):
            return 2
        if any(c in CLOSE_BRK for c in g):
            return 3
        return 0
    if not w:
        return 0
    if w in SENT_FINAL and prev[-1:].isdigit() and wnext[:1].isdigit():
        return 5                                                 # decimal/version period (74.5, 101.1)
    if w in EM_DASH:
        return 6                                                 # em/en dash join
    # period that belongs to an ABBREVIATION (the prev token, or w itself like "Inc.") -> class 4, not terminal
    if w in SENT_FINAL and is_abbrev(prev):
        return 4
    if w.endswith(".") and is_abbrev(w):
        return 4
    if w in SENT_FINAL or (len(w) > 1 and all(c in SENT_FINAL for c in w)):   # "..", "?!", "..."
        return 1
    if w in COMMA_CLASS:
        return 2
    if w in CLOSE_BRK or (w and w[-1] in CLOSE_BRK):                          # boundary-follower (quote/paren)
        return 3
    return 0


class SbdFeaturizer:
    """Builds (lattice cell κ, fine-feature indices) for each inter-token gap. POS tags come from the tagger
    (the coupling): pass per-token tag ids (0..NT-1)."""
    CARDS = [N_ENDPUNCT, N_NEXTCAP, NT, NT1, N_NL]
    PAIRWISE = [(2, 3), (0, 1), (4, 1), (4, 0)]   # POS_i×POS_{i+1}, punct×nextcap, nl×nextcap, nl×endpunct

    def __init__(self, ender_vocab, starter_vocab, n_lenbin=6):
        self.ender_vocab = ender_vocab            # word_i identity (lowercased) -> idx (frequent clause-enders)
        self.starter_vocab = starter_vocab        # word_{i+1} identity -> idx (frequent sentence-starters)
        self.n_lenbin = n_lenbin
        self.b_ender = 0
        self.b_starter = self.b_ender + len(ender_vocab)
        self.b_isstart = self.b_starter + len(starter_vocab)   # is next-word a known STARTER (1 feat)
        self.b_abbrev = self.b_isstart + 1                     # tok_i is an abbreviation (CoreNLP cue)
        self.b_follow = self.b_abbrev + 1                      # boundary-follower after a terminator (." .) )
        self.b_catalog = self.b_follow + 1                    # catalog-tail: long no-punct run + next Title/starter
        self.b_nopred = self.b_catalog + 1                    # CLAUSE-INCOMPLETE: no finite verb since last boundary
        self.b_len = self.b_nopred + 1                        # length-since-last-boundary bin
        self.nf = self.b_len + n_lenbin

    def _nlval(self, newline):
        return int(newline) if isinstance(newline, (int, np.integer)) else (1 if newline else 0)

    def kappa(self, wi, wnext, pos_i, pos_next, newline, prev="", gap=None):
        return np.array([endpunct_class(wi, prev, gap, wnext), nextcap_class(wnext), pos_i,
                         pos_next if pos_next is not None else NT, self._nlval(newline)], np.int16)

    def _lenbin(self, dist):
        if dist <= 2:
            return 0
        if dist <= 5:
            return 1
        if dist <= 10:
            return 2
        if dist <= 20:
            return 3
        if dist <= 40:
            return 4
        return 5

    def _follows_terminator(self, wi, prev, gap):             # single predicate shared by indices & named (exact)
        return gap is None and endpunct_class(wi, prev) == 3 and prev[-1:] in SENT_FINAL

    def _is_catalog_tail(self, wi, wnext, dist, prev, gap):   # long no-punct run + next Title/starter (FN bucket)
        if endpunct_class(wi, prev, gap) != 0 or dist < 12 or not wnext:
            return False
        return wnext[:1].isupper() or wnext.lower() in STARTERS

    def fine_indices(self, wi, wnext, dist, prev="", gap=None, has_pred=True):
        idx = []
        if not has_pred:                                      # no finite verb since last boundary -> clause incomplete
            idx.append(self.b_nopred)
        j = self.ender_vocab.get(wi.lower())
        if j is not None:
            idx.append(self.b_ender + j)
        jn = self.starter_vocab.get(wnext.lower()) if wnext else None
        if jn is not None:
            idx.append(self.b_starter + jn)
        if wnext and wnext.lower() in STARTERS:
            idx.append(self.b_isstart)
        if is_abbrev(wi):                                      # CoreNLP: abbreviation -> not a boundary
            idx.append(self.b_abbrev)
        if self._follows_terminator(wi, prev, gap):           # boundary-follower (quote/paren) after a terminator
            idx.append(self.b_follow)
        if self._is_catalog_tail(wi, wnext, dist, prev, gap):
            idx.append(self.b_catalog)
        idx.append(self.b_len + self._lenbin(dist))
        return idx

    def fine_named(self, wi, wnext, dist, prev="", gap=None, has_pred=True):
        out = []
        if not has_pred:
            out.append(("clause_no_predicate", self.b_nopred))
        j = self.ender_vocab.get(wi.lower())
        if j is not None:
            out.append((f"ender={wi.lower()}", self.b_ender + j))
        jn = self.starter_vocab.get(wnext.lower()) if wnext else None
        if jn is not None:
            out.append((f"starter={wnext.lower()}", self.b_starter + jn))
        if wnext and wnext.lower() in STARTERS:
            out.append(("next_is_starter", self.b_isstart))
        if is_abbrev(wi):
            out.append(("is_abbrev", self.b_abbrev))
        if self._follows_terminator(wi, prev, gap):           # SAME predicate as fine_indices -> explain stays exact
            out.append(("follower_after_terminator", self.b_follow))
        if self._is_catalog_tail(wi, wnext, dist, prev, gap):
            out.append(("catalog_tail", self.b_catalog))
        out.append((f"dist_bin={self._lenbin(dist)}", self.b_len + self._lenbin(dist)))
        return out


class Sbd:
    """Loaded glass-box SBD. predict(tokens, pos, newlines) -> per-gap boundary probability; segment() ->
    list of sentence (start,end) spans; explain() -> exact named additive attribution per gap."""
    AXN = ["endpunct", "nextcap", "POS_i", "POS_next", "newline"]

    def __init__(self, z, fx: SbdFeaturizer, memory=None):
        self.fx = fx
        self.mem = memory                                       # optional glass-box ENGRAM (lexical-tail memory)
        self.g = np.asarray(z["lat_global"], np.float32)        # (1,1)
        self.cards = json.loads(str(z["lat_cards"]))
        self.main = [np.asarray(z[f"lat_main{i}"], np.float32) for i in range(len(self.cards))]
        pw = json.loads(str(z["lat_pairwise"]))
        self.inter = {tuple(ab): np.asarray(z[f"lat_inter_{ab[0]}_{ab[1]}"], np.float32) for ab in pw}
        self.wf = np.asarray(z["wf"], np.float32)               # (nf,)
        self.b = float(z["b"])
        self.thresh = float(z["thresh"]) if "thresh" in z else 0.5
        # PER-CELL thresholds over (endpunct,nextcap,POS_i,POS_next); falls back to global where unsupported
        self.thresh_cell = np.asarray(z["thresh_cell"], np.float32) if "thresh_cell" in z else None

    def _logit_one(self, kap, fidx):
        v = np.ones(1, np.float32)
        th = self.g[:, 0] + sum(self.main[a][kap[a]][:, 0] for a in range(len(kap)))
        for (a, b), I in self.inter.items():
            th = th + I[kap[a], kap[b]][:, 0]
        s = float(th @ v) + self.b
        for j in fidx:
            s += float(self.wf[j])
        return s

    def _cell_thresh(self, kap):
        if self.thresh_cell is None:
            return self.thresh
        return float(self.thresh_cell[kap[0], kap[1], kap[2], kap[3]])

    def _scores(self, tokens, pos, newlines=None, gap_punct=None):
        """Per-gap (boundary probability, per-cell threshold). The single source of truth for predict/segment."""
        L = len(tokens); nl = [0] * L if newlines is None else newlines
        _activate_domain(tokens)                              # in-domain gate (no-op if no medical abbrevs loaded)
        probs = np.zeros(L, np.float32); thr = np.full(L, self.thresh, np.float32); dist = 1; has_pred = False
        for i in range(L):
            if pos[i] in _PRED_POS:                             # predicate seen since the last boundary
                has_pred = True
            wnext = tokens[i + 1] if i + 1 < L else ""
            prev = tokens[i - 1] if i > 0 else ""
            pnext = pos[i + 1] if i + 1 < L else None
            gp = None if gap_punct is None else (gap_punct[i] if i < len(gap_punct) else "")
            kap = self.fx.kappa(tokens[i], wnext, pos[i], pnext, nl[i] if i < len(nl) else 0, prev, gp)
            s = self._logit_one(kap, self.fx.fine_indices(tokens[i], wnext, dist, prev, gp, has_pred))
            if self.mem is not None:                            # + ENGRAM lexical-tail memory delta
                s += self.mem.delta(tokens, pos, i)
            p = 1.0 / (1.0 + np.exp(-s)); t = self._cell_thresh(kap)
            probs[i] = 1.0 if i == L - 1 else p; thr[i] = t
            if p >= t:                                          # per-cell threshold drives length + clause reset
                dist = 1; has_pred = False
            else:
                dist += 1
        return probs, thr

    def predict(self, tokens, pos, newlines=None, gap_punct=None):
        """Boundary probability AFTER each token (len == len(tokens); last is forced ~1).
        gap_punct[i] = the inter-token punctuation string after token i (IE alphanumeric-stream side-channel);
        None = raw-text mode (punctuation is its own token)."""
        return self._scores(tokens, pos, newlines, gap_punct)[0]

    def segment(self, tokens, pos, newlines=None, gap_punct=None):
        """Return sentence spans [(start, end), ...] (half-open) using PER-CELL thresholds."""
        p, thr = self._scores(tokens, pos, newlines, gap_punct); spans = []; start = 0
        for i in range(len(tokens)):
            if p[i] >= thr[i] or i == len(tokens) - 1:
                spans.append((start, i + 1)); start = i + 1
        return spans

    def segment_text(self, raw_text, tagger):
        """RAW-TEXT entry point (IE agent's preferred path b): tokenize raw text KEEPING punctuation as tokens
        (full-punctuation regime), POS-tag it, detect newlines from the source, segment, and return SENTENCE
        spans as CHARACTER offsets [(char_lo, char_hi), ...]. The caller maps these back to their own token
        stream by offset — so the SBD never needs to match the caller's tokenizer. `tagger` = a loaded POS Tagger."""
        import re
        from interpretable_corenlp.posfeats import TAG2I
        rtok = re.compile(r"[A-Za-z0-9]+(?:['\-][A-Za-z0-9]+)*|[^\w\s]")
        toks, cspan = [], []
        for m in rtok.finditer(raw_text):
            toks.append(m.group()); cspan.append((m.start(), m.end()))
        if not toks:
            return []
        _activate_domain(toks)                                # in-domain gate: medical abbrevs only on medical text
        # DECIMAL-MERGE: collapse `digit . digit` / `digit , digit` runs (74.5, 101.1, 69,966) into ONE token, so
        # the internal period/comma is never even a boundary candidate (codex: raw tokenization bypassed endpunct=5).
        i = 0; mt, mc = [], []
        while i < len(toks):
            j = i
            while j + 2 < len(toks) and toks[j + 1] in (".", ",") and toks[j][-1:].isdigit() and toks[j + 2][:1].isdigit():
                j += 2
            mt.append("".join(toks[i:j + 1])); mc.append((cspan[i][0], cspan[j][1])); i = j + 1
        toks, cspan = mt, mc
        # ABBREVIATION-MERGE: collapse dotted-abbreviation runs (`i . v .` -> `i.v.`, `b . i . d .`) that match a
        # known/learned abbreviation, so the trained endpunct=4 abbrev cell suppresses them (our tokenizer otherwise
        # splits on every period). Only merges when the joined form is a KNOWN abbreviation -> never breaks a real
        # boundary like "I. He".
        i = 0; mt, mc = [], []
        while i < len(toks):
            merged = False
            if toks[i].isalpha() and len(toks[i]) <= 2:
                j = i; parts = [toks[i]]
                # extend only across ADJACENT tokens (no whitespace) so "p.o. b.i.d" doesn't merge into "p.o.b"
                while (j + 2 < len(toks) and toks[j + 1] == "." and toks[j + 2].isalpha() and len(toks[j + 2]) <= 2
                       and cspan[j + 1][1] == cspan[j + 2][0] and cspan[j][1] == cspan[j + 1][0]):
                    parts.append(toks[j + 2]); j += 2
                cand = ".".join(p.lower() for p in parts)
                if j > i and (cand in ABBREV or cand in LEARNED_ABBREV):
                    end = j
                    joined = "".join(toks[i:j + 1])
                    if j + 1 < len(toks) and toks[j + 1] == ".":
                        joined += "."; end = j + 1
                    mt.append(joined); mc.append((cspan[i][0], cspan[end][1])); i = end + 1; merged = True
            if not merged:
                mt.append(toks[i]); mc.append(cspan[i]); i += 1
        toks, cspan = mt, mc
        pos_ids = [TAG2I[t] for t in tagger.tag(toks)]
        nl = [0] * len(toks)                                   # newlines from the source: 1 single, 2 blank-line
        for i in range(len(toks) - 1):
            between = raw_text[cspan[i][1]:cspan[i + 1][0]]
            nl[i] = 2 if between.count("\n") >= 2 else (1 if "\n" in between else 0)
        out = []
        for (a, b) in self.segment(toks, pos_ids, newlines=nl):   # raw token-mode (punctuation IS a token)
            out.append((cspan[a][0], cspan[b - 1][1]))
        return out

    def explain(self, tokens, pos, newlines=None, gap_punct=None, topk=8):
        """EXACT additive attribution for each gap's boundary log-odds — the glass-box guarantee."""
        L = len(tokens); nl = [0] * L if newlines is None else newlines; out = []; dist = 1; has_pred = False
        for i in range(L):
            if pos[i] in _PRED_POS:
                has_pred = True
            wnext = tokens[i + 1] if i + 1 < L else ""
            prev = tokens[i - 1] if i > 0 else ""
            pnext = pos[i + 1] if i + 1 < L else None
            gp = None if gap_punct is None else (gap_punct[i] if i < len(gap_punct) else "")
            kap = self.fx.kappa(tokens[i], wnext, pos[i], pnext, nl[i] if i < len(nl) else 0, prev, gp)
            cc = [("global", float(self.g[0, 0]))]
            for a in range(len(kap)):
                nm = f"{self.AXN[a]}={UD_TAGS[kap[a]] if a in (2,3) and kap[a] < NT else kap[a]}"
                cc.append((nm, float(self.main[a][kap[a]][0, 0])))
            for (a, b), I in self.inter.items():
                cc.append((f"{self.AXN[a]}×{self.AXN[b]}", float(I[kap[a], kap[b]][0, 0])))
            for nm, j in self.fx.fine_named(tokens[i], wnext, dist, prev, gp, has_pred):
                cc.append((nm, float(self.wf[j])))
            if self.mem is not None:                            # + ENGRAM lexical-tail memory (named patterns)
                cc.extend(self.mem.named(tokens, pos, i))
            cc.append(("bias", self.b))
            s = sum(v for _, v in cc); p = 1.0 / (1.0 + np.exp(-s))
            cc.sort(key=lambda x: -abs(x[1]))
            if topk and len(cc) > topk:
                resid = sum(v for _, v in cc[topk:])
                cc = cc[:topk] + [(f"(+{len(cc)-topk} smaller)", resid)]
            out.append((tokens[i], round(float(p), 3), cc))
            if p >= self._cell_thresh(kap):
                dist = 1; has_pred = False
            else:
                dist += 1
        return out


class SbdMemory:
    """Loaded glass-box SBD ENGRAM: per-axis {pattern->pid} + per-axis delta vectors. Inference cost is
    O(#axes) dict lookups per gap (additive, no matmul). Each entry = a NAMED lexical pattern -> boundary delta."""
    def __init__(self, z):
        from interpretable_corenlp.sbd_engram import axis_patterns
        self._patterns = axis_patterns
        self.AXES = json.loads(str(z["axes"]))
        keep = json.loads(str(z["keep"]))                       # {axis: {pattern: pid}}
        self.pat2pid = {ax: keep.get(ax, {}) for ax in self.AXES}
        self.E = {ax: np.asarray(z[f"E_{ax}"], np.float32) for ax in self.AXES}

    def delta(self, tokens, pos, i):
        d = 0.0
        pats = self._patterns(tokens, i, pos)
        for ax in self.AXES:
            pid = self.pat2pid[ax].get(pats[ax])
            if pid:                                             # pid 0/None -> no memory
                d += float(self.E[ax][pid])
        return d

    def named(self, tokens, pos, i):
        """The firing memory patterns (name, delta) at gap i — for explain()."""
        out = []; pats = self._patterns(tokens, i, pos)
        for ax in self.AXES:
            pid = self.pat2pid[ax].get(pats[ax])
            if pid:
                out.append((f"mem:{ax}={pats[ax]}", float(self.E[ax][pid])))
        return out


def load_sbd(npz_path, engram_npz=None, abbrev_json=None):
    z = np.load(npz_path, allow_pickle=False)
    fx = SbdFeaturizer(json.loads(str(z["ender_vocab"])), json.loads(str(z["starter_vocab"])),
                       n_lenbin=int(z["n_lenbin"]))
    def _ingest(zz):                                           # populate the domain-split learned-abbrev sets
        if "learned_abbrev_general" in zz.files:
            GENERAL_LEARNED_ABBREV.update(json.loads(str(zz["learned_abbrev_general"])))
        if "learned_abbrev_medical" in zz.files:
            MEDICAL_LEARNED_ABBREV.update(json.loads(str(zz["learned_abbrev_medical"])))
        if "learned_abbrev" in zz.files:                       # legacy single set -> treat as always-on general
            GENERAL_LEARNED_ABBREV.update(json.loads(str(zz["learned_abbrev"])))
    _ingest(z)
    if abbrev_json:
        GENERAL_LEARNED_ABBREV.update(json.load(open(abbrev_json)))
    mem = None
    if engram_npz:
        ez = np.load(engram_npz, allow_pickle=False)
        _ingest(ez)                                            # domain abbreviations shipped IN the engram (lexical mem)
        mem = SbdMemory(ez)
    return Sbd(z, fx, memory=mem)
