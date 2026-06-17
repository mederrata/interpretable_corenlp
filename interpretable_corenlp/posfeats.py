"""Shared glass-box POS featurization — used by BOTH the trainer (train_pos) and the inference module
(tagger), so features align by construction. All features are NAMED and interpretable: suffix-class
(incl. Greco-Latin/biomedical NOUN suffixes for OOD), word-shape, closed-class, fine suffix/prefix
one-hots, and the token's W embedding (POS uses our W — its distributional signature).
"""
from __future__ import annotations
import json, os
import numpy as np


def _load_W(w_dir):
    """Load static embedding W. Supports the int8-quantized release (W_int8.npz: q+scale, ~5x smaller,
    accuracy-free) and the float32 CSR form (W.npz)."""
    import os
    p8 = os.path.join(w_dir, "W_int8.npz")
    if os.path.exists(p8):
        z = np.load(p8); return z["q"].astype(np.float32) * float(z["scale"])
    import scipy.sparse as sp
    return sp.load_npz(os.path.join(w_dir, "W.npz")).toarray().astype(np.float32)

UD_TAGS = ["NOUN", "PUNCT", "ADP", "NUM", "SYM", "SCONJ", "ADJ", "PART", "DET", "CCONJ",
           "PROPN", "PRON", "X", "ADV", "INTJ", "VERB", "AUX"]
FUNCTION = {"ADP", "AUX", "CCONJ", "SCONJ", "DET", "PART", "PRON"}
TAG2I = {t: i for i, t in enumerate(UD_TAGS)}
NT = len(UD_TAGS)
# OOD-robust: Greco-Latin/biomedical NOUN suffixes so splenomegaly/hepatitis tag NOUN by RULE (IE ask)
SUF_CLASSES = ["tion", "sion", "ment", "ness", "ity", "ous", "ive", "ize", "ise", "ate", "ant", "ent",
               "ing", "ed", "ly", "al", "ic", "er", "or", "est", "ful", "less", "able", "ible", "ish",
               "osis", "itis", "emia", "aemia", "oma", "megaly", "pathy", "ectomy", "otomy", "ostomy",
               "plasty", "scopy", "graphy", "logy", "cyte", "penia", "uria", "algia", "rrhea", "trophy",
               "s", "es", "'s", "n't"]
N_SHAPE_CLASS = 15
N_POS = 3                                                       # sentence-position axis: 0=BOS, 1=mid, 2=EOS

# ---- curated CLOSED-class / lexical resources (glass-box, OOD-robust — complement WordNet's open-class coverage) ----
AUX_WORDS = {"be", "am", "is", "are", "was", "were", "been", "being", "have", "has", "had", "do", "does",
             "did", "will", "would", "shall", "should", "can", "could", "may", "might", "must", "ought",
             "'s", "'re", "'ve", "'d", "'ll", "'m", "na", "wo", "ai"}     # aux/modal + contractions -> targets VERB↦AUX
_LEXSETS = {}


def _names_set():
    if "names" not in _LEXSETS:
        try:
            from nltk.corpus import names
            _LEXSETS["names"] = set(n.lower() for n in names.words())
        except Exception:
            _LEXSETS["names"] = set()
    return _LEXSETS["names"]


def _locs_set():
    if "locs" not in _LEXSETS:
        try:
            from nltk.corpus import gazetteers
            _LEXSETS["locs"] = set(g.lower() for g in gazetteers.words())
        except Exception:
            _LEXSETS["locs"] = set()
    return _LEXSETS["locs"]


N_LEXICON = 3                                                   # [is-auxiliary, in-person-names, in-place-gazetteer]


def position_class(i: int, n: int) -> int:
    return 0 if i == 0 else (2 if i == n - 1 else 1)


# ---- WordNet ontology word-class (curated, OOD-robust; the glass-box upgrade of the learned cluster axis) ----
_WN_CACHE = {}
_WN = None


def _wn():
    global _WN
    if _WN is None:
        from nltk.corpus import wordnet as wn
        _WN = wn
    return _WN


def _wn_normalize(w: str) -> str:
    """Backoff normalization for web-text content words WordNet misses as-is: strip possessive ('pic's'->'pic',
    'dogs''->'dogs') and reduce a hyphenated compound to its (head-final) head ('anti-gay'->'gay',
    'counter-terrorism'->'terrorism'). Recovers ~half the content-word coverage gap."""
    if w.endswith("'s") or w.endswith("’s"):
        w = w[:-2]
    elif w.endswith("'") or w.endswith("’"):
        w = w[:-1]
    if "-" in w:
        parts = [p for p in w.split("-") if p]
        if len(parts) > 1:
            w = parts[-1]
    return w


def _posbits(w: str) -> int:
    bits = 0
    try:
        for s in _wn().synsets(w):
            bits |= {"n": 1, "v": 2, "a": 4, "s": 4, "r": 8}.get(s.pos(), 0)
    except Exception:
        bits = 0
    return bits


def wn_posset_id(word: str) -> int:
    """POS-set signature in [0..15]: bit n=1,v=2,adj=4,adv=8 (0 = empty: function word / name / OOV).
    Curated, OOD-robust POS prior; morphy (inside wn.synsets) normalizes inflected forms; a normalization
    BACKOFF (possessive/hyphen) is tried on a miss to recover web-text content words."""
    w = word.lower()
    c = _WN_CACHE.get(w)
    if c is not None:
        return c
    bits = _posbits(w)
    if bits == 0:
        nw = _wn_normalize(w)
        if nw and nw != w:
            bits = _posbits(nw)
    _WN_CACHE[w] = bits
    return bits


N_WN_POSSET = 16                                               # 2^4 subsets of {noun,verb,adj,adv}

# WordNet supersenses = 45 lexicographer files (NAMED semantic classes: noun.animal, verb.motion, ...)
SUPERSENSES = [
    "adj.all", "adj.pert", "adj.ppl", "adv.all", "noun.Tops", "noun.act", "noun.animal", "noun.artifact",
    "noun.attribute", "noun.body", "noun.cognition", "noun.communication", "noun.event", "noun.feeling",
    "noun.food", "noun.group", "noun.location", "noun.motive", "noun.object", "noun.person", "noun.phenomenon",
    "noun.plant", "noun.possession", "noun.process", "noun.quantity", "noun.relation", "noun.shape",
    "noun.state", "noun.substance", "noun.time", "verb.body", "verb.change", "verb.cognition",
    "verb.communication", "verb.competition", "verb.consumption", "verb.contact", "verb.creation",
    "verb.emotion", "verb.motion", "verb.perception", "verb.possession", "verb.social", "verb.stative",
    "verb.weather"]
SS_MAP = {s: i + 1 for i, s in enumerate(SUPERSENSES)}          # 1..45 (0 = no synset: function word / name / OOV)
N_WN_SUPERSENSE = len(SUPERSENSES) + 1
_WN_SS_CACHE = {}


def _ssid(w: str) -> int:
    try:
        syns = _wn().synsets(w)
        if syns:
            return SS_MAP.get(syns[0].lexname(), 0)
    except Exception:
        pass
    return 0


def wn_supersense_id(word: str) -> int:
    """Most-frequent-sense supersense id in [0..45] (0 = no synset). The first synset's lexicographer file
    = a NAMED, curated semantic word-class; same possessive/hyphen normalization backoff as posset."""
    w = word.lower()
    c = _WN_SS_CACHE.get(w)
    if c is not None:
        return c
    sid = _ssid(w)
    if sid == 0:
        nw = _wn_normalize(w)
        if nw and nw != w:
            sid = _ssid(nw)
    _WN_SS_CACHE[w] = sid
    return sid


def prime_wn_caches(w_dir):
    """Load precomputed WordNet lookups (offline) to avoid the per-token live morphy/synsets cost at inference."""
    for kind, cache in (("posset", _WN_CACHE), ("supersense", _WN_SS_CACHE)):
        p = f"{w_dir.rstrip('/')}_wn_{kind}.json"
        if os.path.exists(p):
            try:
                cache.update(json.load(open(p)))
            except Exception:
                pass


def build_wn_caches(w_dir, words, fns=("posset", "supersense")):
    """Precompute posset/supersense for `words` -> json caches next to W (recovers inference speed)."""
    fn = {"posset": wn_posset_id, "supersense": wn_supersense_id}
    for kind in fns:
        d = {w: int(fn[kind](w)) for w in words}
        json.dump(d, open(f"{w_dir.rstrip('/')}_wn_{kind}.json", "w"))
        print(f"[wn] cached {len(d)} {kind} lookups -> {w_dir.rstrip('/')}_wn_{kind}.json", flush=True)


def build_or_load_clusters(w_dir, K=256, dim=32, iters=15, cache=None):
    """word_lower -> cluster id (0..K-1), from a dependency-free numpy spherical k-means on PCA(W). Cached.
    A COARSE K (16-32) gives a generalizing word-CLASS lattice axis; a fine K (256) gives shared engram cells."""
    cache = cache or f"{w_dir.rstrip('/')}_clusters_K{K}.npz"
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=False)
        return {w: int(c) for w, c in zip(json.loads(str(z["words"])), z["clust"])}
    import scipy.sparse as sp
    Wd = _load_W(w_dir)
    vocab = json.load(open(f"{w_dir}/vocab.json"))
    words = [None] * Wd.shape[0]
    for k, idx in vocab.items():
        words[idx] = k.split("=", 1)[-1] if "=" in k else k
    Wm = Wd.mean(0); rng = np.random.default_rng(0)
    sub = rng.choice(Wd.shape[0], min(20000, Wd.shape[0]), replace=False)
    _, _, Vt = np.linalg.svd(Wd[sub] - Wm, full_matrices=False)
    X = (Wd - Wm) @ Vt[:dim].T
    X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)         # spherical (cosine) k-means
    cent = X[rng.choice(X.shape[0], K, replace=False)].copy()
    for _ in range(iters):
        a = (X @ cent.T).argmax(1)
        for k in range(K):
            m = a == k
            if m.any():
                cent[k] = X[m].mean(0)
        cent /= (np.linalg.norm(cent, axis=1, keepdims=True) + 1e-8)
    a = (X @ cent.T).argmax(1).astype(np.int16)
    np.savez(cache, words=json.dumps(words), clust=a, K=K)
    return {w: int(c) for w, c in zip(words, a)}


def suf_class(L: str) -> int:
    best, blen = 0, 0                                           # LONGEST-matching suffix wins (megaly > ly)
    for i, s in enumerate(SUF_CLASSES):
        if len(s) > blen and L.endswith(s) and len(L) > len(s):
            best, blen = i + 1, len(s)
    return best


def shape_class(w: str) -> int:
    if not w:
        return 0
    c0 = "X" if w[0].isupper() else "x" if w[0].islower() else "d" if w[0].isdigit() else "p"
    allcap = w.isupper() and len(w) > 1
    hasdig = any(ch.isdigit() for ch in w)
    return min({"X": 1, "x": 2, "d": 3, "p": 4}[c0] + (5 if allcap else 0) + (10 if hasdig else 0),
               N_SHAPE_CLASS - 1)


def shape6(tok: str) -> str:
    return "".join("X" if c.isupper() else "x" if c.islower() else "d" if c.isdigit() else "-" for c in tok[:6])


class Featurizer:
    """Builds the (fine-linear, W-embedding, lattice-cell) features for a token. Holds the fitted feature
    vocabs + the W lookup so the trainer and tagger emit identical features."""

    def __init__(self, fine_vocab, fine_shapes, closed_words, emb_proj, emb_mean, Wd, wrow, lex_vocab=None,
                 cap_frac=None, clust=None, n_clust=0, wn_kind=None, lexicon_feats=False):
        self.fine_vocab = fine_vocab
        self.fine_shapes = fine_shapes
        self.lex_vocab = lex_vocab or {}                        # frequent-word IDENTITY priors (codex #2)
        self.cap_frac = cap_frac or {}                          # ENTITY-membership proxy: corpus cap-fraction
        self.clust = clust                                      # word->coarse W-cluster (learned class axis)
        self.n_clust = n_clust                                  # 0 = no word-class lattice axis; else its cardinality
        self.wn_kind = wn_kind                                  # None | "posset" — WordNet ontology class axis source
        self.closed = set(closed_words)
        self.P = np.asarray(emb_proj, np.float32)
        self.Wm = np.asarray(emb_mean, np.float32)
        self.Wd = Wd
        self.wrow = wrow
        # NEIGHBOR context block (prev/next shape-class + closed + suffix-class) — the emission needs context
        # (else "the patient underwent" can't disambiguate patient=NOUN, underwent=VERB). IE-spec feature.
        self.n_sufc = len(SUF_CLASSES) + 1
        self.ctx = 2 * (N_SHAPE_CLASS + 1 + self.n_sufc)        # prev + next: shape + closed + sufclass
        self.lex_base = len(fine_vocab) + len(fine_shapes) + self.ctx
        # cap×position (refinement for PROPN over-prediction): 0=initcap_at_BOS (weak PROPN cue —
        # sentence-initial cap), 1=initcap_mid (strong PROPN cue), 2=allcaps_mid (acronym/PROPN)
        self.cappos_base = self.lex_base + len(self.lex_vocab)
        self.capfrac_base = self.cappos_base + 3                # ENTITY proxy: cap-consistency 3 bins
        self.lexicon_feats = lexicon_feats                     # curated AUX/names/places lexicon features
        self.lexfeat_base = self.capfrac_base + 3
        self.nf_fine = self.lexfeat_base + (N_LEXICON if lexicon_feats else 0)
        self.emb_dim = self.P.shape[1]
        self.latent = 1 + self.emb_dim

    def _ctx_block(self, x, base, w):
        """fill one neighbor's (shape-class, closed, suffix-class) one-hots at offset base."""
        if w:
            x[base + shape_class(w)] = 1.0
            x[base + N_SHAPE_CLASS] = 1.0 if w.lower() in self.closed else 0.0
            x[base + N_SHAPE_CLASS + 1 + suf_class(w.lower())] = 1.0

    def _cappos(self, w, prev):
        """cap×position class: None (not cap) / 0 (cap at BOS, weak) / 1 (initcap mid, strong PROPN) /
        2 (allcaps mid, acronym). prev=="" means sentence-initial."""
        if not w or not w[0].isupper():
            return None
        if w.isupper() and len(w) > 1:
            return 2 if prev != "" else 0
        return 1 if prev != "" else 0

    def _capfrac(self, w):
        """ENTITY-membership proxy bin from corpus cap-fraction: 0=common (mostly lower, NOT entity),
        1=mixed, 2=name-like (mostly capitalized -> PROPN). None if unseen."""
        cf = self.cap_frac.get(w.lower())
        if cf is None:
            return None
        return 0 if cf < 0.25 else (2 if cf > 0.75 else 1)

    def _ctx_idx(self, idx, base, w):
        if w:
            idx.append(base + shape_class(w))
            if w.lower() in self.closed:
                idx.append(base + N_SHAPE_CLASS)
            idx.append(base + N_SHAPE_CLASS + 1 + suf_class(w.lower()))

    def _lex_idx(self, L):
        """curated-lexicon active indices: 0=auxiliary/modal (VERB↦AUX), 1=person-name, 2=place (both →PROPN)."""
        out = []
        if L in AUX_WORDS:
            out.append(self.lexfeat_base + 0)
        if L in _names_set():
            out.append(self.lexfeat_base + 1)
        if L in _locs_set():
            out.append(self.lexfeat_base + 2)
        return out

    def fine_indices(self, w, prev="", nxt=""):
        """ACTIVE fine-feature indices (sparse) — for memory-bounded per-batch densification in training."""
        idx = []; L = w.lower()
        for f in (f"suf3={L[-3:]}", f"pre3={L[:3]}", f"suf4={L[-4:]}"):
            j = self.fine_vocab.get(f)
            if j is not None:
                idx.append(j)
        idx.append(len(self.fine_vocab) + self.fine_shapes.get(shape6(w), 0))
        base = len(self.fine_vocab) + len(self.fine_shapes)
        self._ctx_idx(idx, base, prev)
        self._ctx_idx(idx, base + (N_SHAPE_CLASS + 1 + self.n_sufc), nxt)
        li = self.lex_vocab.get(L)
        if li is not None:
            idx.append(self.lex_base + li)
        cp = self._cappos(w, prev)                              # cap×position (PROPN refinement)
        if cp is not None:
            idx.append(self.cappos_base + cp)
        cf = self._capfrac(w)                                   # entity-membership proxy (cap-consistency)
        if cf is not None:
            idx.append(self.capfrac_base + cf)
        if self.lexicon_feats:                                  # curated AUX / names / places (OOD-robust)
            idx.extend(self._lex_idx(L))
        return idx

    def fine_named(self, w: str, prev: str = "", nxt: str = ""):
        """ACTIVE fine features as (NAME, index) — the readable counterpart of fine_indices, for explanations."""
        out = []; L = w.lower()
        for f in (f"suf3={L[-3:]}", f"pre3={L[:3]}", f"suf4={L[-4:]}"):
            j = self.fine_vocab.get(f)
            if j is not None:
                out.append((f, j))
        out.append((f"shape={shape6(w)}", len(self.fine_vocab) + self.fine_shapes.get(shape6(w), 0)))
        base = len(self.fine_vocab) + len(self.fine_shapes)

        def ctx(b, x, who):
            r = []
            if x:
                r.append((f"{who}shape={shape_class(x)}", b + shape_class(x)))
                if x.lower() in self.closed:
                    r.append((f"{who}closed", b + N_SHAPE_CLASS))
                r.append((f"{who}suf={suf_class(x.lower())}", b + N_SHAPE_CLASS + 1 + suf_class(x.lower())))
            return r
        out += ctx(base, prev, "prev:"); out += ctx(base + (N_SHAPE_CLASS + 1 + self.n_sufc), nxt, "next:")
        li = self.lex_vocab.get(L)
        if li is not None:
            out.append((f"word={L}", self.lex_base + li))
        cp = self._cappos(w, prev)
        if cp is not None:
            out.append((f"cap×pos={cp}", self.cappos_base + cp))
        cf = self._capfrac(w)
        if cf is not None:
            out.append((f"capfrac={cf}", self.capfrac_base + cf))
        if self.lexicon_feats:
            nm = {self.lexfeat_base: "is_aux", self.lexfeat_base + 1: "in_names", self.lexfeat_base + 2: "in_places"}
            for j in self._lex_idx(L):
                out.append((nm[j], j))
        return out

    @classmethod
    def from_npz(cls, z, w_dir):
        import scipy.sparse as sp
        Wd = _load_W(w_dir)
        vocab = json.load(open(f"{w_dir}/vocab.json"))
        wrow = {(k.split("=", 1)[-1] if "=" in k else k): idx for k, idx in vocab.items()}
        j = lambda k: json.loads(str(z[k]))
        lexv = j("lex_vocab") if "lex_vocab" in z else {}
        capf = j("cap_frac") if "cap_frac" in z else {}
        clust = None; n_clust = 0; wn_kind = None
        wk = str(z["wn_kind"]) if "wn_kind" in z else ""
        if wk in ("posset", "supersense"):                     # WordNet ontology axis (prime offline cache, no rebuild)
            wn_kind = wk; n_clust = N_WN_POSSET if wk == "posset" else N_WN_SUPERSENSE
            prime_wn_caches(w_dir)
        elif "clust_k" in z and int(z["clust_k"]) > 0:         # learned coarse cluster axis (rebuild from W)
            K = int(z["clust_k"]); clust = build_or_load_clusters(w_dir, K=K); n_clust = K + 1
        lexf = bool(int(z["lexicon_feats"])) if "lexicon_feats" in z else False
        return cls(j("fine_vocab"), j("fine_shapes"), j("closed_words"),
                   z["emb_proj"], z["emb_mean"], Wd, wrow, lex_vocab=lexv, cap_frac=capf,
                   clust=clust, n_clust=n_clust, wn_kind=wn_kind, lexicon_feats=lexf)

    def fine(self, w: str, prev: str = "", nxt: str = "") -> np.ndarray:
        x = np.zeros(self.nf_fine, np.float32); L = w.lower()
        for f in (f"suf3={L[-3:]}", f"pre3={L[:3]}", f"suf4={L[-4:]}"):
            jx = self.fine_vocab.get(f)
            if jx is not None:
                x[jx] = 1.0
        x[len(self.fine_vocab) + self.fine_shapes.get(shape6(w), 0)] = 1.0
        base = len(self.fine_vocab) + len(self.fine_shapes)
        self._ctx_block(x, base, prev)                          # prev-token context
        self._ctx_block(x, base + (N_SHAPE_CLASS + 1 + self.n_sufc), nxt)   # next-token context
        li = self.lex_vocab.get(L)                              # frequent-word identity (codex #2)
        if li is not None:
            x[self.lex_base + li] = 1.0
        cp = self._cappos(w, prev)                              # cap×position (PROPN refinement)
        if cp is not None:
            x[self.cappos_base + cp] = 1.0
        cf = self._capfrac(w)                                   # entity-membership proxy (cap-consistency)
        if cf is not None:
            x[self.capfrac_base + cf] = 1.0
        if self.lexicon_feats:                                  # curated AUX / names / places (OOD-robust)
            for j2 in self._lex_idx(L):
                x[j2] = 1.0
        return x

    def emb(self, w: str) -> np.ndarray:
        v = np.zeros(self.latent, np.float32); v[0] = 1.0
        r = self.wrow.get(w.lower())
        if r is not None:
            v[1:] = (self.Wd[r] - self.Wm) @ self.P
        return v

    def kappa(self, w: str, pos: int = 1) -> tuple:            # +position axis (higher-order lattice)
        L = w.lower()
        base = (suf_class(L), shape_class(w), int(L in self.closed), pos)
        if self.n_clust:                                       # + word-CLASS axis (WordNet ontology OR learned cluster)
            if self.wn_kind == "posset":
                base = base + (wn_posset_id(w),)               # curated POS-set (0=function/name/OOV)
            elif self.wn_kind == "supersense":
                base = base + (wn_supersense_id(w),)           # curated NAMED semantic class (noun.animal, ...)
            else:
                cid = self.clust.get(L) if self.clust else None
                base = base + (0 if cid is None else cid + 1,) # learned W-cluster (0=OOV, else 1..K)
        return base
