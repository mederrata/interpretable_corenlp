"""Glass-box POS tagger INFERENCE + NP-chunker — the IE-agent consumer interface (2026-06-16 contract).

  from interpretable_corenlp.tagger import load_tagger
  pos = load_tagger("data/pos_glassbox.npz", w_dir="data/W_11M_K512_signed")
  pos.tag(tokens)        -> [pos_tag]                  (1:1 with the caller's tokens; aligns by construction)
  pos.np_chunks(tokens)  -> [(i, j)]  half-open spans  (NP chunks: grower-stop + NP candidate gen)

Pure-numpy inference (no JAX/transformer): lattice emission (functional-ANOVA Θ(κ)·[1,W_emb] + fine
linear) + Viterbi over the named tag->tag transition. The model is a glass-box lattice-CRF; every tag is
attributable to named features (suffix-class / shape / closed-class / W-embedding / transition).
"""
from __future__ import annotations
import numpy as np

import json
from interpretable_corenlp.posfeats import Featurizer, UD_TAGS, NT, position_class

# glass-box NP-chunk grammar (interpretable, not learned): [DET]? (ADJ|NUM|NOUN|PROPN)* (NOUN|PROPN)
NP_INTERNAL = {"NOUN", "PROPN", "ADJ", "NUM"}
NP_HEAD = {"NOUN", "PROPN"}


class Tagger:
    def __init__(self, z, fx: Featurizer, memory=None):
        self.fx = fx
        self.Wf = np.asarray(z["Wf"], np.float32)
        self.b = np.asarray(z["b"], np.float32)
        self.T = np.asarray(z["T"], np.float32)
        self.g = np.asarray(z["lat_global"], np.float32)              # (latent, NT)
        self.cards = json.loads(str(z["lat_cards"])) if "lat_cards" in z else [0, 0, 0]
        self.main = [np.asarray(z[f"lat_main{i}"], np.float32) for i in range(len(self.cards))]
        pw = json.loads(str(z["lat_pairwise"])) if "lat_pairwise" in z else [[0, 1]]
        self.inter = {tuple(ab): np.asarray(z[f"lat_inter_{ab[0]}_{ab[1]}"], np.float32) for ab in pw}
        self.qrank = int(z["quad_rank"]) if "quad_rank" in z else 0    # single quadratic attention-like term
        if self.qrank > 0:
            self.qA = np.asarray(z["qA"], np.float32); self.qV = np.asarray(z["qV"], np.float32)
        self.ctx_window = int(z["ctx_window"]) if "ctx_window" in z else 0   # SENTENCE-LEVEL windowed ctx head
        if self.ctx_window > 0:                                        # named per-distance pooling gates (softplus)
            self.wL = np.logaddexp(0.0, np.asarray(z["cwL"], np.float32))   # softplus, stable
            self.wR = np.logaddexp(0.0, np.asarray(z["cwR"], np.float32))
        # optional glass-box ENGRAM memory (additive named-pattern logit deltas + b/T recalibration)
        self.mem = memory                                            # MemoryBank or None
        self.Teff = self.T + memory.dT if memory is not None else self.T
        self.bmem = memory.db if memory is not None else None

    def _emission(self, tokens):
        """(L, NT) emission scores for the token sequence."""
        L = len(tokens)
        E = np.zeros((L, NT), np.float32)
        embs = np.zeros((L, self.fx.latent), np.float32)
        for t, w in enumerate(tokens):
            prev = tokens[t - 1] if t > 0 else ""; nxt = tokens[t + 1] if t < L - 1 else ""
            k = self.fx.kappa(w, position_class(t, L))                # κ axes (incl. position)
            Theta = self.g + sum(self.main[a][k[a]] for a in range(len(k)))   # global + mains
            for (a, b), I in self.inter.items():                      # + 2-way interactions
                Theta = Theta + I[k[a], k[b]]
            embs[t] = self.fx.emb(w)
            E[t] = Theta @ embs[t] + self.fx.fine(w, prev, nxt) @ self.Wf + self.b
        if self.ctx_window > 0:                                       # SENTENCE-LEVEL windowed pooled-context head
            xa = embs[:, 1:] @ self.qA                                # (L,r); W_emb only (drop bias channel)
            ca = np.zeros_like(xa)
            for d in range(1, self.ctx_window + 1):                   # pool ±d neighbors with named gates
                if d < L:
                    ca[d:] += self.wL[d - 1] * xa[:-d]                # left neighbor at distance -d
                    ca[:-d] += self.wR[d - 1] * xa[d:]               # right neighbor at distance +d
            E = E + (xa * ca) @ self.qV                               # glass-box attention: token_i ⊗ pooled-ctx
        elif self.qrank > 0:                                          # single quadratic (prev-token) head
            xa = embs[:, 1:] @ self.qA                                # (L,r); W_emb only (drop bias channel)
            xap = np.zeros_like(xa); xap[1:] = xa[:-1]                # interact with previous token
            E = E + (xa * xap) @ self.qV
        if self.mem is not None:                                      # + ENGRAM: O(#axes) dict lookups/token
            E = E + self.bmem
            self.mem.add_deltas(tokens, E)
        return E

    def tag(self, tokens):
        """POS tags (1:1 with `tokens`) via Viterbi over the glass-box CRF (+ optional Engram memory)."""
        if not tokens:
            return []
        E = self._emission(tokens)
        L = E.shape[0]; T = self.Teff
        dp = E[0].copy(); bp = np.zeros((L, NT), np.int32)
        for t in range(1, L):
            s = dp[:, None] + T                                      # (NT_prev, NT_cur)
            bp[t] = s.argmax(0); dp = E[t] + s.max(0)
        out = np.zeros(L, np.int32); out[-1] = dp.argmax()
        for t in range(L - 2, -1, -1):
            out[t] = bp[t + 1, out[t + 1]]
        return [UD_TAGS[i] for i in out]

    def explain(self, tokens, topk=8):
        """EXACT additive attribution for every token's predicted tag — the glass-box guarantee.
        emission(tok, tag) = global·emb + Σ_axis main_axis·emb + Σ 2-way·emb + Σ fine-feature + bias,
        plus the named tag→tag transition. Returns per token: (word, tag, [(named contribution, value), ...])
        sorted by |value|; the listed values sum EXACTLY to the emission (+ transition) for the chosen tag."""
        from interpretable_corenlp.posfeats import TAG2I
        pred = self.tag(tokens); L = len(tokens)
        axn = ["suffix", "shape", "closed", "position", "wordclass"][:len(self.cards)]
        xa = None                                                  # A-projected W_emb (for ctx/quad attribution)
        if self.qrank > 0 and (self.ctx_window > 0 or self.qrank > 0):
            xa = np.stack([self.fx.emb(w) for w in tokens])[:, 1:] @ self.qA   # (L,r)
        out = []
        for t, w in enumerate(tokens):
            k = TAG2I[pred[t]]; prev = tokens[t - 1] if t > 0 else ""; nxt = tokens[t + 1] if t < L - 1 else ""
            kap = self.fx.kappa(w, position_class(t, L)); emb = self.fx.emb(w); cc = []
            cc.append(("global", float(self.g[k] @ emb)))
            for a in range(len(kap)):                              # named main-effect lattice contributions
                cc.append((f"{axn[a]}={kap[a]}", float(self.main[a][kap[a]][k] @ emb)))
            for (a, b), I in self.inter.items():
                cc.append((f"{axn[a]}×{axn[b]}", float(I[kap[a], kap[b]][k] @ emb)))
            for nm, j in self.fx.fine_named(w, prev, nxt):         # named fine-feature contributions
                cc.append((nm, float(self.Wf[j, k])))
            if xa is not None and self.ctx_window > 0:             # SENTENCE-CONTEXT: one named term per neighbor
                for d in range(1, self.ctx_window + 1):
                    if t - d >= 0:
                        cc.append((f"ctx←{tokens[t-d]}@-{d}", float(self.wL[d - 1] * ((xa[t] * xa[t - d]) @ self.qV)[k])))
                    if t + d < L:
                        cc.append((f"ctx→{tokens[t+d]}@+{d}", float(self.wR[d - 1] * ((xa[t] * xa[t + d]) @ self.qV)[k])))
            elif xa is not None and self.qrank > 0 and t > 0:      # prev-token quad head (exact)
                cc.append((f"quad⊗{tokens[t-1]}", float(((xa[t] * xa[t - 1]) @ self.qV)[k])))
            cc.append(("bias", float(self.b[k])))
            if t > 0:                                              # sequence: the named transition into this tag
                cc.append((f"trans[{pred[t-1]}→]", float(self.Teff[TAG2I[pred[t - 1]], k])))
            cc.sort(key=lambda x: -abs(x[1]))
            if topk and len(cc) > topk:                            # residual keeps the SHOWN list summing exactly (codex)
                resid = sum(v for _, v in cc[topk:])
                out.append((w, pred[t], cc[:topk] + [(f"(+{len(cc)-topk} smaller)", resid)]))
            else:
                out.append((w, pred[t], cc))
        return out

    def marginals(self, tokens):
        """Per-token POS POSTERIOR p(tag | sentence) via CRF forward-backward — (L, NT), each row sums to 1.
        The tagger's UNCERTAINTY (high on ambiguous words = where it errs) → soft, robust POS assignment."""
        from scipy.special import logsumexp
        if not tokens:
            return np.zeros((0, NT), np.float32)
        E = self._emission(tokens); L = E.shape[0]; T = self.Teff
        al = np.zeros((L, NT), np.float64); al[0] = E[0]
        for t in range(1, L):
            al[t] = E[t] + logsumexp(al[t - 1][:, None] + T, axis=0)
        be = np.zeros((L, NT), np.float64)
        for t in range(L - 2, -1, -1):
            be[t] = logsumexp(T + (E[t + 1] + be[t + 1])[None, :], axis=1)
        logp = al + be; logp -= logsumexp(logp, axis=1, keepdims=True)
        return np.exp(logp).astype(np.float32)

    def np_chunks(self, tokens):
        """Maximal NP chunks as half-open `(i, j)` token spans — the grower-stop + NP-candidate interface."""
        tags = self.tag(tokens)
        chunks = []; i = 0; n = len(tags)
        while i < n:
            start = i
            if tags[i] == "DET" and i + 1 < n and tags[i + 1] in NP_INTERNAL:
                i += 1
            if tags[i] in NP_INTERNAL:
                while i < n and tags[i] in NP_INTERNAL:
                    i += 1
                end = i                                              # trim trailing ADJ/NUM back to the noun head
                while end > start and tags[end - 1] not in NP_HEAD:
                    end -= 1
                if end > start and any(tags[k] in NP_HEAD for k in range(start, end)):
                    chunks.append((start, end))
                i = max(i, start + 1)
            else:
                i = start + 1
        return chunks


class MemoryBank:
    """Loaded glass-box ENGRAM: per-axis pattern->pid dicts + (n_pid, NT) delta tables + b/T recal.
    Inference cost is O(#axes) string-keyed dict lookups per token (additive, no matmul)."""
    def __init__(self, z, lex_vocab, clust=None):
        from interpretable_corenlp.engram import axis_patterns
        self._patterns = axis_patterns
        self.lex_vocab = lex_vocab
        self.clust = clust                                          # word->cluster (for the shared cluster axes)
        keep = json.loads(str(z["keep"]))                           # {axis: {pattern: [pid, gtag, n_p]}}
        self.AXES = [ax for ax in json.loads(str(z["axes"]))] if "axes" in z else list(keep.keys())
        self.AXES = [ax for ax in self.AXES if f"E_{ax}" in z]      # robust: only axes actually saved
        self.pat2pid = {ax: {p: v[0] for p, v in keep.get(ax, {}).items()} for ax in self.AXES}
        self.E = {ax: np.asarray(z[f"E_{ax}"], np.float32) for ax in self.AXES}
        self.db = np.asarray(z["db"], np.float32)
        self.dT = np.asarray(z["dT"], np.float32)

    def add_deltas(self, tokens, E):
        """Add the firing memory rows into emission E (L, NT) in place."""
        for t in range(len(tokens)):
            pats = self._patterns(tokens, t, self.lex_vocab, self.clust)
            for ax, pat in pats.items():
                pid = self.pat2pid.get(ax, {}).get(pat)
                if pid:                                              # pid 0 / None = no memory -> skip
                    E[t] += self.E[ax][pid]


def load_tagger(npz_path, w_dir="data/W_11M_K512_signed", memory_npz=None):
    z = np.load(npz_path, allow_pickle=False)
    fx = Featurizer.from_npz(z, w_dir)
    mem = None
    if memory_npz is not None:
        mz = np.load(memory_npz, allow_pickle=False)
        mem = MemoryBank(mz, set(fx.lex_vocab.keys()))
    return Tagger(z, fx, memory=mem)
