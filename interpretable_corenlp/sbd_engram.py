"""Glass-box ENGRAM for the SBD — the lexical-tail memory the boundary lattice can't capture.

The SBD lattice is coarse (5 named axes). It generalizes, but it cannot memorize SPECIFIC lexical patterns,
which is exactly where boundary detection has a long tail:
  - domain abbreviations not in the lexicon (Fig., Eq., vs., clinical p.o.),
  - citation/fixed patterns (et al . Capital -> NOT a boundary),
  - catalog-tail register (wikipedia "External links | ..." runs),
  - collocations (New | York should never split).
So we add a horseshoe-sparse, NAMED additive memory over n-gram cues, fit on the RESIDUAL of the FROZEN
lattice (Brill-style): it only learns patterns the lattice gets wrong. Two-sided — boost real boundaries the
lattice misses (FN), penalize false ones it invents (FP). Forced double descent, glass-box (each entry is a
named pattern -> boundary-logit delta). Mirrors pos/engram.py but BINARY (no CRF/transition).

  .venv-rocm/bin/python -m bq_embedding.pos.sbd_engram --sbd <sbd.npz> --pos <model.npz> --out <mem.npz>
"""
from __future__ import annotations
import argparse, json, time
from collections import defaultdict, Counter
import numpy as np

from interpretable_corenlp.posfeats import TAG2I, UD_TAGS
from interpretable_corenlp.sbd import load_sbd, nextcap_class

# cue axes for boundary memory (the literal lexical patterns the coarse lattice abstracts away)
AXES = ["prevword", "nextword", "bigram", "prevword_nextcap", "pos_nextword"]


def axis_patterns(toks, i, pos_ids):
    """The named patterns firing at gap-after-token-i (boundary candidate)."""
    wi = toks[i].lower(); wn = (toks[i + 1].lower() if i + 1 < len(toks) else "")
    nc = nextcap_class(toks[i + 1] if i + 1 < len(toks) else "")
    pi = UD_TAGS[pos_ids[i]]
    return {
        "prevword": wi,
        "nextword": wn,
        "bigram": f"{wi}|{wn}",
        "prevword_nextcap": f"{wi}|{nc}",
        "pos_nextword": f"{pi}|{wn}",
    }


WORD_AXES = {"prevword", "nextword", "bigram", "prevword_nextcap"}   # word-keyed: must RECUR across contexts


def generate_candidates(docs, p_back, *, n_min, err_floor, pos_frac, neg_frac, ctx_min=3):
    """Per (axis, pattern): support, lattice-error count, boundary-rate, DISTINCT contexts. Keep error-driven,
    concentrated patterns. Word-keyed axes must recur across >= ctx_min distinct contexts (the tail-recurrence
    rule, mirroring pos/engram.py) — kills over-specific singletons like `prevword_nextcap=indianapolis|1`."""
    agg = {ax: defaultdict(lambda: [0, 0, 0, set()]) for ax in AXES}  # [support, n_boundary, n_wrong, contexts]
    for d, (toks, pos, bnd) in enumerate(docs):
        pb = p_back[d]
        for i in range(len(toks)):
            g = int(bnd[i]); wrong = int((pb[i] >= 0.5) != (g == 1))
            ctx = (toks[i - 1].lower() if i else "", toks[i].lower(),
                   toks[i + 1].lower() if i + 1 < len(toks) else "")   # the literal window = one distinct context
            for ax, pat in axis_patterns(toks, i, pos).items():
                rec = agg[ax][pat]; rec[0] += 1; rec[1] += g; rec[2] += wrong; rec[3].add(ctx)
    keep = {ax: {} for ax in AXES}; pid = {ax: 1 for ax in AXES}  # pid 0 = "no pattern" (frozen zero delta)
    for ax in AXES:
        for pat, (n_p, nb, nw, ctxs) in agg[ax].items():
            if n_p < n_min or nw / n_p < err_floor:
                continue
            if ax in WORD_AXES and len(ctxs) < ctx_min:           # must generalize across contexts, not memorize one
                continue
            rate = nb / n_p                                       # boundary rate at this pattern
            if rate >= pos_frac or rate <= (1 - neg_frac):        # concentrated -> boundary (boost) or not (penalize)
                keep[ax][pat] = (pid[ax], rate, n_p); pid[ax] += 1
    return keep


def build_pids(docs, keep):
    pids = []
    for toks, pos, _ in docs:
        a = {ax: np.zeros(len(toks), np.int32) for ax in AXES}
        for i in range(len(toks)):
            for ax, pat in axis_patterns(toks, i, pos).items():
                hit = keep[ax].get(pat)
                if hit is not None:
                    a[ax][i] = hit[0]
        pids.append(a)
    return pids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sbd", required=True, help="frozen SBD lattice model (the backbone)")
    ap.add_argument("--pos", required=True)
    ap.add_argument("--w", default="data/W_11M_K512_signed")
    ap.add_argument("--gold", default="data/ud_full_native.jsonl")
    ap.add_argument("--max-sents", type=int, default=0)
    ap.add_argument("--n-min", type=int, default=6, help="min pattern support")
    ap.add_argument("--err-floor", type=float, default=0.25, help="min lattice error-rate at the pattern")
    ap.add_argument("--pos-frac", type=float, default=0.85, help="boost if boundary-rate >= this")
    ap.add_argument("--neg-frac", type=float, default=0.85, help="penalize if NON-boundary-rate >= this")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--hs", type=float, default=0.5)
    ap.add_argument("--warmup", type=float, default=0.3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()

    from interpretable_corenlp.cotrain import load_ud, build_documents, retag_with_pos
    from interpretable_corenlp.tagger import load_tagger
    sbd = load_sbd(a.sbd); tagger = load_tagger(a.pos, w_dir=a.w)
    rng = np.random.default_rng(13)
    sents = load_ud(a.gold, a.max_sents)
    docs_raw = build_documents(sents, rng)
    retag_with_pos(docs_raw, tagger)
    docs = [(d["tokens"], d["pos"], d["bnd"]) for d in docs_raw]
    ntr = int(0.9 * len(docs)); tr, te = docs[:ntr], docs[ntr:]
    print(f"[sbd-eng] {len(docs)} docs ({sum(len(t) for t,_,_ in docs):,} gaps)", flush=True)

    def backbone(dlist):
        out = []
        for toks, pos, _ in dlist:
            out.append(sbd.predict(toks, pos))                   # frozen lattice boundary prob (raw token-mode)
        return out
    pb_tr = backbone(tr); pb_te = backbone(te)
    keep = generate_candidates(tr, pb_tr, n_min=a.n_min, err_floor=a.err_floor,
                               pos_frac=a.pos_frac, neg_frac=a.neg_frac)
    ksz = {ax: len(keep[ax]) for ax in AXES}
    print(f"[sbd-eng] memory patterns: " + ", ".join(f"{ax}={ksz[ax]}" for ax in AXES), flush=True)

    # backbone logit (invert the prob) per gap; fit additive memory deltas with horseshoe (binary BCE)
    def logit(p):
        p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))
    pids_tr = build_pids(tr, keep); pids_te = build_pids(te, keep)

    import jax, jax.numpy as jnp, optax
    print(f"[sbd-eng] backend={jax.default_backend()}", flush=True)
    # flatten gaps
    def flatten(dlist, pb, pids):
        Z, Y, P = [], [], {ax: [] for ax in AXES}
        for d, (toks, pos, bnd) in enumerate(dlist):
            z = logit(pb[d])
            for i in range(len(toks)):
                Z.append(z[i]); Y.append(bnd[i])
                for ax in AXES:
                    P[ax].append(pids[d][ax][i])
        return (np.asarray(Z, np.float32), np.asarray(Y, np.float32),
                {ax: np.asarray(P[ax], np.int32) for ax in AXES})
    Ztr, Ytr, Ptr = flatten(tr, pb_tr, pids_tr)
    Zte, Yte, Pte = flatten(te, pb_te, pids_te)
    params = {f"E_{ax}": jnp.zeros(max(1, ksz[ax] + 1), jnp.float32) for ax in AXES}
    opt = optax.adam(a.lr); st = opt.init(params)
    pw = float((1 - Ytr.mean()) / max(Ytr.mean(), 1e-6))

    def emit(p, Z, P):
        e = Z
        for ax in AXES:
            e = e + p[f"E_{ax}"][P[ax]]
        return e

    @jax.jit
    def step(p, st):
        def loss(p):
            z = emit(p, jnp.asarray(Ztr), {ax: jnp.asarray(Ptr[ax]) for ax in AXES})
            w = 1.0 + (pw - 1.0) * jnp.asarray(Ytr)
            return jnp.mean(w * optax.sigmoid_binary_cross_entropy(z, jnp.asarray(Ytr)))
        l, g = jax.value_and_grad(loss)(p)
        upd, st2 = opt.update(g, st); return optax.apply_updates(p, upd), st2, l

    warm = int(a.warmup * a.epochs)
    for ep in range(a.epochs):
        params, st, l = step(params, st)
        if ep >= warm and a.hs > 0:                              # lazy horseshoe prox on memory rows (spike-and-slab)
            for ax in AXES:
                E = params[f"E_{ax}"]; fac = jnp.clip(1.0 - a.lr * a.hs / (E ** 2 + 1e-8), 0.0, 1.0)
                E = (E * fac).at[0].set(0.0); params = {**params, f"E_{ax}": E}
    # eval lattice vs lattice+engram on held-out
    def f1(z, Y):
        pred = (1 / (1 + np.exp(-z)) >= 0.5).astype(np.float32)
        tp = ((pred == 1) & (Y == 1)).sum(); fp = ((pred == 1) & (Y == 0)).sum(); fn = ((pred == 0) & (Y == 1)).sum()
        pr = tp / (tp + fp + 1e-9); rc = tp / (tp + fn + 1e-9); return 2 * pr * rc / (pr + rc + 1e-9), pr, rc
    base_f1 = f1(Zte, Yte)[0]
    ze = np.asarray(emit(params, jnp.asarray(Zte), {ax: jnp.asarray(Pte[ax]) for ax in AXES}))
    eng_f1, pr, rc = f1(ze, Yte)
    nnz = sum(int((np.abs(np.asarray(params[f"E_{ax}"])) > 1e-4).sum()) for ax in AXES)
    print(f"[sbd-eng] held-out boundary F1: lattice {base_f1:.3f} -> +engram {eng_f1:.3f} "
          f"(P={pr:.3f} R={rc:.3f}); {nnz} live memory rows", flush=True)

    np.savez(a.out, axes=json.dumps(AXES),
             keep=json.dumps({ax: {p: v[0] for p, v in keep[ax].items()} for ax in AXES}),
             **{f"E_{ax}": np.asarray(params[f"E_{ax}"]) for ax in AXES})
    print(f"[sbd-eng] wrote SBD engram -> {a.out}  ({time.time()-t0:.0f}s)", flush=True)
    # readout: strongest memory patterns
    for ax in AXES:
        E = np.asarray(params[f"E_{ax}"]); inv = {v[0]: p for p, v in keep[ax].items()}
        order = np.argsort(-np.abs(E))[:6]
        items = [f"{inv.get(int(j),'?')}:{E[j]:+.1f}" for j in order if j > 0 and abs(E[j]) > 1e-3]
        if items:
            print(f"  [{ax}] " + ", ".join(items), flush=True)


if __name__ == "__main__":
    main()
