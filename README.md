# interpretable-corenlp

**Intrinsically interpretable English NLP — a part-of-speech tagger, a sentence boundary detector, and the
static word embedding they share.** Each model is a *glass box*: every prediction is an **exact sum of named
feature contributions** you can read off (`explain()`), not a post-hoc approximation. No attention, no depth —
a shallow kernel machine over a frozen, inspectable embedding — yet it **matches or beats Stanford CoreNLP,
spaCy, and a fine-tuned RoBERTa** at a fraction of the size and far higher speed.

> **English only.** All models are trained and evaluated on English.

## Why "interpretable"

These are not black boxes probed after the fact. The architecture *is* the explanation:

- The **embedding** `W` is a sparse PPMI matrix factorization with **named dimensions** — each coordinate is a
  coherent, nameable factor (a topic / part-of-speech / sense signal), not a rotation-arbitrary latent.
- The **POS tagger** and **sentence detector** are *functional-ANOVA lattices* (a generalized additive model
  over **named** features: suffix-class × shape × closed-class × position × WordNet-POS-set, projecting the
  embedding), plus a named tag→tag transition. The emission for any tag is literally
  `Θ_global·emb + Σ_axis Θ_axis·emb + fine-feature weights + bias + transition`.
- So `model.explain(tokens)` returns, for every token, the **exact** list of named contributions that produced
  the decision — the values sum to the score (to floating-point error). You can see *why* `Apple→PROPN` or *why*
  a period was/wasn't a sentence boundary.

```python
from interpretable_corenlp import load_tagger
tagger = load_tagger("models/pos/model.npz", w_dir="models/embedding")
for word, tag, contribs in tagger.explain("Apple unveiled the new iPhone today .".split()):
    print(word, "->", tag, contribs[:4])
# unveiled -> VERB  [('wordclass=6', +5.05), ('shape=2', +1.71), ('suffix=14', +1.69), ('position=1', +1.44)]
```

## Models

| model | task | tags / output | size |
|---|---|---|---|
| `models/pos/model.npz` | POS tagging | 17 Universal-Dependencies UPOS tags | 3.3 MB |
| `models/sbd/sbd.npz` (+`engram.npz`) | sentence boundary detection | char-span sentences | 0.3 MB |
| `models/embedding/W_int8.npz` | static word embedding | 300,000 words × 512 dims | 108 MB (int8) |

The embedding is shared by both models (they read it via a table lookup). It ships **int8-quantized** — 5×
smaller than float32 (554 MB → 108 MB) and **accuracy-free** (POS UPOS 0.9364 → 0.9358, within noise).

## Speed & accuracy vs baselines

**POS tagging** (Universal Dependencies English-EWT test, UPOS):

| model | UPOS | tokens/s | model size |
|---|---|---|---|
| **interpretable-corenlp** | **0.9364** | 19,628 | 3.3 MB + shared W |
| spaCy `en_core_web_trf` (RoBERTa) | 0.9326 | 694 | 501 MB |
| spaCy `en_core_web_sm` (CNN) | 0.9140 | 13,589 | 15 MB |
| Stanford CoreNLP | 0.8675\* | 5,560 | 488 MB |

We **beat a fine-tuned RoBERTa on UPOS at ~28× its speed**, and beat the CNN on both. (\*CoreNLP is natively
PTB-tagged; the UPOS number reflects a lossy PTB→UPOS mapping that conflates AUX/VERB and ADP/SCONJ — its
native PTB accuracy is ~97%.)

**Sentence boundary detection** (sentence-level exact-match / boundary-F1):

| model | GUM (formal) | EWT (informal web) | model size |
|---|---|---|---|
| **interpretable-corenlp** | **0.848 / 0.926** | **0.634 / 0.818** | **0.3 MB** |
| Stanford CoreNLP `ssplit` | 0.832 / 0.915 | 0.595 / 0.793 | 488 MB |
| spaCy `sentencizer` | 0.835 / 0.911 | 0.621 / 0.808 | 15 MB |
| NLTK `punkt` | 0.803 / 0.892 | 0.574 / 0.777 | few MB |

A **0.3 MB glass-box segmenter beats Stanford CoreNLP and spaCy.** (Accuracy tracks how much punctuation the
text actually has; on the informal web half it is bounded by genuinely unpunctuated boundaries.)

## Install & use

```bash
git lfs install && git clone <this repo>     # the 108 MB embedding is tracked with git-lfs
pip install numpy scipy nltk                 # then: python -c "import nltk; nltk.download('wordnet')"
```

```python
from interpretable_corenlp import load_tagger, load_sbd

tagger = load_tagger("models/pos/model.npz", w_dir="models/embedding")
sbd    = load_sbd("models/sbd/sbd.npz", engram_npz="models/sbd/engram.npz")

tagger.tag("The model beats CoreNLP .".split())                   # -> ['DET','NOUN','VERB','PROPN','PUNCT']
sbd.segment_text("Dr. Smith joined in 2009. It works.", tagger)   # -> [(0,25),(26,35)] char spans
```

See `examples/run_pos.py`, `examples/run_sentence.py`, `examples/interpretability.py`.

## How it works (the framework)

A gradient-trained network is approximately a kernel machine; attention and depth are *an implementation* of a
kernel, not a requirement. These models build the kernel machine on purpose: a shallow, non-deep-learning
factorization of the PMI kernel (the embedding `W`) read by a **functional-ANOVA lattice** — a generalized
additive model whose parameters are decomposed over a lattice of named features, with
generalization-preserving (horseshoe-sparse) regularization. The result is piecewise-linear, modular, and
fully auditable. Details and theory are in the papers below.

## Data sources

All English. Models trained on:

- **Embedding `W`** — [C4](https://www.tensorflow.org/datasets/catalog/c4) (Colossal Clean Crawled Corpus;
  ODC-BY), as a quilted/sparse PPMI factorization.
- **POS tagger & sentence detector** — [Universal Dependencies](https://universaldependencies.org) English
  treebanks: **GUM** (CC BY-SA), **EWT**, **LinES**, **ParTUT**, **Atis** (training); evaluated on UD-EWT test
  and UD-GUM. Plus rule-augmented synthetic biomedical examples.
- **Named features** — [Princeton WordNet](https://wordnet.princeton.edu) (POS-set / supersense lattice axes)
  and NLTK person/place gazetteers.
- **Domain abbreviations** (the optional clinical-text segmentation aid) — learned unsupervised
  (Punkt LLR) from C4 + PubMed abstracts ([MedRAG/pubmed](https://huggingface.co/datasets/MedRAG/pubmed)).

Please respect the upstream licenses of these corpora.

## Citation

If you use these models or the framework, please cite both papers:

```bibtex
@article{interpretable_embedding,
  title = {An interpretable static word embedding via sparse quilted matrix factorization},
  note  = {arXiv:2605.05493},
  url   = {https://arxiv.org/abs/2605.05493}
}
@inproceedings{interpretable_lattice,
  title = {Glass-box lattice models},
  note  = {OpenReview},
  url   = {https://openreview.net/forum?id=D_KeYoqCYC}
}
```

## License

Code: see `LICENSE`. Models are released for research and practical use; the training corpora retain their
upstream licenses (see Data sources).
