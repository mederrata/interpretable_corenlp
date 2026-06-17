# Intrinsically interpretable core NLP models

**These models share a static word embedding that we are sharing using git lfs (for now, until it becomes cost prohibitive). 
Each model is an intriniscally interpretable *glass box*: every prediction is an **exact sum of named
feature contributions** you can read off (`explain()`). To reiterate, the explainer is **exact** and is NOT not a post-hoc approximation. 
The accuracy of these models is competitive with transformers-based counterparts while offering orders of magnitude efficiency gains.

> **English only.** All models are trained and evaluated on English. We'll work on other languages eventually, with your support.

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

| model | GUM (formal) | EWT (informal web) | speed (chars/s) | model size |
|---|---|---|---|---|
| **interpretable-corenlp** | **0.848 / 0.926** | **0.634 / 0.818** | 116k | **0.3 MB** |
| Stanford CoreNLP `ssplit` | 0.832 / 0.915 | 0.595 / 0.793 | 132k | 488 MB |
| spaCy `sentencizer` | 0.835 / 0.911 | 0.621 / 0.808 | 101k | 15 MB |
| NLTK `punkt` | 0.803 / 0.892 | 0.574 / 0.777 | 14.6M | few MB |

A **0.3 MB glass-box segmenter beats Stanford CoreNLP and spaCy** at comparable speed. Our segmenter runs a
POS pass (the cost), so it is in the same speed class as CoreNLP/spaCy and ~1500× smaller; NLTK `punkt` is far
faster because it is pure rules with no tagging, but it is the least accurate. (Accuracy tracks how much
punctuation the text actually has; on the informal web half it is bounded by genuinely unpunctuated boundaries.)

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

Neural networks are spline regression models. All regression models are kernel machines. The [Bayesianquilts framework](https://github.com/mederrata/bayesianquilts) provides a set of techniques to adapt hierarchical mixed effects regression to the same task with the explicit constraint of being interpretable. This framework is anti- deep learning and in particular anti- transformers.
As a bonus, models fitted with this technique have very quick forward computation (so-called inference - side note, inference means learning something from data, why computing a forward pass of a neural network is called inference is bewildering).

## Data sources

All models are English-only for now. Since all regression models are kernel machines, knowing the data that went into learning them can inform on any blind spots. These datasets are:

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

@misc{chang2026renormalizationgroupinspiredlatticebasedframework,
      title={A renormalization-group inspired lattice-based framework for piecewise generalized linear models}, 
      author={Joshua C. Chang},
      year={2026},
      eprint={2605.05493},
      archivePrefix={arXiv},
      primaryClass={stat.ME},
      url={https://arxiv.org/abs/2605.05493}, 
}

@article{chang2024interpretable,
  title={Interpretable (not just posthoc-explainable) medical claims modeling for discharge placement to reduce preventable all-cause readmissions or death},
  author={Chang, Ted L and Xia, Hongjing and Mahajan, Sonya and Mahajan, Rohit and Maisog, Jose and others},
  journal={PLOS ONE},
  volume={19},
  number={5},
  pages={e0302871},
  year={2024},
  publisher={Public Library of Science}
}

@article{chang2024gradient,
  title={Gradient-flow adaptive importance sampling for Bayesian leave one out cross-validation with application to sigmoidal classification models},
  author={Chang, Joshua C and Li, Xu and Xu, Shuang and Yao, Howard R and Porcino, John and Chow, Carson C},
  journal={arXiv preprint arXiv:2402.08151},
  year={2024}
}
```

## License

Code: see `LICENSE`. Models are released for research and practical use; the training corpora retain their
upstream licenses (see Data sources).
