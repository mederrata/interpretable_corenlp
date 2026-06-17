"""The frozen static embedding W (interpretable, glass-box). Public release is int8-quantized (W_int8.npz):
300,000 words x 512 dims, ~113 MB (vs 554 MB float32), accuracy-free (POS UPOS 0.9364 -> 0.9358).
Embeddings ARE matrix factorization (Levy & Goldberg); this W is a sparse/quilted PPMI factorization with
NAMED, inspectable dimensions — the shared representation every model here reads via a table lookup."""
import os, json
import numpy as np


def load_embedding(w_dir):
    """-> (W float32 [V,512], vocab {word: row}). Reads int8 W_int8.npz if present, else float32 W.npz."""
    p8 = os.path.join(w_dir, "W_int8.npz")
    if os.path.exists(p8):
        z = np.load(p8); W = z["q"].astype(np.float32) * float(z["scale"])
    else:
        import scipy.sparse as sp
        W = sp.load_npz(os.path.join(w_dir, "W.npz")).toarray().astype(np.float32)
    vocab = json.load(open(os.path.join(w_dir, "vocab.json")))
    return W, {(k.split("=", 1)[-1] if "=" in k else k): i for k, i in vocab.items()}
