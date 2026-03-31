"""
learned_model.py — MLP that predicts a per-chunk decay constant λ.

Architecture: chunk-level λ predictor.

The model takes features known at ingestion time and outputs λ,
which is then stored in Redis as the chunk's decay parameter.
At query time, the standard exponential formula e^(-λΔt) is used
unchanged — no ML inference at query time, no latency penalty.

Input features (9-dim):
    doc_type_onehot    [4]  — news | research | documentation | generic
    source_domain_onehot [3] — arxiv | github-docs | news-site (others=generic)
    text_length_norm   [1]  — len(text) / 2000, clipped to [0, 1]
    feedback_ratio     [1]  — used / (used + ignored), default 0.5 cold start

Output:
    λ ∈ [λ_min, λ_max] via sigmoid scaling
    λ_min = 1e-8  (~20 year half-life, landmark papers)
    λ_max = 1e-4  (~2 hour half-life, breaking news)

Training:
    Labels come from benchmark results — specifically, for each chunk
    we compute the λ that would have maximised nDCG on the queries
    that retrieved it. See train_mlp.py for the training loop.

Cold start:
    Before training, the model is initialised with weights that
    approximate the rule-based classifier's priors:
      news        → λ ≈ 1e-5
      research    → λ ≈ 1e-7
      documentation → λ ≈ 5e-7
      generic     → λ ≈ 1e-6
    This means an untrained model is no worse than the rule-based baseline.
"""

import math
import numpy as np

# λ range — defines the output space of the model.
# All λ values are clipped to this range after sigmoid scaling.
LAMBDA_MIN = 1e-8
LAMBDA_MAX = 1e-4

# Feature dimensions
DOC_TYPE_DIM    = 4    # news, research, documentation, generic
SOURCE_DIM      = 3    # arxiv, github-docs, news-site (other = [0,0,0])
TEXT_LEN_DIM    = 1
FEEDBACK_DIM    = 1
INPUT_DIM       = DOC_TYPE_DIM + SOURCE_DIM + TEXT_LEN_DIM + FEEDBACK_DIM  # = 9
HIDDEN_DIM      = 16

# Doc type and source domain index maps for one-hot encoding
DOC_TYPE_INDEX = {"news": 0, "research": 1, "documentation": 2, "generic": 3}
SOURCE_INDEX   = {"arxiv": 0, "github-docs": 1, "arxiv-news": 2}


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _scale_lambda(sigmoid_output: float) -> float:
    """
    Maps sigmoid output ∈ (0, 1) to λ ∈ [LAMBDA_MIN, LAMBDA_MAX].
    Uses log-space interpolation so the model operates on the natural
    scale of λ — a unit change in the middle of log space is the same
    as a unit change at the extremes.
    """
    log_min = math.log10(LAMBDA_MIN)
    log_max = math.log10(LAMBDA_MAX)
    log_lambda = log_min + sigmoid_output * (log_max - log_min)
    return 10 ** log_lambda


class DecayMLP:
    """
    Minimal 2-layer MLP implemented in pure NumPy.

    No PyTorch dependency at inference time — the model runs everywhere.
    Training uses PyTorch (see train_mlp.py) and exports weights as
    numpy arrays that this class loads directly.

    Architecture:
        input (9) → Linear → ReLU → hidden (16) → Linear → Sigmoid → output (1)

    The output is passed through _scale_lambda() to get λ.
    """

    def __init__(self):
        # Weights initialised to approximate rule-based priors (cold start).
        # W1: (INPUT_DIM, HIDDEN_DIM), b1: (HIDDEN_DIM,)
        # W2: (HIDDEN_DIM, 1),         b2: (1,)
        self.W1 = np.zeros((INPUT_DIM, HIDDEN_DIM), dtype=np.float32)
        self.b1 = np.zeros(HIDDEN_DIM, dtype=np.float32)
        self.W2 = np.zeros((HIDDEN_DIM, 1), dtype=np.float32)
        self.b2 = np.zeros(1, dtype=np.float32)

        self._init_cold_start_weights()

    def _init_cold_start_weights(self) -> None:
        """
        Initialise weights so the untrained model approximates the
        rule-based classifier's λ priors.

        Target mapping (sigmoid output → λ):
          news          → λ=1e-5 → sigmoid output ≈ 0.75
          research      → λ=1e-7 → sigmoid output ≈ 0.25
          documentation → λ=5e-7 → sigmoid output ≈ 0.375
          generic       → λ=1e-6 → sigmoid output ≈ 0.50

        We set W2 and b2 so the final layer produces these outputs
        from a near-zero hidden layer (W1 stays near zero initially).
        The bias b2 acts as the "default" output (generic = 0.5).
        """
        # b2 = logit(0.5) = 0.0 → generic λ = 1e-6 by default
        self.b2[0] = 0.0

        # Nudge the first 4 hidden units to respond to doc_type one-hot:
        # unit 0 → news (+0.75 target), unit 1 → research (-0.25 target)
        # These are soft priors — training will override them.
        for i in range(min(4, HIDDEN_DIM)):
            self.W1[i, i] = 0.5    # doc_type input i → hidden unit i
            self.b1[i]    = -0.25  # slight negative bias

        target_outputs = [0.75, 0.25, 0.375, 0.50]  # news, research, docs, generic
        for i, target in enumerate(target_outputs):
            if i < HIDDEN_DIM:
                # W2[i] set so sigmoid(W2[i] * relu(0.25) + 0) ≈ target
                self.W2[i, 0] = (math.log(target / (1 - target))) / 0.25

    def forward(self, x: np.ndarray) -> float:
        """
        Forward pass. x: (INPUT_DIM,) float32 array.
        Returns λ as a float.
        """
        h = np.maximum(0, x @ self.W1 + self.b1)          # (HIDDEN,)
        out = _sigmoid(float((h @ self.W2).squeeze() + self.b2[0]))
        return _scale_lambda(out)

    def load_weights(self, path: str) -> None:
        """Load weights exported by train_mlp.py."""
        data = np.load(path)
        self.W1 = data["W1"]
        self.b1 = data["b1"]
        self.W2 = data["W2"]
        self.b2 = data["b2"]

    def save_weights(self, path: str) -> None:
        """Save weights for use by DecayMLP.load_weights()."""
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)


# --------------------------------------------------------------------------- #
#  Feature extraction                                                          #
# --------------------------------------------------------------------------- #

def extract_features(
    doc_type:      str,
    source_domain: str,
    text:          str,
    feedback_used:    int = 0,
    feedback_ignored: int = 0,
) -> np.ndarray:
    """
    Converts chunk metadata into the 9-dim input vector for DecayMLP.

    Args:
        doc_type:         "news" | "research" | "documentation" | "generic"
        source_domain:    e.g. "arxiv", "github-docs", "arxiv-news"
        text:             raw chunk text (used for length feature)
        feedback_used:    number of times chunk was used (from Redis)
        feedback_ignored: number of times chunk was ignored (from Redis)

    Returns:
        np.ndarray shape (INPUT_DIM,) dtype float32
    """
    features = np.zeros(INPUT_DIM, dtype=np.float32)

    # doc_type one-hot [0:4]
    dt_idx = DOC_TYPE_INDEX.get(doc_type, DOC_TYPE_INDEX["generic"])
    features[dt_idx] = 1.0

    # source_domain one-hot [4:7]
    src_idx = SOURCE_INDEX.get(source_domain)
    if src_idx is not None:
        features[DOC_TYPE_DIM + src_idx] = 1.0

    # text length, normalised [7]
    features[DOC_TYPE_DIM + SOURCE_DIM] = min(len(text) / 2000.0, 1.0)

    # feedback ratio [8]: used / total, default 0.5 at cold start
    total = feedback_used + feedback_ignored
    features[DOC_TYPE_DIM + SOURCE_DIM + TEXT_LEN_DIM] = (
        feedback_used / total if total > 0 else 0.5
    )

    return features


# --------------------------------------------------------------------------- #
#  Singleton engine                                                            #
# --------------------------------------------------------------------------- #

class LearnedDecayEngine:
    """
    Singleton wrapper around DecayMLP.
    Loaded once at import time. Use predict_lambda() everywhere.
    """

    def __init__(self):
        self.model = DecayMLP()
        self._weights_loaded = False

    def try_load_weights(self, path: str = "decay_mlp.npz") -> bool:
        """
        Attempts to load trained weights. Falls back to cold-start
        priors silently if the file doesn't exist yet.
        Returns True if weights were loaded.
        """
        import os
        if os.path.exists(path):
            self.model.load_weights(path)
            self._weights_loaded = True
            return True
        return False

    def predict_lambda(
        self,
        doc_type:      str,
        source_domain: str,
        text:          str,
        feedback_used:    int = 0,
        feedback_ignored: int = 0,
    ) -> float:
        """
        Predicts the decay constant λ for a chunk.
        Called at ingestion time. Result stored in Redis.
        """
        features = extract_features(
            doc_type, source_domain, text,
            feedback_used, feedback_ignored,
        )
        return self.model.forward(features)


# Global singleton — imported by learned.py and pipeline.py
LEARNED_ENGINE = LearnedDecayEngine()
# Silently try to load trained weights if they exist
LEARNED_ENGINE.try_load_weights()