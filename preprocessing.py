"""
Shared preprocessing transformers.
Both the training pipeline (Colab notebook) and the Flask app import these
classes from this exact module path, so joblib can unpickle them correctly.
Keep this file identical on both sides.
"""
import pandas as pd


class ZeroImputer:
    """Replace biologically-implausible zeros with a training-set median.

    Deliberately NOT conditioned on the Outcome label. Grouping the median
    by target class (as a naive version of this step often does) leaks the
    label into a feature at training time, and can't be reproduced at
    inference anyway -- a new patient's Outcome is exactly what you're
    trying to predict, so it's never available to condition on.
    """
    def __init__(self, cols):
        self.cols = cols
        self.medians_ = {}

    def fit(self, Xdf):
        for c in self.cols:
            valid = Xdf.loc[Xdf[c] != 0, c]
            self.medians_[c] = valid.median()
        return self

    def transform(self, Xdf):
        Xdf = Xdf.copy()
        for c in self.cols:
            Xdf.loc[Xdf[c] == 0, c] = self.medians_[c]
        return Xdf


class IQRCapper:
    """Cap outlier columns using training-set IQR fences (winsorizing)."""
    def __init__(self, cols, k=1.5):
        self.cols = cols
        self.k = k
        self.bounds_ = {}

    def fit(self, Xdf):
        for c in self.cols:
            q1, q3 = Xdf[c].quantile([0.25, 0.75])
            iqr = q3 - q1
            self.bounds_[c] = (q1 - self.k * iqr, q3 + self.k * iqr)
        return self

    def transform(self, Xdf):
        Xdf = Xdf.copy()
        for c, (lo, hi) in self.bounds_.items():
            Xdf[c] = Xdf[c].clip(lo, hi)
        return Xdf
