"""Unfitted estimator factory. Pass raw development inputs to five-fold CV."""
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


def make_pipeline():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('logistic', LogisticRegression(max_iter=2000, random_state=42)),
    ])


def make_cv():
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
