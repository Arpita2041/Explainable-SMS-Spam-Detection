"""Sparse text features, linear classifiers, and exact feature contributions."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score, log_loss)
from .data import normalize

def candidates(seed=42):
    def word():
        return TfidfVectorizer(preprocessor=normalize, lowercase=False, ngram_range=(1, 2),
                               min_df=2, max_df=.98, sublinear_tf=True, max_features=20000)
    return {
        "prevalence_baseline": Pipeline([("features", word()),
                                         ("classifier", DummyClassifier(strategy="prior"))]),
        "word_naive_bayes": Pipeline([("features", word()),
                                      ("classifier", MultinomialNB(alpha=.5))]),
        "word_logistic": Pipeline([("features", word()),
                                   ("classifier", LogisticRegression(C=3, max_iter=1500,
                                                                     random_state=seed))]),
        "character_logistic": Pipeline([
            ("features", TfidfVectorizer(preprocessor=normalize, lowercase=False,
                                        analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                                        sublinear_tf=True, max_features=30000)),
            ("classifier", LogisticRegression(C=5, max_iter=1500, random_state=seed))])}

def choose_threshold(y, p):
    choices = np.unique(np.r_[.5, np.quantile(p, np.linspace(0, 1, 201))])
    scores = np.array([f1_score(y, p >= t, zero_division=0) for t in choices])
    tied = np.flatnonzero(np.isclose(scores, scores.max()))
    return float(choices[tied[np.argmin(np.abs(choices[tied] - .5))]])

def metrics(y, p, threshold):
    prediction = p >= threshold
    return {"accuracy": accuracy_score(y, prediction),
            "spam_f1": f1_score(y, prediction, zero_division=0),
            "macro_f1": f1_score(y, prediction, average="macro", zero_division=0),
            "spam_precision": precision_score(y, prediction, zero_division=0),
            "spam_recall": recall_score(y, prediction, zero_division=0),
            "average_precision": average_precision_score(y, p),
            "roc_auc": roc_auc_score(y, p), "log_loss": log_loss(y, p, labels=[0, 1])}

def explain(model, text, top_k=8):
    features, classifier = model.named_steps["features"], model.named_steps["classifier"]
    x = features.transform([text])
    if hasattr(classifier, "coef_"):
        coefficient = classifier.coef_[0]
        intercept = float(classifier.intercept_[0])
    elif hasattr(classifier, "feature_log_prob_"):
        coefficient = classifier.feature_log_prob_[1] - classifier.feature_log_prob_[0]
        intercept = float(classifier.class_log_prior_[1] - classifier.class_log_prior_[0])
    else:
        return {"features": [], "note": "This prevalence baseline has no text feature weights."}
    contributions = x.data * coefficient[x.indices]
    names = features.get_feature_names_out()
    order = np.argsort(-np.abs(contributions))[:top_k]
    return {"intercept": intercept, "log_odds": float(intercept + contributions.sum()),
            "features": [{"feature": str(names[x.indices[i]]),
                          "contribution": float(contributions[i])} for i in order],
            "meaning": "Positive contributions increase spam log-odds; negative ones decrease them."}
