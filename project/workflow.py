from pathlib import Path
import re
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, ConfusionMatrixDisplay, PrecisionRecallDisplay
from threadpoolctl import threadpool_limits
from .data import load_data, obtain_data, partition
from .models import candidates, choose_threshold, metrics, explain
from .common import output_dirs, provenance, save_json, html_report, dataset_label

def perturb(text):
    substitutions = {"free": "fr33", "prize": "pr1ze", "claim": "cl@im", "winner": "w1nner"}
    return re.sub(r"\b(free|prize|claim|winner)\b",
                  lambda m: substitutions[m.group(0).lower()], str(text), flags=re.I)

@threadpool_limits.wrap(limits=1)
def train(mode="full", seed=42, data=None, output="outputs/full"):
    source = Path(data) if data else obtain_data()
    dataset = dataset_label(source, "UCI SMS Spam Collection", supplied=data is not None)
    frame, data_audit = load_data(source, mode)
    splits = partition(frame, seed)
    out = output_dirs(output)
    pd.concat([frame.iloc[ids][["source_row", "normalized"]].assign(split=name)
               for name, ids in splits.items()]).to_csv(out / "audit/split_registry.csv", index=False)
    tr, va, te = (splits[name] for name in ("train", "validation", "test"))
    fitted, thresholds, validation = {}, {}, []
    for name, model in candidates(seed).items():
        model.fit(frame.text.iloc[tr], frame.y.iloc[tr])
        p = model.predict_proba(frame.text.iloc[va])[:, 1]
        thresholds[name] = choose_threshold(frame.y.iloc[va], p)
        validation.append({"model": name, "validation_average_precision":
                           average_precision_score(frame.y.iloc[va], p),
                           "validation_spam_f1": metrics(frame.y.iloc[va], p, thresholds[name])["spam_f1"],
                           "threshold": thresholds[name]})
        fitted[name] = model
    leaderboard = pd.DataFrame(validation).sort_values(
        "validation_average_precision", ascending=False, kind="stable")
    selected = str(leaderboard.iloc[0].model)
    artifact = {"model": fitted[selected], "threshold": thresholds[selected],
                "selected": selected, "seed": seed, "training_data_source": dataset}
    joblib.dump(artifact, out / "models/model.joblib")
    rows = []
    for condition in ("clean", "keyword_spelling_perturbation"):
        texts = frame.text.iloc[te] if condition == "clean" else frame.text.iloc[te].map(perturb)
        for name, model in fitted.items():
            rows.append({"model": name, "condition": condition, "threshold": thresholds[name],
                         **metrics(frame.y.iloc[te], model.predict_proba(texts)[:, 1], thresholds[name])})
    measured = pd.DataFrame(rows)
    measured["data_source"] = dataset
    measured.to_csv(out / "reports/metrics.csv", index=False)
    leaderboard.to_csv(out / "reports/validation.csv", index=False)
    p = fitted[selected].predict_proba(frame.text.iloc[te])[:, 1]
    predictions = pd.DataFrame({"source_row": frame.source_row.iloc[te].to_numpy(),
                                "text": frame.text.iloc[te].to_numpy(),
                                "target": frame.y.iloc[te].to_numpy(),
                                "spam_probability": p,
                                "prediction": (p >= thresholds[selected]).astype(int)})
    predictions.to_csv(out / "reports/test_predictions.csv", index=False)
    predictions.loc[predictions.target != predictions.prediction].to_csv(
        out / "reports/error_analysis.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ConfusionMatrixDisplay.from_predictions(frame.y.iloc[te], p >= thresholds[selected],
                                            display_labels=["ham", "spam"], ax=axes[0],
                                            colorbar=False)
    PrecisionRecallDisplay.from_predictions(frame.y.iloc[te], p, ax=axes[1])
    if dataset.startswith("SIMULATED"):
        fig.suptitle("Simulated software demonstration")
    fig.tight_layout()
    fig.savefig(out / "reports/evaluation.png", dpi=150)
    plt.close(fig)
    record = provenance(mode, seed, source)
    record.update({"dataset": dataset,
                   "selected": selected, "threshold": thresholds[selected],
                   "data_audit": data_audit, "split_sizes": {k: len(v) for k, v in splits.items()},
                   "selection": "maximum validation average precision",
                   "threshold_selection": "maximum validation spam F1"})
    save_json(out / "audit/run.json", record)
    html_report(out / "reports/index.html", "Explainable SMS spam detection",
                f"Data: {dataset}. Selected model: {selected}. Normalized duplicate texts cannot cross splits. "
                "Model and threshold selection use validation only. The keyword spelling "
                "condition is a diagnostic perturbation of the same test messages.",
                [("Held-out test metrics", measured), ("Validation selection", leaderboard)],
                [("evaluation.png", "Selected model confusion matrix and precision-recall curve.")])
    print(measured.to_string(index=False))
    print(f"Report: {out / 'reports/index.html'}")
    return out

def predict(model_path, text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Provide a nonempty text message.")
    if len(text) > 10000:
        raise ValueError("The message exceeds the 10000-character input limit.")
    artifact = joblib.load(model_path)
    p = float(artifact["model"].predict_proba([text])[0, 1])
    return {"label": "spam" if p >= artifact["threshold"] else "ham",
            "spam_probability": p, "threshold": artifact["threshold"],
            "selected": artifact["selected"], "training_data_source": artifact["training_data_source"],
            "explanation": explain(artifact["model"], text)}
