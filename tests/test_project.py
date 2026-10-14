import tempfile
import unittest
from pathlib import Path
import numpy as np
from project.data import normalize, load_data, partition
from project.models import candidates, explain
from project.workflow import train, predict

def fixture(n=360):
    """Original simulated messages for software tests only."""
    rows = []
    for i in range(n):
        if i % 3 == 0:
            rows.append(f"spam\tClaim your free prize today with winner reference {i}")
        elif i % 3 == 1:
            rows.append(f"ham\tPlease bring the meeting notes to room {i} tomorrow")
        else:
            rows.append(f"ham\tAre we still meeting for dinner at table {i} tonight")
    return "\n".join(rows) + "\n"

class TextTests(unittest.TestCase):
    def test_normalization_and_duplicate_isolation(self):
        self.assertEqual(normalize("  MEETING \t Tomorrow "), normalize("meeting tomorrow"))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "messages"
            path.write_text(fixture() + "ham\tPlease bring the meeting notes to room 1 tomorrow\n")
            frame, audit = load_data(path)
            self.assertEqual(audit["duplicates_removed"], 1)
            splits = partition(frame)
            seen = set()
            for ids in splits.values():
                keys = set(frame.normalized.iloc[ids])
                self.assertFalse(seen & keys)
                seen |= keys

    def test_conflicting_labels_removed(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "messages"
            path.write_text(fixture() + "ham\tClaim your free prize today with winner reference 0\n")
            frame, audit = load_data(path)
            self.assertEqual(audit["conflicting_label_rows_removed"], 2)
            self.assertNotIn(normalize("Claim your free prize today with winner reference 0"),
                             set(frame.normalized))

    def test_training_vocabulary_and_exact_explanations(self):
        messages = ["free prize claim today", "free prize offer today", "meeting notes tomorrow",
                    "meeting notes today", "claim your free prize", "read the meeting notes"] * 3
        labels = [1, 1, 0, 0, 1, 0] * 3
        for name in ("word_naive_bayes", "word_logistic", "character_logistic"):
            model = candidates()[name].fit(messages, labels)
            model.predict_proba(["neverobservedvalidationtoken"])
            self.assertNotIn("neverobservedvalidationtoken", model.named_steps["features"].vocabulary_)
            text = "claim free prize today"
            probability = model.predict_proba([text])[0, 1]
            expected = np.log(probability / (1 - probability))
            self.assertAlmostEqual(explain(model, text)["log_odds"], expected, places=5)

    def test_complete_fixture_run_and_prediction(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, output = Path(temporary) / "messages", Path(temporary) / "run"
            path.write_text(fixture(), encoding="utf-8")
            train("quick", 42, path, output)
            result = predict(output / "models/model.joblib", "Please bring the meeting notes")
            self.assertIn(result["label"], ("ham", "spam"))
            self.assertTrue(0 <= result["spam_probability"] <= 1)
            self.assertTrue((output / "reports/error_analysis.csv").exists())
            with self.assertRaises(ValueError):
                predict(output / "models/model.joblib", "")

if __name__ == "__main__":
    unittest.main()
