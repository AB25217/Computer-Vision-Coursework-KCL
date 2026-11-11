"""
The scoring script the teaching staff will use to compute the score on the test data. The script loads in your `test_set_predictions.csv` and compares it against the hidden ground truth. It computes the macro-averaged F1 score to give a snapshot of precision and recall in a single numeric indicator.

Both files are headerless CSVs of `uuid,numeric_label` rows. The rows of the predictions must be in the same order as the ground truth (sorted by uuid).
"""

import argparse
import csv
from pathlib import Path

from sklearn.metrics import f1_score

REPO_DIR = Path(__file__).resolve().parent.parent
GROUND_TRUTH = REPO_DIR / "test_ground_truth.csv"


def read_labelled_rows(path: Path) -> tuple[list[str], list[int]]:
    with path.open(newline="") as f:
        rows = [(uuid, int(label)) for uuid, label in csv.reader(f)]
    uuids, labels = zip(*rows)
    return list(uuids), list(labels)


def check_same_uuids(truth_uuids: list[str], predicted_uuids: list[str]) -> None:
    if len(truth_uuids) != len(predicted_uuids):
        raise ValueError(f"expected {len(truth_uuids)} predictions, found {len(predicted_uuids)}")
    for i, (expected, found) in enumerate(zip(truth_uuids, predicted_uuids)):
        if expected != found:
            raise ValueError(f"row {i + 1}: expected uuid {expected}, found {found}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path, help="CSV of predicted `uuid,numeric_label` rows")
    parser.add_argument("--ground-truth", type=Path, default=GROUND_TRUTH)
    args = parser.parse_args()

    truth_uuids, truth_labels = read_labelled_rows(args.ground_truth)
    predicted_uuids, predicted_labels = read_labelled_rows(args.predictions)
    check_same_uuids(truth_uuids, predicted_uuids)

    print(f"macro F1: {f1_score(truth_labels, predicted_labels, average='macro'):.4f}")


if __name__ == "__main__":
    main()
