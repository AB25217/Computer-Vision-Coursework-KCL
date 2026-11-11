"""
Pseudo-code showing how your `run_model.py` will be evaluated. Use it to structure your own script.

Your submission is a ZIP file containing:
    run_model.py            <-----This script
    trained_model.pkl       <-----The trained model (format does not have to be pkl)
    feature_extractor.py    <-----Your feature extractor, called from run_model.py

We will run:
    python run_model.py IMAGE_DIR GROUND_TRUTH_CSV

where IMAGE_DIR is a directory of unlabeled images and GROUND_TRUTH_CSV is a headerless CSV of
`uuid,numeric_label` rows (the uuid is the image filename without its extension). The script must:
1. extract features from every image in IMAGE_DIR into a feature matrix,
2. run your trained model on the feature matrix,
3. print the macro-averaged F1 score and the total runtime to the command line.

You can change just about anything in this pseudocode EXCEPT for the command line input structure, the computation of the F1 score, the timer and the print statements. 

The whole script should run in under 5 minutes on CPU only. Paths to your own files (such as the
model) must be relative to this script, not to the directory it is run from.
"""

# Add other import commands as needed. Do not install additional dependencies.
import argparse
import csv
import time
from pathlib import Path

import joblib
from sklearn.metrics import f1_score

# Your module. It must provide `extract_features(image_paths)`, which returns a 2D array with one
# row of features per image, in the same order as `image_paths`.
from feature_extractor import extract_features

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "trained_model.pkl"


def list_images(image_dir):
    # Every file in the directory, sorted by filename.
    return sorted(f for f in Path(image_dir).iterdir() if f.is_file())


def read_ground_truth(path):
    # Returns {uuid: numeric_label} from a headerless `uuid,numeric_label` CSV.
    with open(path, newline="") as f:
        return {uuid: int(label) for uuid, label in csv.reader(f)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_dir", help="directory of images to classify")
    parser.add_argument("ground_truth", help="CSV of true `uuid,numeric_label` rows")
    args = parser.parse_args()

    start = time.perf_counter()

    # 1. Feature matrix: one row per image.
    image_paths = list_images(args.image_dir)
    features = extract_features(image_paths)

    # 2. Classifier: load the trained model and predict a numeric label for each row.
    model = joblib.load(MODEL_PATH)
    predicted_labels = model.predict(features)

    # 3. Score against the ground truth, matching images by uuid.
    truth = read_ground_truth(args.ground_truth)
    true_labels = [truth[path.stem] for path in image_paths]
    print(f"macro F1: {f1_score(true_labels, predicted_labels, average='macro'):.4f}")

    print(f"runtime: {time.perf_counter() - start:.1f} s")


if __name__ == "__main__":
    main()
