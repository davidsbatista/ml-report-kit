from ml_report_kit import MLReport

import csv
import random
import numpy as np


def generate_probabilities(pred_label, classes):
    """
    Function to generate probabilities.

    Generate probabilities such the highest probability is the predicted class in pred_label.
    The highest probability is not 1.0 and the sum of the probabilities is 1.0

    :param pred_label: The predicted class
    :param classes: List of classes
    :return:
    """
    # Initialize probabilities with small random values
    probs = np.random.rand(len(classes))
    probs /= probs.sum()  # Normalize to ensure sum is 1.0

    # Assign the highest probability to the predicted label
    pred_idx = classes.index(pred_label)
    max_prob = np.random.uniform(0.5, 0.9)  # Ensure the highest probability is not 1.0
    probs[pred_idx] = max_prob

    # re-normalize remaining probabilities to sum to 1 - max_prob
    other_indices = [i for i in range(len(classes)) if i != pred_idx]
    remaining_prob = 1 - max_prob
    remaining_probs = probs[other_indices]
    remaining_probs /= remaining_probs.sum()  # Normalize
    remaining_probs *= remaining_prob

    # insert the remaining probabilities back
    probs[other_indices] = remaining_probs

    # round to the 4th decimal place
    probs = np.round(probs, 4)

    # ensure the sum is exactly 1.0 by adjusting the highest probability if needed
    probs[pred_idx] = 1.0 - np.round(probs[other_indices].sum(), 4)

    return probs.tolist()


def generate_random_labels(num_samples, classes):
    return [random.choice(classes) for _ in range(num_samples)]


def make_report(class_names, y_id=None, present=None, num_samples=100):
    """A report over random predictions, optionally drawing labels from a subset of the classes."""
    labels_seen = present if present is not None else class_names
    y_true = generate_random_labels(num_samples, labels_seen)
    y_label_pred = generate_random_labels(num_samples, labels_seen)
    y_pred_probs = [generate_probabilities(pred, class_names) for pred in y_label_pred]
    if y_id is True:
        y_id = [f"sample_{i}" for i in range(num_samples)]
    return MLReport(y_true, y_label_pred, y_pred_probs, class_names, y_id=y_id)


def test_every_report_is_written(tmp_path):
    class_names = ["class_a", "class_b", "class_c"]
    make_report(class_names).run(str(tmp_path), fold_nr=1)

    written = {f.name for f in (tmp_path / "fold_1").iterdir()}
    expected = {
        "classification_report.txt",
        "confusion_matrix.png",
        "confusion_matrix.txt",
        "prediction_scores.csv",
    }
    for name in class_names:
        expected |= {f"precision_recall_threshold_{name}.csv", f"precision_recall_threshold_{name}.png"}
    assert written == expected


def test_prediction_scores_are_written_with_one_row_per_sample(tmp_path):
    class_names = ["class_a", "class_b", "class_c"]
    make_report(class_names, num_samples=20).run(str(tmp_path), fold_nr=1)

    rows = list(csv.DictReader((tmp_path / "fold_1" / "prediction_scores.csv").open(encoding="utf8")))
    assert len(rows) == 20
    # no leading index column, and one probability column per class
    assert list(rows[0]) == ["true_y", "pred_label"] + class_names


def test_prediction_scores_carry_the_id_when_one_is_given(tmp_path):
    """Without this column the predictions cannot be joined back to what produced them."""
    class_names = ["class_a", "class_b", "class_c"]
    make_report(class_names, y_id=True, num_samples=20).run(str(tmp_path), fold_nr=1)

    rows = list(csv.DictReader((tmp_path / "fold_1" / "prediction_scores.csv").open(encoding="utf8")))
    assert list(rows[0]) == ["id", "true_y", "pred_label"] + class_names
    assert [row["id"] for row in rows] == [f"sample_{i}" for i in range(20)]


def test_a_fold_missing_one_of_the_classes_still_reports(tmp_path):
    """Cross-validation over rare classes produces folds where one of them never shows up."""
    class_names = ["class_a", "class_b", "class_c", "rare_class"]
    make_report(class_names, present=["class_a", "class_b", "class_c"]).run(str(tmp_path), fold_nr=1)

    assert (tmp_path / "fold_1" / "confusion_matrix.png").exists()
    assert (tmp_path / "fold_1" / "prediction_scores.csv").exists()


def test_final_report_goes_to_its_own_directory(tmp_path):
    class_names = ["class_a", "class_b", "class_c"]
    make_report(class_names).run(str(tmp_path), final_report=True)

    assert (tmp_path / "final_report" / "classification_report.txt").exists()
