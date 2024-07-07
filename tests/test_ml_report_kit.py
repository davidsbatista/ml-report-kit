from src.ml_report_kit import MLReport

import random
import numpy as np


def generate_probabilities(pred_label, classes):
    """
    Function to generate probabilities
    # generate probabilities such the highest probability is the predicted class in y_label_pred
    # make sure the highest probability is not 1.0
    # make sure the sum of the probabilities is 1.0
    # go up to the 4th decimal place
    # store the probabilities in a list of lists

    :param pred_label:
    :param classes:
    :return:
    """
    # Initialize probabilities with small random values
    probs = np.random.rand(len(classes))
    probs /= probs.sum()  # Normalize to ensure sum is 1.0

    # Assign the highest probability to the predicted label
    pred_idx = classes.index(pred_label)
    max_prob = np.random.uniform(0.5, 0.9)  # Ensure the highest probability is not 1.0
    probs[pred_idx] = max_prob

    # Renormalize remaining probabilities to sum to 1 - max_prob
    other_indices = [i for i in range(len(classes)) if i != pred_idx]
    remaining_prob = 1 - max_prob
    remaining_probs = probs[other_indices]
    remaining_probs /= remaining_probs.sum()  # Normalize
    remaining_probs *= remaining_prob

    # Insert the remaining probabilities back
    probs[other_indices] = remaining_probs

    # Round to the 4th decimal place
    probs = np.round(probs, 4)

    # Ensure the sum is exactly 1.0 by adjusting the highest probability if needed
    probs[pred_idx] = 1.0 - np.round(probs[other_indices].sum(), 4)

    return probs.tolist()


def generate_random_labels(num_samples, classes):
    return [random.choice(classes) for _ in range(num_samples)]


def test_save_fold_info():
    class_names = ["class_a", "class_b", "class_c"]
    num_samples = 100
    y_true = generate_random_labels(num_samples, class_names)
    y_label_pred = generate_random_labels(num_samples, class_names)
    y_pred_probs = [generate_probabilities(pred, class_names) for pred in y_label_pred]

    out_path: str = "test"
    fold_nr: int = 1

    ml_report = MLReport(y_true, y_label_pred, y_pred_probs, class_names, out_path, fold_nr)
    ml_report.run()
