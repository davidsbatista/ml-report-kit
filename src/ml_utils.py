from inspect import signature
from pathlib import Path
from typing import List, Union, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, classification_report, confusion_matrix

base_dir = "plots/"




def plot_precision_recall_vs_threshold(precisions, recalls, thresholds, filename):
    plt.figure(figsize=(10, 8))
    plt.title("Precision and Recall Scores as a function of the decision threshold")
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")

    # loc = plticker.MultipleLocator(base=1.0)  # this locator puts ticks at regular intervals
    # plt.xaxis.set_major_locator(loc)

    plt.grid()
    plt.ylabel("Score")
    plt.xlabel("Decision Threshold")
    plt.legend(loc="best")

    with open(base_dir + filename + ".png", "wb") as f_out:
        plt.savefig(f_out, bbox_inches="tight")
    plt.close()


def plot_precision_recall_curve(predictions_prob, test_y, pos_label, filename):
    precision, recall, thresholds = precision_recall_curve(test_y, predictions_prob[:, 1:], pos_label=pos_label)

    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument

    step_kwargs = {"step": "post"} if "step" in signature(plt.fill_between).parameters else {}
    plt.step(recall, precision, color="b", alpha=0.2, where="post")
    plt.fill_between(recall, precision, alpha=0.2, color="b", **step_kwargs)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    # plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
    with open(base_dir + filename + ".png", "wb") as f_out:
        plt.savefig(f_out)
    plt.close()

    return precision, recall, thresholds




def save_fold_info(
    fold_nr: str,
    y_pred_probs: List[List[float]],
    y_label_pred: List,
    y_true: List,
    class_names: List[str],
    out_path: Path,
    y_id: Optional[List] = None
) -> None:
    """
    ToDo: optionally save to S3

    For a given fold inside a cross-validation training, reports the following:

    - A classification report for each class
    - Precision and Recall vs threshold plot for a specified class
    - Saves .CSV files with the precision, recall, thresholds used to generate the plot
    - Saves .CSV files with the true_y and predicted probabilities for each class

    The class for each we want to plot the Precision and Recall vs threshold is given by two parameters:

    - 'idx_pos_label': inside the y_pred_prob List there are the predictions scores for each class,
                       this parameter indicates the idx of the class we are interested in
    - 'pos_label': the label of the positive class

    :param fold_nr: the fold number
    :param y_pred_probs: each element in the List is another List with probabilities for each class
    :param y_label_pred: the predicted label for each sample
    :param y_true:: the true label for each sample
    :param y_id:: the id of each sample.
    :param class_names::
    :param out_path: the output Path where to store the results
    """

    # classification report
    report = classification_report(y_true, y_label_pred, zero_division=0.0)
    out_path.mkdir(parents=True, exist_ok=True)
    out_file = out_path.joinpath(Path(f"classification_report_{fold_nr}.txt"))
    with open(out_file, "wt", encoding="utf8") as f:
        print(report, file=f)

    # confusion matrix
    labels = sorted(list(set(y_true)))
    cm = confusion_matrix(y_true, y_label_pred, labels=labels)
    print_cm(cm, labels, out_path)

    # precision and recall as a function of threshold value
    for idx in range(0, len(y_pred_probs[0])):
        pos_class_prob = [y_prob_pred[idx] for y_prob_pred in y_pred_probs]
        tmp = np.array([1 if sample == class_names[idx] else 0 for sample in y_true])
        precisions, recalls, thresholds = precision_recall_curve(tmp, np.array(pos_class_prob))

        # save precisions, recalls, thresholds - allows for threshold tuning based on precision and/or recall
        df = pd.DataFrame(list(zip(precisions, recalls, thresholds)), columns=["precision", "recall", "thresholds"])
        f_out = out_path.joinpath(Path(f"precision_recall_threshold_{class_names[idx]}.csv"))
        df.to_csv(f_out, index=False)

        # generate the precision-recall vs threshold plot
        f_out = out_path.joinpath(Path(f"precision_recall_threshold_{class_names[idx]}.png"))
        plot_curve(thresholds, precisions, recalls, f_out)

    preds = np.array(y_pred_probs)
    if y_id:
        np.hstack(np.array(y_id, y_true, preds))
    else:
        preds_array = np.array(y_true).reshape(len(y_true), 1)
        labels_array = np.array(y_label_pred).reshape(len(y_true), 1)
        data = np.hstack((preds_array, labels_array, preds))
        labels = ["true_y", "pred_label"] + class_names

        df = pd.DataFrame(data, columns=labels)
        df.to_csv(out_path.joinpath(Path(f"prediction_scores_{fold_nr}.csv", index=False)))
