from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
from numpy import hstack, array
from pandas import DataFrame
from sklearn.metrics import classification_report, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay


class MLReport:
    def __init__(
        self,
        y_true: List[str],
        y_label_pred: List[str],
        y_pred_probs: List[List[float]],
        class_names: List[str],
        y_id: Optional[List] = None,
    ):
        self.y_true = y_true
        self.y_label_pred = y_label_pred
        self.y_pred_probs = y_pred_probs
        self.class_names = class_names
        self.y_id = y_id

    def plot_confusion_matrix(self, cm, out_path: Path, xticks_rotation='vertical'):
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.class_names)
        final_out = out_path.joinpath('confusion_matrix.png')
        fig, ax = plt.subplots(figsize=(10, 10))
        disp.plot(xticks_rotation=xticks_rotation, ax=ax).figure_.savefig(str(final_out))

    @staticmethod
    def print_cm(
        cm,
        labels,
        print_reports,
        out_path: Path,
        hide_zeroes=False,
        hide_diagonal=False,
        hide_threshold=None,
        save_to_file: bool = True,
    ):
        """
        Pretty print for confusion matrices.

        Taken from here: https://gist.github.com/zachguo/10296432
        """

        if print_reports:
            column_width = max([len(x) for x in labels] + [5])  # 5 is value length
            empty_cell = " " * column_width

            # Begin CHANGES
            fst_empty_cell = (column_width - 3) // 2 * " " + "t/p" + (column_width - 3) // 2 * " "

            if len(fst_empty_cell) < len(empty_cell):
                fst_empty_cell = " " * (len(empty_cell) - len(fst_empty_cell)) + fst_empty_cell
            # Print header
            print("    " + fst_empty_cell, end=" ")
            # End CHANGES

            for label in labels:
                print("%{0}s".format(column_width) % label, end=" ")

            print()
            # Print rows
            for i, label1 in enumerate(labels):
                print("    %{0}s".format(column_width) % label1, end=" ")
                for j in range(len(labels)):
                    cell = "%{0}.1f".format(column_width) % cm[i, j]
                    if hide_zeroes:
                        cell = cell if float(cm[i, j]) != 0 else empty_cell
                    if hide_diagonal:
                        cell = cell if i != j else empty_cell
                    if hide_threshold:
                        cell = cell if cm[i, j] > hide_threshold else empty_cell
                    print(cell, end=" ")
                print()

        if save_to_file:
            # ToDo: this can be improved, remove the duplication of code
            # same as above, but output goes to a text file instead of stdout
            column_width = max([len(x) for x in labels] + [5])  # 5 is value length
            empty_cell = " " * column_width
            fst_empty_cell = (column_width - 3) // 2 * " " + "t/p" + (column_width - 3) // 2 * " "
            out_file = out_path.joinpath(Path("confusion_matrix.txt"))
            with open(out_file, "wt") as f:
                f.write("    " + fst_empty_cell + " ")
                for label in labels:
                    f.write("%{0}s".format(column_width) % label + " ")

                f.write("\n")
                for i, label1 in enumerate(labels):
                    f.write("    %{0}s".format(column_width) % label1 + " ")
                    for j in range(len(labels)):
                        cell = "%{0}.1f".format(column_width) % cm[i, j]
                        if hide_zeroes:
                            cell = cell if float(cm[i, j]) != 0 else empty_cell
                        if hide_diagonal:
                            cell = cell if i != j else empty_cell
                        if hide_threshold:
                            cell = cell if cm[i, j] > hide_threshold else empty_cell
                        f.write(cell + " ")
                    f.write("\n")

    @staticmethod
    def plot_curve(thresholds, precisions, recalls, f_out: Path) -> None:
        """
        Plots the precision-recall curve for a given classifier.

        :param thresholds: the thresholds used to generate the precision and recall scores
        :param precisions: the precision scores
        :param recalls: the recall scores
        :param f_out: the output file where to save the plot
        """

        plt.figure(figsize=(10, 8))
        plt.title("Precision and Recall Scores as a function of the decision threshold")
        plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
        plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
        plt.grid()
        plt.ylabel("Score")
        plt.xlabel("Decision Threshold")
        plt.legend(loc="best")
        plt.savefig(f_out, bbox_inches="tight")
        plt.close()

    def run(self, results_path: str, fold_nr: int = 0, print_reports=False):
        """
        For a given fold inside a cross-validation training, reports the following:

        - A classification report for each class
        - Precision and Recall vs threshold plot for a specified class
        - Saves .CSV files with the precision, recall, thresholds used to generate the plot
        - Saves .CSV files with the true_y and predicted probabilities for each class

        The class for each we want to plot the Precision and Recall vs threshold is given by two parameters:

        - 'idx_pos_label': inside the y_pred_prob List there are the predictions scores for each class,
                           this parameter indicates the idx of the class we are interested in
        - 'pos_label': the label of the positive class

        :param results_path: the path where to save the reports
        :param fold_nr: the fold number
        :param print_reports: if True, print the classification report
        :return:
        """

        # classification report
        report = classification_report(self.y_true, self.y_label_pred, zero_division=0.0)
        out_path = Path(results_path)
        fold_nr = Path(str(f"fold_{fold_nr}"))
        final_path = out_path.joinpath(fold_nr)
        final_path.mkdir(parents=True, exist_ok=True)
        out_file = final_path.joinpath(Path(f"classification_report.txt"))
        with open(out_file, "wt", encoding="utf8") as f:
            print(report, file=f)
        if print_reports:
            print(report)

        # confusion matrix
        labels = sorted(list(set(self.y_true)))
        cm = confusion_matrix(self.y_true, self.y_label_pred, labels=labels)
        self.print_cm(cm, labels, print_reports, final_path)
        self.plot_confusion_matrix(cm, final_path)

        # precision and recall as a function of threshold value
        for idx in range(0, len(self.y_pred_probs[0])):
            pos_class_prob = [y_prob_pred[idx] for y_prob_pred in self.y_pred_probs]
            tmp = array([1 if sample == self.class_names[idx] else 0 for sample in self.y_true])
            precisions, recalls, thresholds = precision_recall_curve(tmp, array(pos_class_prob))

            # save precisions, recalls, thresholds - allows for threshold tuning based on precision and/or recall
            df = DataFrame(list(zip(precisions, recalls, thresholds)), columns=["precision", "recall", "thresholds"])
            f_out = final_path.joinpath(Path(f"precision_recall_threshold_{self.class_names[idx]}.csv"))
            df.to_csv(f_out, index=False)

            # generate the precision-recall vs threshold plot
            f_out = final_path.joinpath(Path(f"precision_recall_threshold_{self.class_names[idx]}.png"))
            self.plot_curve(thresholds, precisions, recalls, f_out)

        preds = array(self.y_pred_probs)
        if self.y_id:
            # ToDo: do this without using hstack, to remove one dependency
            hstack(array(self.y_id, self.y_true, self.preds))
        else:
            preds_array = array(self.y_true).reshape(len(self.y_true), 1)
            labels_array = array(self.y_label_pred).reshape(len(self.y_true), 1)
            data = hstack((preds_array, labels_array, preds))
            labels = ["true_y", "pred_label"] + self.class_names

            df = DataFrame(data, columns=labels)
            df.to_csv(final_path.joinpath(Path(f"prediction_scores.csv", index=False)))
