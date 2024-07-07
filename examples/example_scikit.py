import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src.ml_report_kit import MLReport


def main():

    # Load the 20 newsgroups dataset
    dataset = fetch_20newsgroups(subset='all', shuffle=True, random_state=42)

    # Split the dataset into 3 folds
    k_folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    folds = {}
    for fold_nr, (train_index, test_index) in enumerate(k_folds.split(dataset.data, dataset.target)):
        np_data = np.array(dataset.data)
        np_target = np.array(dataset.target)
        x_train, x_test = np_data[train_index], np_data[test_index]
        y_train, y_test = np_target[train_index], np_target[test_index]
        folds[fold_nr] = {"x_train": x_train, "x_test": x_test, "y_train": y_train, "y_test": y_test}

    # Train a classifier and generate a report for each fold
    for fold_nr in folds.keys():
        clf = Pipeline([('tfidf', TfidfVectorizer()), ('clf', LogisticRegression(class_weight='balanced'))])
        clf.fit(folds[fold_nr]["x_train"], folds[fold_nr]["y_train"])
        y_pred = clf.predict(folds[fold_nr]["x_test"])
        y_pred_prob = clf.predict_proba(folds[fold_nr]["x_test"])

        y_true_label = [dataset.target_names[sample] for sample in folds[fold_nr]["y_test"]]
        y_pred_label = [dataset.target_names[sample] for sample in y_pred]
        ml_report = MLReport(y_true_label, y_pred_label, list(y_pred_prob), dataset.target_names, y_id=None)
        ml_report.run(results_path="results", fold_nr=fold_nr)


if __name__ == "__main__":
    main()
