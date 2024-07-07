# Machine Learning Report Toolkit

Generating evaluating metrics reports for machine learning models in two lines of code.


```python
from ml_report import MLReport

report = MLReport(y_true_label, y_pred_label, y_pred_prob, class_names)
report.generate_report()
```

This will generate a report for each fold, containing the following:

- A Classification Report with Precision, Recall, F1-Score, Support
- A Confusion Matrix
- Precision and Recall curves as a function of the threshold for each class
- A `.csv` file with precision, recall, at different thresholds
- A `.csv` file with predictions scores for each class for each sample

## Example

Install the package and dependencies:

```bash
pip install ml-report-kit
pip install scikit-learn
```

Run the following code:

```python
    
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from ml_report import MLReport

dataset = fetch_20newsgroups(subset='all', shuffle=True, random_state=42)
k_folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
folds = {}

# create folds
for fold_nr, (train_index, test_index) in enumerate(k_folds.split(dataset.data, dataset.target)):
    x_train, x_test = np.array(dataset.data)[train_index], np.array(dataset.data)[test_index]
    y_train, y_test = np.array(dataset.target)[train_index], np.array(dataset.target)[test_index]
    folds[fold_nr] = {"x_train": x_train, "x_test": x_test, "y_train": y_train, "y_test": y_test}


for fold_nr in folds.keys():
    clf = Pipeline([('tfidf', TfidfVectorizer()), ('clf', LogisticRegression(class_weight='balanced'))])
    clf.fit(folds[fold_nr]["x_train"], folds[fold_nr]["y_train"])
    y_pred = clf.predict(folds[fold_nr]["x_test"])
    y_pred_prob = clf.predict_proba(folds[fold_nr]["x_test"])
    y_true_label = [dataset.target_names[sample] for sample in folds[fold_nr]["y_test"]]
    y_pred_label = [dataset.target_names[sample] for sample in y_pred]
    
    report = MLReport(y_true_label, y_pred_label, y_pred_prob, dataset.target_names)
    report.generate_report()
```