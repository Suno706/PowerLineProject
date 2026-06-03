"""
Train a Random Forest classifier on the windowed feature matrix.

Pipeline:
  1. Load the raw CSV.
  2. Build the 156 x 36 feature matrix with feature_extraction.py.
  3. 80/20 train-test split.
  4. Fit a Random Forest (class_weight='balanced' for the heavy
     'No Fault' class).
  5. Print accuracy, classification report, and confusion matrix.
  6. Save the trained model with joblib.
"""

import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from feature_extraction import extract_features


DATA_PATH  = 'data/power_line_fault_dataset.csv'
MODEL_PATH = 'models/rf_fault_classifier.pkl'


def main():
    print('Loading dataset...')
    df = pd.read_csv(DATA_PATH)
    print(f'Raw rows: {len(df)}')

    print('\nExtracting sliding-window features...')
    X, y = extract_features(df, window_size=50, n_samples=7800)
    print(f'Feature matrix: {X.shape}')
    print(f'Class counts:\n{y.value_counts()}')

    print('\nSplitting train/test (80/20)...')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f'Train: {X_train.shape}, Test: {X_test.shape}')

    print('\nTraining Random Forest (n_estimators=200)...')
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    print('\n--- Results ---')
    y_pred = clf.predict(X_test)
    print(f'Test accuracy: {accuracy_score(y_test, y_pred):.4f}')

    print('\nClassification report:')
    print(classification_report(y_test, y_pred, zero_division=0))

    print('Confusion matrix:')
    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    print(pd.DataFrame(cm, index=labels, columns=labels))

    print('\nTop 10 most important features:')
    importance = pd.Series(clf.feature_importances_, index=X.columns)
    print(importance.sort_values(ascending=False).head(10).round(4))

    os.makedirs('models', exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    print(f'\nModel saved to {MODEL_PATH}')


if __name__ == '__main__':
    main()
