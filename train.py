"""
Training and evaluation pipeline for the Power Grid Fault Detection model.

This script runs the full experimental protocol:

    1. Load the 50k-row CSV of 3-phase measurements.
    2. Build the windowed feature matrix (156 windows x 36 features).
    3. Compare four classifiers under 5-fold stratified cross-validation:
         - Random Forest
         - Gradient Boosting
         - Logistic Regression
         - Naive Bayes
    4. Retrain the winning model on the full training set.
    5. Evaluate on a held-out 20% test set with:
         - Accuracy + macro-F1
         - Per-class precision / recall / F1
         - Confusion matrix
         - One-vs-rest ROC curves and AUC
         - Feature importance ranking
    6. Save the trained model and every plot under reports/.
"""

import json
import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, train_test_split
)
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
)

from feature_extraction import extract_features

warnings.filterwarnings('ignore', category=UserWarning)

DATA_PATH    = 'data/power_line_fault_dataset.csv'
MODEL_PATH   = 'models/rf_fault_classifier.pkl'
REPORTS_DIR  = 'reports'
RANDOM_STATE = 42


def candidate_models():
    """Return the four candidate pipelines we'll compare with CV."""
    return {
        'Random Forest': RandomForestClassifier(
            n_estimators=300, class_weight='balanced',
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=200, random_state=RANDOM_STATE,
        ),
        'Logistic Regression': Pipeline([
            ('scale', StandardScaler()),
            ('clf', LogisticRegression(
                max_iter=2000, class_weight='balanced',
                random_state=RANDOM_STATE,
            )),
        ]),
        'Naive Bayes': Pipeline([
            ('scale', StandardScaler()),
            ('clf', GaussianNB()),
        ]),
    }


def cross_validate_all(X, y):
    """5-fold stratified CV for every candidate model."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, model in candidate_models().items():
        scores = cross_val_score(model, X, y, cv=cv, scoring='f1_macro', n_jobs=-1)
        rows.append({
            'model': name,
            'f1_macro_mean': scores.mean(),
            'f1_macro_std':  scores.std(),
        })
        print(f'  {name:<22s}  F1-macro = {scores.mean():.3f} +/- {scores.std():.3f}')
    return pd.DataFrame(rows).sort_values('f1_macro_mean', ascending=False)


def plot_cv_comparison(cv_df, path):
    fig, ax = plt.subplots(figsize=(8, 4))
    order = cv_df['model'].tolist()
    ax.barh(
        order, cv_df['f1_macro_mean'],
        xerr=cv_df['f1_macro_std'],
        color='steelblue', edgecolor='black', capsize=4,
    )
    ax.set_xlabel('Macro-F1 (5-fold CV)')
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    for i, (m, s) in enumerate(zip(cv_df['f1_macro_mean'], cv_df['f1_macro_std'])):
        ax.text(m + 0.01, i, f'{m:.3f} +/- {s:.3f}', va='center', fontsize=9)
    ax.set_title('Cross-validation comparison')
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_confusion(y_true, y_pred, labels, path):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=labels, yticklabels=labels, ax=ax,
        cbar=False,
    )
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion matrix (test set)')
    plt.xticks(rotation=30, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_feature_importance(model, feature_names, path, top_n=15):
    imp = pd.Series(model.feature_importances_, index=feature_names)
    top = imp.sort_values(ascending=True).tail(top_n)
    fig, ax = plt.subplots(figsize=(8, 6))
    top.plot.barh(ax=ax, color='darkorange', edgecolor='black')
    ax.set_xlabel('Importance')
    ax.set_title(f'Top {top_n} features (Random Forest)')
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_roc_curves(model, X_test, y_test, classes, path):
    y_bin = label_binarize(y_test, classes=classes)
    y_score = model.predict_proba(X_test)

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, cls in enumerate(classes):
        if y_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        auc = roc_auc_score(y_bin[:, i], y_score[:, i])
        ax.plot(fpr, tpr, label=f'{cls} (AUC = {auc:.2f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4)
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.set_title('One-vs-rest ROC curves')
    ax.legend(loc='lower right', fontsize=9)
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs('models', exist_ok=True)

    print('1. Loading dataset...')
    df = pd.read_csv(DATA_PATH)
    print(f'   Raw rows: {len(df):,}')

    print('\n2. Building windowed feature matrix...')
    X, y = extract_features(df, window_size=50, n_samples=7800)
    print(f'   Shape: {X.shape}')
    print(f'   Classes: {y.nunique()}')

    print('\n3. 5-fold cross-validation across candidate models...')
    cv_df = cross_validate_all(X, y)
    cv_df.to_csv(f'{REPORTS_DIR}/cv_scores.csv', index=False)
    plot_cv_comparison(cv_df, f'{REPORTS_DIR}/cv_comparison.png')
    winner = cv_df.iloc[0]['model']
    print(f'\n   Winner: {winner}')

    print('\n4. Retraining Random Forest on 80% train split...')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE,
    )
    model = RandomForestClassifier(
        n_estimators=300, class_weight='balanced',
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_PATH)

    print('\n5. Evaluating on held-out test set...')
    y_pred = model.predict(X_test)
    acc  = accuracy_score(y_test, y_pred)
    f1m  = f1_score(y_test, y_pred, average='macro')
    print(f'   Accuracy:   {acc:.4f}')
    print(f'   Macro-F1:   {f1m:.4f}')
    print(f'   Baseline:   {1/y.nunique():.4f}  (random guessing)')

    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    pd.DataFrame(report_dict).T.round(4).to_csv(f'{REPORTS_DIR}/classification_report.csv')

    classes = sorted(y.unique())
    plot_confusion(y_test, y_pred, classes, f'{REPORTS_DIR}/confusion_matrix.png')
    plot_feature_importance(model, X.columns, f'{REPORTS_DIR}/feature_importance.png')
    plot_roc_curves(model, X_test, y_test, classes, f'{REPORTS_DIR}/roc_curves.png')

    summary = {
        'n_windows':        int(X.shape[0]),
        'n_features':       int(X.shape[1]),
        'n_classes':        int(y.nunique()),
        'test_accuracy':    round(float(acc), 4),
        'test_macro_f1':    round(float(f1m), 4),
        'random_baseline':  round(1 / y.nunique(), 4),
        'cv_winner':        winner,
    }
    with open(f'{REPORTS_DIR}/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print('\n6. Artifacts written:')
    for fn in sorted(os.listdir(REPORTS_DIR)):
        print(f'   reports/{fn}')
    print(f'   {MODEL_PATH}')
    print('\nDone.')


if __name__ == '__main__':
    main()
