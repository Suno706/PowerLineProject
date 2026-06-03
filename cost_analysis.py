"""
Cost-weighted evaluation for the fault classifier.

In a real power system, classification mistakes are not equal. Missing
a Three-Phase fault (LLL) for several minutes can cascade into a
black-out; raising a false alarm on a No-Fault window just dispatches
an unnecessary inspection.

This module defines a per-fault cost model and provides utilities to:

  - Compute the total expected cost of a confusion matrix.
  - Sweep a confidence threshold and produce a cost-vs-coverage curve.
  - Recommend a threshold that minimises total expected cost.

The cost numbers below are order-of-magnitude estimates from public
reliability literature (NERC / IEEE outage cost studies). They are
configurable; the methodology is what matters.
"""

import numpy as np
import pandas as pd


# Cost per missed fault (false negative). Dollars per event.
# A missed LLL fault can cascade into a wide-area outage; an LG is
# usually self-clearing.
FN_COST = {
    'No Fault':                        0,
    'Single Line to Ground (LG)':   2_500,
    'Line to Line (LL)':            8_000,
    'Double Line to Ground (LLG)': 15_000,
    'Three Phase Fault (LLL)':     60_000,
    'Open Circuit':                12_000,
}

# Cost per false alarm (false positive). Dollars per event - mostly
# inspection crew + downtime spent on a non-event.
FP_COST = 800


def expected_cost(y_true, y_pred):
    """Total expected cost for one set of predictions."""
    total = 0.0
    for actual, pred in zip(y_true, y_pred):
        if actual == pred:
            continue
        if actual == 'No Fault':
            # Predicted a fault when there wasn't one -> false alarm
            total += FP_COST
        else:
            # Missed an actual fault -> pay the full fault cost
            total += FN_COST.get(actual, 0)
    return total


def cost_per_class(y_true, y_pred):
    """Break down the realised cost by actual class."""
    rows = []
    for cls in sorted(set(y_true)):
        mask = np.array(y_true) == cls
        sub_true = np.array(y_true)[mask]
        sub_pred = np.array(y_pred)[mask]
        rows.append({
            'class': cls,
            'count': int(mask.sum()),
            'errors': int((sub_true != sub_pred).sum()),
            'cost_usd': expected_cost(sub_true, sub_pred),
        })
    return pd.DataFrame(rows)


def threshold_sweep(model, X_test, y_test, thresholds=None):
    """
    Sweep a 'minimum confidence' threshold and report the trade-off.

    For each threshold tau:
      - If max class probability >= tau, accept the model's prediction.
      - Otherwise, abstain and route to a human operator (we model the
        human as always-correct: zero misclassification cost, but each
        abstention has a small handling cost).

    Returns a DataFrame with cost, coverage (fraction auto-decided),
    and accuracy on the auto-decided slice.
    """
    if thresholds is None:
        thresholds = np.linspace(0.20, 0.95, 16)

    proba = model.predict_proba(X_test)
    classes = model.classes_
    pred_idx = proba.argmax(axis=1)
    pred_conf = proba.max(axis=1)
    preds = classes[pred_idx]

    HUMAN_HANDLING_COST = 150  # $ per abstention

    rows = []
    for tau in thresholds:
        auto_mask = pred_conf >= tau
        n_auto = int(auto_mask.sum())
        coverage = n_auto / len(preds) if len(preds) else 0
        auto_cost = expected_cost(
            np.array(y_test)[auto_mask].tolist(),
            preds[auto_mask].tolist(),
        )
        abstain_cost = (len(preds) - n_auto) * HUMAN_HANDLING_COST
        rows.append({
            'threshold':    round(float(tau), 2),
            'coverage':     round(coverage, 3),
            'auto_acc':     round(
                (preds[auto_mask] == np.array(y_test)[auto_mask]).mean()
                if n_auto else 0.0, 3,
            ),
            'auto_cost_usd':    round(auto_cost, 0),
            'abstain_cost_usd': round(abstain_cost, 0),
            'total_cost_usd':   round(auto_cost + abstain_cost, 0),
        })
    return pd.DataFrame(rows)


def recommend_threshold(sweep_df):
    """Pick the threshold that minimises total expected cost."""
    row = sweep_df.loc[sweep_df['total_cost_usd'].idxmin()]
    return {
        'threshold': row['threshold'],
        'coverage':  row['coverage'],
        'total_cost_usd': row['total_cost_usd'],
        'reason': (
            f'At tau = {row["threshold"]:.2f} the model auto-decides '
            f'{row["coverage"]*100:.0f}% of windows; the remaining '
            f'{(1-row["coverage"])*100:.0f}% are routed to a human. '
            f'Total expected cost is minimised at '
            f'${row["total_cost_usd"]:,.0f} per test set.'
        ),
    }
