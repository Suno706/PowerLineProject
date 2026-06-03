"""
Sliding-window feature extraction for 3-phase power grid signals.

For each window of W consecutive samples we compute 6 statistics
on each of the 6 channels (Va, Vb, Vc, Ia, Ib, Ic), giving 36
features per window.

Stats: RMS, Mean, Std Dev, Peak-to-Peak, Skewness, Kurtosis.
"""

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis


CHANNELS = [
    'voltage_phase_a', 'voltage_phase_b', 'voltage_phase_c',
    'current_phase_a', 'current_phase_b', 'current_phase_c',
]

STAT_NAMES = ['rms', 'mean', 'std', 'ptp', 'skew', 'kurt']


def _window_stats(x):
    """Six statistics for a 1-D numpy array."""
    return [
        np.sqrt(np.mean(x * x)),   # RMS
        float(np.mean(x)),         # Mean
        float(np.std(x)),          # Std dev
        float(np.ptp(x)),          # Peak-to-peak (max - min)
        float(skew(x)),            # Skewness
        float(kurtosis(x)),        # Kurtosis (Fisher; normal -> 0)
    ]


def extract_features(df, window_size=50, n_samples=7800, random_state=42):
    """
    Build a (windows x 36) feature matrix.

    The raw CSV is time-sorted, so the first chunk is dominated by
    'No Fault'. We first stratified-sample `n_samples` rows so all
    six fault classes are represented in roughly equal numbers,
    then slide a non-overlapping `window_size` window over the sample.

    The label for each window is the majority fault_type within it.

    Returns
    -------
    X : pandas.DataFrame, shape (n_windows, 36)
    y : pandas.Series,    shape (n_windows,)
    """
    classes = df['fault_type'].unique()
    per_class = n_samples // len(classes)
    pieces = []
    for cls in classes:
        cls_df = df[df['fault_type'] == cls]
        take = min(per_class, len(cls_df))
        pieces.append(cls_df.sample(n=take, random_state=random_state))
    df = (pd.concat(pieces)
            .sample(frac=1, random_state=random_state)   # shuffle
            .reset_index(drop=True)
            .iloc[:n_samples])
    n_windows = len(df) // window_size

    feature_cols = [f'{ch}_{stat}' for ch in CHANNELS for stat in STAT_NAMES]
    X = np.zeros((n_windows, len(feature_cols)), dtype=float)
    y = []

    for w in range(n_windows):
        start = w * window_size
        end = start + window_size
        window = df.iloc[start:end]

        row = []
        for ch in CHANNELS:
            row.extend(_window_stats(window[ch].to_numpy()))
        X[w] = row

        # Window label = most common fault_type in this window
        y.append(window['fault_type'].mode().iat[0])

    return pd.DataFrame(X, columns=feature_cols), pd.Series(y, name='fault_type')


if __name__ == '__main__':
    df = pd.read_csv('data/power_line_fault_dataset.csv')
    X, y = extract_features(df)
    print(f'Feature matrix: {X.shape}')
    print(f'Labels:         {y.shape}')
    print('\nClass distribution:')
    print(y.value_counts())
    print('\nFirst 3 feature columns:')
    print(X.iloc[:3, :6].round(3))
