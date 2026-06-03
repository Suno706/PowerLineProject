# Power Grid Fault Detection System

Detects fault type in a 3-phase electrical power line from raw voltage
and current measurements. Built end-to-end in Python with scikit-learn:
sliding-window feature extraction, statistical feature engineering and a
Random Forest classifier.

## Dataset

A 50,000-row dataset of 3-phase power line measurements collected
across 20 grid sections. Each row has six raw signal channels:

- `voltage_phase_a`, `voltage_phase_b`, `voltage_phase_c`
- `current_phase_a`, `current_phase_b`, `current_phase_c`

plus context columns (weather, frequency, temperature, load, etc.) and
a `fault_type` label with six classes:

| Class                       | Rows  |
|-----------------------------|------:|
| No Fault                    | 27574 |
| Single Line to Ground (LG)  |  9063 |
| Line to Line (LL)           |  4942 |
| Double Line to Ground (LLG) |  3950 |
| Three Phase Fault (LLL)     |  2505 |
| Open Circuit                |  1966 |

## Pipeline

1. **Stratified sample** 7,800 rows (1,300 per class) so all six fault
   types are represented. The raw CSV is time-sorted and heavily
   dominated by "No Fault", so naive sampling won't work.
2. **Sliding window** the sample into 156 non-overlapping windows of
   50 rows each.
3. For every window, compute six statistics on each of the six
   channels: **RMS, Mean, Std Dev, Peak-to-Peak, Skewness, Kurtosis**.
   That gives `6 stats x 6 channels = 36 features per window`.
4. Result: a `156 x 36` feature matrix. Each window's label is the
   majority `fault_type` of its 50 rows.
5. 80/20 train-test split (stratified). Train a Random Forest with
   `class_weight='balanced'`.

## Why these features

Each statistic captures a different physical aspect of the signal:

| Stat            | What it tells you about the signal              |
|-----------------|-------------------------------------------------|
| RMS             | Effective magnitude (energy content)            |
| Mean            | DC offset / average level                       |
| Std Dev         | Spread / variability                            |
| Peak-to-Peak    | Max swing of the signal (transients)            |
| Skewness        | Asymmetry of the distribution                   |
| Kurtosis        | "Tailedness" - high values mean sharp spikes    |

Faults produce characteristic distortions in these statistics. For
example, a Line-to-Line fault drops voltage on two phases and spikes
the current, which shows up as high std dev and peak-to-peak on those
channels.

## Results

On the held-out test set (32 windows):

- **Test accuracy: 0.59** (random baseline for 6 classes = 0.17)
- Strongest features: `current_phase_{a,b,c}_std` and
  `voltage_phase_{a,b,c}_rms` - current variance is the dominant
  fault signal.
- Per-class F1 ranges from 0.22 (LLG) to 0.86 (Open Circuit). The
  hardest classes to separate are LL and LLG, which share similar
  current-imbalance signatures.

## Project layout

```
power-grid-fault-detection/
  data/
    power_line_fault_dataset.csv     50,000 rows of 3-phase samples
  models/                            saved .pkl files (after training)
  feature_extraction.py              sliding-window + 36-stat features
  train.py                           training + evaluation pipeline
  requirements.txt
  README.md
```

## How to run

```bash
pip install -r requirements.txt
python train.py
```

The script loads the CSV, builds the feature matrix, trains the
model, prints accuracy + classification report + confusion matrix,
and saves the trained model to `models/rf_fault_classifier.pkl`.

## What's next

This is Module 1 of a larger plan:

- **Module 1: Random Forest** (done) - works on hand-engineered window stats.
- **Module 2: LSTM** - feed the raw 50-step window directly to a small
  recurrent network instead of summarising to 36 features. Should pick
  up temporal patterns the stats throw away.
- **Module 3: Isolation Forest** - unsupervised anomaly detection for
  fault types the supervised model has never seen.
