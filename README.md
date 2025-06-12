# Air Quality Analysis

## Overview

This repository contains a comprehensive exploration of air quality prediction using both classification and regression approaches. The primary aim is to forecast carbon monoxide levels (CO(GT)) using sensor data, and to evaluate the effectiveness of linear models and neural networks.

---

## Repository Structure

```
.
├── data/
│   └── sensor_data.xlsx       # Raw dataset of hourly‑averaged measurements
├── air_quality_analysis.ipynb  # Single notebook containing all code and experiments
└── README.md                   
```

---

## Dataset

* **Source**: Multisensor device measurements at road level in a polluted city.
* **Instances**: 3,304 hourly‐averaged samples.
* **Features**:

  * Spectrometer analyzers (GT\*)
  * Metal oxide detectors (PT08.S\*)
  * Temperature (T), Relative Humidity (RH), Absolute Humidity (AH)
* **Target**: CO(GT) – carbon monoxide concentration.
* **Missing values** flagged by `-999`.

---

## Environment & Dependencies

* Python 3.7+
* `numpy`, `pandas`, `matplotlib`, `scikit-learn`
* `hyperopt` (for hyperparameter tuning; optional)

Install required packages:

```bash
pip install numpy pandas matplotlib scikit-learn hyperopt
```

---

## 1. Data Preparation

1. Load `sensor_data.xlsx` into a DataFrame.
2. Replace `-999` with NaN.
3. Impute missing values using mean or KNN imputation.
4. Standardise or robustly scale features.
5. Split into training and test sets (e.g. 85%/15%).

---

## 2. Binary Classification

* **Task**: Predict whether CO(GT) > 4.5 (`True` for poor air quality).
* **Model**: Linear classifier trained by gradient descent on hinge loss with L2 regularisation.
* **Function**: `linear_gd_train(data, labels, c, n_iters, learning_rate)`.

### Outputs

* Training cost vs iterations.
* Training accuracy vs iterations.
* Test set accuracy and F1 score.

---

## 3. Learning Rate Analysis

* Experiment with different learning rates (`0.1`, `0.01`, `0.001`, `0.0001`).
* Plot training cost and accuracy curves for each rate.
* Report test accuracy and F1 for each.

---

## 4. MLP Regression Model Selection

* **Task**: Predict continuous CO(GT) values.
* **Model**: `MLPRegressor` with grid search over:

  * Hidden layer sizes: (3,), (100,), (3,3), (100,100)
  * Activation: `relu`, `logistic`
* **Metric**: Mean Squared Error (MSE) via 5-fold cross validation.

### Outputs

* Best hyperparameters.
* CV MSE ± standard deviation.
* Test set MSE and R² score.

---

## 5. Solver Comparison (SGD vs ADAM)

* **Model**: MLP with two hidden layers (100,100), `relu` activation.
* **Solvers**: `adam` vs `sgd` (adaptive learning rate, momentum 0.95).
* **Procedure**: Incremental training for 300 iterations with `warm_start=True`.

### Outputs

* Training loss curve for each solver.
* Train and test MSE over iterations.
* Final test MSE and R² score.

---

## 6. Robust MLP Regressor

* **Objective**: Develop a robust model handling missing and noisy data.
* **Techniques**:

  * KNN imputation.
  * Robust scaling.
  * Hyperparameter optimisation via Hyperopt.
* **Script**: `scripts/robust_mlp_regressor.py`

### Outputs

* Best hyperparameters.
* Test set MSE and R² score.

---

## Usage

Each notebook and script can be run independently:

1. Open the relevant notebook in JupyterLab.
2. Execute cells sequentially.
3. Review plots, metrics and report findings.

For the final robust model:

```bash
python scripts/robust_mlp_regressor.py
```

---

## Licence

MIT Licence. Feel free to use and adapt for your own analyses. Safe air!
