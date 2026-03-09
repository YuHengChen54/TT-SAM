# Model Training & Experiment Tracking

This directory contains the scripts for training **TT-SAM**. The training logic is designed to handle seismic data imbalances and to ensure model robustness through spatio-temporal augmentation.

---

## 🚀 Getting Started

### 1. Prerequisites

**Option A: Use Demo Dataset (Quick Start)**

If you want to test training without preprocessing, download our pre-processed demo dataset from Zenodo:

- **Zenodo Repository**: [10.5281/zenodo.18885307](https://doi.org/10.5281/zenodo.18885307)
- Place `TSMIP_2016_demo.hdf5` into `data/processed/` directory.

**Option B: Full Pipeline**

Complete the [Data Preprocessing](../preprocess/README.md) and prepare your own HDF5 file.

The training script by default uses `data/processed/TSMIP_2016_demo.hdf5`.

### 2. Training with MLflow

We use **MLflow** to track hyperparameters, loss curves, and model artifacts.

`multi_station_training.py` sets:
- tracking URI: `http://localhost:5000`
- experiment name: `TT-SAM Training` (must exist before training)

So please start MLflow server first:

```bash
mlflow server --host localhost --port 5000
```

Then create/select the experiment `TT-SAM Training` in MLflow UI, and start training:

```bash
cd src/training
python multi_station_training.py
```

To visualize runs in browser:
Then open `http://localhost:5000` in your browser.

---

## 📂 Script Overview

### `multi_station_training.py`
- **Purpose:** The main entry point for training.
- **Functionality:** Handles data loading (via `multiple_sta_dataset.py`), defines the optimizer, and executes the training loop.
- **Features:**
  - Early stopping mechanism to prevent overfitting.
  - MLflow integration for experiment tracking.
  - Checkpoint saving at regular intervals.

### `predict_ensemble_merge_info.py`
- **Purpose:** Used for generating ensemble predictions.
- **Use case:** Post-training analysis.

---

## ⚙️ Configuration

You can adjust the following parameters directly in `multi_station_training.py`:

- **`batch_size`**: Default is set for optimal GPU memory usage (typically 16).
- **`num_epochs`**: The number of passes through the entire dataset (default: 300).
- **`learning_rate`**: Managed by an automated scheduler with patience-based early stopping.
- **`mask_waveform_sec`**: Time-masking window in seconds (e.g., 3, 5, 7...).
- **`oversample`**: Multiplication factor for magnitude-based oversampling.

Example configuration loop:

```python
for batch_size in [16]:
    for learning_rate in [5e-05, 2.5e-05]:
        for run_index in range(3):
            # Training with these hyperparameters
```

---

## 🔗 Related Files

- [Data Preprocessing](../preprocess/README.md): Generate HDF5 datasets before training.
- [Model Architecture](../models/README.md): Details on the CNN-Transformer-MDN architecture.
