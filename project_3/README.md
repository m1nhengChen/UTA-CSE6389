## MRI Disease Classification with RNN

This project uses a RNN model (Deep LSTM, [reference](https://github.com/sarasoltani80/RL-and-RNN-on-Time-Series-fMRI-data)) to classify brain connectivity patterns derived from fMRI signals into two categories: Health and AD (Alzheimer's Disease). Functional connectivity is captured over time and processed as sequential data, allowing the LSTM to learn temporal dependencies across regions of interest (ROIs). The model is trained and evaluated using k-fold cross-validation with metrics such as accuracy, sensitivity, specificity, F1 score, and confusion matrix visualizations.
### Requirements

Ensure you have the following Python packages installed:

```bash
pip install torch scikit-learn numpy matplotlib seaborn
```


### Directory Structure

The project directory should be structured as follows:
```
project/
│
├── CN/                # Healthy subjects' fMRI data
│   ├── subject_1/
│       └── fmri_average_signal/
│           ├── raw_fmri_feature_matrix_0.txt
│           └── ... (up to raw_fmri_feature_matrix_99.txt)
│   └── ...            # Additional healthy subjects
│
├── AD/                # Alzheimer's Disease subjects' fMRI data
│   ├── subject_1/
│       └── fmri_average_signal/
│           ├── raw_fmri_feature_matrix_0.txt
│           └── ... (up to raw_fmri_feature_matrix_99.txt)
│   └── ...            # Additional AD subjects
│
├── train.py           # Main Python script for training and evaluation
├── README.md          # This file
└── deep_lstm_model.pth# Saved LSTM model (generated after training)
```
Each subject’s folder contains 100 time points of fMRI signals across 150 ROIs stored as .txt files.
### Usage

The script supports training and evaluating the model using k-fold cross-validation. Data augmentation techniques (e.g., Gaussian noise, jitter, scaling) can be applied to the input data to improve generalization.
#### Training the Model

To train the model, use the following command:
```bash
python train.py --num_epochs 30 --k_folds 5 --batch_size 8 --hidden_size 128 --learning_rate 0.001
```
#### Options:

    --num_epochs: Number of epochs for training (default: 30).
    --k_folds: Number of folds for cross-validation (default: 5).
    --batch_size: Batch size for training (default: 8).
    --hidden_size: Number of hidden units in the LSTM (default: 128).
    --learning_rate: Learning rate for the optimizer (default: 0.001).
    --num_layers: Number of LSTM layers (default: 2).
    --augment: Apply data augmentation (optional).
    --std_threshold: Standard deviation threshold for early stopping based on training loss stability (default: 0.005).

### Methodology

The Deep LSTM model is designed to process sequential fMRI data, capturing temporal dependencies across brain regions:
- LSTM Layers: Two LSTM layers with ReLU activation.
- Fully Connected Layer: Output layer for binary classification (Health vs. AD).

The LSTM leverages the sequential nature of fMRI data to improve the classification of health vs. AD, learning both spatial and temporal patterns across time points.

#### Data Preprocessing and Augmentation

To enhance training data, augmentation techniques are used:
- Gaussian Noise: Adds random noise to data.
- Time Warping: Slightly shifts signals along the temporal axis.
- Scaling: Randomly scales the amplitude of signals.
- Jittering: Adds small random values to each time point.

#### Seed Reproducibility

To ensure reproducibility, seeds are set for:
- NumPy
- PyTorch
- CUDA (when available)

#### Early Stopping Strategy

The training process includes an early stopping mechanism based on the stability of the training loss. If the standard deviation of the training loss across the last 10 epochs falls below a set threshold, training halts early to prevent overfitting.

### Results
#### Metrics

The following metrics are computed for each fold:
- Accuracy: Percentage of correctly classified samples.
- Sensitivity: True Positive Rate (recall) for AD cases.
- Specificity: True Negative Rate for Healthy cases.
- F1 Score: Harmonic mean of precision and recall.
- Confusion Matrix: Visualized using a heatmap for classification insights.



#### Files Generated

After running the model, the following files are generated:
- Loss Plots: Training loss plotted over epochs for each fold.
- Confusion Matrices: Confusion matrices visualized for each fold.
- Metric Summaries: Detailed output of metrics for each fold and the overall mean performance.

### Example Command
```bash
python train.py --num_epochs 20 --k_folds 5 --batch_size 16 --hidden_size 128 --learning_rate 0.001 --augment --std_threshold 0.001
```
This command runs the model for 20 epochs with 5-fold cross-validation, a batch size of 16, LSTM hidden size of 128, and data augmentation enabled.