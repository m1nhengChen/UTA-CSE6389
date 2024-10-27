## MRI Disease Classification with GCN

This project uses a Graph Convolutional Network (GCN) for classifying brain connectivity patterns derived from MRI data into two categories: Health and AD (Alzheimer's Disease). Functional and structural connectivity matrices are utilized to capture connectivity information, and the model is trained and evaluated using k-fold cross-validation. Metrics including accuracy, sensitivity, specificity, F1 score, and AUC are used to assess performance.

### Requirements

Ensure you have the following Python packages installed:

```bash
pip install torch torchvision scikit-learn numpy matplotlib seaborn
```
### Directory Structure
```
project/
│
├── Health/            # Healthy subjects' connectivity data
│   ├── subject_1/
│       ├── FunctionalConnectivity.txt
│       └── StructuralConnectivity.txt
│   └── ...            # Additional healthy subjects
│
├── AD/                # Alzheimer's Disease subjects' connectivity data
│   ├── subject_1/
│       ├── FunctionalConnectivity.txt
│       └── StructuralConnectivity.txt
│   └── ...            # Additional AD subjects
│
├── train.py           # Main Python script for training and evaluation
├── README.md          # This file
└── gcn_model.pth      # Saved GCN model (generated after training)
```
### Usage
The script supports training and evaluating the model using k-fold cross-validation.

#### Training the Model
To train the model, use the following command:

```bash
python train.py --epochs 20 --k_folds 5
```
--epochs: Specifies the number of training epochs (default is 10).
--k_folds: Specifies the number of folds for cross-validation (default is 5).

Each fold's metrics and confusion matrix will be printed and visualized.


### Model Architecture
The GCN model has two parallel branches: one for functional connectivity and one for structural connectivity.

- Functional Connectivity Branch: Two GCN layers followed by ReLU and mean pooling.
- Structural Connectivity Branch: Two GCN layers followed by ReLU and mean pooling.
- Concatenation: Outputs from both branches are concatenated and passed to a fully connected layer for binary classification.
Each branch processes the corresponding connectivity matrix, leveraging shared features to improve classification performance.

### Data Preprocessing
The MRI images are read in .nii.gz format using nibabel, converted to NumPy arrays, and normalized. The MRI volumes are expanded along the channel dimension to make them compatible with 3D convolutions in PyTorch.

### Seed Reproducibility
For reproducibility, the script sets seeds for the following:


- NumPy
- PyTorch
- CUDA
### Metrics
The following metrics are computed for each fold:

- Accuracy: Percentage of correctly classified samples.
- Sensitivity: True Positive Rate (recall).
- Specificity: True Negative Rate.
- F1 Score: Harmonic mean of precision and recall.
- AUC: Area under the ROC curve for binary classification.
### Results
After each fold, the model outputs:

- Confusion Matrix: Visualized using a heatmap.
- Metric Scores: Includes accuracy, F1 score, AUC, sensitivity, and specificity.
- Mean Metrics: Mean values of each metric across all folds, providing a robust assessment of the model's generalization.
### Files Generated
- Confusion Matrices: Confusion matrix for each fold visualized.
- Metric Summaries: Detailed output of all metrics per fold and the overall mean performance.