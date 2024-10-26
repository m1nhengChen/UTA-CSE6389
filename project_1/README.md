## MRI Disease Classification with CNN

This project implements a Convolutional Neural Network (CNN) for classifying MRI images into two categories: `health` and `patient`. The images are in `.nii.gz` format, and the CNN model is trained using 3D MRI images. The project includes functionalities for training, testing, and visualizing the model's predictions.

### Requirements

Ensure you have the following Python packages installed:

```bash
pip install torch torchvision nibabel numpy scikit-learn matplotlib seaborn

```
### Directory Structure
```
project/
│
├── Training/          # Contains training data
│   ├── health/        # Healthy subject images in .nii.gz format
│   └── patient/       # Patient images in .nii.gz format
│
├── Testing/           # Contains test data
│   ├── health/        # Healthy subject images in .nii.gz format
│   └── patient/       # Patient images in .nii.gz format
│
├── train.py           # Main Python script for training and inference
├── README.md          # This file
└── cnn_model.pth      # Saved CNN model (generated after training)
```
### Usage
The script can be run in two modes: train and inference.

#### 1. Training the Model
To train the model, use the following command:

```bash
python train.py --mode train --epochs 10 --model_path cnn_model.pth
```
--mode train: Runs the script in training mode.
--epochs 10: Specifies the number of training epochs (default is 10).
--model_path cnn_model.pth: Path to save the trained model (default is cnn_model.pth).

The training process will output:

Loss values at each step and epoch.
A plot of the training loss over epochs (loss.png).
A confusion matrix and the accuracy on the test set (confusion_matrix.png).
Visualization of some MRI predictions (visualization.png).

#### 2. Inference
To run inference on the test set using a pre-trained model:

```bash
python train.py --mode inference --model_path cnn_model.pth
```
--mode inference: Runs the script in inference mode.
--model_path cnn_model.pth: Path to the saved model to load (default is cnn_model.pth).

The inference process will:
Output the confusion matrix and accuracy on the test set.
Visualize the predictions on some MRI images.

### Model Architecture
The CNN model consists of two 3D convolutional layers followed by max-pooling, and two fully connected layers. It accepts 3D MRI volumes and outputs a classification into two categories: health or patient.

- Conv Layer 1: 16 filters, kernel size 3x3x3, followed by ReLU and max pooling.
- Conv Layer 2: 32 filters, kernel size 3x3x3, followed by ReLU and max pooling.
- FC Layer 1: Fully connected layer with 128 units.
- FC Layer 2: Fully connected layer with 2 output units (health or patient).


### Data Preprocessing
The MRI images are read in .nii.gz format using nibabel, converted to NumPy arrays, and normalized. The MRI volumes are expanded along the channel dimension to make them compatible with 3D convolutions in PyTorch.

### Seed Reproducibility
For reproducibility, the script sets seeds for the following:


- NumPy
- PyTorch
- CUDA

### Results
After training, the model's performance is evaluated on the test set, and results are visualized through:

- A confusion matrix: This shows the model's classification performance.
- Accuracy: Overall classification accuracy.
- MRI prediction visualization: Shows the MRI slices with the predicted and true labels.
### Files Generated
- cnn_model.pth: Saved model after training.
- loss.png: Plot of training loss over epochs.
- confusion_matrix.png: Confusion matrix visualization after testing.
- visualization.png: Visualization of MRI predictions.
