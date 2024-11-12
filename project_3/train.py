import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import shift
from collections import deque
import matplotlib.pyplot as plt
import seaborn as sns

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define paths for data
CN_PATH = './CN'
AD_PATH = './AD'

# Augmentation Functions
def add_gaussian_noise(data, std_dev=0.01):
    noise = np.random.normal(0, std_dev, data.shape)
    return data + noise

def time_warp(data, max_warp=0.1):
    shift_val = np.random.uniform(-max_warp, max_warp) * data.shape[0]
    return shift(data, (int(shift_val), 0), mode='nearest')

def scale(data, scale_range=(0.9, 1.1)):
    factor = np.random.uniform(scale_range[0], scale_range[1])
    return data * factor

def jitter(data, jitter_range=0.01):
    jitter_vals = np.random.uniform(-jitter_range, jitter_range, data.shape)
    return data + jitter_vals

# Custom Dataset to load fMRI signals for each subject with augmentation
class FMRI_Dataset(Dataset):
    def __init__(self, root_dir, labels, augment=False, augment_prob=0.5):
        self.root_dir = root_dir
        self.labels = labels
        self.augment = augment
        self.augment_prob = augment_prob
        self.data, self.targets = self.load_data()

    def load_data(self):
        data = []
        targets = []
        
        for label, folder in enumerate(['CN', 'AD']):
            label_dir = os.path.join(self.root_dir, folder)
            subjects = os.listdir(label_dir)
            
            for subject in subjects:
                subject_dir = os.path.join(label_dir, subject, "fmri_average_signal")
                time_series = []
                for i in range(100):
                    file_path = os.path.join(subject_dir, f"raw_fmri_feature_matrix_{i}.txt")
                    time_point_data = np.loadtxt(file_path)
                    time_series.append(time_point_data)
                
                data.append(np.stack(time_series))  # Shape: (100, 150)
                targets.append(label)
        
        return torch.tensor(data, dtype=torch.float32), torch.tensor(targets, dtype=torch.long)
    
    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.targets[idx]

        if self.augment and np.random.rand() < self.augment_prob:
            data = self.apply_augmentations(data.numpy())
            data = torch.tensor(data, dtype=torch.float32)

        return data, label

    def apply_augmentations(self, data):
        augmentations = [add_gaussian_noise, time_warp, scale, jitter]
        np.random.shuffle(augmentations)
        
        for aug in augmentations:
            if np.random.rand() < 0.5:
                data = aug(data)
        return data

# Define the DeepLSTM model for fMRI data classification
class DeepLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(DeepLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Training and Evaluation function with early stopping based on training loss standard deviation
def train_and_evaluate(root_dir, num_epochs, k_folds, batch_size, hidden_size, learning_rate, num_layers, augment, std_threshold=0.001):
    dataset = FMRI_Dataset(root_dir, labels=[0, 1], augment=augment)
    labels = dataset.targets
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    all_accuracies, all_precisions, all_recalls, all_f1_scores = [], [], [], []
    criterion = nn.CrossEntropyLoss()

    for fold, (train_idx, test_idx) in enumerate(skf.split(dataset.data, labels)):
        print(f'Fold {fold + 1}/{k_folds}')
        
        train_sampler = torch.utils.data.Subset(dataset, train_idx)
        test_sampler = torch.utils.data.Subset(dataset, test_idx)
        train_loader = DataLoader(train_sampler, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_sampler, batch_size=batch_size, shuffle=False)
        
        model = DeepLSTM(input_size=150, hidden_size=hidden_size, output_size=2, num_layers=num_layers).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Track the last 10 training losses for early stopping
        recent_losses = deque(maxlen=10)
        loss_values = []  # For visualization of training loss

        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_epoch_loss = epoch_loss / len(train_loader)
            recent_losses.append(avg_epoch_loss)
            loss_values.append(avg_epoch_loss)  # Store for visualization
            print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_epoch_loss:.4f}')
            
            # Early Stopping Check
            if len(recent_losses) == 10 and np.std(recent_losses) < std_threshold:
                print("Early stopping triggered based on training loss stability.")
                break

        # Visualize training loss
        plt.plot(range(len(loss_values)), loss_values, label=f'Fold {fold + 1} Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training Loss for Fold {fold + 1}')
        plt.legend()
        plt.savefig(f'./loss_fold_{fold + 1}.png')
        # plt.show()

        # Evaluation
        model.eval()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        
        all_accuracies.append(accuracy)
        all_precisions.append(precision)
        all_recalls.append(recall)
        all_f1_scores.append(f1)

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["CN", "AD"], yticklabels=["CN", "AD"])
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Confusion Matrix for Fold {fold + 1}")
        plt.savefig(f'./confusion_matrix_fold_{fold + 1}.png')
        # plt.show()
        plt.close()

        print(f'Fold {fold + 1} - Accuracy: {accuracy * 100:.2f}% | Precision: {precision:.2f} | Recall: {recall:.2f} | F1 Score: {f1:.2f}')
    
    # Display average metrics across all folds
    print(f'\nMean Accuracy over {k_folds} folds: {np.mean(all_accuracies) * 100:.2f}%')
    print(f'Mean Precision over {k_folds} folds: {np.mean(all_precisions):.2f}')
    print(f'Mean Recall over {k_folds} folds: {np.mean(all_recalls):.2f}')
    print(f'Mean F1 Score over {k_folds} folds: {np.mean(all_f1_scores):.2f}')

# Set seed for reproducibility
def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Main execution with argument parser for hyperparameters
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=30, help='Number of epochs for training')
    parser.add_argument('--k_folds', type=int, default=5, help='Number of folds for cross-validation')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--hidden_size', type=int, default=128, help='Number of hidden units in LSTM')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--root_dir', type=str, default='./', help='Root directory for dataset')
    parser.add_argument('--augment', action='store_true', help='Apply data augmentation')
    parser.add_argument('--std_threshold', type=float, default=0.005, help='Standard deviation threshold for early stopping')
    args = parser.parse_args()

    # Seed for reproducibility
    seed_everything(75)
    
    # Train and evaluate the model
    train_and_evaluate(
        root_dir=args.root_dir,
        num_epochs=args.num_epochs,
        k_folds=args.k_folds,
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
        learning_rate=args.learning_rate,
        num_layers=args.num_layers,
        augment=args.augment,
        std_threshold=args.std_threshold
    )
