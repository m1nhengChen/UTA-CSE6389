import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score, roc_curve
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
device = torch.device('cpu' if torch.cuda.is_available() else 'cuda')

def min_max_normalize_rows(matrix, epsilon=1e-5):
    # Apply min-max normalization to each row individually
    row_min = matrix.min(axis=1, keepdims=True)
    row_max = matrix.max(axis=1, keepdims=True)
    row_range = np.where(row_max - row_min == 0, 1, row_max - row_min)  # Avoid division by zero
    normalized_matrix = (matrix - row_min) / row_range
    return normalized_matrix

# Laplacian normalization
def normalize_adjacency(adj, epsilon=1e-5):
    adj_hat = min_max_normalize_rows(adj)
    D_hat = np.diag(np.sum(adj_hat, axis=1)+epsilon)
    D_hat_inv_sqrt = np.linalg.inv(np.sqrt(D_hat))
    adj_norm = D_hat_inv_sqrt @ adj_hat @ D_hat_inv_sqrt
    # print("norm_fc",adj_norm)
    return torch.tensor(adj_norm, dtype=torch.float32)

# Row normalization and Laplacian normalization for structural connectivity
def normalize_structural_adjacency(adj):
    I = np.eye(adj.shape[0])
    adj_hat = adj+I
    # adj_hat = min_max_normalize_rows(adj_hat)
    # Row normalization (make each row sum to 1)
    row_sums = adj_hat.sum(axis=1, keepdims=True)
    adj_hat = adj_hat / row_sums  # Broadcasting for row normalization
    
    # Compute the degree matrix after row normalization
    D = np.diag(np.sum(adj_hat, axis=1))
    D_inv_sqrt = np.linalg.inv(np.sqrt(D))
    
    # Apply Laplacian normalization
    adj_norm = D_inv_sqrt @ adj_hat @ D_inv_sqrt
    # print("norm_sc",adj_norm)
    return torch.tensor(adj_norm, dtype=torch.float32)

# Dataset for loading connectivity matrices
class ConnectivityDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.data = []
        self.labels = []
        self.load_data()

    def load_data(self):
        for label, folder in enumerate(['Health', 'AD']):
            folder_path = os.path.join(self.root_dir, folder)
            for subject in os.listdir(folder_path):
                subject_path = os.path.join(folder_path, subject)
                func_conn = np.loadtxt(os.path.join(subject_path, "FunctionalConnectivity.txt"))
                struct_conn = np.loadtxt(os.path.join(subject_path, "StructuralConnectivity.txt"))
                # print(subject)
                # print("fc",func_conn)
                # print("sc",struct_conn)
                # Normalize matrices
                func_conn = normalize_adjacency(func_conn)
                struct_conn = normalize_structural_adjacency(struct_conn)
                
                self.data.append((func_conn, struct_conn))
                self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        func_conn, struct_conn = self.data[idx]
        label = self.labels[idx]
        return func_conn, struct_conn, label

# Define GCN Layer
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, adj, x):
        x = torch.matmul(adj, x)
        return self.linear(x)

# GCN Model for classification
class GCNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCNModel, self).__init__()
        self.gcn1 = GCNLayer(input_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim)
        self.gcn3 = GCNLayer(input_dim, hidden_dim)
        self.gcn4 = GCNLayer(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, func_conn, struct_conn):
        # Functional GCN branch
        x1 = torch.relu(self.gcn1(func_conn, func_conn))
        x1 = torch.relu(self.gcn2(func_conn, x1))
        
        # Structural GCN branch
        x2 = torch.relu(self.gcn3(struct_conn, struct_conn))
        x2 = torch.relu(self.gcn4(struct_conn, x2))

        # Flatten and concatenate
        x1 = torch.mean(x1, dim=1)
        x2 = torch.mean(x2, dim=1)
        x = torch.cat([x1, x2], dim=1)
        
        # Fully connected output layer
        return self.fc(x)

def train_and_evaluate(root_dir, num_epochs=10, k_folds=5):
    dataset = ConnectivityDataset(root_dir)
    labels = dataset.labels  # Get labels to apply stratified k-fold
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    all_accuracies = []
    criterion = nn.CrossEntropyLoss()
    all_accuracies, all_f1_scores, all_aucs, all_sensitivities, all_specificities = [], [], [], [], []
    for fold, (train_idx, test_idx) in enumerate(skf.split(dataset.data, labels)):
        print(f'Fold {fold + 1}/{k_folds}')
        
        # Sample elements for each fold
        train_subsampler = torch.utils.data.Subset(dataset, train_idx)
        test_subsampler = torch.utils.data.Subset(dataset, test_idx)
        
        train_loader = DataLoader(train_subsampler, batch_size=4, shuffle=True)
        test_loader = DataLoader(test_subsampler, batch_size=4, shuffle=False)

        # Initialize the model
        model = GCNModel(input_dim=150, hidden_dim=64, output_dim=2).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=0.1)

        # Training loop
        for epoch in range(num_epochs):
            model.train()
            for func_conn, struct_conn, labels in train_loader:
                func_conn, struct_conn, labels = func_conn.to(device), struct_conn.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(func_conn, struct_conn)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}')

        # Evaluation on test set
        model.eval()
        all_preds, all_labels, all_probs = [], [], []
        
        with torch.no_grad():
            for func_conn, struct_conn, labels in test_loader:
                func_conn, struct_conn, labels = func_conn.to(device), struct_conn.to(device), labels.to(device)
                outputs = model(func_conn, struct_conn)
                probs = torch.softmax(outputs, dim=1)[:, 1]  # Probability for the positive class
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        # Compute metrics
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        auc = roc_auc_score(all_labels, all_probs)
        cm = confusion_matrix(all_labels, all_preds)
        
        # Sensitivity and Specificity calculation
        tp = cm[1, 1]
        tn = cm[0, 0]
        fp = cm[0, 1]
        fn = cm[1, 0]
        sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0

        # Store metrics for each fold
        all_accuracies.append(accuracy)
        all_f1_scores.append(f1)
        all_aucs.append(auc)
        all_sensitivities.append(sensitivity)
        all_specificities.append(specificity)

        print(f'Fold {fold + 1} - Accuracy: {accuracy * 100:.2f}% | F1 Score: {f1:.2f} | AUC: {auc:.2f} | Sensitivity: {sensitivity:.2f} | Specificity: {specificity:.2f}')
        sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
        plt.title(f"Confusion Matrix for Fold {fold + 1}")
        plt.show()

    # Print mean metrics across all folds
    print(f'Mean Accuracy: {np.mean(all_accuracies) * 100:.2f}%')
    print(f'Mean F1 Score: {np.mean(all_f1_scores):.4f}')
    print(f'Mean AUC: {np.mean(all_aucs):.4f}')
    print(f'Mean Sensitivity: {np.mean(all_sensitivities):.4f}')
    print(f'Mean Specificity: {np.mean(all_specificities):.4f}')

# Set the seed for reproducibility
def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Run cross-validation training and evaluation
if __name__ == "__main__":
    seed_everything(830)
    parser = argparse.ArgumentParser()
        # Set the number of epochs for training
    parser.add_argument(
        "-epochs", "--epochs", type=int, default=10, help="Number of training epochs"
    )
        # Set the number of folds for cross-validation
    parser.add_argument(
        "-k_folds", "--k_folds", type=int, default=5, help="Number of folds for cross-validation"
    )
    args = parser.parse_args()
    train_and_evaluate(root_dir='./', num_epochs=args.epochs, k_folds=args.k_folds)