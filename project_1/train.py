import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import nibabel as nib
import numpy as np
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import argparse
import seaborn as sns

# Define the device to use (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MRIDataset(Dataset):
    def __init__(self, img_dir, labels, transform=None):
        self.img_dir = img_dir
        self.labels = labels
        self.transform = transform
        self.data = []
        self.targets = []
        self.load_data()

    # Load the data from the specified directories
    def load_data(self):
        for label, folder in enumerate(['health', 'patient']):
            folder_path = os.path.join(self.img_dir, folder)
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                data = read_nifti_file(file_path)
                self.data.append(data)
                self.targets.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        img = np.expand_dims(img, axis=0)  # Add channel dimension

        # Convert numpy array to torch tensor
        img = torch.tensor(img, dtype=torch.float32)
        
        label = self.targets[idx]
        
        # Apply any transformations if provided
        if self.transform:
            img = self.transform(img)
        
        return img, torch.tensor(label, dtype=torch.long)

# Function to read NIfTI files
def read_nifti_file(nifti_file):
    nii_image = nib.load(nifti_file)
    nii_data = nii_image.get_fdata()  
    return nii_data

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool3d(2, 2)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 35 * 49 * 47, 128)  # Adjust based on your MRI data shape
        self.fc2 = nn.Linear(128, 2)  # 2 output classes: health and patient

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 35 * 49 * 47)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Training function
def train(model, criterion, optimizer, dataloader, epochs=10):
    model.train()
    train_losses = []  # Save training loss for each epoch
    for epoch in range(epochs):
        running_loss = 0.0
        epoch_loss = 0.0
        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            epoch_loss += loss.item()

            if i % 10 == 9:  # Print loss every 10 steps
                print(f'Epoch {epoch + 1}, Step {i + 1}, Loss: {running_loss / 10}')
                running_loss = 0.0

        avg_epoch_loss = epoch_loss / len(dataloader)  # Calculate average loss per epoch
        train_losses.append(avg_epoch_loss)
        print(f'Epoch {epoch + 1}, Average Loss: {avg_epoch_loss:.4f}')
    
    # Plot the loss curve
    plt.figure()
    plt.plot(range(1, epochs+1), train_losses, marker='o', label='Training Loss')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('./loss.png')
    plt.show()

    return train_losses

# Function to evaluate the model on the test set
def test(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            # print(outputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print(f'Confusion Matrix:\n{cm}')
    
    # Visualize confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Health', 'Patient'], yticklabels=['Health', 'Patient'])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig('./confusion_matrix.png')
    plt.show()

    # Calculate and print accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    print(f'Test Accuracy: {accuracy * 100:.2f}%')
    
# Function to visualize model predictions on test images
def visualize_predictions(model, dataloader, num_images=5):
    model.eval()
    images_so_far = 0
    plt.figure(figsize=(12, 10))  # Set figure size
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            for i in range(inputs.size(0)):
                if images_so_far == num_images:
                    break
                
                # Get MRI slice (assuming 3D MRI images)
                img = inputs[i].cpu().numpy()[0]  
                
                plt.subplot(num_images // 2, 5, images_so_far + 1)
                plt.imshow(np.rot90(img[:,70],k=1), cmap='gray')  # Rotate 270 degrees for display
                plt.title(f'Pred: {preds[i].item()}, Label: {labels[i].item()}')
                plt.axis('off')
                
                images_so_far += 1
        plt.savefig('./visualization.png')
        plt.show()

# Save model to file
def save_model(model, path='cnn_model.pth'):
    torch.save(model.state_dict(), path)
    print(f'Model saved to {path}')

# Load model from file
def load_model(model, path='cnn_model.pth'):
    model.load_state_dict(torch.load(path))
    model.eval()  # Set to evaluation mode
    print(f'Model loaded from {path}')
    return model

# Inference function
def inference(model_path):
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize images
    ])
    test_dataset = MRIDataset(img_dir='./Testing', labels=[0, 1], transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    # Initialize the model
    model = CNNModel().to(device)
    
    # Load the saved model
    load_model(model, model_path)
    
    # Test the model
    test(model, test_loader)
    
    # Visualize predictions
    visualize_predictions(model, test_loader, num_images=10)  # Show ten images

# Function to train the model
def training_model(model_save_path,epochs):
    # Data transformation
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize images
    ])

    # Load training and testing datasets
    train_dataset = MRIDataset(img_dir='./Training', labels=[0, 1], transform=transform)
    test_dataset = MRIDataset(img_dir='./Testing', labels=[0, 1], transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    # Initialize the model
    model = CNNModel().to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_losses = train(model, criterion, optimizer, train_loader, epochs=epochs)
    save_model(model, model_save_path)
    
    # Test the model
    test(model, test_loader)
    
    # Visualize predictions
    visualize_predictions(model, test_loader, num_images=10)

# Set the seed for reproducibility
def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Main program entry
if __name__ == "__main__":
    seed_everything(830)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-mode", "--mode", type=str, default="train", help="train/inference"
    )
    # Model save or load path
    parser.add_argument(
        "-model_path", "--model_path", type=str, default="cnn_model.pth", help="Path to save or load the model"
    )

    # Set the number of epochs for training
    parser.add_argument(
        "-epochs", "--epochs", type=int, default=10, help="Number of training epochs"
    )
    args = parser.parse_args()
    if args.mode=='train':
        training_model(model_save_path=args.model_path, epochs=args.epochs)
    elif args.mode=='inference':
        inference(model_path=args.model_path)
    else:
         print(f"Error: Invalid mode '{args.mode}'. Please choose 'train' or 'inference'.")
