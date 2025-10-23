import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, roc_auc_score, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt

# Data Reading and Preprocessing
data_dir = './UCMerced_LandUse/Images'
batch_size = 32

# Data augmentation for the training set
train_transform = transforms.Compose([
    transforms.RandomResizedCrop((256, 256), scale=(0.5, 1.0)),
    transforms.RandomRotation(degrees=360),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Validation transform (no augmentation)
val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Datasets
dataset = datasets.ImageFolder(data_dir, transform=None)  # No transform applied initially
train_size = int(0.5 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Apply transformations
train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform = val_transform

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Capsule Layers
class PrimaryCaps(nn.Module):
    def __init__(self, in_channels, num_capsules, capsule_dim, kernel_size, stride):
        super(PrimaryCaps, self).__init__()
        self.capsules = nn.Conv2d(in_channels, num_capsules * capsule_dim, kernel_size, stride)
        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim

    def forward(self, x):
        x = self.capsules(x)
        batch_size = x.size(0)
        x = x.view(batch_size, self.num_capsules, self.capsule_dim, -1)
        x = x.permute(0, 1, 3, 2).contiguous()
        x = x.view(batch_size, self.num_capsules, -1)
        return self.squash(x)

    @staticmethod
    def squash(x):
        norm = torch.norm(x, dim=-1, keepdim=True)
        scale = (norm ** 2) / (1 + norm ** 2)
        return scale * (x / (norm + 1e-8))

class DigitCaps(nn.Module):
    def __init__(self, in_capsules, in_dim, num_capsules, capsule_dim):
        super(DigitCaps, self).__init__()
        self.num_capsules = num_capsules
        self.in_capsules = in_capsules
        self.in_dim = in_dim
        self.capsule_dim = capsule_dim

        self.W = nn.Parameter(0.01 * torch.randn(1, in_capsules, num_capsules, capsule_dim, in_dim))

    def forward(self, x):
        batch_size = x.size(0)
        x = x.unsqueeze(2).unsqueeze(4)
        W = self.W.expand(batch_size, -1, -1, -1, -1)
        u_hat = torch.matmul(W, x).squeeze(4)

        b = torch.zeros_like(u_hat[:, :, :, 0])

        for _ in range(2):
            c = torch.softmax(b, dim=2)
            s = (c.unsqueeze(-1) * u_hat).sum(dim=1)
            v = self.squash(s)
            b = b + (u_hat * v.unsqueeze(1)).sum(-1)

        return v

    @staticmethod
    def squash(x):
        norm = torch.norm(x, dim=-1, keepdim=True)
        scale = (norm ** 2) / (1 + norm ** 2)
        return scale * (x / (norm + 1e-8))

# CapsNet Definition
class CapsNet(nn.Module):
    def __init__(self, num_classes):
        super(CapsNet, self).__init__()
        self.vgg = models.vgg16(pretrained=True)
        self.vgg.classifier = nn.Identity()

        self.primary_caps = PrimaryCaps(in_channels=512, num_capsules=32, capsule_dim=8, kernel_size=8, stride=2)
        self.dropout = nn.Dropout(p=0.5)
        self.digit_caps = DigitCaps(in_capsules=32, in_dim=8, num_capsules=num_classes, capsule_dim=16)

    def forward(self, x):
        x = self.vgg.features(x)
        x = self.primary_caps(x)
        x = self.dropout(x)
        x = self.digit_caps(x)
        return x

class MarginLoss(torch.nn.Module):
    def __init__(self, m_plus=0.9, m_minus=0.1, lambda_=0.5):
        super(MarginLoss, self).__init__()
        self.m_plus = m_plus
        self.m_minus = m_minus
        self.lambda_ = lambda_

    def forward(self, v, labels):
        """
        Args:
            v (torch.Tensor): Capsule output vectors of shape (batch_size, num_classes, capsule_dim).
            labels (torch.Tensor): One-hot encoded labels of shape (batch_size, num_classes).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        v_norm = torch.norm(v, dim=-1)  # Compute length of capsule vectors (batch_size, num_classes)

        # Loss for the correct class
        positive_loss = F.relu(self.m_plus - v_norm).pow(2)
        # Loss for incorrect classes
        negative_loss = F.relu(v_norm - self.m_minus).pow(2)

        # Combine positive and negative losses
        loss = labels * positive_loss + self.lambda_ * (1 - labels) * negative_loss

        # Average over all classes and batch samples
        return loss.mean()

# Training and Validation with Integer Epoch Labels
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = len(dataset.classes)
model = CapsNet(num_classes=num_classes).to(device)

criterion = MarginLoss(m_plus=0.9, m_minus=0.1, lambda_=0.5)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 50

# To store losses for plotting
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        one_hot_labels = F.one_hot(labels, num_classes=num_classes).float()
        loss = criterion(outputs, one_hot_labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    # Validation step
    # Validation step with AUROC computation
    # Validation step with Macro AUROC computation
    model.eval()
    val_loss = 0
    correct = 0
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_loss += criterion(outputs, F.one_hot(labels, num_classes=num_classes).float()).item()
            preds = outputs.norm(dim=-1).argmax(dim=1)
            correct += (preds == labels).sum().item()
    
            # Collect ground truth and predicted probabilities
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(outputs.norm(dim=-1).cpu().numpy())
    
    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    accuracy = correct / len(val_dataset)
    
    # Compute Macro AUROC
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    one_hot_labels = np.eye(num_classes)[all_labels]
    macro_auroc = roc_auc_score(one_hot_labels, all_probs, average="macro", multi_class="ovr")
    
    # Display metrics
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
          f"Accuracy: {accuracy:.4f}, Macro AUROC: {macro_auroc:.4f}")
    
    # Live plot after each epoch (train/val losses, accuracy, etc.)
    plt.plot(train_losses, '-or', label='Train Loss')
    plt.plot(val_losses, '-ob', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend(loc='upper right')
    plt.xticks(np.arange(0, num_epochs + 1, max(1, num_epochs // 10)))  # Ensure integer labels
    plt.savefig('loss_plot.png', dpi=200)
    plt.close()
    
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = outputs.norm(dim=-1).argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Create confusion matrix
cm = confusion_matrix(all_labels, all_preds, labels=np.arange(num_classes))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dataset.classes)

# Plot confusion matrix with improved readability
fig, ax = plt.subplots(figsize=(12, 12))  # Increase figure size for better clarity
disp.plot(ax=ax, cmap='Blues', values_format='d')

# Rotate x-axis labels by 45 degrees
plt.xticks(rotation=45, ha='right')  # Adjust the alignment to make the labels readable

# Add padding around the plot
plt.tight_layout(pad=3.0)

plt.title("Confusion Matrix for Validation Set")
plt.savefig('confusion_matrix_updated.png', dpi=200)
plt.show()