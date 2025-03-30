import os
import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

# Define the paths
data_dir = './mini_speech_commands'
commands = [command for command in os.listdir(data_dir) if command != "README.md"]

# Load and preprocess the data
def load_audio(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    return waveform, sample_rate

# STFT transform to convert audio waveform to spectrogram
def get_spectrogram(waveform, sample_rate, n_fft=400, win_length=400, hop_length=160, n_mels=23, target_width=100):
    # STFT to Mel Spectrogram
    mel_spectrogram = T.MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, win_length=win_length, 
                                       hop_length=hop_length, n_mels=n_mels)
    spectrogram = mel_spectrogram(waveform)
    
    # Padding or cropping to the target width
    _, _, width = spectrogram.shape
    if width < target_width:
        # Padding with zeros to the target width
        pad_width = target_width - width
        spectrogram = F.pad(spectrogram, (0, pad_width), mode='constant', value=0)
    elif width > target_width:
        # Cropping to the target width
        spectrogram = spectrogram[:, :, :target_width]
    
    return spectrogram


# Create Dataset class
class SpeechCommandsDataset(Dataset):
    def __init__(self, data_dir, commands, split='train', target_width=100):
        self.data_dir = data_dir
        self.commands = commands
        self.split = split
        self.target_width = target_width
        self.file_paths = []
        self.labels = []

        for idx, command in enumerate(commands):
            command_path = os.path.join(data_dir, command)
            for file in os.listdir(command_path):
                file_path = os.path.join(command_path, file)
                self.file_paths.append(file_path)
                self.labels.append(idx)

        # Split into train, validation, test
        train_files, test_files, train_labels, test_labels = train_test_split(self.file_paths, self.labels, test_size=0.2, random_state=42)
        val_files, test_files, val_labels, test_labels = train_test_split(test_files, test_labels, test_size=0.5, random_state=42)

        if self.split == 'train':
            self.file_paths = train_files
            self.labels = train_labels
        elif self.split == 'val':
            self.file_paths = val_files
            self.labels = val_labels
        else:
            self.file_paths = test_files
            self.labels = test_labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        waveform, sample_rate = load_audio(file_path)
        spectrogram = get_spectrogram(waveform, sample_rate, target_width=self.target_width)
        return spectrogram, label

# Create DataLoader
train_dataset = SpeechCommandsDataset(data_dir, commands, split='train')
val_dataset = SpeechCommandsDataset(data_dir, commands, split='val')
test_dataset = SpeechCommandsDataset(data_dir, commands, split='test')

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

class SpeechCommandModel(nn.Module):
    def __init__(self, num_classes=8):
        super(SpeechCommandModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # This reduces the height and width by a factor of 2
        self.fc1 = nn.Linear(64 * 12 * 12, 128)  # Adjusted size based on the actual output dimensions
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 23 * 23)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Initialize the model
model = SpeechCommandModel(num_classes=len(commands))

# Set the device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0
    for spectrograms, labels in train_loader:
        spectrograms = spectrograms.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(spectrograms)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track metrics
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct_preds += (predicted == labels).sum().item()
        total_preds += labels.size(0)

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * correct_preds / total_preds

    print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')

def evaluate(model, data_loader):
    model.eval()
    correct_preds = 0
    total_preds = 0
    with torch.no_grad():
        for spectrograms, labels in data_loader:
            spectrograms = spectrograms.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(spectrograms)
            _, predicted = torch.max(outputs, 1)

            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)

    accuracy = 100 * correct_preds / total_preds
    return accuracy

# Evaluate on validation and test sets
val_accuracy = evaluate(model, val_loader)
test_accuracy = evaluate(model, test_loader)

print(f'Validation Accuracy: {val_accuracy:.2f}%')
print(f'Test Accuracy: {test_accuracy:.2f}%')

