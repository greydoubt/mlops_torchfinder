import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from deep_learning.model import PathFindingCNN
from deep_learning.data_loader import ImageDataset

class Trainer:
    def __init__(self, dataset_path, model_save_path, epochs=10, batch_size=32, learning_rate=0.001):
        self.dataset_path = dataset_path
        self.model_save_path = model_save_path
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = PathFindingCNN().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.CrossEntropyLoss()

    def train(self):
        train_dataset = ImageDataset(self.dataset_path, train=True)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(1, self.epochs+1):
            print(f'Epoch {epoch}/{self.epochs}:')

            epoch_loss = 0
            epoch_accuracy = 0
            self.model.train()

            for i, (images, labels) in enumerate(tqdm(train_loader)):
                images = images.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()

                self.optimizer.step()

                _, predicted = torch.max(outputs.data, 1)
                epoch_accuracy += (predicted == labels).sum().item()
                epoch_loss += loss.item()

            epoch_loss /= len(train_loader.dataset)
            epoch_accuracy /= len(train_loader.dataset)
            print(f'Training loss: {epoch_loss:.4f}, accuracy: {epoch_accuracy*100:.2f}%')

        torch.save(self.model.state_dict(), self.model_save_path)
