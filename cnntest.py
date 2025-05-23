import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


import matplotlib.pyplot as plt
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#hyper-parameter 
num_epochs = 4
batch_size = 4
lr = 10**-3


# input image [0, 255] -> [0, 1] -> Normalization [-1, 1]
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
    ]
)
#dataset
train_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

#dataloader
train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle=True)

#define CIFAR10 10 classes
classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


class ConvNet(nn.Module):

    def __init__(self): 
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120) #Flatten
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)



    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.contiguous().view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        return x

#define model
model = ConvNet().to(device)

#loss, optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr)

#steps
steps = len(train_dataloader)

#train
for epoch in range(num_epochs):

    model.train()
    for i, (images, labels) in enumerate(train_dataloader):

        images = images.to(device)
        labels = labels.to(device)

        output = model(images)
        loss = loss_fn(output, labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (i+1) % 2000 == 0 : # i+1 : steps
            print(f"Epoch {epoch+1} / {num_epochs}, Step [{i+1} / {steps}], Loss {loss.item():.4f} ")

print("Finished Training")

model.eval()
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]

    for images, labels in test_dataloader:
        images = images.to(device)
        labels = labels.to(device)
        output = model(images)

        _, predicted = torch.max(output, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]

            if (label == pred):
                n_class_correct[label] += 1
                
            n_class_samples[label] += 1
    
    acc = 100.0 * n_correct / n_samples
    print(f"accuracy of the network : {acc}%")

    for i in range(10):
        acc = 100.0 * n_class_correct / n_class_samples
        print(f"accuracy of {classes[i]} : {acc}%")





    
    

