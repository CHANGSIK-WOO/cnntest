#library_torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

#library_torchvision for CIFAR10
import torchvision
import torchvision.transforms as transforms

#numpy for preprocessing and visualize
import numpy as np
import matplotlib.pyplot as plt

#config arguments parsing
import argparse
import yaml
import os



#parsing
parser = argparse.ArgumentParser()

parser.add_argument('--config', type=str, default=None)
parser.add_argument('--num_epoch', type=int, default=None)
parser.add_argument('--batch_size', type=int, default=None)
parser.add_argument('--lr', type=float, default=None)
parser.add_argument('--fc1_dim', type=int, default=None)
parser.add_argument('--fc2_dim', type=int, default=None)
parser.add_argument('--conv1_out', type=int, default=None)
parser.add_argument('--conv2_out', type=int, default=None)
parser.add_argument('--dropout', type=float, default=None)
args = parser.parse_args()

# Load config from YAML
if args.config is not None:
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
else:
    config = {}

# 🧠 핵심: config 값으로 args 덮어쓰기
for k, v in config.items():
    if getattr(args, k) is None:
        setattr(args, k, v)


#check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#input image [0, 255] -> [0, 1] -> Normalization [-1, 1] : (x-mean) / std
transform = transforms.Compose(
    [ 
        transforms.ToTensor(), # PILImage to Tensor [0,255] --> [0,1]
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) # Each Channel Normalzation [0,1] --> [-1,1]
    ]
)


#dataset
train_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.CIFAR10(root="./data", train=False, transform=transform, download=True)

#dataloader for batch
train_dataloader = DataLoader(dataset = train_dataset, batch_size = args.batch_size, shuffle = True)
test_dataloader = DataLoader(dataset = test_dataset, batch_size = args.batch_size, shuffle = True)

#CIFAR10 Classes
classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

#Model ConvNet
class Convnet(nn.Module):

    def __init__(self, conv1_out, conv2_out, fc1_dim, fc2_dim, dropout):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=conv1_out, kernel_size=5), # [3, 32, 32] --> [6, 28, 28]
            nn.BatchNorm2d(num_features = conv1_out),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # [6, 28, 28] --> 6, 14, 14]
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=conv1_out, out_channels=conv2_out, kernel_size=5), # [6, 14, 14] --> [16, 10, 10]
            nn.BatchNorm2d(num_features = conv2_out),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # [16, 10, 10] --> [16, 5, 5]
        ) 
        
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=conv2_out * 5 * 5, out_features=fc1_dim), # dim = 400 --> dim = 200
            nn.ReLU(),
            nn.Dropout(p = dropout)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(in_features=fc1_dim, out_features=fc2_dim), # dim = 200 --> dim = 100
            nn.ReLU(),
            nn.Dropout(p = dropout)
        )

        self.fc3 = nn.Linear(in_features = fc2_dim, out_features=10) # dim = 200 --> dim = 100
        # CrossEntropyLoss가 마지막에서 softmax(logits) 취급해서 raw score(logit) 그대로 주는 게 맞음. ReLU 쓰면 음수 점수 다 0돼서 정보 손실. Dropout도 마지막엔 잘 안 씀
         

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x) # [batch, 10]. x의 각 원소가 바로 logits!

        return x

#define model
model = Convnet(
    conv1_out=args.conv1_out,
    conv2_out=args.conv2_out,
    fc1_dim=args.fc1_dim,
    fc2_dim=args.fc2_dim,
    dropout=args.dropout
).to(device)

#loss, optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(params = model.parameters(), lr = args.lr)

#steps
steps = len(train_dataloader)

#train_net
for epoch in range(args.num_epoch):

    model.train()
    for i, (images, labels) in enumerate(train_dataloader):

        images = images.to(device)
        labels = labels.to(device)

        pred = model(images)
        loss = loss_fn(pred, labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (i+1) % 2000 == 0 :
            print(f"Epochs : {epoch} / {args.num_epoch}, Steps : [{i+1} / {steps}], loss : {loss.item():.4f}")

print("Finish Training Model")

#test
model.eval()
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]

    for images, labels in test_dataloader:
        images = images.to(device)
        labels = labels.to(device)
        output = model(images) # [bs, 10]

        _, predicted = torch.max(output, dim=-1) # [max_value, max_index]
        n_samples += labels.shape[0] # batch_size summation
        n_correct += (predicted == labels).sum().item()

        for i in range(labels.shape[0]):
            label = labels[i].item()
            pred = predicted[i].item()   

            if label == pred :
                n_class_correct[label] += 1
            
            n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples    
    print(f"accuracy of the network : {acc:.2f}%")

    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f"accuracy of {classes[i]} : {acc:.2f}%")