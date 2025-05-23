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

def imshow(img):
    img = img / 2 * 0.5 #unnormalized
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, [1,2,0]))
    plt.show()


dataiter = iter(train_dataloader)
images, labels = next(dataiter) 

imshow(torchvision.utils.make_grid(images))

conv1 = nn.Conv2d(3, 6, 5)
pool = nn.MaxPool2d(2, 2)
conv2 = nn.Conv2d(6, 16, 5)
fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
fc2 = nn.Linear(in_features=120, out_features=84)
fc3 = nn.Linear(in_features=84, out_features=10)

print(images.shape)

# images -> conv1
x = conv1(images)
print(x.shape)

# conv1 -> pool
x = pool(x)
print(x.shape)

# pool -> conv2
x = conv2(x)
print(x.shape)

# conv1 -> pool
x = pool(x)
print(x.shape)

# flatten
x = x.contiguous().view(-1, 16 * 5 * 5)
print(x.shape)

# fc1
x = fc1(x)
print(x.shape)

# fc2
x = fc2(x)
print(x.shape)

# fc3
x = fc3(x)
print(x.shape)


