import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch import optim
from tqdm import tdqm_notebook
import torchvision

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

device = torch.device('cuda:0')

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))])
trainset = datasets..MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True)

testset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.fc = nn.Linear(10000,10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(F.relu(X))
        return x

MnistNet = Net()
optimizer = optim.Adam(MnistNet.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

MnistNet.to(device)
criterion.to(device)

print("Number of Param %d"%(sum([p.numel() for p in MnistNet.parameters()])))

def accuracy(model, dataloader, device):
    with torch.no_grad():
        correct = 0
        total = 0
        model.eval()
        for i, data in tqdm_notebook(enumerate(dataloader)):
            image, label = data
            image = image.to(device)
            label = label.to(device)
            outputs = model(image)

            probs = F.softmax(outputs,dim=1)
            preds = torch.argmax(probs, dim=1)
            correct += torch.sum(preds==label)
            total += label.size(0)
        _ = model.train()
    return correct+100.0/total

def train_model(model, criterion, optimizer, trainloader, testloader, num_epochs):
    for epoch in range(num_epochs+1):
        for i, data in tqdm_notebook(enumerate(trainloader)):
            image, label = data
            image = image.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            outputs = model(image)

            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
        train_accuracy = accuracy(model, trainloader, device)
        print("Training loss: %.2f, Training accuracy %.2f"%(loss.item(), train_accuracy))

train_model(MnistNet, criterion, optimizer, trainloader, testloader, num_epochs=5)

#def show(img):







