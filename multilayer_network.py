import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

from torchvision import datasets, transforms
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
#print(help(trainloader))

print("Train loader dataset ", trainloader.dataset)
dataloader_iter = iter(trainloader)
X, y = dataloader_iter.next()

print("Shape of X array ", X.shape)
print("Batch size of obtained", len(X))

print("Shape of image label ", y.shape)
print("Printing labels ", y)

print("Image flattened size = ", X[0].flatten().shape)

np.random.seed(0)

length_of_dict_class = 10
dict_class = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8:"eight", 9:"nine"}

X = X.squeeze()

def visualize_images(examples_per_class):
    for cls in range(length_of_dict_class):
        idxs = np.where((y==cls))[0]
        if len(idxs) < examples_per_class:
            continue
        idxs = np.random.choice(idxs, examples_per_class, replace=False)
        for i, idx in enumerate(idxs):
            plt.subplot(examples_per_class, length_of_dict_class, i * length_of_dict_class + cls + 1)
            plt.imshow(X[idx])
            plt.axis('off')
            if i==0:
                plt.title(dict_class[cls])
    plt.savefig('plt.png', dpi=300)

visualize_images(5)
