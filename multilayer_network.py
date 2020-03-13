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

#multi-later perceptron

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear_layer_one = torch.nn.Linear(784,128)
        self.linear_layer_two = torch.nn.Linear(128, 64)
        self.linear_layer_three = torch.nn.Linear(64,10)

        print("Base model defined")

    def forward(self, X):
        X_layer1 = F.relu(self.linear_layer_one(X))
        X_layer2 = F.relu(self.linear_layer_two(X_layer1))
        X_layer3 = self.linear_layer_three(X_layer2)
        return X_layer2

network = Model().to(device)
print(network)

cross_entropy_loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(network.parameters(), lr=0.1)

def cal_acc(probs, target):
    # probs: probability that each image is labeled as 1
    # target: ground truth label
    with torch.no_grad():
        prediction = torch.argmax(probs, axis=-1)
        acc = torch.sum(prediction == target)
    return acc.item() / len(target) * 100

def train_one_pass(p_network, p_optim):
    # Input: Network and Optimizer
    # Output: Averge accuracy , Avergae loss in the pass
    p_network.train()
    acc_one_pass=[]
    loss_one_pass=[]
    for i, batch in enumerate(iter(trainloader)):
        X = batch[0].squeeze()
        X = X.view(X.shape[0], -1)
        X = torch.tensor(X).to(device)
        y = torch.tensor(batch[1]).to(device)

        pred = p_network(X)

        loss = cross_entropy_loss(pred,y)

        p_optim.zero_grad()
        loss.backward()

        p_optim.step()
        if i % 100 == 0:
            print("Iter: {:.2f}. ".format(i / len(trainloader)),
                    "Train Total Loss: {:.2f}. ".format(loss.item()),
                    "Train Accuracy: {:.2f}% ".format(cal_acc(pred, y))
                    )
        acc_one_pass.append(cal_acc(pred, y))
        loss_one_pass.append(loss.item())
    return sum(acc_one_pass) / len(acc_one_pass), sum(loss_one_pass)/ len(loss_one_pass)


#avg_acc, avg_loss = train_one_pass(network, optimizer)

def test_one_pass(p_network):
    # Input: Network and Optimizer
    # Output: Averge accuracy , Avergae loss in the pass
    p_network.eval()
    acc_one_pass=[]
    loss_one_pass = []
    for i, batch in enumerate(iter(testloader)):
        X = batch[0].squeeze()
        X = X.view(X.shape[0], -1)
        X = torch.tensor(X).to(device)
        y = torch.tensor(batch[1]).to(device)
        with torch.no_grad():
            pred = p_network(X)
            loss = cross_entropy_loss(pred, y)

        acc_one_pass.append(cal_acc(pred, y))
        loss_one_pass.append(loss.item())
    return sum(acc_one_pass)/len(acc_one_pass), sum(loss_one_pass)/len(acc_one_pass)

def multiple_pass(p_network, p_optim):
    # Input: p_network, p_optim
    # Output: Output a chart accuracy
    all_epochs_acc=[]
    all_epochs_loss=[]

    for epoch in range(50):
        avg_acc, avg_loss = train_one_pass(p_network, p_optim)
        print('\033[92m',
          "Epoch: {:d}. ".format(epoch),
          "Training Total Loss: {:.2f}. ".format(avg_loss),
          "Training Accuracy: {:.2f}% ".format(avg_acc),
          '\033[0m')

    test_accuracy, test_loss = test_one_pass(p_network)
    print('\033[92m',
            "Epoch: {:d}. ".format(epoch),
            "Test Total Loss: {:.2f}. ".format(test_loss),
            "Test Accuracy: {:.2f}% ".format(test_accuracy),
            '\033[0m')

    all_epochs_acc.append(test_accuracy)
    all_epochs_loss.append(test_loss)

    #plot
    plt.subplot(1,2,1)
    plt.plot(all_epochs_acc)
    plt.title("test accuracy")

    plt.subplot(1,2,2)
    plt.plot(all_epochs_loss)
    plt.title("test loss")
    plt.savefig("test-loss-acc-multipass.png", dpi=300)

#print("Running the network with SGD optimizer with LR=0.1")
#multiple_pass(network, optimizer)


# ADAM optimizer
adam_network = Model().to(device)
adam_optim = torch.optim.Adam(adam_network.parameters(), lr=0.1, betas=(0.9,0.999))
print("Adam optimizer with same network")
#multiple_pass(adam_network, adam_optim)

def init_weights(m):
    print(m)
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        print(m.weight)

network_init = Model().to(device)
network_init.apply(init_weights)
adam_optim = torch.optim.Adam(network_init.parameters(), lr=0.01, betas=(0.9, 0.999))

print("Running the differently initialized model with Adam Optimizer")
#multiple_pass(network_init, adam_optim)


# New Model with Batch Norm and ReLU 

class Model_Normalization(nn.Module):
    def __init__(self):
        super(Model_Normalization, self).__init__()
        self.linear_layer_one = torch.nn.Linear(784, 128)
        self.bn1 = nn.BatchNorm1d(num_features=128)
        self.linear_layer_two = torch.nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(num_features=64)
        self.linear_layer_three = torch.nn.Linear(64, 10)
        self.bn3 = nn.BatchNorm1d(num_features=10)
        
        print("Base model defined")

    def forward(self, X):
        X_layer1 = F.relu(self.bn1(self.linear_layer_one(X)))
        X_layer2 = F.relu(self.bn2(self.linear_layer_two(X_layer1)))
        X_layer3 = self.bn3(self.linear_layer_three(X_layer2))
        return X_layer3

    def accuracy(self, probs, target):
    # probs: probability that each image is labeled as 1
    # target: ground truth label
        with torch.no_grad():
            prediction = torch.argmax(probs, axis=-1)
            acc = torch.sum(prediction == target)
        return acc.item() / len(target) * 100

network_bn = Model_Normalization().to(device)
print(network_bn)
adam_optim = torch.optim.Adam(network_bn.parameters(), lr=0.01, betas=(0.9, 0.999))

print("Running BN network with Adam Opt.")
multiple_pass(network_bn, adam_optim)
