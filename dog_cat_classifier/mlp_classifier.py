import torch
from torch import nn, optim
from torchvision import datasets, transforms

from network import MLPClassifier
import matplotlib.pyplot as plt

import helper

train_transform = transforms.Compose([transforms.RandomRotation(30),
                                transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5],
                                                     [0.5, 0.5, 0.5])])

trainset = datasets.ImageFolder("data", transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

test_transform = transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor()])
testset = datasets.ImageFolder("data", transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True)

images, labels = next(iter(trainloader))

model = MLPClassifier()
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 2
train_losses, test_losses = list(), list()

for e in range(epochs):
    model.train()
    total_train_loss = 0
    for images, labels in trainloader:
        logps = model.forward(images)
        optimizer.zero_grad()
        loss = criterion(logps, labels)
        loss.backward()
        total_train_loss += loss.item()

    else:
        total_test_loss = 0
        test_accuracy = 0
        model.eval()
        for images, labels in testloader:
            logps = model.forward(images)
            loss = criterion(logps, labels)
            total_test_loss += loss.item()

            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            test_accuracy += equals.sum().item()

        # Get mean loss to enable comparison between train and test sets
        train_loss = total_train_loss / len(trainloader.dataset)
        test_loss = total_test_loss / len(testloader.dataset)

        # At completion of epoch
        train_losses.append(train_loss)
        test_losses.append(test_loss)

        print("Epoch: {}/{}.. ".format(e+1, epochs),
              "Training Loss: {:.3f}.. ".format(train_loss),
              "Test Loss: {:.3f}.. ".format(test_loss),
              "Test Accuracy: {:.3f}".format(test_accuracy / len(testloader.dataset)))

plt.plot(train_losses, label='Training loss')
plt.plot(test_losses, label='Validation loss')
plt.legend(frameon=False)
plt.show()

"""

fig, ax = plt.subplots(figsize=(10, 4), ncols=4)

for i in range(4):
    print(images[i].shape)
    helper.imshow(images[i], ax=ax[i], normalize=False)

plt.show()
"""

images, labels = next(iter(trainloader))

# Testing the model
with torch.no_grad():
    outps = model(images[0].view(1, 784))

# Plotting the result
helper.view_classify(images[0].view(28, 28), torch.exp(outps))