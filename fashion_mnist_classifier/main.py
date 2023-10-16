import torch
from torch import nn, optim
from torchvision import datasets, transforms

from network import MLPClassifier
from helper import view_classification

import matplotlib.pyplot as plt

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize([0.5], [0.5])])

trainset = datasets.FashionMNIST('Fashion_MNIST', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = datasets.FashionMNIST('Fashion_MNIST', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

model = MLPClassifier()

criterion = nn.NLLLoss()

optimizer = optim.SGD(model.parameters(), lr=0.01)

epochs = 30

train_losses, test_losses = list(), list()

for e in range(epochs):
    # Training mode
    total_training_loss = 0
    model.train()
    for images, labels in trainloader:
        images = images.view(-1, 784)

        optimizer.zero_grad()

        logps = model.forward(images)

        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        total_training_loss += loss.item()

    else:
        # Eval mode
        total_test_loss = 0
        test_correct = 0
        model.eval()
        with torch.no_grad():
            for images, labels in testloader:
                logps = model(images)
                loss = criterion(logps, labels)
                total_test_loss += loss.item()

                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                test_correct += equals.sum().item()

        train_loss = total_training_loss/len(trainloader.dataset)
        test_loss = total_test_loss/len(testloader.dataset)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        print("Epoch number: {}/{}".format(e+1, epochs),
              "Train loss: {}".format(train_loss),
              "Test loss: {}".format(test_loss),
              "Test accuracy: {}".format(test_correct/len(testloader.dataset)))

plt.plot(train_losses, label="Training Loss")
plt.plot(test_losses, label="Validation loss")
plt.legend(frameon=False)
plt.show()

# To save the model for future references
checkpoint = {'input_size' : 784,
              'output_size': 10,
              'hidden_layers': [each.out_features for each in [model.fc1, model.fc2, model.fc3]],
              'state_dict': model.state_dict()}
torch.save(checkpoint, "checkpoint.pth")
images, labels = next(iter(trainloader))

# Testing the model
with torch.no_grad():
    outps = model(images[0].view(1, 784))

# Plotting the result
view_classification(images[0].view(28, 28), torch.exp(outps))
