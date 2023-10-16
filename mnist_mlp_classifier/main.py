import torch
from torch import nn, optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from network import MLPClassifier
from helper import view_classification

# Apply transformation to images to normalize it
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize([0.5], [0.5])])

# Download the data and load it to trainloader for training
trainset = datasets.MNIST('MNIST_data', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Download the data and load it to testloader for testing
testset = datasets.MNIST('MNIST_data', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

# Define the model
model = MLPClassifier()

# Define the loss
criterion = nn.NLLLoss()

# Define optimizers with model parameters and learning rate
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Define number of epochs
epochs = 15

train_losses, test_losses = list(), list()

# Training and validating the data
for e in range(epochs):
    tot_test_loss = 0

    # Model training mode
    model.train()
    for images, labels in trainloader:
        # Predict Log probabilities
        logps = model(images)

        # Reset grad
        optimizer.zero_grad()

        # Calculate loss
        loss = criterion(logps, labels)

        # Back propagation
        loss.backward()

        # Update parameters
        optimizer.step()

        tot_test_loss += loss.item()

    else:
        # Model validation mode
        tot_train_loss = 0
        test_correct = 0
        model.eval()
        with torch.no_grad():
            for images, labels in testloader:
                logps = model(images)
                loss = criterion(logps, labels)
                tot_train_loss += loss.item()

                # Selecting the high probabilty as predicted label
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape) 
                test_correct += equals.sum().item()

        # Get mean loss to enable comparison between train and test sets
        train_loss = tot_train_loss / len(trainloader.dataset)
        test_loss = tot_test_loss / len(testloader.dataset)

        # At completion of epoch
        train_losses.append(train_loss)
        test_losses.append(test_loss)

        print("Epoch: {}/{}.. ".format(e+1, epochs),
              "Training Loss: {:.3f}.. ".format(train_loss),
              "Test Loss: {:.3f}.. ".format(test_loss),
              "Test Accuracy: {:.3f}".format(test_correct / len(testloader.dataset)))

plt.plot(train_losses, label='Training loss')
plt.plot(test_losses, label='Validation loss')
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