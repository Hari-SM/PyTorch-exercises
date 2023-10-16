import torch
from torchvision import datasets, transforms

from network import MLPClassifier
from helper import view_classification

# Apply transformation to images to normalize it
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize([0.5], [0.5])])

# Download the data and load it to testloader for testing
testset = datasets.MNIST('MNIST_data', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

# Define the model 
model = MLPClassifier()

# Loading the state dict
state_dict = torch.load('checkpoint.pth')
model.load_state_dict(state_dict['state_dict'])

# Testing the model
images, labels = next(iter(testloader))
with torch.no_grad():
    outps = model(images[0].view(1, 784))

# Plotting the result
view_classification(images[0].view(28, 28), torch.exp(outps))