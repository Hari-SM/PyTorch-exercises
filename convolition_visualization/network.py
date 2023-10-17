from torch import nn
import torch.nn.functional as F

class Network(nn.Module):
    """
    Define a neural network with convolutional layer with four filters 
    and a pooling layer of size (2, 2)
    """
    def __init__(self, weights):
        super().__init__()
        ## Initialise the kernal values
        k_height, k_weight = 4, 4 # weights.shape[2:]

        ## Creating convolutional layer for a 1 channel image, assuming 4 kernals
        self.conv = nn.Conv2d(1, 4, kernel_size=(k_height, k_weight), bias=False)
        self.conv.weight = nn.Parameter(weights)

        ## Define a pooling layer
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        ## Calculating the output of a convolutional layer
        ## pre- and post-activation
        conv_x = self.conv(x)
        activated_x = F.relu(conv_x)

        ## Applies pooling layer
        pooled_x = self.pool(activated_x)

        return conv_x, activated_x, pooled_x