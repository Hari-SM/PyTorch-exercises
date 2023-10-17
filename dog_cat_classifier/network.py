from torch import nn
import torch.nn.functional as F

class MLPClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        input_units = 224*224*3
        output_units = 2

        # Hyperparameters to the network
        hidden_layer1 = 256
        hidden_layer2 = 128
        hidden_layer3 = 64

        # Hidden layers
        self.fc1 = nn.Linear(input_units, hidden_layer1)
        self.fc2 = nn.Linear(hidden_layer1, hidden_layer2)
        self.fc3 = nn.Linear(hidden_layer2, hidden_layer3)

        # Dropout module
        self.dropout = nn.Dropout(0.2)

        # Output layers
        self.output = nn.Linear(hidden_layer3, output_units)

    def forward(self, x):
        # To flatten the image
        # x = torch.flatten(x, start_dim=1, end_dim=3)
        x = x.view(x.shape[0], -1)
        
        # Forward pass with activation function
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = F.log_softmax(self.output(x), dim=1)
        return x
