import torch
from torch import nn
import torch.nn.functional as F

class MLPClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        input_units = 28*28
        output_units = 10

        hidden_layer1 = 512
        hidden_layer2 = 256
        hidden_layer3 = 64

        self.fc1 = nn.Linear(input_units, hidden_layer1)
        self.fc2 = nn.Linear(hidden_layer1, hidden_layer2)
        self.fc3 = nn.Linear(hidden_layer2, hidden_layer3)

        self.dropout = nn.Dropout(0.2)

        self.output = nn.Linear(hidden_layer3, output_units)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))

        x = F.log_softmax(self.output(x), dim=1)
        return x