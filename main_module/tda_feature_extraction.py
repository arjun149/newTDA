import torch
import torch.nn as nn

class DynamicNN(nn.Module):
    def __init__(self, input_size, num_layers=3):
        super(DynamicNN, self).__init__()
        
        # Dynamic layer generation
        self.layers = nn.ModuleList([nn.Linear(input_size, input_size) for _ in range(num_layers)])
        
        # Output layer (binary classification)
        self.output_layer = nn.Linear(input_size, 1)  # Binary output (1 for up, 0 for down)
        
    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        x = self.output_layer(x)
        return x
