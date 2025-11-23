import torch
import torch.optim as optim
import torch.nn as nn
from tda_module.tda_model import DynamicNN

# Cross-entropy loss for binary classification
criterion = nn.BCEWithLogitsLoss()

# Optimizer (Adam)
optimizer = optim.Adam

def train_model(model, X, Y, optimizer, epochs=100):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        outputs = model(X)  # Forward pass
        loss = criterion(outputs, Y)  # Compute loss
        
        loss.backward()  # Backpropagation
        optimizer.step()  # Optimize
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
    
    return model
