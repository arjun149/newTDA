import numpy as np
import torch
from tda_module.tda_feature_extraction import compute_persistent_homology
from tda_module.tda_model import DynamicNN
from tda_module.train import train_model
from tda_module.evaluate import evaluate_model

# Example time-series data (e.g., stock price movements)
time_series_data = np.random.rand(100, 2)  # 100 random 2D points (this should be your actual data)

# Step 1: Extract TDA features
tda_features = compute_persistent_homology(time_series_data)

# Combine TDA features into a feature matrix (for simplicity, we'll assume 1 sample)
X_tda = np.array([tda_features])  
Y = np.random.randint(0, 2, (1, 1))  # Random binary label (e.g., stock price movement up/down)

# Step 2: Convert to torch tensors
X = torch.tensor(X_tda, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.float32)

# Step 3: Define and train the model
input_size = len(tda_features)  # Number of TDA features (Betti numbers)
num_layers = 3  # Number of layers in the network

# Create and initialize the model
model = DynamicNN(input_size=input_size, num_layers=num_layers)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
train_model(model, X, Y, optimizer)

# Step 4: Evaluate the model
evaluate_model(model, X, Y)
