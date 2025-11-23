import torch

def evaluate_model(model, X, Y):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        outputs = model(X)
        predictions = torch.sigmoid(outputs) > 0.5  # Convert logits to binary predictions
        accuracy = (predictions == Y).float().mean()  # Calculate accuracy
        print(f"Accuracy: {accuracy.item() * 100}%")
        return accuracy.item()
