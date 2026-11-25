import torch

def evaluate(model, test_loader):
    model.eval()
    
    # Test model
    correct_predictions = 0
    total_samples = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted_labels = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted_labels == labels).sum().item()

    return(f"Accuracy of test set: {100 * correct_predictions / total_samples}%")