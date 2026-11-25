import torch

from train import data_load, train
from model import MNISTClassifier
from evaluate import evaluate

train_loader, test_loader = data_load()

model = MNISTClassifier()

train_model = train(model, train_loader)

vyhodnotenie = evaluate(model, test_loader)
print(vyhodnotenie)

# Save the model
torch.save(model.state_dict(), './model/mnist_model.pth')

# Load the model
loaded_model = MNISTClassifier()
loaded_model.load_state_dict(torch.load('./model/mnist_model.pth'))

# Set the model to evaluation mode
loaded_model.eval()
