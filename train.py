import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import MNISTClassifier


# transform -> modul sluziaci na upravu, pripravu obrazkov
def data_load():
    #pipeline 
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)), # hodnota pixelov
        
    ])

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.RandomRotation(10)
    ]) 

    #nacitanie MNIST datasetu
    train_data = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=train_transform
    )

    test_data = datasets.MNIST(
        root='./data',
        train=False,
        transform=test_transform
    )

    # batching and shuffle
    train_loader = DataLoader(
        train_data,
        shuffle=True,
        batch_size=64
    )
    test_loader = DataLoader(
        test_data,
        shuffle=False,
        batch_size=64
    )

    return train_loader, test_loader
'''
1. forward pass
2. calculate loss
3. backpropagation
4. updating 
'''
def train(model, train_loader, epochs=5):
    define_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=0.001 # learning rate
    ) # minimalizovanie lossu

    for epoch in range(epochs):
        total_loss = 0 

        for image, label in train_loader:
            modelik = model(image)
            loss = define_loss(modelik, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} - Average Loss: {avg_loss:.4f}")

    