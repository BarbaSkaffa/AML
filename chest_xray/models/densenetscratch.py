from torchvision import models
import torch
from chest_xray.models.train import ModelTrainer
from chest_xray.data.labels import CLASSES

# https://deepwiki.com/andreasveit/densenet-pytorch/6.2-training-configuration
# 161 model has the best top-1 and top-5 accuracy 



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

densenet = models.densenet161(weights=None)
densenet.classifier = torch.nn.Linear(densenet.classifier.in_features, len(CLASSES))
densenet = densenet.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(densenet.parameters(), lr=0.001)

modelTrainer = ModelTrainer(densenet, criterion, optimizer, device)

def trainModel():
    train_loader, val_loader = modelTrainer.load_data()
    modelTrainer.train(num_epochs=10, train_loader=train_loader, val_loader=val_loader)
    
if __name__ == "__main__":
    trainModel()