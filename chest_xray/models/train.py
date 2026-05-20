import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from pathlib import Path
from chest_xray.data.chestdataset import ChestXRayDataset
from chest_xray.data.labels import CLASSES

class ModelTrainer:
    def __init__(self, model, criterion, optimizer, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.root = Path(__file__).parent.parent.parent
        self.image_root = self.root / "data" / "images"
        self.classes = CLASSES
        self.batch_size = 16
        self.image_size = 512
        self.seed = 42
        
    def load_csv(self):
        csv_file = self.root / "data" / "lists" / "Data_Entry_2017.csv"
        return pd.read_csv(csv_file)
    
    def transform_images(self, image_size):
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=3),  # Convert to grayscale 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.449], std=[0.226]),
        ])
        
    def create_dataloaders(self, data, image_root, classes, transform):
        unique_patients = data["Patient ID"].unique()
        
        generator = torch.Generator().manual_seed(self.seed)
        perm = torch.randperm(len(unique_patients), generator=generator)
        
        unique_patients = unique_patients[perm.numpy()]
        train_patient_count = int(0.8 * len(unique_patients))
        
        train_patients = set(unique_patients[:train_patient_count])
        val_patients = set(unique_patients[train_patient_count:])
        
        train_data = data[data["Patient ID"].isin(train_patients)].reset_index(drop=True)
        val_data = data[data["Patient ID"].isin(val_patients)].reset_index(drop=True)
        
        train_dataset = ChestXRayDataset(train_data, image_root, classes, transform)
        val_dataset = ChestXRayDataset(val_data, image_root, classes, transform)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
        return train_loader, val_loader

    def train_one_epoch(self, epoch, num_epochs, phase, dataloader):
        self.model.train()

        running_loss = 0.0

        progress_bar = tqdm(
            dataloader,
            total=len(dataloader),
            desc=f"Epoch [{epoch + 1}/{num_epochs}] {phase} Train",
            unit="batch",
            dynamic_ncols=True
        )

        for batch_idx, (images, labels) in enumerate(progress_bar, start=1):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            average_loss = running_loss / batch_idx

            progress_bar.set_postfix(loss=f"{average_loss:.4f}")

        return running_loss / len(dataloader)


    def validate_one_epoch(self, epoch, num_epochs, phase, dataloader):
        self.model.eval()

        running_loss = 0.0

        progress_bar = tqdm(
            dataloader,
            total=len(dataloader),
            desc=f"Epoch [{epoch + 1}/{num_epochs}] {phase} Val",
            unit="batch",
            dynamic_ncols=True
        )

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(progress_bar, start=1):
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item()
                average_loss = running_loss / batch_idx

                progress_bar.set_postfix(loss=f"{average_loss:.4f}")

        return running_loss / len(dataloader)
    
    def load_data(self):
        data = self.load_csv()
        transform = self.transform_images(self.image_size)
        train_loader, val_loader = self.create_dataloaders(data, self.image_root, self.classes, transform)
        return train_loader, val_loader

    def train(self, num_epochs, train_loader, val_loader):
        for epoch in range(num_epochs):
            train_loss = self.train_one_epoch(epoch, num_epochs, "Phase 1", train_loader)
            val_loss = self.validate_one_epoch(epoch, num_epochs, "Phase 1", val_loader)
            print(f"Epoch [{epoch + 1}/{num_epochs}] Phase 1 Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")