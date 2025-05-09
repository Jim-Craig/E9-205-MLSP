

import numpy as np
from PIL import Image
import glob
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms,models
import torch.nn as nn
import torch
from PIL import Image
import glob
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from collections import Counter
import gc
import cv2


#Global Varibales
gpu_id = 1  # Change this to select a different GPU
device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"

class EarlyStopping:
    def __init__(self, model, patience=5, min_delta=0.001):
        """
        Args:
            patience (int): Number of epochs to wait after last improvement.
            min_delta (float): Minimum change to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False
        self.model = model

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0  # Reset counter if loss improves
            self.model = model
        else:
            self.counter += 1  # Increase counter if no improvement
            if self.counter >= self.patience:
                self.early_stop = True


class CLIPDataset(Dataset):
    def __init__(self, image_paths, labels, transform, USE_DFT, USE_CROSS_DIFF):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.USE_DFT = USE_DFT
        self.USE_CROSS_DIFF = USE_CROSS_DIFF

    def __len__(self):
        return len(self.image_paths)

    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image_np = np.array(image)

        if self.USE_DFT:
            image_np = np.array(image)
            dft = np.fft.fft2(image_np, axes=(0, 1))
            dft_shifted = np.fft.fftshift(dft)
            magnitude_spectrum = 20 * np.log(np.abs(dft_shifted) + 1)
            image_np = np.uint8(np.clip(magnitude_spectrum, 0, 255))
            image = Image.fromarray(image_np.astype(np.uint8))

        if self.USE_CROSS_DIFF:
            image_np = np.array(image)
            blurred = cv2.GaussianBlur(image_np, (3, 3), 0)
            diff = cv2.absdiff(image_np, blurred)
            image = Image.fromarray(diff.astype(np.uint8))

        image = self.transform(image)
        label = self.labels[idx]
        return image, label

# Load pretrained ResNet-50 model
class ResNet50MultiClassClassifier(nn.Module):
    def __init__(self, pretrained=True, num_classes=11):
        super(ResNet50MultiClassClassifier, self).__init__()
        self.model = models.resnet50(pretrained=pretrained)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)  # 11 classes (10 AI-generated types + 1 real image)

    def forward(self, x):
        return self.model(x)

# Define transformations (data augmentation + normalization)
def get_transforms():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to ResNet-50 input size
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform


# Define the file paths for training set
def get_dataset(transform, USE_DFT, USE_CROSS_DIFF):
    file_paths = [
        '/data/home/saisuchithm/godwin/mlsp/project/diffusion_model_deepfakes_lsun_bedroom/train/ADM/1_fake/*.png',
        '/data/home/saisuchithm/godwin/mlsp/project/diffusion_model_deepfakes_lsun_bedroom/train/DDPM/1_fake/*.png',
        '/data/home/saisuchithm/godwin/mlsp/project/diffusion_model_deepfakes_lsun_bedroom/train/Diff-ProjectedGAN/1_fake/*.png',
        '/data/home/saisuchithm/godwin/mlsp/project/diffusion_model_deepfakes_lsun_bedroom/train/Diff-StyleGAN2/1_fake/*.png',
        '/data/home/saisuchithm/godwin/mlsp/project/diffusion_model_deepfakes_lsun_bedroom/train/IDDPM/1_fake/*.png',
        '/data/home/saisuchithm/godwin/mlsp/project/diffusion_model_deepfakes_lsun_bedroom/train/LDM/1_fake/*.png',
        '/data/home/saisuchithm/godwin/mlsp/project/diffusion_model_deepfakes_lsun_bedroom/train/PNDM/1_fake/*.png',
        '/data/home/saisuchithm/godwin/mlsp/project/diffusion_model_deepfakes_lsun_bedroom/train/ProGAN/1_fake/*.png',
        '/data/home/saisuchithm/godwin/mlsp/project/diffusion_model_deepfakes_lsun_bedroom/train/ProjectedGAN/1_fake/*.png',
        '/data/home/saisuchithm/godwin/mlsp/project/diffusion_model_deepfakes_lsun_bedroom/train/StyleGAN/1_fake/*.png',
        '/data/home/saisuchithm/godwin/mlsp/project/diffusion_model_deepfakes_lsun_bedroom/train/Real/0_real/*.jpg'
    ]

    label_ADM = 0
    label_DDPM = 1
    label_Diff_Pro_GAN = 2
    label_Diff_Style_GAN = 3
    label_IDDPM = 4
    label_LDM = 5
    label_PNDM = 6
    label_ProGAN = 7
    label_ProjectedGAN = 8
    label_StyleGAN = 9
    label_Real = 10

    # List of labels
    labels = [label_ADM, label_DDPM, label_Diff_Pro_GAN, label_Diff_Style_GAN, label_IDDPM, label_LDM, label_PNDM, label_ProGAN, label_ProjectedGAN, label_StyleGAN, label_Real]


    # Collect all image paths and corresponding labels
    image_paths = []
    image_labels = []

    for path, label in zip(file_paths, labels):
        images = glob.glob(path)  # Get all image file paths in the folder
        image_paths.extend(images)
        image_labels.extend([label] * len(images))  # Assign the same label to all images in that folder

    # Check if images and labels are aligned
    assert len(image_paths) == len(image_labels)

    # Example usage
    dataset = CLIPDataset(image_paths, image_labels, transform, USE_DFT, USE_CROSS_DIFF)

    # Define the file paths for validation set
    # Define the file paths for training set
    val_file_paths = [
        '/data/home/saisuchithm/godwin/mlsp/project/diffusion_model_deepfakes_lsun_bedroom/val/ADM/1_fake/*.png',
        '/data/home/saisuchithm/godwin/mlsp/project/diffusion_model_deepfakes_lsun_bedroom/val/DDPM/1_fake/*.png',
        '/data/home/saisuchithm/godwin/mlsp/project/diffusion_model_deepfakes_lsun_bedroom/val/Diff-ProjectedGAN/1_fake/*.png',
        '/data/home/saisuchithm/godwin/mlsp/project/diffusion_model_deepfakes_lsun_bedroom/val/Diff-StyleGAN2/1_fake/*.png',
        '/data/home/saisuchithm/godwin/mlsp/project/diffusion_model_deepfakes_lsun_bedroom/val/IDDPM/1_fake/*.png',
        '/data/home/saisuchithm/godwin/mlsp/project/diffusion_model_deepfakes_lsun_bedroom/val/LDM/1_fake/*.png',
        '/data/home/saisuchithm/godwin/mlsp/project/diffusion_model_deepfakes_lsun_bedroom/val/PNDM/1_fake/*.png',
        '/data/home/saisuchithm/godwin/mlsp/project/diffusion_model_deepfakes_lsun_bedroom/val/ProGAN/1_fake/*.png',
        '/data/home/saisuchithm/godwin/mlsp/project/diffusion_model_deepfakes_lsun_bedroom/val/ProjectedGAN/1_fake/*.png',
        '/data/home/saisuchithm/godwin/mlsp/project/diffusion_model_deepfakes_lsun_bedroom/val/StyleGAN/1_fake/*.png',
        '/data/home/saisuchithm/godwin/mlsp/project/diffusion_model_deepfakes_lsun_bedroom/val/Real/0_real/*.jpg'
    ]

    label_ADM = 0
    label_DDPM = 1
    label_Diff_Pro_GAN = 2
    label_Diff_Style_GAN = 3
    label_IDDPM = 4
    label_LDM = 5
    label_PNDM = 6
    label_ProGAN = 7
    label_ProjectedGAN = 8
    label_StyleGAN = 9
    label_Real = 10

    # List of labels
    val_labels = [label_ADM, label_DDPM, label_Diff_Pro_GAN, label_Diff_Style_GAN, label_IDDPM, label_LDM, label_PNDM, label_ProGAN, label_ProjectedGAN, label_StyleGAN, label_Real]

    # Collect all image paths and corresponding labels
    val_image_paths = []
    val_image_labels = []

    for path, label in zip(val_file_paths, val_labels):
        images = glob.glob(path)  # Get all image file paths in the folder
        val_image_paths.extend(images)
        val_image_labels.extend([label] * len(images))

    # Check if images and labels are aligned
    assert len(val_image_paths) == len(val_image_labels)

    val_dataset = CLIPDataset(val_image_paths, val_image_labels, transform, USE_DFT, USE_CROSS_DIFF)

    return dataset, val_dataset

def checkDatapoints(dataset, val_dataset):
    label_idx_to_name = {
        0: "Picture of AI generated image, from ADM method",
        1: "Picture of AI generated image, from DDPM method",
        2: "Picture of AI generated image, from Diff-ProjectedGAN method",
        3: "Picture of AI generated image, from Diff-StyleGAN",
        4: "Picture of AI generated image, from IDDPM method",
        5: "Picture of AI generated image, from LDM method",
        6: "Picture of AI generated image, from PNDM method",
        7: "Picture of AI generated image, from ProGAN method",
        8: "Picture of AI generated image, from ProjectedGAN method",
        9: "Picture of AI generated image, from StyleGAN method",
        10: "Picture of a Real Image"
    }

    print(f"Number of datapoints per label in training set")
    label_counts = Counter(dataset.labels)
    for label, count in label_counts.items():
        print(f"{label_idx_to_name.get(label)}: {count} images")

    from collections import Counter
    print(f"Number of datapoints per label in validation set")
    label_counts = Counter(val_dataset.labels)
    for label, count in label_counts.items():
        print(f"{label_idx_to_name.get(label)}: {count} images")


# Training Loop
def train_CNN(model, train_dummy_loader, val_dummy_loader, criterion, optimizer, num_epochs):
    early_stopping = EarlyStopping(model,patience=3, min_delta=0.001)
    training_loss_list = []
    training_accuracy_list = []
    val_loss_list = []
    val_accuracy_list = []
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_accuracy = 0
        for batch_idx, (images, labels) in enumerate(train_dummy_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            predicted = torch.argmax(outputs, dim=1)
            #Calcualte the accuracy
            accuracy = (predicted == labels).sum().item() / len(labels)
            total_accuracy += accuracy
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_dummy_loader)}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}", end='\r')
            if batch_idx > 2000:
                break
        avg_loss = total_loss / len(train_dummy_loader)
        avg_accuracy = total_accuracy / len(train_dummy_loader)
        training_loss_list.append(avg_loss)
        training_accuracy_list.append(avg_accuracy)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}, Average Accuracy: {avg_accuracy:.4f}", end='\r')

        # Validation 
        model.eval()
        with torch.no_grad():
            total_loss = 0.0
            total_accuracy = 0.0
            for val_idx, (images, labels) in enumerate(val_dummy_loader):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                predicted = torch.argmax(outputs, dim=1)
                #Calcualte the accuracy
                accuracy = (predicted == labels).sum().item() / len(labels)
                total_accuracy += accuracy
                if val_idx > 2000:
                    break
            avg_loss = total_loss / len(val_dummy_loader)
            avg_accuracy = total_accuracy / len(val_dummy_loader)
            val_loss_list.append(avg_loss)
            val_accuracy_list.append(avg_accuracy)
            early_stopping(avg_loss, model)
            print(f"\nValidation Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}\n", end='\r')
            
            if early_stopping.early_stop:
                print("Early stopping triggered!")
                break
    return training_accuracy_list, training_loss_list, val_loss_list, val_accuracy_list, early_stopping.model

def plotFig(train_loss, val_loss, name, label1, label2):
  #Plot the training and validation loss
    plt.figure()
    plt.plot(train_loss, label=label1)
    plt.plot(val_loss, label=label2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    path = '/data/home/saisuchithm/godwin/mlsp/project/'
    location = path + name + '.png'
    plt.savefig(location)
    plt.show()


if __name__ == "__main__":
    #hyperParameters
    num_epochs = 5
    learning_rate = 1e-4
    weight_deacy = 1e-5
    batch_size = 32

    # Initialize model, loss function, and optimizer
    model = ResNet50MultiClassClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = weight_deacy)
    transform = get_transforms()
    dataset, val_dataset = get_dataset(transform, USE_DFT=True, USE_CROSS_DIFF=False)
    
    #Define the dataloaders
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model_name = 'CNN_AI_Detect_DFT'
    print(f"Start Training the model")
    training_accuracy_list, training_loss_list, val_loss_list, val_accuracy_list, model = train_CNN(model=model,
                                                                                                    train_dummy_loader=train_dataloader,
                                                                                                    val_dummy_loader=val_dataloader,
                                                                                                    criterion=criterion,
                                                                                                    optimizer=optimizer,
                                                                                                    num_epochs=num_epochs)

    print("Dummy training and validation completed successfully!")
    plotFig(train_loss=training_loss_list, 
            val_loss=val_loss_list, 
            name="CNN_DFT_Loss", 
            label1='Training Loss',
            label2='Validation Loss')
    
    plotFig(train_loss=training_accuracy_list, 
            val_loss=val_accuracy_list, 
            name="CNN__DFT_Accuracy",
            label1='Training Accuracy',
            label2='Validation Accuracy')


    # Save model
    torch.save(model.state_dict(), "MLSP_CNN_DFT.pth")
    print("Model saved successfully!")
    
    del model
    del criterion
    del optimizer
    del train_dataloader
    del val_dataloader 

    gc.collect()
    torch.cuda.empty_cache()