import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import io
import glob
import clip  # OpenAI's CLIP package
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

#Global Variables
gpu_id = 2  # Change this to select a different GPU
device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"

#Function Definitions
# Define the transformations
# Function to apply JPEG compression artifacts to PNG images
def apply_jpeg_artifacts(img, quality=75):
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)  # Convert PNG to JPEG in memory
    return Image.open(buffer)

    # Define the transformations
# Define transformations (data augmentation + normalization)
def get_transform():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to ResNet-50 input size
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform
# Custom dataset
class CLIPImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, USE_DFT=False):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.USE_DFT=USE_DFT

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

        if self.transform != None:
            image = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label
    
# Simple linear head for classification
class CLIPClassifier(nn.Module):
    def __init__(self, clip_model, num_classes=11):
        super().__init__()
        self.clip_model = clip_model
        self.classifier = nn.Sequential(
                            nn.LayerNorm(512),
                            nn.Linear(512, 128),
                            nn.ReLU(),
                            nn.Linear(128, num_classes)
                            )

    def forward(self, x):
        with torch.no_grad():
            features = self.clip_model.encode_image(x)
        features = features.float()
        return self.classifier(features)
    
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

    
def get_datsets(transform, USE_DFT = False):
    # Define the file paths for training set
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
    dataset = CLIPImageDataset(image_paths, image_labels, transform, USE_DFT)

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

    val_dataset = CLIPImageDataset(val_image_paths, val_image_labels, transform, USE_DFT)

    return dataset, val_dataset

# Training Loop
def train_CLIP(model, train_dummy_loader, val_dummy_loader, criterion, optimizer, num_epochs):
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
            if batch_idx % 1000 == 0:
                torch.save(model.state_dict(), "/data/home/saisuchithm/godwin/mlsp/project/checkpoint.pth")
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_dummy_loader)}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}", end='\r')
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
            avg_loss = total_loss / len(val_dummy_loader)
            avg_accuracy = total_accuracy / len(val_dummy_loader)
            val_loss_list.append(avg_loss)
            val_accuracy_list.append(avg_accuracy)
            early_stopping(avg_loss, model)
            print(f"\n Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}\n", end='\r')
            
            
    return training_accuracy_list, training_loss_list, val_loss_list, val_accuracy_list, early_stopping.model

def plotFig(datapoints, name, label, y_label):
  #Plot the training and validation loss
    plt.figure()
    plt.plot(datapoints, label=label)
    plt.xlabel('Epoch')
    plt.ylabel(y_label)
    plt.legend()
    path = '/data/home/saisuchithm/godwin/mlsp/project/'
    location = path + name + '.png'
    plt.savefig(location)
    plt.show()

if __name__ == "__main__":
    print("Loading the VIT CLIP Model")
    model, preprocess = clip.load("ViT-B/32", device=device)
    # transform = get_transform()
    print("Loading Dataset")
    dataset, val_dataset = get_datsets(transform = preprocess, USE_DFT=False)
    batch_size = 32
    lr = 1e-4
    num_epochs = 11
    print("Loading Dataloader")
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # small_dataset = torch.utils.data.Subset(dataset, range(8))
    # train_loader = DataLoader(small_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    print("Loading Classifier")
    classifier = CLIPClassifier(model).to(device)
    print("Loading optimizer")
    optimizer = torch.optim.Adam(classifier.classifier.parameters(), lr=lr, weight_decay=0.1 * lr)
    print("Loading criterion")
    criterion = nn.CrossEntropyLoss()
    
    #Load the weights if exists
    path = "/data/home/saisuchithm/godwin/mlsp/project/checkpoint.pth"
    # Load checkpoint if found
    if os.path.exists(os.path.join(path)):
        print('Loading checkpoint as found one')
        classifier.load_state_dict(torch.load(os.path.join(path), map_location=device))

    print("Start Training")
    training_accuracy_list, training_loss_list, val_loss_list, val_accuracy_list, model_trainined = train_CLIP(model=classifier,
                                                                                                    train_dummy_loader=train_loader,
                                                                                                    val_dummy_loader=val_loader,
                                                                                                    criterion=criterion,
                                                                                                    optimizer=optimizer,
                                                                                                    num_epochs=num_epochs)

    print("End Training")
    torch.save(model_trainined.state_dict(), "Trained_CLIP_final.pth")
    print("Model saved successfully!")

    plotFig(datapoints=training_loss_list, 
            name="CLIP_Training_Loss",
            label="Training Loss", 
            y_label="Training Loss")
    plotFig(datapoints=val_loss_list, 
            name="CLIP_Validation_Loss",
            label="Validation Loss", 
            y_label="Validation Loss")
    plotFig(datapoints=training_accuracy_list, 
            name="CLIP_Training_accuracy",
            label="Training accuracy", 
            y_label="Training accuracy")
    plotFig(datapoints=val_accuracy_list, 
            name="CLIP_Validation_Accuracy",
            label="Val Loss", 
            y_label="Val Loss")

    