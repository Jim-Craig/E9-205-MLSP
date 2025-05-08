#Import block
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import librosa
from tqdm import tqdm
import torch.nn.functional as f

#Global Variables
DEVICE="cuda:2" if torch.cuda.is_available() else "cpu"
file_path = "/data/home/saisuchithm/godwin/mlsp/Assignment/Assignment_3/Results_CNN.txt"

"""### Dataset and Dataloader"""
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

# Define a training method to train the model
def trainModel(trainingDataloader, validationDataloader, model, optimizer, criterion, max_epochs) :
  # Run the training loop for max_epochs
  train_loss = []
  val_loss = []
  train_acc = []
  val_acc = []
  early_stopping = EarlyStopping(model,patience=3, min_delta=0.001)
  for epoch in range(max_epochs):
    model.train()
    batch_train_loss = []
    batch_train_acc = []

    for batch_index, (batch_data, batch_labels) in enumerate(trainingDataloader):

      optimizer.zero_grad()
      output = model(batch_data)
      loss = criterion(output, batch_labels)
      loss.backward()
      optimizer.step()
      batch_train_loss.append(loss.item())
      #Calculate the accuracy
      predicted = torch.argmax(output, dim=1)
      correct = (predicted == batch_labels).sum().item()
      accuracy = correct / len(batch_labels)
      batch_train_acc.append(accuracy)
      print(f"Epoch {epoch+1}/{max_epochs} batch {batch_index+1}/{len(trainingDataloader)} train loss {np.mean(batch_train_loss)}, train acc {accuracy}", end='\r')
    print(f"\nEpoch {epoch+1}/{max_epochs} train loss {np.mean(batch_train_loss)}, train acc {np.mean(batch_train_acc)}\n", end='\r')
    train_loss.append(np.mean(batch_train_loss))
    train_acc.append(np.mean(batch_train_acc))

    #test the model on valiation data if validation_dataloader is not None
    if validationDataloader is not None:
      model.eval()
      with torch.no_grad():
        batch_val_loss = []
        batch_val_acc = []
        for batch_index, (batch_data, batch_labels) in enumerate(validationDataloader):
          output = model(batch_data)
          loss = criterion(output, batch_labels)
          batch_val_loss.append(loss.item())
          #Calculate the accuracy
          predicted = torch.argmax(output, dim=1)
          correct = (predicted == batch_labels).sum().item()
          accuracy = correct / len(batch_labels)
          batch_val_acc.append(accuracy)
        print(f"\nEpoch {epoch+1}/{max_epochs} val loss {np.mean(batch_val_loss)}, val acc {np.mean(batch_val_acc)}\n", end='\r')
        val_loss.append(np.mean(batch_val_loss))
        val_acc.append(np.mean(batch_val_acc))
        # Check early stopping
        early_stopping(np.mean(batch_val_loss), model)

        #Commenting this since, we don't need to early stop, just save the best model
        # if early_stopping.early_stop:
        #     print("\nEarly stopping triggered!")
        #     break  # Stop training
  if(validationDataloader is None):
    return train_loss, val_loss, train_acc, val_acc, model
  return train_loss, val_loss, train_acc, val_acc, early_stopping.model
  
  

  

#Test the model on testing data
def testModel(testDataloader, model, name, criterion):
  message = "Starting Testing for " + name
  writeResults(message = message)
  model.eval()
  with torch.no_grad():
    batch_test_loss = []
    batch_test_acc = []
    for batch_index, (batch_data, batch_labels) in enumerate(testDataloader):
      output = model(batch_data)
      loss = criterion(output, batch_labels)
      batch_test_loss.append(loss.item())
      #Calculate the accuracy
      predicted = torch.argmax(output, dim=1)
      correct = (predicted == batch_labels).sum().item()
      accuracy = correct / len(batch_labels)
      batch_test_acc.append(accuracy)
    message_test = f"\nTest loss {np.mean(batch_test_loss)}, Test acc {np.mean(batch_test_acc)}\n"
    writeResults(message=message_test)
    print(message_test, end='\r')

def plotFig(train_loss, val_loss, name):
  #Plot the training and validation loss
    plt.figure()
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    path = '/data/home/saisuchithm/godwin/mlsp/Assignment/Assignment_3/'
    location = path + name + '.png'
    plt.savefig(location)
    plt.show()

def writeResults(message):
  # Save to a file
  with open(file_path, "a") as f:
      f.write(message + "\n")

"""## 3.0 CNN Model
Make a CNN model architecture for this 10-class classification setting with the following details, 
two layers of 2-D CNN with 16 filters 3 x3 size, with stride of 1 x1 and with max-pooling of 3x3. 
Flatten the CNN outputs and use 2 fully connected layers of hidden dimensions 128 and then 
classification with softmax non-linearity for 10 classes. Use the cross-entropy loss for training the models.
### Prepare the Model and the train function
"""
def extract_mel_spectrogram(ap, n_mels=128, win_ms=25, hop_ms=10, duration=5, sr=44100):

    y, sr = librosa.load(ap, sr=None, duration=duration)
    win_length = int(win_ms * sr / 1000)
    hop_length = int(hop_ms * sr / 1000)
    
    mel_spectrogram = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels,
        win_length=win_length,
        hop_length=hop_length,
    )
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    
    return mel_spectrogram_db

# Create a dataset dictionary
def process_df(df, audio_path):
    features = []
    labels = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        file_path = os.path.join(audio_path, row['filename'])
        mel = extract_mel_spectrogram(file_path)
        mel = mel[:, :500]  # Ensure shape is (128, 500)
        features.append(mel)
        labels.append(row['category'])
    return np.expand_dims(np.array(features), axis=1), np.array(labels)

#Create a custom dataset for the mel spectogram
class customDatasetCNN(torch.utils.data.Dataset):
  def __init__(self, features, labels):
    self.features = features
    self.labels = labels

  def __len__(self):
    return len(self.features)
  def __getitem__(self, index):
    feature = self.features[index]
    label = self.labels[index]
    feature = torch.tensor(feature, dtype = torch.float32, device = DEVICE)
    label = torch.tensor(label, dtype = torch.long, device = DEVICE)
  
    return feature, label
def createDataloader_CNN(batch_size = 16):
    meta_df = pd.read_csv('/data/home/saisuchithm/godwin/mlsp/Assignment/Assignment_3/ESC-50-master/meta/esc50.csv')
    # Filter for ESC-10
    esc10_df = meta_df[meta_df['esc10'] == True]
    esc10_df = esc10_df.reset_index(drop=True)
    #Encode the labels of the sentiment (positive and negative) to 0 and 1
    le = LabelEncoder()
    le.fit(esc10_df['category'])
    esc10_df['category'] = le.transform(esc10_df['category'])
    # Split data
    train_df = esc10_df[esc10_df['fold'].isin([1, 2, 3])]
    val_df   = esc10_df[esc10_df['fold'] == 4]
    test_df  = esc10_df[esc10_df['fold'] == 5]
    
    #Define the audio file path
    audio_path = '/data/home/saisuchithm/godwin/mlsp/Assignment/Assignment_3/ESC-50-master/audio'

    # Extract all features
    X_train, y_train = process_df(train_df, audio_path)
    X_val, y_val     = process_df(val_df, audio_path)
    X_test, y_test   = process_df(test_df, audio_path)

    #Make the train, validation and test dataset
    train_dataset = customDatasetCNN(X_train, y_train)
    val_dataset = customDatasetCNN(X_val, y_val)
    test_dataset = customDatasetCNN(X_test, y_test)

    #Make the train, validation and test dataset
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader

# CNN model for this 10 classification setting
#Based on the type of norm{no_norm, batch_norm, layer_norm}, add that to the flatten output
class CNN(nn.Module):
  def __init__(self, num_classes, norm = 'no_norm'):
    super(CNN, self).__init__()
    #Conv-2d layer with  16 filters 3 ×3 size, with stride of 1 ×1
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1)
    #max-pooling of 3×3.
    self.pool1 = nn.MaxPool2d(kernel_size=3)
    #Conv-2 layer with  16 filters 3 ×3 size, with stride of 1 ×1
    self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1)
    #max-pooling of 3×3.
    self.pool2 = nn.MaxPool2d(kernel_size=3)
    #  fully connected layers of hidden dimensions 128
    self.fc1 = nn.Linear(in_features=11232, out_features=128)
    self.fc2 = nn.Linear(in_features=128, out_features=num_classes)
    # Choose the normalization
    self.norm = nn.Identity()
    if norm == 'batch_norm':
      self.norm = nn.BatchNorm1d(11232)
    elif norm == 'layer_norm':
      self.norm = nn.LayerNorm(11232)
    self.flatten = nn.Flatten()
  def forward(self, x):
    #Conv-2d layer with 16 filters 3 ×3 size, with stride of 1 ×1
    x = f.relu(self.conv1(x))
    #max-pooling of 3×3.
    x = self.pool1(x)
    #Conv-2d layer with 16 filters 3 ×3 size, with stride of 1 ×1
    x = f.relu(self.conv2(x))
    #max-pooling of 3×3.
    x = self.pool2(x)
    #Flatten the CNN outputs
    x = self.flatten(x)
    #If the norm is not None, appy the normalization
    x = self.norm(x)

    #pass throught the fully connected layer
    x = f.relu(self.fc1(x))
    x = self.fc2(x)
    return x
  
  '''3.a Compare different optimizers
  Compare the training and validation loss curves for training with (a) SGD, (b) SGD
  with momentum (factor of 0.9), (c) RMSprop (with default parameters) and (d)
  Adam optimizer.   '''
#Function to check CNN training on different opimizers
def optimizersCheck(train_dataloader, validation_dataloader, test_dataloader, model, criterion, max_epochs):
  writeResults(message="Results for 3.a")
  optimizer1 = torch.optim.SGD(model.parameters(), lr=1e-4)
  optimizer2 = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
  optimizer3 = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
  optimizer4 = torch.optim.RMSprop(model.parameters(), lr = 1e-4)
  

  optimizers = [optimizer1, optimizer2, optimizer3, optimizer4]
  names = ["SGD", "SGD with momentum", "Adam", "RMSProp"]

  # optimizers = [optimizer1, optimizer2, optimizer4]
  # names = ["SGD", "SGD with momentum", "Adam"]

  for i in range(len(optimizers)):
    model_copy = CNN(num_classes=10)
    model_copy = model.to(device=DEVICE)
    optimizer = optimizers[i]
    name = names[i]
    print(f"Training with {name} optimizer")
    print("Start Training")
    train_loss, val_loss, _, _, model_copy = trainModel(trainingDataloader=train_dataloader
                                                        , validationDataloader=validation_dataloader
                                                        , model=model_copy
                                                        , optimizer=optimizer
                                                        , criterion=criterion
                                                        , max_epochs=max_epochs)
    print("End Training")

    plotFig(train_loss=train_loss, val_loss=val_loss, name=name)
    print("Start testing the model")
    #Test the model on testing data
    testModel(testDataloader=test_dataloader, model=model_copy, name = name, criterion=criterion)
    print("End model testing")
    del model_copy

  '''
  3.b Comparing the normalization:
  At the flattened output and at the input of 2 fully connected layers, compare the
  training and validation loss curves for :
  a) No-norm
  b) Layer norm 
  c) Batch norm. 
  '''
  #Fucntion to run the model on the 3 normalization options: no_norm, batch_norm and layer_norm
def compareNormalization(train_dataloader, validation_dataloader, test_dataloader, max_epochs):
  writeResults(message="Results for 3.b")
  normalization = ['no_norm', 'batch_norm', 'layer_norm']
  #Iterate through the normalization and calculate the model performance
  for norm in normalization:
    model = CNN(num_classes=10, norm = norm).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    print(f"Training with {norm} normalization")
    print("Start Training")
    train_loss, val_loss, _, _, model_copy = trainModel(train_dataloader
                                                        , validationDataloader=validation_dataloader
                                                        , model=model
                                                        , optimizer=optimizer
                                                        , criterion=criterion
                                                        , max_epochs=max_epochs)
    print("End Training")

    plotFig(train_loss=train_loss, val_loss=val_loss, name=norm)

    print("Start testing the model")
    #Test the model on testing data
    testModel(testDataloader=test_dataloader, model=model_copy, name = norm, criterion=criterion)
    print("End model testing")
    del model
    del optimizer
    del criterion

  '''
  3.C Ensemble of methods
  For the first model, use SGD training without any normalization. 
  For the second model use the RMSprop
  optimizer with Layer norm. 
  For the third model, use Adam optimizer with Batch norm. 
  '''
  #function to train the ensemble methods
def trainEnsembleMethods(train_dataloader, validation_dataloader, max_epochs):
  #
  # CNN Model with no normalization and SGD Training
  model_1 = CNN(num_classes=10)
  model_1 = model_1.to(device=DEVICE)
  optimizer1 = torch.optim.SGD(model_1.parameters(), lr=1e-3)
  criterion = nn.CrossEntropyLoss()
  print("START Training the model_1 models")
  _, _, _, _, model_1 = trainModel(trainingDataloader = train_dataloader
                                   , validationDataloader = validation_dataloader
                                   , model = model_1
                                   , optimizer = optimizer1
                                   , criterion = criterion
                                   , max_epochs=max_epochs)
  print("END Training the model_1 models")

  #CNN Model with layer norm and RMSProp training
  model_2 = CNN(num_classes=10, norm = 'layer_norm')
  model_2 = model_2.to(device=DEVICE)
  optimizer2 = torch.optim.RMSprop(model_2.parameters(), lr=1e-4)
  print("START Training the model_2 models")
  _, _, _, _, model_2 = trainModel(trainingDataloader = train_dataloader,
                                    validationDataloader = validation_dataloader
                                    , model = model_2
                                    , optimizer = optimizer2
                                    , criterion = criterion
                                    , max_epochs=max_epochs)
  print("END Training the model_2 models")

  #CNN Model with batch norm and Adam training
  model_3 = CNN(num_classes=10, norm = 'batch_norm')
  model_3 = model_3.to(device=DEVICE)
  optimizer3 = torch.optim.Adam(model_3.parameters(), lr=1e-4, weight_decay=1e-5)
  print("START Training the model_3 models")
  _, _, _, _, model_3 = trainModel(
     trainingDataloader = train_dataloader,
     validationDataloader = validation_dataloader,
     model = model_3,
     optimizer = optimizer3,
     criterion = criterion,
     max_epochs=max_epochs
  )
  print("END Training the model_3 models")
  del optimizer1
  del optimizer2
  del optimizer3
  return model_1, model_2, model_3
'''
Function to train and evaluate the ensemble average
'''
#  Ensemble the model outputs using output averaging of the posterior model outputs from the three model outputs
def ensembleAveraging(train_dataloader, validation_dataloader, test_dataloader, criterion, max_epochs):
  print("START Training the Ensemble models")
  model_1, model_2, model_3 = trainEnsembleMethods(train_dataloader=train_dataloader
                                                   , validation_dataloader= validation_dataloader
                                                   , max_epochs= max_epochs)
  print("END Training the Ensemble models")
  model_1.eval()
  model_2.eval()
  model_3.eval()
  startMessage = f"\nStart Evaluation of Ensemble average"
  print(startMessage)
  writeResults(message = startMessage)
  with torch.no_grad():
    batch_test_loss = []
    batch_test_acc = []
    for batch_index, (batch_data, batch_labels) in enumerate(test_dataloader):
      output_1 = model_1(batch_data)
      output_2 = model_2(batch_data)
      output_3 = model_3(batch_data)
      output = (output_1 + output_2 + output_3) / 3
      loss = criterion(output, batch_labels)
      batch_test_loss.append(loss.item())
      #Calculate the accuracy
      predicted = torch.argmax(output, dim=1)
      correct = (predicted == batch_labels).sum().item()
      accuracy = correct / len(batch_labels)
      batch_test_acc.append(accuracy)
    
    endMessage = f"Ensemble Average test loss {np.mean(batch_test_loss)}, Ensemble Average test acc {np.mean(batch_test_acc)}"
    print(endMessage)
    writeResults(message = endMessage)
    del model_1
    del model_2
    del model_3

#Ensemble the model outputs using with optimal linear weighted combination of the three model outputs.
#Define a new model that takes the model_1, model_2 and model_3 and gives out the weighted combination of the output
class EnsembleWeightedAverage(nn.Module):
  def __init__(self, model_1, model_2, model_3):
    super(EnsembleWeightedAverage, self).__init__()
    self.model_1 = model_1
    self.model_2 = model_2
    self.model_3 = model_3
    #Define 3 learnable parameters alpha, beta and gamma as the cofficients of the outputs of the model
    self.weight = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]))
    #Freeze the weights for model_1, model_2, model_3
    for model in [self.model_1, self.model_2, self.model_3]:
       for param in model.parameters():
          param.requires_grad = False
    self.softmax = nn.Softmax()

  def forward(self, x):
    output1 = self.model_1(x)
    output2 = self.model_2(x)
    output3 = self.model_3(x)

    normalized_weight = self.softmax(self.weight)
    output = (normalized_weight[0] * output1 + normalized_weight[1] * output2 + normalized_weight[2] * output3)
    return output 
  
#Write a function to weighted Average ensemble method
def trainEnsembleWeightedAverage(train_dataloader, validation_dataloader, max_epochs):
  print("START Training the Ensemble models\n")
  model_1, model_2, model_3 = trainEnsembleMethods(train_dataloader=train_dataloader
                                                   , validation_dataloader= validation_dataloader
                                                   , max_epochs= max_epochs)
  print("END Training the Ensemble models\n")
  model_1.eval()
  model_2.eval()
  model_3.eval()
  print("START Training the Ensemble Weighted Average model\n")
  model_weighted = EnsembleWeightedAverage(model_1, model_2, model_3)
  optimizer = torch.optim.Adam(model_weighted.parameters(), lr=1e-4, weight_decay=1e-5)
  criterion = nn.CrossEntropyLoss()
  _, _, _, _, model_weighted = trainModel(trainingDataloader = validation_dataloader
                                          , validationDataloader = None
                                          , model = model_weighted
                                          , optimizer = optimizer
                                          , criterion = criterion
                                          , max_epochs=max_epochs)
  print("END Training the Ensemble Weighted Average model\n")
  return model_weighted

'''
Function to train and evaluate the ensemble weighted average
'''
#Ensemble the model outputs using with optimal linear weighted combination of the three model outputs.
def ensembleWeightedAverage(train_dataloader, validation_dataloader, test_dataloader, max_epochs):
  #Define the ensemble weighted average model 
  model_weighted = trainEnsembleWeightedAverage(train_dataloader=train_dataloader
                                                , validation_dataloader=validation_dataloader
                                                , max_epochs=max_epochs)
  model_weighted.eval()
  startMessage = f"\nStart Evaluation of Ensemble weighted average"
  print(startMessage)
  writeResults(message = startMessage)
  with torch.no_grad():
    batch_test_loss = []
    batch_test_acc = []
    for batch_index, (batch_data, batch_labels) in enumerate(test_dataloader):
      output = model_weighted(batch_data)
      loss = criterion(output, batch_labels)
      batch_test_loss.append(loss.item())
      #Calculate the accuracy
      predicted = torch.argmax(output, dim=1)
      correct = (predicted == batch_labels).sum().item()
      accuracy = correct / len(batch_labels)
      batch_test_acc.append(accuracy)
    weights = torch.nn.functional.softmax(model_weighted.weight)  
    endMessage = f"Ensemble test loss {np.mean(batch_test_loss)}, Ensemble test acc {np.mean(batch_test_acc)}, weights are {weights}"
    print(endMessage)
    writeResults(message = endMessage)
    del model_weighted



"""### Train the model !!!"""
if __name__ == "__main__":
    learning_rate = 1e-4
    weight_decay = 1e-5
    num_classes = 10
    model = CNN(num_classes=num_classes)
    model = model.to(device=DEVICE)
    max_epochs = 10
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr = 1e-4)
    
    criterion = nn.CrossEntropyLoss()

    print("Creating DataLoader")
    train_dataloader, validation_dataloader, test_dataloader = createDataloader_CNN()
  
    #3.a Compare the Optimizers
    optimizersCheck(train_dataloader, validation_dataloader, test_dataloader,model, criterion, max_epochs)

    #3.b Compare the Normalization
    compareNormalization(train_dataloader, validation_dataloader, test_dataloader, max_epochs)

    #3.c Compare the Ensemble methods
    ##Ensemble Average
    ensembleAveraging(train_dataloader=train_dataloader
                      , validation_dataloader=validation_dataloader
                      , test_dataloader= test_dataloader
                      , criterion= criterion
                      , max_epochs= max_epochs)
    ##Ensemble Weighted Average
    ensembleWeightedAverage(train_dataloader=train_dataloader
                            , validation_dataloader=validation_dataloader
                            , test_dataloader=test_dataloader
                            , max_epochs=max_epochs)


    #Clear GPU
    del model
    del optimizer
    del criterion
    del train_dataloader
    del validation_dataloader 
    del test_dataloader