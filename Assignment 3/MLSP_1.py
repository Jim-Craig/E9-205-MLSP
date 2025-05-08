#Import block
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from random import sample
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import torch
import torch.nn as nn
from gensim.models import KeyedVectors
import nltk

#Global Variables
w2v_model = KeyedVectors.load_word2vec_format('word2vec-google-news-300.bin', binary=True)
DEVICE="cuda:2" if torch.cuda.is_available() else "cpu"

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

def prepData():
  f = pd.read_csv('/data/home/saisuchithm/godwin/mlsp/Assignment/Assignment_3/IMDB Dataset.csv')
  #Encode the labels of the sentiment (positive and negative) to 0 and 1
  le = LabelEncoder()
  le.fit(f['sentiment'])
  f['sentiment'] = le.transform(f['sentiment'])
  #Split the data into training, validation and test
  train_data, test_data, val_data = np.split(f.sample(frac=1), [int(.8*len(f)), int(.9*len(f))])
  #split the data into 2 types, one for the review and one for the sentiment to create a data and label sort of structure
  train_review = train_data['review']
  train_sentiment = train_data['sentiment']

  test_review = test_data['review']
  test_sentiment = test_data['sentiment']

  val_review = val_data['review']
  val_sentiment = val_data['sentiment']
  return train_review, train_sentiment, test_review, test_sentiment, val_review, val_sentiment

#Create a custom dataset to take the sentances from the pandas dataframe, tokenize it and then put the word embeddings and the labels
class customDataset(torch.utils.data.Dataset):
  def __init__(self, review, sentiment, w2v_model, device):
    self.review = review
    self.sentiment = sentiment
    self.w2v_model = w2v_model
    self.device = device

  def __len__(self):
    return len(self.review)

  def __getitem__(self, index):
    review = self.review.iloc[index]
    sentiment = self.sentiment.iloc[index]
    #tokenize the review sentance
    review = RegexpTokenizer(r'\w+').tokenize(review)
    #convert to word embeddings
    review = [self.w2v_model[word] for word in review if word in self.w2v_model]
    #convert list to numpy arrays to numpy arrays
    review_numpy = np.array(review)
    #convert the numpy arrays to tensors
    review = torch.tensor(review_numpy, dtype = torch.float32, device = DEVICE)
    return review, sentiment

#We also define a new collate function to take care of different sequence lengths
def custom_collate(batch):
    reviews, sentiments = zip(*batch)
    #pack the reviews
    device = reviews[0].device
    reviews = torch.nn.utils.rnn.pack_sequence(reviews, enforce_sorted=False)
    sentiments = torch.tensor(sentiments, dtype = torch.float32, device=device)
    return reviews, sentiments

def createDataloader(device, w2v_model):
  #Prepare the data
  train_review, train_sentiment, test_review, test_sentiment, val_review, val_sentiment = prepData()
  #Before creating the dataloader, let's fix the index of the train and validation data
  train_review = train_review.reset_index(drop=True)
  train_sentiment = train_sentiment.reset_index(drop=True)

  val_review = val_review.reset_index(drop=True)
  val_sentiment = val_sentiment.reset_index(drop=True)
  #Create a train and validation dataloader for this dataset
  train_dataloader = torch.utils.data.DataLoader(customDataset(train_review, train_sentiment, w2v_model,device), batch_size=32, shuffle=True, collate_fn=custom_collate)
  validation_dataloader = torch.utils.data.DataLoader(customDataset(val_review, val_sentiment, w2v_model, device), batch_size=32, shuffle=False, collate_fn=custom_collate)
  test_dataloader = torch.utils.data.DataLoader(customDataset(test_review, test_sentiment, w2v_model, device), batch_size=32, shuffle=False, collate_fn=custom_collate)
  return train_dataloader, validation_dataloader, test_dataloader

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
      predicted = torch.round(torch.sigmoid(output))
      correct = (predicted == batch_labels).sum().item()
      accuracy = correct / len(batch_labels)
      batch_train_acc.append(accuracy)
      print(f"Epoch {epoch+1}/{max_epochs} batch {batch_index+1}/{len(trainingDataloader)} train loss {np.mean(batch_train_loss)}, train acc {accuracy}", end='\r')
    print(f"\nEpoch {epoch+1}/{max_epochs} train loss {np.mean(batch_train_loss)}, train acc {np.mean(batch_train_acc)}\n", end='\r')
    train_loss.append(np.mean(batch_train_loss))
    train_acc.append(np.mean(batch_train_acc))

    #test the model on valiation data
    model.eval()
    with torch.no_grad():
      batch_val_loss = []
      batch_val_acc = []
      for batch_index, (batch_data, batch_labels) in enumerate(validationDataloader):
        output = model(batch_data)
        loss = criterion(output, batch_labels)
        batch_val_loss.append(loss.item())
        #Calculate the accuracy
        predicted = torch.round(torch.sigmoid(output))
        correct = (predicted == batch_labels).sum().item()
        accuracy = correct / len(batch_labels)
        batch_val_acc.append(accuracy)
      print(f"\nEpoch {epoch+1}/{max_epochs} val loss {np.mean(batch_val_loss)}, val acc {np.mean(batch_val_acc)}\n", end='\r')
      val_loss.append(np.mean(batch_val_loss))
      val_acc.append(np.mean(batch_val_acc))
      # Check early stopping
      early_stopping(np.mean(batch_val_acc), model)
      #Commenting this since, we don't need to early stop, just save the best model
      # if early_stopping.early_stop:
      #     print("\nEarly stopping triggered!")
      #     break  # Stop training
  return train_loss, val_loss, train_acc, val_acc, early_stopping.model


#Test the model on testing data
def testModel(testDataloader, model, name, file_path, criterion):
  message = "Starting Testing for " + name
  writeResults(file_path = file_path, message = message)
  model.eval()
  with torch.no_grad():
    batch_test_loss = []
    batch_test_acc = []
    for batch_index, (batch_data, batch_labels) in enumerate(testDataloader):
      output = model(batch_data)
      loss = criterion(output, batch_labels)
      batch_test_loss.append(loss.item())
      #Calculate the accuracy
      predicted = torch.round(torch.sigmoid(output))
      correct = (predicted == batch_labels).sum().item()
      accuracy = correct / len(batch_labels)
      batch_test_acc.append(accuracy)
    message_test = f"\nTest loss {np.mean(batch_test_loss)}, Test acc {np.mean(batch_test_acc)}\n"
    writeResults(message=message_test, file_path=file_path)
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

def writeResults(file_path, message):
  # Save to a file
  with open(file_path, "a") as f:
      f.write(message + "\n")
"""
1.1 PCA Semantic Clustering
Use pretrained word2vec to create 300 word embeddings
use PCA to reduce dimentions from 300 to 2
Make a scatter plot for 10 validation sentances
"""
#Given the sentances, remove the stop words and the redundant words and embedded the unique words to form a matrix
#Sentances is a pandas dataframe of sentances, model is the word to Vector model
def sentence_to_matrix(sentences, model, emb_dim=300):
  cachedStopWords = stopwords.words("english")

  #Generate a list of unique words by removing all the stop words
  unique_words = []
  indices = sentences.index.values
  for i in indices:
    sentance  = RegexpTokenizer(r'\w+').tokenize(sentences[i])
    [unique_words.append(word.lower()) for word in sentance if (word not in cachedStopWords and word.lower() not in unique_words)]

  #Embedd the tokens
  embedded_tokens = np.zeros((len(unique_words), emb_dim))
  for i, word in enumerate(unique_words):
    if word in model:
      embedded_tokens[i] = model[word]
  return embedded_tokens, unique_words

# Write a PCA function to reduce the embedding dimentions to 2
def pca(data, n_components=2):
  # Calculate the covariance matrix
  cov_matrix = np.cov(data.T)

  # Calculate the eigenvalues and eigenvectors of the covariance matrix
  eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

  #Generate the transformation matrix
  transformation_matrix = eigenvectors[:, :n_components]

  #Transform the data
  transformed_data = np.dot(data, transformation_matrix)

  return transformed_data, transformation_matrix

# Write a PCA function to reduce the embedding dimentions to 2
def pca(data, n_components=2):
  # Calculate the covariance matrix
  cov_matrix = np.cov(data.T)

  # Calculate the eigenvalues and eigenvectors of the covariance matrix
  eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

  #Generate the transformation matrix
  transformation_matrix = eigenvectors[:, :n_components]

  #Transform the data
  transformed_data = np.dot(data, transformation_matrix)

  return transformed_data, transformation_matrix

def createData():
  print("Read Data")
  f = pd.read_csv('/data/home/saisuchithm/godwin/mlsp/Assignment/Assignment_3/IMDB Dataset.csv')
  #Encode the labels of the sentiment (positive and negative) to 0 and 1
  print("Encoding the labels")
  le = LabelEncoder()
  le.fit(f['sentiment'])
  f['sentiment'] = le.transform(f['sentiment'])
  
  #Split the data into training, validation and test
  print("train val test split")
  train_data, test_data, val_data = np.split(f.sample(frac=1), [int(.8*len(f)), int(.9*len(f))])
  
  #split the data into 2 types, one for the review and one for the sentiment to create a data and label sort of structure
  print("X and Y split")
  train_review = train_data['review']
  train_sentiment = train_data['sentiment']

  test_review = test_data['review']
  test_sentiment = test_data['sentiment']

  val_review = val_data['review']
  val_sentiment = val_data['sentiment']

  #Generate the 100 dimention word-embedings for the training, validation and the test datasets
  print("sentance to matrix on training")
  train_embed, unique_words_train = sentence_to_matrix(train_review, w2v_model)

  #Generate the 100 dimention word-embedings for the validation
  print("sentance to matrix on validation")
  valid_embed, unique_words_valid = sentence_to_matrix(val_review, w2v_model)

  #Generate the 100 dimention word-embedings for the test
  print("sentance to matrix on test")
  test_embed, unique_words_test= sentence_to_matrix(test_review, w2v_model)

  return train_embed, valid_embed, unique_words_valid

def train_pca():
  print("Creating Data")
  train_embed, valid_embed, unique_words_valid = createData()
  print("PCA begin on trainging data")
  train_embed_2d, transformation_matrix = pca(train_embed)
  # perform PCA on the val dataset and plot the 2d plot for the words
  print("PCA on validation")
  val_embed_2d = np.dot(valid_embed, transformation_matrix)

  #Take maximum 20 words to show
  unique_words_short = unique_words_valid
  val_embed_2d_short = val_embed_2d

  # Plot the val_embed_2d embedding points with the text
  print("Plotting")
  plt.figure(figsize=(40,25))
  plt.scatter(val_embed_2d_short[:, 0], val_embed_2d_short[:, 1])
  for i, word in enumerate(unique_words_short):
      plt.annotate(word, (val_embed_2d_short[i, 0], val_embed_2d_short[i, 1]))
  plt.show()
  path = '/data/home/saisuchithm/godwin/mlsp/Assignment/Assignment_3/'
  location = path + "cluster" + '.png'
  plt.savefig(location)
  plt.show()




"""## 1.2 LSTM Model
Create a LSTM model with 2 hidden layers, with 256 cells, followed by average
pooling and one-classification layer. It should take word2vec embedding as input.

### Prepare the Model and the train function
"""

#The model should consist of an LSTM Model with 2 hidden dimentions and hidden size of 256,followed by average-pooling and one-classification layer
class LSTMModel(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, output_size):
    super(LSTMModel, self).__init__()
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
    self.avg_pool = nn.AdaptiveAvgPool1d(1)
    self.fc = nn.Linear(hidden_size, output_size)

  #Forward function
  def forward(self, x, hidden=None):
    #if hidden is None then pass only the inputs to the lstm
    if hidden is None:
      output, output_hidden = self.lstm(x, hidden)
    else:
      output, output_hidden = self.lstm(x, hidden)
    #pad the packed sequence that comes out of the lstm
    output = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)[0] #dim (batch_size, max_seq_len, hidden_size)
    output = output.permute(0, 2, 1) #dim (batch_size, hidden_size, max_seq_len)
    #pass the output through the average-pooling
    pooled_output = self.avg_pool(output)#dim (batch_size, hidden_size, 1)
    pooled_output = pooled_output.squeeze()#dim (batch_size, hidden_size)
    # project the pooled output to the one dimention
    projected_output = self.fc(pooled_output)
    return projected_output.squeeze()


"""## 1.3 Attention Based LSTM Model
Replace the average pooling layer in the above question with an attention based pooling after the 2-layer LSTM model.
"""

#Same LSTM model as the above but the average pooling is replaced with attenion based pooling
class LSTMAttention(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, output_size):
    super(LSTMAttention, self).__init__()
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
    #Attention-based pooling
    self.weight = nn.Linear(hidden_size,1)
    self.fc = nn.Linear(hidden_size, output_size)

  def forward(self, x, hidden = None):
    if hidden is None:
      output, output_hidden = self.lstm(x, hidden)
    else:
      output, output_hidden = self.lstm(x, hidden)
    #pad the packed sequence that comes out of the lstm
    output = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)[0] #dim (batch_size, max_seq_len, hidden_size)
    #pass the output through the attention-based pooling
    attention_weights = torch.softmax(self.weight(output), dim=-1) #dim (batch_size, max_seq_len, 1)
    pooled_output = torch.sum(output * attention_weights, dim=1) #dim (batch_size, hidden_size)
    # project the pooled output to the one dimention
    projected_output = self.fc(pooled_output).squeeze()
    return projected_output

"""## 1.4 Transformer Based LSTM Model
Replace the two-layer LSTM model with a single transformer encoder layer with 256
hidden dimensions, followed by average pooling and classification with BCE loss.
"""

#Alternate implementation of multi-head self attention head
#Unlike the original implementation, which used the self attention head, I'd try to calculate the attention scores for the attention head without explicetly calling the self attention head.
# I would try to ge the scores via matrix multiplicaiton itself.
class MultiHeadSelfAttention_2(torch.nn.Module):
  def __init__(self, input_dim, num_heads):
    super().__init__()
    self.num_heads = num_heads
    self.input_dim = input_dim
    self.embedding_dim = input_dim//num_heads
    self.weight_Q = nn.Linear(input_dim, self.num_heads * self.embedding_dim)
    self.weight_K = nn.Linear(input_dim, self.num_heads * self.embedding_dim)
    self.weight_V = nn.Linear(input_dim, self.num_heads * self.embedding_dim)

  #function to convert the Key, Query and Value matrices of dim (N, T, num_heads * embedding_dim) to dim dim (N, num_heads, T, embedding_dim)
  def change_dimentions(self, matrix) :
    new_shape = matrix.shape[:-1] + (self.num_heads, self.embedding_dim)
    return matrix.reshape(new_shape).permute(0, 2, 1, 3)

  #Define the forward function for the the input X which should be of the dimention (N, T, input_dim)
  def forward(self, X):
    # Define the Query, Key and Value vectors
    Q = self.weight_Q(X)
    K = self.weight_K(X)
    V = self.weight_V(X)

    # Change the dimentions of the Query, Key and Value vectors to account for the multi-head system
    Q = self.change_dimentions(Q)
    K = self.change_dimentions(K)
    V = self.change_dimentions(V)

    #Calculate the attention scores
    scale = torch.sqrt(torch.tensor(self.embedding_dim, dtype=torch.float32, device=X.device))
    attention_scores = Q @ K.transpose(-2, -1) / scale
    self_attention_scores = nn.functional.softmax(attention_scores, dim = -1)# Should be of the dim (N, num_heads, T, T)

    #Calculate the output embeddings
    output_embeddings = self_attention_scores @ V # Should be of the dim (N, num_heads, T, embedding_dim)

    #Reshape back to the original dimentions: dim (N, T, num_heads * embedding_dim) = dim (N, T, input_dim)
    output_embeddings = output_embeddings.permute(0, 2, 1, 3)
    output_embeddings = output_embeddings.reshape(output_embeddings.shape[:-2] + (self.num_heads * self.embedding_dim,))
    return output_embeddings, self_attention_scores
  
#Now let's define a Multi-Layer Perceptron
class MLP(torch.nn.Module):

  def __init__(self, input_dim, hidden_dim, output_dim):
    super().__init__()
    self.fc1 = nn.Linear(input_dim, hidden_dim)
    self.activation = nn.GELU()
    self.fc2 = nn.Linear(hidden_dim, output_dim)
    self.dropout = nn.Dropout(0.2)
#Define the forward pass with the the above defined weights
# X is of the dim (N, T, input_dim)
  def forward(self, X):
    output = self.fc1(X)
    output = self.activation(output)
    output = self.fc2(output)
    output = self.dropout(output)
    return output # Output would also be of dim (T x input_size)
  
#Now we're ready to define a the Transformer encoder
class TransformerEncoder(torch.nn.Module):
  '''
  # Define the init function with the
  # input_dim being the dimention of the single input patch,
  # embedding_dim being the dimention of the encoder context word
  # num_heads being the number of the encoder blocks
  # hidden_dim being the dim of the hidden layer of the MLP
  # output_dim being the number of the classes
  '''
  def __init__(self, input_dim, num_heads, hidden_dim):
    super().__init__()
    self.input_dim = input_dim
    self.num_heads = num_heads
    self.multihead_self_attention = MultiHeadSelfAttention_2(input_dim, num_heads)
    self.layer_norm1 = nn.LayerNorm(input_dim)
    self.mlp = MLP(input_dim, hidden_dim, input_dim)
    self.layer_norm2 = nn.LayerNorm(input_dim)


  # Define the forward function
  # X would have the dim (N, T, input_dim)
  def forward(self, X):
    layer_normalization_1 = self.layer_norm1(X)#dim (batch_size, max_seq_len, input_dim)

    #Calculate the Multi-head Attention Encoding for the patches
    multihead_self_attention_output, self_attention_scores = self.multihead_self_attention(layer_normalization_1)
    #dim (batch_size, max_seq_len, input_dim)

    # Compute the residual Connection, will need to make sure that the dimentions of X and that of MSA match
    residual_connection_1 = X + multihead_self_attention_output #dim (batch_size, max_seq_len, input_dim)

    layer_normalization_2 = self.layer_norm2(residual_connection_1) #dim (batch_size, max_seq_len, input_dim)

    #Calculate the MLP output from the Multi-layer perceptron
    mlp_output = self.mlp(layer_normalization_2)#dim (batch_size, max_seq_len, input_dim)

    # Compute the residual Connection, will need to make sure that the dimentions of X and that of MLP match
    residual_connection_2 = residual_connection_1 + mlp_output#dim (batch_size, max_seq_len, input_dim)
    return residual_connection_2, self_attention_scores

#Same LSTM model as the above but the average pooling is replaced with attenion based pooling
class LSTMTransformer(nn.Module):
  def __init__(self, input_size, hidden_size, num_head, output_size, embedding_size = 256, max_seq_len = 512):
    super(LSTMTransformer, self).__init__()
    self.hidden_size = hidden_size
    self.max_seq_len = max_seq_len
    self.projection = nn.Linear(input_size, embedding_size)
    self.encoder = TransformerEncoder(embedding_size, num_head, hidden_size)
    self.embedding = nn.Embedding(max_seq_len, embedding_size)
    #Average Pooling
    self.avg_pool = nn.AdaptiveAvgPool1d(1)
    self.fc = nn.Linear(embedding_size, output_size)

  #Dim of x = (batch_size, seq_len, input_dim)
  def forward(self, x, hidden = None):
    #pad the packed sequence that comes out of the lstm
    x = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)[0] #dim (batch_size, seq_len, input_size)

    #limit the sequence length of the input to max_seq_len
    seq_len = x.shape[1]
    if(seq_len > self.max_seq_len):
      x = x[:, :self.max_seq_len, :] #dim (batch_size, max_seq_len, input_size)

    #project the data to embedding_dim 
    x = self.projection(x) #dim (batch_size, max_seq_len, embedding_size)

    # Add the Projection embeddings
    projection_vectors = torch.tensor([[j for j in range(x.shape[1])] for i in range (x.shape[0])], device = x.device) #dim (batch_size, max_seq_len)
    projection_embeddings = self.embedding(projection_vectors)
    x = x + projection_embeddings #dim (batch_size, max_seq_len, embedding_dim)
   #pass the input throught the encoder
    output = self.encoder(x)[0]#dim (batch_size, max_seq_len, embedding_dim)
    
    output = output.permute(0, 2, 1) #dim (batch_size, embedding_dim, max_seq_len)
    #pass the output through the average-pooling
    pooled_output = self.avg_pool(output)#dim (batch_size, embedding_dim, 1)
    pooled_output = pooled_output.squeeze(-1)#dim (batch_size, embedding_dim)
    # project the pooled output to the one dimention
    projected_output = self.fc(pooled_output)
    return projected_output.squeeze()


"""### Train the model !!!"""
if __name__ == "__main__":
  '''1.a: PCA Clustering'''
  # train_pca()

  #__________________________________________________________________________________________
  
  input_size = 300
  hidden_size = 256
  num_layers = 2
  output_size = 1
  max_epochs = 1
  learning_rate = 1e-4
  weight_decay = 1e-5
  file_path = "/data/home/saisuchithm/godwin/mlsp/Assignment/Assignment_3/Results.txt"

  print("Initializing the word2vec")
  nltk.download('punkt_tab')
  nltk.download('stopwords')


  #Prepare the data
  print("Creating DataLoader")
  train_dataloader, validation_dataloader, test_dataloader = createDataloader(DEVICE, w2v_model)
  #__________________________________________________________________________________________
  '''1.b: LSTM with Average Pool'''
  print("Working with LSTM Avg Pool")
  print("Creating model")
  model = LSTMModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size)
  model = model.to(device=DEVICE)
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
  criterion = nn.BCEWithLogitsLoss()
  #Prepare the data
  print("Creating DataLoader")
  train_dataloader, validation_dataloader, test_dataloader = createDataloader(DEVICE, w2v_model)
  model_name = "LSTM_AVG_POOL"
  print("Start Training")
  train_loss, val_loss, train_acc, val_acc, model = trainModel(train_dataloader, validation_dataloader, model, optimizer, criterion, max_epochs)
  print("End Training")
  
  plotFig(train_loss=train_loss, val_loss=val_loss, name=model_name)
  print("Start testing the model")

  #Test the model on testing data
  testModel(test_dataloader, model,name = model_name, file_path = file_path, criterion=criterion)
  print("End model testing")

  del model
  del optimizer
  del criterion
  del train_dataloader
  del validation_dataloader 
  del test_dataloader
 #__________________________________________________________________________________________
  '''1.c: LSTM with Attention Model'''
  print("Working with LSTM Attention model")
  print("Creating model")
  model = LSTMAttention(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size)
  model = model.to(device=DEVICE)
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
  criterion = nn.BCEWithLogitsLoss()
  #Prepare the data
  print("Creating DataLoader")
  train_dataloader, validation_dataloader, test_dataloader = createDataloader(DEVICE, w2v_model)
  model_name = "LSTM_ATTENTION"

  print("Start Training")
  train_loss, val_loss, train_acc, val_acc, model = trainModel(train_dataloader, validation_dataloader, model, optimizer, criterion, max_epochs)
  print("End Training")

  plotFig(train_loss=train_loss, val_loss=val_loss, name=model_name)
  print("Start testing the model")
  #Test the model on testing data
  testModel(test_dataloader, model, name = model_name, file_path = file_path, criterion=criterion)
  print("End model testing")
  #Clean the GPU
  del model
  del optimizer
  del criterion
  del train_dataloader
  del validation_dataloader 
  del test_dataloader
  #__________________________________________________________________________________________
  '''1.d: LSTM with Transformer Encoder'''
  print("Working with Transformer based LSTM Model")
  print("Creating the model")
  model = LSTMTransformer(input_size=300, hidden_size=512, num_head=4, output_size=1)
  model = model.to(device=DEVICE)
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
  criterion = nn.BCEWithLogitsLoss()
  print("Creating DataLoader")
  train_dataloader, validation_dataloader, test_dataloader = createDataloader(DEVICE, w2v_model)
  model_name = "LSTM_TRANSFORMER"

  print("Start Training")
  train_loss, val_loss, train_acc, val_acc, model = trainModel(train_dataloader, validation_dataloader, model, optimizer, criterion, max_epochs)
  print("End Training")

  plotFig(train_loss=train_loss, val_loss=val_loss, name=model_name)
  print("Start testing the model")
  #Test the model on testing data
  testModel(testDataloader=test_dataloader, model=model, name = model_name, file_path = file_path, criterion=criterion)
  print("End model testing")

  del model
  del optimizer
  del criterion
  del train_dataloader
  del validation_dataloader 
  del test_dataloader