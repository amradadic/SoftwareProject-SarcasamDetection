#%%
#BiLSTM PYTORCH
from nltk.corpus import stopwords 
from collections import Counter
import string
import re
from unicodedata import bidirectional
import numpy as np
import pandas as pd
from utils import preprocessing
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report
import torch #pytorch
import torch.nn as nn
from torch.autograd import Variable 
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import (
    DataLoader, TensorDataset
) 
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

#%%
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")
    
#%%

df_SPIRS_sarcastic = pd.read_csv('SPIRS-sarcastic (1).csv')
df_SPIRS_non_sarcastic = pd.read_csv('SPIRS-non-sarcastic (1).csv')

#remove NAs from sar_text
df_SPIRS_sarcastic = preprocessing.remove_na_from_column(df_SPIRS_sarcastic, 'sar_text')
df_SPIRS_non_sarcastic = preprocessing.remove_na_from_column(df_SPIRS_non_sarcastic, 'sar_text')

#fill na from other columns
df_SPIRS_sarcastic = preprocessing.fill_na_from_column(df_SPIRS_sarcastic, 'eli_text')
df_SPIRS_non_sarcastic = preprocessing.fill_na_from_column(df_SPIRS_non_sarcastic, 'eli_text')

df_SPIRS_sarcastic = preprocessing.fill_na_from_column(df_SPIRS_sarcastic, 'obl_text')
df_SPIRS_non_sarcastic = preprocessing.fill_na_from_column(df_SPIRS_non_sarcastic, 'obl_text')

df_SPIRS_sarcastic = preprocessing.fill_na_from_column(df_SPIRS_sarcastic, 'cue_text')
#non sar has no cue text

#get context
df_SPIRS_sarcastic = preprocessing.get_df_context(df_SPIRS_sarcastic, cue = False)
df_SPIRS_non_sarcastic = preprocessing.get_df_context(df_SPIRS_non_sarcastic, cue = False)


#df_SPIRS_sarcastic = preprocessing.remove_na_from_column(df_SPIRS_sarcastic, 'eli_text')
#df_SPIRS_non_sarcastic = preprocessing.remove_na_from_column(df_SPIRS_non_sarcastic, 'eli_text')

#df_SPIRS_sarcastic = preprocessing.remove_na_from_column(df_SPIRS_sarcastic, 'obl_text')
#df_SPIRS_non_sarcastic = preprocessing.remove_na_from_column(df_SPIRS_non_sarcastic, 'obl_text')

#df_SPIRS_sarcastic = preprocessing.remove_na_from_column(df_SPIRS_sarcastic, 'cue_text')
#non sar has no cue text


#preprocess columns
df_SPIRS_sarcastic = preprocessing.preprocess_tweets(df_SPIRS_sarcastic, 'sar_text')
df_SPIRS_non_sarcastic = preprocessing.preprocess_tweets(df_SPIRS_non_sarcastic, 'sar_text')

df_SPIRS_sarcastic = preprocessing.preprocess_tweets(df_SPIRS_sarcastic, 'eli_text')
df_SPIRS_non_sarcastic = preprocessing.preprocess_tweets(df_SPIRS_non_sarcastic, 'eli_text')

df_SPIRS_sarcastic = preprocessing.preprocess_tweets(df_SPIRS_sarcastic, 'obl_text')
df_SPIRS_non_sarcastic = preprocessing.preprocess_tweets(df_SPIRS_non_sarcastic, 'obl_text')

#df_SPIRS_sarcastic = preprocessing.preprocess_tweets(df_SPIRS_sarcastic, 'cue_text')
#non sar has no cue text

#without context
#df_SPIRS_sarcastic = df_SPIRS_sarcastic[['sar_text']]
#df_SPIRS_non_sarcastic = df_SPIRS_non_sarcastic[['sar_text']]

#with context

#add labels
df_SPIRS_sarcastic = df_SPIRS_sarcastic.assign(label=1)
df_SPIRS_non_sarcastic = df_SPIRS_non_sarcastic.assign(label=0)

#concat
df_SPIRS = pd.concat([df_SPIRS_sarcastic, df_SPIRS_non_sarcastic], ignore_index=True)

#test train split

#for context
df_SPIRS_X = preprocessing.concat_df(df_SPIRS.loc[:, ~df_SPIRS.columns.isin(['sar_id', 'label'])], 'sar_text')
df_SPIRS_Y = df_SPIRS[['label']]

x, x_test2, y, y_test2 = train_test_split(df_SPIRS_X,df_SPIRS_Y,test_size=0.1,train_size=0.9, shuffle=True)
x_train2, x_val2, y_train2, y_val2 = train_test_split(x,y,test_size = 0.1,train_size =0.9,shuffle=True)
#%%
#x, x_test2, y, y_test2 = train_test_split(df_SPIRS.loc[:, ~df_SPIRS.columns.isin(['sar_id', 'label'])],df_SPIRS[['label']],test_size=0.1,train_size=0.9)
#x_train2, x_val2, y_train2, y_val2 = train_test_split(x,y,test_size = 0.1,train_size =0.9)

#%%
print(f'shape of train data is {x_train2.shape}')
print(f'shape of val data is {x_val2.shape}')
print(f'shape of test data is {x_test2.shape}')
# %%
def preprocess_string(s):
    # Remove all non-word characters (everything except numbers and letters)
    s = re.sub(r"[^\w\s]", '', s)
    # Replace all runs of whitespaces with no space
    s = re.sub(r"\s+", '', s)
    # replace digits with no space
    s = re.sub(r"\d", '', s)

    return s

def tockenize(x_train,y_train,x_val,y_val):
    word_list = []

    stop_words = set(stopwords.words('english')) 
    for sent in x_train:
        for word in sent.lower().split():
            word = preprocess_string(word)
            if word not in stop_words and word != '':
                word_list.append(word)
  
    corpus = Counter(word_list)
    # sorting on the basis of most common words
    corpus_ = sorted(corpus,key=corpus.get,reverse=True)[:1000]
    # creating a dict
    onehot_dict = {w:i+1 for i,w in enumerate(corpus_)}
    
    # tockenize
    final_list_train,final_list_test = [],[]
    for sent in x_train:
            final_list_train.append([onehot_dict[preprocess_string(word)] for word in sent.lower().split() 
                                     if preprocess_string(word) in onehot_dict.keys()])
    for sent in x_val:
            final_list_test.append([onehot_dict[preprocess_string(word)] for word in sent.lower().split() 
                                    if preprocess_string(word) in onehot_dict.keys()])
            
    encoded_train = [1 if label ==1 else 0 for label in y_train]  
    encoded_test = [1 if label ==1 else 0 for label in y_val] 
    return np.array(final_list_train), np.array(encoded_train),np.array(final_list_test), np.array(encoded_test),onehot_dict

#%%
#x_train,y_train,x_val,y_val,vocab = tockenize(x_train2['sar_text'],y_train2['label'],x_val2['sar_text'],y_val2['label'])
#x_temp,y_temp,x_test,y_test,vocab = tockenize(x_train2['sar_text'],y_train2['label'],x_test2['sar_text'],y_test2['label'])

# %%
tokenizer, dictionary = preprocessing.get_dictionary(x_train2, 5000)
embedding = preprocessing.get_glove_embedding_BiLSTM(dictionary)

vocab = dictionary
print(f'Length of vocabulary is {len(vocab)}')
#%%
X_train_indices = tokenizer.texts_to_sequences(x_train2['sar_text'])
X_val_indices = tokenizer.texts_to_sequences(x_val2['sar_text'])
X_test_indices = tokenizer.texts_to_sequences(x_test2['sar_text'])

# %%
def padding_(sentences, seq_len):
    features = np.zeros((len(sentences), seq_len),dtype=int)
    for ii, review in enumerate(sentences):
        if len(review) != 0:
            features[ii, -len(review):] = np.array(review)[:seq_len]
    return features

#we have very less number of reviews with length > 500.
#So we will consideronly those below it.
#x_train_pad = padding_(x_train,100)
#x_val_pad = padding_(x_val,100)
#x_test_pad = padding_(x_test,100)

x_train_pad = padding_(np.array(X_train_indices),100)
x_val_pad = padding_(np.array(X_val_indices),100)
x_test_pad = padding_(np.array(X_test_indices),100)
# %%
# create Tensor datasets
train_data = TensorDataset(torch.from_numpy(x_train_pad), torch.from_numpy(y_train2['label'].to_numpy()))
valid_data = TensorDataset(torch.from_numpy(x_val_pad), torch.from_numpy(y_val2['label'].to_numpy()))
test_data = TensorDataset(torch.from_numpy(x_test_pad), torch.from_numpy(y_test2['label'].to_numpy()))

# dataloaders
batch_size = 64

# make sure to SHUFFLE your data
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)
valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size, drop_last=True)
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size, drop_last=True)
# %%
class SentimentRNN(nn.Module):
    def __init__(self,no_layers,vocab_size,hidden_dim,embedding_dim,drop_prob=0.5):
        super(SentimentRNN,self).__init__()
 
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
 
        self.no_layers = no_layers
        self.vocab_size = vocab_size
    
        # embedding and LSTM layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = nn.Parameter(torch.from_numpy(embedding).float(),requires_grad=True)
        
        
        #lstm
        self.lstm = nn.LSTM(input_size=embedding_dim,hidden_size=self.hidden_dim,
                           num_layers=no_layers, batch_first=True, bidirectional = True)
        
        self.maxpool = nn.MaxPool1d(4) # Where 4 is kernal size
        
        # dropout layer
   #     self.dropout = nn.Dropout(0.3)
   # 
        # linear and sigmoid layer
        self.fc = nn.Linear(self.hidden_dim, output_dim)
        self.sig = nn.Sigmoid()
        
    def forward(self,x,hidden):
        batch_size = x.size(0)
        # embeddings and lstm_out
        embeds = self.embedding(x)  # shape: B x S x Feature   since batch = True
        #print(embeds.shape)  #[50, 500, 1000]
        lstm_out, hidden = self.lstm(embeds, hidden)
        
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim) 
        
    #    # dropout and fully connected layer
   #     out = self.dropout(lstm_out)
        out = self.fc(lstm_out)
        
        # sigmoid function
        sig_out = self.sig(out)
        
        # reshape to be batch_size first
        sig_out = sig_out.view(batch_size, -1)

        sig_out = sig_out[:, -1] # get last batch of labels
        
        # return last sigmoid output and hidden state
        return sig_out, hidden
  
        
        
    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        h0 = torch.zeros((self.no_layers * 2,batch_size,self.hidden_dim)).to(device)
        c0 = torch.zeros((self.no_layers * 2,batch_size,self.hidden_dim)).to(device)
        hidden = (h0,c0)
        return hidden

#%%
no_layers = 2
vocab_size = len(vocab) + 1 #extra 1 for padding
embedding_dim = 100
output_dim = 1
hidden_dim = 128


model = SentimentRNN(no_layers,vocab_size,hidden_dim,embedding_dim,drop_prob=0.5)

#moving to gpu
model.to(device)

print(model)
# %%
# loss and optimization functions
lr=0.0005

criterion = nn.BCELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# function to predict accuracy
def acc(pred,label):
    pred = torch.round(pred.squeeze())
    return torch.sum(pred == label.squeeze()).item()

#%%
clip = 5
epochs = 5
valid_loss_min = np.Inf
# train for some number of epochs
epoch_tr_loss,epoch_vl_loss = [],[]
epoch_tr_acc,epoch_vl_acc = [],[]

for epoch in range(epochs):
    train_losses = []
    train_acc = 0.0
    model.train()
    # initialize hidden state 
    h = model.init_hidden(batch_size)
    for inputs, labels in train_loader:
        
        inputs, labels = inputs.to(device), labels.to(device)   
        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        h = tuple([each.data for each in h])
        
        model.zero_grad()
        output,h = model(inputs,h)
        
        # calculate the loss and perform backprop
        loss = criterion(output.squeeze(), labels.float())
        loss.backward()
        train_losses.append(loss.item())
        # calculating accuracy
        accuracy = acc(output,labels)
        train_acc += accuracy
        #`clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
 
    
        
    val_h = model.init_hidden(batch_size)
    val_losses = []
    val_acc = 0.0
    model.eval()
    for inputs, labels in valid_loader:
            val_h = tuple([each.data for each in val_h])

            inputs, labels = inputs.to(device), labels.to(device)

            output, val_h = model(inputs, val_h)
            val_loss = criterion(output.squeeze(), labels.float())

            val_losses.append(val_loss.item())
            
            accuracy = acc(output,labels)
            val_acc += accuracy
            
    epoch_train_loss = np.mean(train_losses)
    epoch_val_loss = np.mean(val_losses)
    epoch_train_acc = train_acc/len(train_loader.dataset)
    epoch_val_acc = val_acc/len(valid_loader.dataset)
    epoch_tr_loss.append(epoch_train_loss)
    epoch_vl_loss.append(epoch_val_loss)
    epoch_tr_acc.append(epoch_train_acc)
    epoch_vl_acc.append(epoch_val_acc)
    print(f'Epoch {epoch+1}') 
    print(f'train_loss : {epoch_train_loss} val_loss : {epoch_val_loss}')
    print(f'train_accuracy : {epoch_train_acc*100} val_accuracy : {epoch_val_acc*100}')
    if epoch_val_loss <= valid_loss_min:
        torch.save(model.state_dict(), './state_dict.pt')
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,epoch_val_loss))
        valid_loss_min = epoch_val_loss
    print(25*'==')
    
#%%
fig = plt.figure(figsize = (20, 6))
plt.subplot(1, 2, 1)
plt.plot(epoch_tr_acc, label='Train Acc')
plt.plot(epoch_vl_acc, label='Validation Acc')
plt.title("Accuracy")
plt.legend()
plt.grid()
    
plt.subplot(1, 2, 2)
plt.plot(epoch_tr_loss, label='Train loss')
plt.plot(epoch_vl_loss, label='Validation loss')
plt.title("Loss")
plt.legend()
plt.grid()

plt.show()


#%%

model.load_state_dict(torch.load('./state_dict.pt'), strict=False)

test_losses = []
num_correct = 0
h = model.init_hidden(batch_size)

model.eval()
for inputs, labels in test_loader:
    h = tuple([each.data for each in h])
    inputs, labels = inputs.to(device), labels.to(device)
    output, h = model(inputs, h)
    test_loss = criterion(output.squeeze(), labels.float())
    test_losses.append(test_loss.item())
    pred = torch.round(output.squeeze())  # Rounds the output to 0/1
    correct_tensor = pred.eq(labels.float().view_as(pred))
    correct = np.squeeze(correct_tensor.cpu().numpy())
    num_correct += np.sum(correct)

print("Test loss: {:.3f}".format(np.mean(test_losses)))
test_acc = num_correct/len(test_loader.dataset)
print("Test accuracy: {:.3f}%".format(test_acc*100))
# %%

#%%
from sklearn.model_selection import KFold
