#%%
#BiLSTM PYTORCH
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

maxlen = 50

#%%

df_SPIRS_sarcastic = pd.read_csv('SPIRS-sarcastic.csv')
df_SPIRS_non_sarcastic = pd.read_csv('SPIRS-non-sarcastic.csv')

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

#preprocess columns
df_SPIRS_sarcastic = preprocessing.preprocess_tweets(df_SPIRS_sarcastic, 'sar_text')
df_SPIRS_non_sarcastic = preprocessing.preprocess_tweets(df_SPIRS_non_sarcastic, 'sar_text')

df_SPIRS_sarcastic = preprocessing.preprocess_tweets(df_SPIRS_sarcastic, 'eli_text')
df_SPIRS_non_sarcastic = preprocessing.preprocess_tweets(df_SPIRS_non_sarcastic, 'eli_text')

df_SPIRS_sarcastic = preprocessing.preprocess_tweets(df_SPIRS_sarcastic, 'obl_text')
df_SPIRS_non_sarcastic = preprocessing.preprocess_tweets(df_SPIRS_non_sarcastic, 'obl_text')

df_SPIRS_sarcastic = preprocessing.preprocess_tweets(df_SPIRS_sarcastic, 'cue_text')
#non sar has no cue text

#without context
df_SPIRS_sarcastic = df_SPIRS_sarcastic[['sar_text']]
df_SPIRS_non_sarcastic = df_SPIRS_non_sarcastic[['sar_text']]

#with context
#get context
#df_SPIRS_sarcastic = preprocessing.get_df_context(df_SPIRS_sarcastic, cue = True)
#df_SPIRS_non_sarcastic = preprocessing.get_df_context(df_SPIRS_non_sarcastic, cue = True)

#add labels
df_SPIRS_sarcastic = df_SPIRS_sarcastic.assign(label=1)
df_SPIRS_non_sarcastic = df_SPIRS_non_sarcastic.assign(label=0)

#concat
df_SPIRS = pd.concat([df_SPIRS_sarcastic, df_SPIRS_non_sarcastic], ignore_index=True)

#test train split
x_train, x_test, y_train, y_test = train_test_split(df_SPIRS.loc[:, ~df_SPIRS.columns.isin(['sar_id', 'label'])], df_SPIRS[['label']], test_size=0.2, random_state=123, shuffle=True)
#%%

#%%
x, x_test2, y, y_test2 = train_test_split(df_SPIRS.loc[:, ~df_SPIRS.columns.isin(['sar_id', 'label'])],df_SPIRS[['label']],test_size=0.1,train_size=0.9)
x_train2, x_val2, y_train2, y_val2 = train_test_split(x,y,test_size = 0.1,train_size =0.9)

#%%
tokenizer, dictionary = preprocessing.get_dictionary(x_train2, 25000)
embedding = preprocessing.get_glove_embedding_BiLSTM(dictionary)
  
#%%
X_train_indices = tokenizer.texts_to_sequences(x_train2['sar_text'][0:12000])
X_train_indices = pad_sequences(X_train_indices, maxlen=maxlen, padding='post')

X_val_indices = tokenizer.texts_to_sequences(x_val2['sar_text'][0:4000])
X_val_indices = pad_sequences(X_val_indices, maxlen=maxlen, padding='post')

X_test_indices = tokenizer.texts_to_sequences(x_test2['sar_text'][0:4000])
X_test_indices = pad_sequences(X_test_indices, maxlen=maxlen, padding='post')

#%%

train_data = TensorDataset(torch.from_numpy(X_train_indices), torch.from_numpy(y_train2['label'][0:12000].to_numpy()))
val_data = TensorDataset(torch.from_numpy(X_val_indices), torch.from_numpy(y_val2['label'][0:4000].to_numpy()))
test_data = TensorDataset(torch.from_numpy(X_test_indices), torch.from_numpy(y_test2['label'][0:4000].to_numpy()))


#128, 64
batch_size = 64

train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size,  drop_last=True)
val_loader = DataLoader(val_data, shuffle=True, batch_size=batch_size,  drop_last=True)
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size,  drop_last=True)
# %%
# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    
    
#%%
class SentimentNet(nn.Module):
    def __init__(self, vocab_size, output_size, embedding_dim, embedding, hidden_dim, n_layers, drop_prob=0.5):
        super(SentimentNet, self).__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        
      #  self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(embedding).float())
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
   #     self.embedding.weight = nn.Parameter(torch.from_numpy(embedding).float(),requires_grad=False)
   #     model.embedding.weight.data.copy_(torch.from_numpy(embedding))
       
  #      self.embedding.load_state_dict({'weight': torch.from_numpy(embedding).float()})
       
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, hidden):
        batch_size = x.size(0)
        x = x.long()
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        
        out = self.dropout(lstm_out)
        out = self.fc(out)
        out = self.sigmoid(out)
        
        out = out.view(batch_size, -1)
        out = out[:,-1]
        return out, hidden
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
        return hidden
# %%
vocab_size = len(dictionary)
output_size = 1
embedding_dim = 100
hidden_dim = 512
n_layers = 2

model = SentimentNet(vocab_size, output_size, embedding_dim, embedding, hidden_dim, n_layers)
model.to(device)
#%%
lr=0.0005
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# %%
epochs = 10
counter = 0
print_every = 15
clip = 5
valid_loss_min = np.Inf

model.train()

#%%

for i in range(epochs):
    h = model.init_hidden(batch_size)
    
    for inputs, labels in train_loader:
        counter += 1
     #   print(counter)
        h = tuple([e.data for e in h])
        inputs, labels = inputs.to(device), labels.to(device)
        model.zero_grad()
        output, h = model(inputs, h)
        loss = criterion(output.squeeze(), labels.float())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        if counter%print_every == 0:
            val_h = model.init_hidden(batch_size)
            val_losses = []
            model.eval()
            for inp, lab in val_loader:
                val_h = tuple([each.data for each in val_h])
                inp, lab = inp.to(device), lab.to(device)
                out, val_h = model(inp, val_h)
                val_loss = criterion(out.squeeze(), lab.float())
                val_losses.append(val_loss.item())
                
            model.train()
            print("Epoch: {}/{}...".format(i+1, epochs),
                  "Step: {}...".format(counter),
                  "Loss: {:.6f}...".format(loss.item()),
                  "Val Loss: {:.6f}".format(np.mean(val_losses)))
            if np.mean(val_losses) <= valid_loss_min:
                torch.save(model.state_dict(), './state_dict.pt')
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,np.mean(val_losses)))
                valid_loss_min = np.mean(val_losses)
# %%
# Loading the best model
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
