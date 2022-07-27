#%%
#BiLSTM PYTORCH
import numpy as np
import pandas as pd
from utils import preprocessing
import torch #pytorch
import torch.nn as nn
from torch.utils.data import (
    DataLoader, TensorDataset
) 
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

x_train2 = pd.read_csv('split/x_train_new.csv',converters = {'sar_text': str})
x_test2 = pd.read_csv('split/x_test_new.csv',converters = {'sar_text': str})
x_val2 = pd.read_csv('split/x_val_new.csv',converters = {'sar_text': str})
y_train2 = pd.read_csv('split/y_train_new.csv')
y_test2 = pd.read_csv('split/y_test_new.csv')
y_val2 = pd.read_csv('split/y_val_new.csv')

print(f'shape of train data is {x_train2.shape}')
print(f'shape of val data is {x_val2.shape}')
print(f'shape of test data is {x_test2.shape}')
print(f'shape of train data is {y_train2.shape}')
print(f'shape of val data is {y_val2.shape}')
print(f'shape of test data is {y_test2.shape}')


#%%
#no context
x_train2 = x_train2[['sar_text']]
x_test2 = x_test2[['sar_text']]
x_val2 = x_val2[['sar_text']]

#context
#x_train2 = x_train2[['sar_text', 'eli_text', 'obl_text']]
#x_test2 = x_test2[['sar_text', 'eli_text', 'obl_text']]
#x_val2 = x_val2[['sar_text', 'eli_text', 'obl_text']]

#%%
#x_train2 = preprocessing.concat_df(x_train2, 'sar_text')
#x_val2 = preprocessing.concat_df(x_val2, 'sar_text')
#x_test2 = preprocessing.concat_df(x_test2, 'sar_text')

#%%
#creating glove embedding
tokenizer, dictionary = preprocessing.get_dictionary(x_train2)
embedding = preprocessing.get_glove_embedding_BiLSTM(dictionary)

vocab = dictionary
print(f'Length of vocabulary is {len(vocab)}')
#%%
#train, validation and test tensors
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

#we have very less number of reviews with length > 50.

#x_train_pad = padding_(x_train,100)
#x_val_pad = padding_(x_val,100)
#x_test_pad = padding_(x_test,100)

x_train_pad = padding_(np.array(X_train_indices),50)
x_val_pad = padding_(np.array(X_val_indices),50)
x_test_pad = padding_(np.array(X_test_indices),50)
# %%
# create Tensor datasets
train_data = TensorDataset(torch.from_numpy(x_train_pad), torch.from_numpy(y_train2['label'].to_numpy()))
valid_data = TensorDataset(torch.from_numpy(x_val_pad), torch.from_numpy(y_val2['label'].to_numpy()))
test_data = TensorDataset(torch.from_numpy(x_test_pad), torch.from_numpy(y_test2['label'].to_numpy()))

# dataloaders
batch_size = 64

train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)
valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size, drop_last=True)
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size, drop_last=True)
# %%
#BiLSTM network definition
class SentimentRNN(nn.Module):
    def __init__(self,no_layers,vocab_size,hidden_dim,embedding_dim,drop_prob=0.5):
        super(SentimentRNN,self).__init__()
 
        #output_dim = 1, hidden_dim = 1
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
 
        #no_layers = 2, vocab_size = len(dictionary)
        self.no_layers = no_layers
        self.vocab_size = vocab_size
    
        # embedding and LSTM layers
        #embedding_dim = 200
        #embedding - pretrained glove - matrix (30688, 200)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = nn.Parameter(torch.from_numpy(embedding).float(),requires_grad=False)
        
        
        #lstm
        self.lstm = nn.LSTM(input_size=embedding_dim,hidden_size=self.hidden_dim,
                           num_layers=no_layers, batch_first=True, bidirectional = True)
        
        self.maxpool = nn.MaxPool1d(4) # Where 4 is kernal size
        
        # dropout layer
   #     self.dropout = nn.Dropout(0.3)
   # 
        # linear and sigmoid layer
        self.fc = nn.Linear(self.hidden_dim//4, output_dim)
        self.sig = nn.Sigmoid()
        
    def forward(self,x,hidden):
        batch_size = x.size(0)
        # embeddings and lstm_out
        embeds = self.embedding(x) 
        
        lstm_out, hidden = self.lstm(embeds, hidden)
        
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim) 
        
        pooled = self.maxpool(lstm_out)
        
        #linear layer
      #  out = self.fc(lstm_out)
        out = self.fc(pooled)
        
        # sigmoid function
        sig_out = self.sig(out)
        
        # reshape to be batch_size first
        sig_out = sig_out.view(batch_size, -1)

        sig_out = sig_out[:, -1] # get last batch of labels
        
        # return last sigmoid output and hidden state
        return sig_out, hidden
  
        
        
    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        h0 = torch.zeros((self.no_layers * 2,batch_size,self.hidden_dim)).to(device)
        c0 = torch.zeros((self.no_layers * 2,batch_size,self.hidden_dim)).to(device)
        hidden = (h0,c0)
        return hidden

#%%
#hyperparameters
no_layers = 2
vocab_size = len(vocab)
embedding_dim = 200
output_dim = 1
hidden_dim = 256


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
#training
clip = 5
epochs = 5
valid_loss_min = 0
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
     #   train_losses.append(loss.item())
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
            
 #   epoch_train_loss = np.mean(train_losses)
    epoch_train_loss = 0
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
    if epoch_val_acc >= valid_loss_min:
        torch.save(model.state_dict(), './state_dict.pt')
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,epoch_val_loss))
        valid_loss_min = epoch_val_acc
    print(25*'==')
    
#%%
#plotting loss and accuracy
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
#loading the best model and testing
model.load_state_dict(torch.load('./state_dict.pt'), strict=False)

test_losses = []
num_correct = 0
h = model.init_hidden(batch_size)
TP = 0
TN = 0
FP = 0
FN = 0

bilstm_predictions = []
true_labels = []

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
    
  #  print(pred.cpu().detach().numpy())
    
    #calculating metrics
    for i in range(0, len(pred)):
        prediction = pred.cpu().detach().numpy()[i]
        true = labels[i]
        bilstm_predictions.append(prediction)
        true_labels.append(true)
        
        if prediction == 1 and true == 1 :
            TP += 1
        elif prediction == 0 and true == 0 :
            TN += 1
        elif prediction == 1 and true == 0 :
            FP += 1
        elif prediction == 0 and true == 1:
            FN += 1
            
    num_correct += np.sum(correct)

print("Test loss: {:.3f}".format(np.mean(test_losses)))
test_acc = num_correct/len(test_loader.dataset)
print("Test accuracy: {:.3f}%".format(test_acc*100))
print("TP: ", TP)
print("TN: ", TN)
print("FP: ", FP)
print("FN: ", FN)
precision = TP / (TP + FP)
print("Precision: ", precision)
recall = TP / (TP + FN)
print("Recal:", recall)
print("F1", 2 * (precision * recall) / (precision + recall))


#%%
#JSON file
#metrics_context = pf.metrics(true_labels, bilstm_predictions, target_names=['Non-sarcastic', 'Sarcastic'])
#print(metrics_context)

#df = pd.DataFrame({'sar_id': df_SPIRS['sar_id'][len(x_train2['sar_text'])], 'label': true_labels, 'predicted_value': bilstm_predictions})
#pf.json_metrics("json\SVMmetrics_GloVe_context.json", "SVM - LinearSVC with context", "Glove", metrics_context, df)
# %%
