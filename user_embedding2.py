#%%
import zipfile
import pandas as pd
from csv import reader
import re
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from utils import preprocessing

#%%
file = None

with zipfile.ZipFile('spirs_history.zip') as zip:
  file = zip.open('spirs_history/SPIRS-sarcastic-history.txt', mode='r')

dictionary = {}
old_user = None
sentences = []

i = 0
n_user_tweets = 0

for line in file:

    if i > 0:
  
        user_id, tweet_id, tweet = re.split(r'\t+', line.decode('utf-8'))

        if user_id in dictionary:
        #RAM limit
    #    if n_user_tweets < 500:
        #    n_user_tweets += 1
            dictionary[user_id] = dictionary[user_id] + preprocessing.process_tweet(tweet) + '\t'
        else:
        #    n_user_tweets = 1
            dictionary[user_id] = preprocessing.process_tweet(tweet) + '\t'
      

    if i == 500000:
      break

    i += 1
# %%
import random

for key in dictionary:
  tweets = dictionary[key].split('\t')
  if len(tweets) > 1000 :
    tweets = random.sample(tweets, 1000)

    dictionary[key] = '\t'.join(tweets);


#%%
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

from csv import writer

out_file = 'user_embeddings.txt'
with open(out_file, 'a') as f:

#with open('user_embedding.csv', newline='', mode='a', encoding='utf-8') as f_object:

  # Pass this file object to csv.writer() and get a writer object
  #writer_object = writer(f_object)

  for key in dictionary:
    # Tokenize sentences
    encoded_input = tokenizer(dictionary[key].split('\t'), padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
      model_output = model(**encoded_input)

    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

    #storing the embedding
   # writer_object.writerow([key, torch.mean(sentence_embeddings, 0).numpy()])
    f.write(str(key) + ', ' + str(torch.mean(sentence_embeddings, 0).numpy()).replace('\n','') + '\\n')
  #  print(str(torch.mean(sentence_embeddings, 0).numpy()).replace('\n','') + '\n')

    del encoded_input, model_output, sentence_embeddings

    print(key)

  #Close the file object
  #f_object.close()
# %%
