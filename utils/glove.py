#%%
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
import numpy as np

import os

#converting the GloVe file containing the word embeddings to the word2vec format
#creating a model 
def load_glove() :
    
    dirname = os.path.dirname(__file__)
    filepath = os.path.join(dirname, '..\\glove.6B.100d.txt')
    
    word2vec_output_file = 'glove.6B.100d' +'.word2vec'
    glove2word2vec(filepath, word2vec_output_file)
    
    model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
    
    return model

#model = load_glove('glove.6B.100d', filepath)

class GloveVectorizer:
    def __init__(self, model):
        print("Loading in word vectors...")
        self.word_vectors = model
        print("Finished loading in word vectors")

    def fit(self, data):
        pass

    def transform(self, data):
        
        # determine the dimensionality of vectors
        self.D = self.word_vectors.get_vector('king').shape[0]

        # the final vector
        X = np.zeros((len(data), self.D))
        n = 0
        emptycount = 0
        
        for sentence in data:
            tokens = sentence.split()
            vecs = []
            m = 0
            
        #    if n < 1:
        #        print(sentence + '\n\n')
            
            for word in tokens:
                try:
                # throws KeyError if word not found
                #get the vector of a current word and append it to vecs
                    vec = self.word_vectors.get_vector(word)
                    vecs.append(vec)
                    m += 1
                except KeyError:
                    pass
                
            if len(vecs) > 0:
                vecs = np.array(vecs)
                #each element i is the mean value of the ith column of vecs (meann value of ith dimension of every word)
                X[n] = vecs.mean(axis=0)
            else:
                emptycount += 1
                
            n += 1
            
        print("Numer of samples with no words found: %s / %s" % (emptycount, len(data)))
        
        return X


    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
    

# %%
