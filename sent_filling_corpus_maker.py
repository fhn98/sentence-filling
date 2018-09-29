
# coding: utf-8

# In[1]:


import numpy as np
import pickle
from urllib import request
from nltk import word_tokenize
from nltk import sent_tokenize
import nltk


# In[2]:


def open_file (file_name):
    with open (file_name , 'rb') as pkl:
        ret = pickle.load(pkl)
    return ret


# In[3]:


def dump (data , file_name):
     with open (file_name,'wb') as pkl:
        pickle.dump(data , pkl)     


# In[4]:


def convert (book , stoi , itos , vectors):
    """
    description: converting words of the book to indexes.
                 if there was an unknown word, we add it to the end of our dicitionary and use a random vector for its embedding
    
    inputs: stoi: a dictionary from words to their indices in GloVe 
            itos: a list in which index of a word is its index in GloVe vectors
            vectors: a vocab_size*embedding_length array containing vector representation of each word in its corresponding row
    
    outputs: stoi , itos , vectors after adding new words
             book_by_index: our corpus after converting words to their GloVe indices
    """
    
    book_by_index=[]
    i = len(stoi)
    
    for sent in book:
        listt=[]
        for word in sent:
            if word.lower() in stoi:#if the word already exists in out vocabulary
                listt.append(stoi[word.lower()])
            else:
                stoi[word.lower()]=i
                itos.append(word.lower())
                new = np.random.rand(1,vectors.shape[1])
                vectors = np.concatenate((vectors , new) , axis = 0)#adding a random as the embedding of the new word to vectors
                listt.append(i)
                i+=1
        book_by_index.append(listt)

    return book_by_index , stoi , itos , vectors


# In[5]:


def make_corpus(books , stoi, itos , vectors):
    """
    description: this function receives book_urls and tokenizes their text. then  it calls 'convert' function.
    
    inputs: books: books which we are going to use as our corpus
            stoi: a dictionary from words to their indices in GloVe 
            itos: a list in which index of a word is its index in GloVe vectors
            vectors: a vocab_size*embedding_length array containing vector representation of each word in its corresponding row
    
    outputs: stoi , itos , vectors after adding new words
             book_by_index: our corpus after converting words to their GloVe indices
    """

    book_by_index , stoi , itos , vectors = convert(books , stoi , itos , vectors)
    

    return stoi , itos , vectors , book_by_index


# In[ ]:


def download_books(book_urls):
    sents=[]
    corpus=[]
    
    for book_url in book_urls:
        response = request.urlopen(book_url)
        raw = response.read().decode('utf8')
        sents = list(sent_tokenize(raw))
        for sent in sents:
            corpus.append(list(word_tokenize(sent)))
    
    return corpus


# In[7]:


if __name__=='__main__':
    
    stoi = open_file('stoi.pkl')
    itos = open_file('itos.pkl')
    vectors = open_file('vectors.pkl')
    
    books = download_books(['http://www.gutenberg.org/cache/epub/730/pg730.txt'] )

    stoi , itos , vectors , book_by_index = make_corpus(books , stoi , itos , vectors)
    
    print (vectors.shape[0])
    
    #writing back data
    dump (stoi , 'stoi.pkl')
    dump (itos , 'itos.pkl')
    dump (vectors , 'vectors.pkl')
    dump (book_by_index , 'sent_train.pkl')   

