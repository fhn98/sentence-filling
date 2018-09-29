
# coding: utf-8

# In[2]:


import numpy as np
import pickle


# In[3]:


def load_file(filename):
    with open (filename , 'rb') as pkl:
        file = pickle.load(pkl)
    
    return file


# In[4]:


def make_array(file_length , batch_size): 
    sents_array = np.arange(file_length , dtype = np.int32)
    
    batch_len = file_length//batch_size
    
    reshaped_array = (sents_array[:batch_len*batch_size]).reshape(batch_size , batch_len)
    
    return reshaped_array , batch_len


# In[5]:


def get_words (index , file):
    #getting words of the sentence with index index in the file
    return file[index]


# In[9]:


def make_tensor_of_words (array , file):
    list_of_words=[] #a list of lists in which each element is a list of words of the corresponding row
    max_len=0 #length of the longest row
    
    batch_size=array.shape[0]
    
    for row in range (batch_size) :
        row_list=[] #a list containing all the words of each row in the array
        for col in array[row,:] :
            row_list=row_list+get_words(col , file)
        max_len = max(max_len , len(row_list))
        list_of_words.append(row_list)
        
    ret = np.zeros ([array.shape[0] , max_len] , dtype=np.int32)
    lengths = np.zeros(array.shape[0] , dtype=np.int32) #length of each row
    
    for i , row in enumerate (list_of_words):
        lengths[i]=len(row)
        ret[i,:lengths[i]]=np.array(row)
        
    return ret,lengths


# In[32]:


def get_batch(file , batch_size , num_steps_before , num_steps_after , epochs , vocab_size):
    """
    description: generating training batches for sentence filling
    
    inputs: file: training file
            num_steps_before: number of sentences before the blank
            num_steps_after: number of sentences after the blank
            vocab_size: size of GloVe
            
    outputs: Xs: inputs [batch_size X num_steps]
             Ys: targets [batch_size X 1]
             ratio: a [batch_size X 1] array with each cell showing how much we want our target have impact on gradient decent.
                    ',' , '.' have value of 0.6. 'the' , 'and' , 'of' , 'to' have 0.8.
    """
    arr , batch_len = make_array(len(file) , batch_size)
    
    print (arr)
    #for each batch, we use arr[:,window_size].
    
    for epoch in range (epochs):
        for time_step in range (batch_len-num_steps_before-num_steps_after):
            before_indices = np.copy(arr[:,time_step:time_step+num_steps_before])
            blank_indices = (np.copy(arr[:,time_step+num_steps_before])).reshape(-1,1)
            after_indices = np.copy(arr[:,time_step+num_steps_before+1 : time_step+num_steps_before+num_steps_after+1])
            
            before_words , steps_before = make_tensor_of_words(before_indices , file)
            blank_sent , blank_steps = make_tensor_of_words(blank_indices , file)
            after_words , steps_after = make_tensor_of_words(after_indices , file)
         
            yield before_words , after_words , blank_sent , steps_before , steps_after , blank_steps


# In[33]:


if __name__=='__main__':
    with open ('vectors.pkl' , 'rb') as pkl:
        embd = pickle.load(pkl)
        
  
    with open ('sent_train.pkl' , 'rb') as pkl:
        file = pickle.load(pkl)
        
    generated_batch=get_batch(batch_size=100 , epochs=1 , file=file , num_steps_after=2 , 
                                  num_steps_before=2 , vocab_size=embd[0])
    
    
    for before_batch , after_batch , target_batch , before_length , after_length , target_length in generated_batch:
        print (target_batch)

