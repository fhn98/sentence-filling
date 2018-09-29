
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import pickle
import import_ipynb

from sent_filling_batch_generator import get_batch
from sentence_filler_attempt1 import sentence_filler


# In[2]:


def train (file , batch_size , steps_before , steps_after , epochs , embedding_array , 
           learning_rate , lr_decay , hidden_len , num_layers , savepath , blank_index):
    
    """
    description: training out sentence filling model.
    
    inputs: file: training file
            steps_before: steps before the blank sentence which are going through the encoder
            steps_after: steps after the blank sentence which are going through the encoder
            embedding_array: an array with each row as the vector representation of its corresponding index
            itos: a list with index of each word as its index in GloVe
            stoi: a dictionary from each word to its index in GloVe
            lr_decay: after each 10000 steps, learning_rate is reassigned to learning_rate*learning_decay
            save_path: the path in which trained model is going to be saved
            blank_index: index of an additional word in GloVe which represents separator (separator is used for separating sentences before
                         blank and after blank when they are passed through the encoder).
            
    outputs: savepath: address of the saved trained model
    
    """
    
    
    #making an instance of our word_filling model
    my_model = sentence_filler (batch_size=batch_size , blank_index=blank_index , embd_len=embedding_array.shape[1] , hidden_len=hidden_len 
                                , keep_prob=0.8 ,num_layers=num_layers 
                                , train_mode=True , vocab_size=embedding_array.shape[0] )
    saver = tf.train.Saver()
    
    with tf.Session() as sess :
        #used for updating learning_rate
        tmp=learning_rate
        lr = tf.Variable(0.0, trainable=False)
        lr_new_value = tf.placeholder(tf.float32 ,[])
        lr_update=tf.assign(lr,lr_new_value)
        
        #clipping gradients
        global_step = tf.Variable(0 , trainable=False)
        params = tf.trainable_variables()
        clipped = [tf.clip_by_norm(g,5) for g in tf.gradients(my_model.batch_loss , params)]
        
        #declaring the optimizer
        optimizer = tf.train.AdamOptimizer(lr)
        train_opt = optimizer.apply_gradients(zip(clipped , params) , global_step=global_step)

#         train_opt = tf.train.AdamOptimizer(learning_rate).minimize(my_model.batch_loss)
        
        #initializing global variables
        sess.run(tf.global_variables_initializer())
        #initializing embedding tensor in our model with loaded pretrained vectors
        sess.run(my_model.embedding_init , feed_dict={my_model.weights: embd})
        
        #in each 500 steps, we run our model on a test corpus and measure its perplexity
        #it works just like running our model on train data, except that total loss is mean of all batch_losses
        
       
        
        #using batch generator defined in word_filling_batch_generator1
        generated_batch=get_batch(batch_size=batch_size , epochs=epochs , file=file , num_steps_after=steps_after , 
                                  num_steps_before=steps_before , vocab_size=embedding_array.shape[0])
        
        step=0
      
        for before_batch , after_batch , target_batch , before_length , after_length , target_length in generated_batch:
            _,batch_loss = sess.run([train_opt , my_model.batch_loss] , 
                                    feed_dict={my_model.before: before_batch , my_model.strictly_after:after_batch ,
                                              my_model.target: target_batch , my_model.before_length: before_length,
                                              my_model.strictly_after_length: after_length , my_model.target_length: target_length})
            
            if step%10==0:
                print (str(step)+ " train loss: " +':'+str(batch_loss))

            if step%500==0:
                tmp=tmp*lr_decay
                sess.run(lr_update , feed_dict={lr_new_value:tmp})
                
            step+=1
        
        saver.save(sess, './'+savepath+'.ckpt')
        print("Model saved in file: %s" % savepath) 
        
        return embd.shape[0] ,'./'+savepath+'.ckpt'


# In[ ]:


if __name__=="__main__":
    with open ('vectors.pkl' , 'rb') as pkl:
        embd = pickle.load(pkl)
        
  
    with open ('sent_train.pkl' , 'rb') as pkl:
        file = pickle.load(pkl)
        
    train(file=file , batch_size=10 , blank_index=407371 , embedding_array=embd , epochs=1 , hidden_len=20 , learning_rate=0.02 , lr_decay=0.5 , num_layers=1 , savepath='sent_model1' , steps_after=1 , steps_before=1 , )

