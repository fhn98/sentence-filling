
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
from tensorflow.python.layers.core import Dense


# In[3]:


class sentence_filler(object):
    def __init__(self  , batch_size  , hidden_len  , num_layers , vocab_size , keep_prob , embd_len , train_mode , blank_index , start_token=0 , end_token=0):
        """
        description: initializing class fields and then calling construct_model for constructing the model_graph

        inputs:

        outputs:
        """
        self.batch_size = batch_size
        self.hidden_len = hidden_len
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.keep_prob = keep_prob
        self.embd_len = embd_len
        self.start_token = start_token
        self.end_token = end_token
        self.train_mode = train_mode
        self.blanks= blank_index*tf.ones([batch_size,1] , dtype=tf.int32)
        
        self.construct_model()
        
    def construct_model(self):
        """
        description: first, we use encoder on context and then use attention on hidden of each context word.
                     attention mechanism is luong attention. training_helper is used for decoder.
                     loss is cross entropy loss.
        inputs:
        outputs:
        """
        
        #context words and words of the target sentence.
        self.before = tf.placeholder(tf.int32 , [self.batch_size ,None] , name='b')
        self.strictly_after = tf.placeholder(tf.int32 , [self.batch_size,None] , name='a')
        self.target = tf.placeholder(tf.int32 , [self.batch_size , None] , name='t')
        
        #number of words in each row of context tensor and target tensor
        self.before_length = tf.placeholder(tf.int32 , [self.batch_size,] , name = 'bl')
        self.strictly_after_length = tf.placeholder(tf.int32 , [self.batch_size,] , name='al')
        self.target_length = tf.placeholder(tf.int32 , [self.batch_size,] , name = 'tl')
        
        #adding separator to the beginning of words after the blank (used for separating words before and after the blank in the rnn)
        self.after = tf.concat ([self.blanks,self.strictly_after] , axis=1)
        self.after_length = self.strictly_after_length+1
        self.context_length=tf.add(self.before_length , self.strictly_after_length)
        
        #here we use a placeholder to load the embedding vectors in it with feed_dict
        #then we assign values of the placeholder into a variable and use that variable for embedding_lookup
        #separator is a learnable vector used for representing blanks. we attach it to the end of our embedding tensor
        #separator is used for separating sentences before and after the blank
        embedding = tf.Variable (tf.constant (0.0 ,shape=[self.vocab_size , self.embd_len]) , trainable=False , name='embedding')
        self.separator = tf.Variable(tf.random_uniform([1,self.embd_len] , -1.0 , 1.0))
        self.weights = tf.placeholder(tf.float32 , [self.vocab_size , self.embd_len])
        self.embedding_init = embedding.assign(self.weights)
        new_embedding = tf.concat([embedding , self.separator] , axis = 0)
        
   
        before_embedded_weights = tf.nn.embedding_lookup(new_embedding , self.before)
        after_embedded_weights = tf.nn.embedding_lookup(new_embedding , self.after)
        target_embedded_weight = tf.nn.embedding_lookup(new_embedding , self.target)
    
        #basic lstm cell with dropout wrapper. used for encoder
        def basic_encoder_cell():
            cell = tf.contrib.rnn.LSTMCell(self.hidden_len , forget_bias=1.0)
            if self.train_mode==True:
                cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
            return cell
        
        with tf.variable_scope ('forward_cell',reuse=True):
            #multilayer forward cell used for encoder
            encoder_forward_cell = tf.contrib.rnn.MultiRNNCell([basic_encoder_cell() for _ in range (self.num_layers)])

        #passing context words through the encoder
        before_output ,before_state = tf.nn.dynamic_rnn(cell=encoder_forward_cell , inputs= before_embedded_weights , sequence_length=self.before_length , dtype=tf.float32 ,time_major=False )
        after_output ,_ =  tf.nn.dynamic_rnn(cell=encoder_forward_cell , inputs= after_embedded_weights , sequence_length=self.after_length , initial_state=before_state , time_major=False)

        context_output = tf.concat([before_output , after_output[:,1:]],axis=1)
        #attention on outputs of encoder
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(memory = context_output , num_units=self.hidden_len , memory_sequence_length=self.context_length)
        
        #basic lstm cell used for decoder
        def basic_decoder_cell():
            cell = tf.contrib.rnn.LSTMCell(self.hidden_len , forget_bias=1.0)
            return cell
        
        
        
        #making multilayer decoder cell. last layer is attention cell
        decoder_layers = [basic_decoder_cell() for _ in range (self.num_layers)]
        
        decoder_layers[-1]= tf.contrib.seq2seq.AttentionWrapper(cell=decoder_layers[-1] , attention_mechanism=attention_mechanism , initial_cell_state=before_state[-1], attention_layer_size=self.hidden_len)
        
        initial_state = [state for state in before_state]
        initial_state[-1] = decoder_layers[-1].zero_state(
        batch_size=self.batch_size, dtype=tf.float32)
        decoder_initial_state = tuple(initial_state)
        
        decoder_cell = tf.contrib.rnn.MultiRNNCell(decoder_layers)
        
        #training helper
        helper = tf.contrib.seq2seq.TrainingHelper(inputs=target_embedded_weight ,sequence_length=self.target_length)
        
        #projection layer for output of decoder
        output_logits = Dense (self.vocab_size , use_bias=False) 
        
        #making the decoder
        decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell , helper=helper , initial_state=decoder_initial_state, output_layer=output_logits)
    
        #getting outputs of decoder after unrolling it
        decoder_output,_,_ = tf.contrib.seq2seq.dynamic_decode(decoder,...)
        
        logits = decoder_output.rnn_output

        
        #a mask used for masking rest of output array
        mask = tf.to_float(tf.sequence_mask(self.target_length))
        
        self.batch_loss = tf.contrib.seq2seq.sequence_loss(logits=logits, 
                                          targets=self.target,
                                          weights=mask,
                                          average_across_timesteps=True,
                                          average_across_batch=True,)
        


# In[1]:


if __name__=='_main__':
    sentence_filler(batch_size=64 , blank_index=4000001 , embd_len=4000000 , hidden_len=200 
                                , keep_prob=0.8 ,num_layers=1 
                                , train_mode=True , vocab_size=50)

