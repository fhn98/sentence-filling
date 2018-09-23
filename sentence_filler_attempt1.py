
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
from tensorflow.python.layers.core import Dense


# In[2]:


class sentence_filler(object):
    def __init__(self , batch_size  , hidden_len  , num_encoder_layers , num_decoder_layers , vocab_size , keep_prob , embd_len , start_token , end_token , train_mode):
    """
    description: initializing class fields and then calling construct_model for constructing the model_graph
    
    inputs:
    
    outputs:
    """
        self.batch_size = batch_size
        self.hidden_len = hidden_len
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.vocab_size = vocab_size
        self.keep_prob = keep_prob
        self.embd_len = embd_len
        self.start_token = start_token
        self.end_token = end_token
        self.train_mode = train_mode
        
        self.construct_model()
        
    def construct_model(self):
        """
        description: first, we use encoder on context and then use attention on hidden of each context word.
                     attention mechanism is luong attention. training_helper is used for decoder.
                     loss is squared difference of means of target and output
        inputs:
        outputs:
        """
        
        #context words and words of the target sentence.
        self.context = tf.placeholder(tf.int32 , [self.batch_size ,None])
        self.target = tf.placeholder(tf.int32 , [self.batch_size , None])
        
        #number of words in each row of context tensor and target tensor
        self.context_length = tf.placeholder(tf.int32 , [self.batch_size])
        self.target_length = tf.placeholder(tf.int32 , [self.batch_size])
        
        #here we use a placeholder to load the embedding vectors in it with feed_dict
        #then we assign values of the placeholder into a variable and use that variable for embedding_lookup
        #blank_vector is a learnable vector used for representing blanks. we attach it to the end of our embedding tensor 
        embedding = tf.Variable (tf.constant (0.0 ,shape=[self.vocab_size , self.embd_len]) , trainable=False , name='embedding')
        self.seperator = tf.Variable(tf.random_uniform(self.embd_len)  ,trainable=False)
        self.weights = tf.placeholder(tf.float32 , [self.vocab_size , self.embd_len])
        self.embedding_init = embedding.assign(self.weights)
        new_embedding = tf.concat([embedding , self.blank_vector] , axis = 0)
        
        
   
        context_embedded_weights = tf.nn.embedding_lookup(new_embedding , self.inputs)
    
    
        #basic lstm cell with dropout wrapper. used for encoder
        def basic_encoder_cell():
            cell = tf.contrib.rnn.LSTMCell(self.hidden_len , forget_bias=1.0)
            if self.train_mode==True:
                cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
            return cell
        
        #multilayer forward cell used for encoder
        forward_cell = tf.contrib.rnn.MultiRNNCell([basic_cell() for _ in range (self.num_encoder_layers)])
        
        #passing context words through the encoder
        context_output ,_ = tf.nn.dynamic_rnn(cell=forward_cell , inputs= context_embedded_weights , sequence_length=self.context_length)
        
        #attention on outputs of encoder
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(memory = context_outputs , num_units=self.hidden_len , memory_sequence_length=self.context_length)
        
        #basic lstm cell used for decoder
        def basic_decoder_cell():
            cell = tf.contrib.rnn.LSTMCell(self.hidden_len , forget_bias=1.0)
            return cell
        
        #basic lstm cell with attention wrapper. used for the last layer of decoder
        def attention_decoder_cell():
            cell = tf.contrib.rnn.LSTMCell(self.hidden_len , forget_bias=1.0)
            cell = tf.contrib.seq2seq.AttentionWrapper(cell=decoder_cell , attention_mechanism=attention_mechanism , attention_layer_size=hidden_len)
            return cell
        
        #making multilayer decoder cell. last layer is attention cell
        decoder_cell = tf.contrib.rnn.MultiRNNCell([basic_decoder_cell() for _ in range (num_decoder_layers-1)].append(attention_decoder_cell()))
        
        #training helper
        helper = tf.contrib.seq2seq.TrainingHelper(inputs=self.target ,sequence_length=self.target_length)
        
        #projection layer for output of decoder
        output_logits = Dense (self.vocab_size , use_bias=False) 
        
        #making the decoder
        decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell , helper=helper , initial_state=prev_state , output_layer=output_logits)
        
        #getting outputs of decoder after unrolling it
        decoder_output,_,_ = tf.contrib.seq2seq.dynamic_decode(decoder,...)
        
        #getting output of the decoder as word embeddings
        self.output_words = tf.argmax(input=decoder_output , axis=2)
        output_embeddings = tf.nn.embedding_lookup(embedding , self.output_words)
        
        #a mask used for masking rest of output array
        mask = tf.sequence_mask(self.target_length)
        
        #mean of words of each decoder output
        output_embedding_mean = tf.reduce_mean(tf.multiply(output_embeddings,mask) , axis=1)
        
        #mean of words of each target
        target_embedding_mean = tf.reduce_mean(tf.multiply(tf.nn.embedding_lookup(embedding , self.target),mask),axis=1)
        
        
        loss = tf.square(target_embedding_mean-output_embedding_mean)
        
        self.batch_loss = tf.reduce_mean(loss , axis = 0)
        
        

