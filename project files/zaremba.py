# Copyright 2015-2020 David Demeter. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import time

import socket
import numpy as np
import tensorflow as tf
import collections
import os
import math
import io
import random
import sys
import shutil

class MyParameters(object):

    mode = 2
    time_name = 'example'
    
    if mode == 2:
        dim = 200
        threshold = 3
        knn = -1
        regularizer = 0
        keep_prob = 0.50
        init_scale = 0.05
        max_epochs = 20
        alpha_decay = 0.80
        alpha_start = 6
        interval = 100000
        bCorpus = 2             # 1 = PTB, 2 = Wiki2, 3 = Wiki103
        num_layers = 2
        max_tokens = 500000000
        batch_size = 20
        num_steps = 30
        max_grad = 2.0       

    alpha_initial = 1.0
    alpha_mode = 0
    alpha_min = 0.0
    bValid = True;
    bTest = True;
    save_net = True
    recall_net = False
    bWET = True
    bBias = True
    bSaveEmbed = False
    
    vocab_size = 0
    epoch = 0
    gpu_mem = 0.90
    precision = tf.float32    
    
    
def OutText2(text,filename=None,screen=True):
    if screen:
        print(text)
    if filename:
        outFile = open(filename,"a+")
        outFile.write(text+"\n")
        
def ReadCorpus(file_name,words,vocab,params,src):
    
    if src == 0:
        temp = dict()
        last = 0
        total_tokens = 0
        with open(file_name,"r") as f:
            for line in f:
#                line = line.replace(" "+chr(8211)+" "," - ")
                tokens = line.replace("\n", " </s> ").split()
                total_tokens = total_tokens + len(tokens)

                if (total_tokens - last) > 10000000:
                    print(total_tokens)
                    last = total_tokens

                for t in tokens:
                    if t == '"':
                        t = '<quote>'
                    try:
                        elem = temp[t]
                    except:
                        elem = [0,0]
                    elem[1] = elem[1] + 1
                    temp[t] = elem
                    
        wNextID = 0
        words = dict()
        words['<unk>'] = [wNextID,0]
        wNextID = wNextID + 1
                
        for t in temp:
            elem = temp[t]
            if elem[1] >= params.threshold:
                words[t] = [wNextID,elem[1]]
                wNextID = wNextID + 1
                
        vocab = list()
        vocab.append(' ')
        for w in words:
            vocab.append(' ')
        for w in words:
            elem = words[w]
            vocab[elem[0]] = w

    corpus = list()
    garbage = dict()
        
    last = 0
    total_tokens = 0
    with open(file_name,"r") as f:
        for line in f:
#            line = line.replace(" "+chr(8211)+" "," - ")
            tokens = line.replace("\n", " </s> ").split()
            total_tokens = total_tokens + len(tokens)

            if (total_tokens - last) > 10000000:
                print(total_tokens)
                last = total_tokens

            for t in tokens:
                if t == '"':
                    t = '<quote>'
                try:
                    elem = words[t]
                except:
                    try:
                        g = garbage[t]
                    except:
                        g = 0
                    g = g + 1
                    garbage[t] = g
                    elem = words['<unk>']
#                elem[1] = elem[1] + 1
#                words[t] = elem
                corpus.append(elem[0])
       
    return corpus, words, vocab, garbage

def Load_LSTM(count,offs,batch_size,num_steps,corpus,vocab,c,t,i):
    last = len(corpus)//batch_size
    for s in range(num_steps):
        for b in range(batch_size):
            i[b,s] = -1
    for s in range(num_steps):
        for b in range(batch_size):
            c[b,s] = corpus[offs[b]+count]
            t[b,s] = corpus[offs[b]+count+1]
            i[b,s] = offs[b]+count+1
        count = count + 1
        if count >= (last - 1):
            return c,t,i,-1
    return c,t,i,count

def Alpha_Algo(alpha,params):
    if params.alpha_mode == 0:
        if params.epoch >= params.alpha_start:
            alpha = alpha * params.alpha_decay
        if alpha < params.alpha_min:
            alpha = params.alpha_min
            
    return alpha

class Model(object):
    
    def __init__(self,is_training,params):

        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(params.dim, forget_bias=0.0, state_is_tuple=False)
        if is_training and params.keep_prob < 1:
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=params.keep_prob)
        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * params.num_layers, state_is_tuple=False)
        self._in_state = cell.zero_state(params.batch_size, tf.float32)
        
        self._c = tf.placeholder(tf.int32,shape=[params.batch_size,params.num_steps],name="context")
        self._t = tf.placeholder(tf.int32,shape=[params.batch_size,params.num_steps],name = "targets")
        self._lr = tf.placeholder(params.precision,shape=[1],name="alpha")

        self._embed = tf.get_variable("embed", [params.vocab_size, params.dim], dtype=params.precision)
        self._biases = tf.get_variable("biases", [params.vocab_size, 1], dtype=params.precision)
        
        inputs = tf.nn.embedding_lookup(self._embed,self._c[:,:])
        if is_training and params.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, params.keep_prob)
        state = self._in_state
        outputs = []
        with tf.variable_scope("RNN"):
            for s in range(params.num_steps): 
                if s > 0: tf.get_variable_scope().reuse_variables()
                (cell_output,state) = cell(inputs[:,s,:],state)
                outputs.append(cell_output)
        self._fin_state = state
        
        output = tf.reshape(tf.concat(outputs,1), [-1, params.dim])        
        print('output = ',output)
        
        self._logits = tf.matmul(output, tf.transpose(self._embed)) + tf.transpose(self._biases)
        print('logits = ',self._logits)
        self._logits = tf.exp(self._logits)          
        denom = tf.reduce_sum(self._logits,1)
        numer = tf.diag_part(tf.nn.embedding_lookup(tf.transpose(self._logits),tf.reshape(self._t,[-1])))
        self._ppl = -tf.log(tf.div(numer,denom))
        print('ppl = ',self._ppl)
        loss = tf.reduce_sum(self._ppl,0)
        cost = tf.reduce_sum(loss) / params.batch_size 
        print('cost = ',cost)

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),params.max_grad)
        optimizer = tf.train.GradientDescentOptimizer(self._lr[0])
        self._train_op = optimizer.apply_gradients(zip(grads, tvars))
        
        self._saver = tf.train.Saver();
            
        tf.initialize_all_variables().run()
        self._st_zero = self._in_state.eval()

    @property
    def hidden(self):
        return self._hidden
    @property
    def logits(self):
        return self._logits
    @property
    def prob(self):
        return self._prob
    @property
    def ppl(self):
        return self._ppl
    @property
    def train_op(self):
        return self._train_op
    @property
    def st_zero(self):
        return self._st_zero
    @property
    def in_state(self):
        return self._in_state
    @property
    def fin_state(self):
        return self._fin_state
    @property
    def lr(self):
        return self._lr
    @property
    def c(self):
        return self._c
    @property
    def t(self):
        return self._t
    @property
    def saver(self):
        return self._saver
    @property
    def embed(self):
        return self._embed
    @property
    def biases(self):
        return self._biases
    @property
    def data(self):
        return self._data
    @property
    def saver(self):
        return self._saver
    
def main(_):

    # establish log directory and create back-up of source file(s)
    print(socket.gethostname())
    server = socket.gethostname()
    if server == 'steve':
        data_dir = '../..'
    else:
        data_dir = '//scratch//ddemeter'
        
    params = MyParameters()    
    time_name = time.strftime("%Y%m%d_%H%M")
    if len(params.time_name) > 1:
        time_name = params.time_name
    dir_name = data_dir + "//dwd//msai337//" + time_name
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    source_name = sys.argv[0]
    dir_name = dir_name + "//"
    shutil.copy(source_name,dir_name + source_name)
    log_file = dir_name + "logfile.txt"
    ckpt_dir = dir_name + "ckpt"
    
    text = "\n\nTIMESTAMP: %s\n\n" %(time_name)
    OutText2(text,log_file)
    # read corpus files

    if params.bCorpus == 1:
        train,words,vocab,train_g=ReadCorpus(data_dir + "//dwd//ptb.train.txt",None,None,params,0)
        valid,words,vocab,valid_g=ReadCorpus(data_dir + "//dwd//ptb.valid.txt",words,vocab,params,1)
        test,words,vocab,test_g=ReadCorpus(data_dir + "//dwd//ptb.test.txt",words,vocab,params,2) 
        params.vocab_size = len(vocab)       
    if params.bCorpus == 2:
        train,words,vocab,train_g=ReadCorpus(data_dir + "//dwd//wiki.train.txt",None,None,params,0)
        valid,words,vocab,valid_g=ReadCorpus(data_dir + "//dwd//wiki.valid.txt",words,vocab,params,1)
        test,words,vocab,test_g=ReadCorpus(data_dir + "//dwd//wiki.test.txt",words,vocab,params,2) 
        params.vocab_size = len(vocab)       
    if params.bCorpus == 3:
        train,words,vocab,train_g=ReadCorpus(data_dir + "//dwd//wiki.train.tokens",None,None,params,0)
        valid,words,vocab,valid_g=ReadCorpus(data_dir + "//dwd//wiki.valid.tokens",words,vocab,params,1)
        test,words,vocab,test_g=ReadCorpus(data_dir + "//dwd//wiki.test.tokens",words,vocab,params,2) 
        params.vocab_size = len(vocab)       
        
    text = 'vocab_size = %d  train_size = %d' % (params.vocab_size,len(train))
    OutText2(text,log_file)
            
    c_loc = np.zeros((params.batch_size,params.num_steps),dtype=np.int)
    t_loc = np.zeros((params.batch_size,params.num_steps),dtype=np.int)
    i_loc = np.zeros((params.batch_size,params.num_steps),dtype=np.int)
    alpha = params.alpha_initial
    
    stop_words = ['he','she','the','a','of','and','his','her','them','it','.']
               
    gpu_opt = tf.GPUOptions(per_process_gpu_memory_fraction=params.gpu_mem)
    with tf.Graph().as_default(), tf.Session(config=tf.ConfigProto(gpu_options=gpu_opt)) as sess:

        tf.logging.set_verbosity(tf.logging.ERROR)

        initializer = tf.random_uniform_initializer(-params.init_scale,params.init_scale)
        with tf.variable_scope("Model", reuse = None, initializer = initializer):
            m = Model(is_training = True, params = params)
        with tf.variable_scope("Model", reuse = True, initializer = initializer):
            v = Model(is_training = False, params = params)
            
        if params.recall_net:
            params.max_epochs = -1
        for epoch in range(0,params.max_epochs):
            ppl = []
            obs = 0
            count = 0
            last_obs = obs
            st = m.st_zero
            start_time = time.time()
            offs = []
            for i in range(params.batch_size):
                offs.append((len(train)//params.batch_size)*i)
            while count != -1:            # and obs < 100005:
                c_loc,t_loc,i_loc,count = Load_LSTM(count,offs,params.batch_size,params.num_steps,train,vocab,c_loc,t_loc,i_loc)
                st,pw,to = sess.run([m.fin_state,m.ppl,m.train_op],{m.c: c_loc,m.t: t_loc,m.lr: [alpha],m.in_state: st})
                for p in pw:
                    ppl.append(p)
                obs = obs + params.num_steps * params.batch_size
                if obs - last_obs >= params.interval:
                    last_obs = obs
                    text = " a: %.5f p[%7d]: %7.1f spd: %.1f wps e: %d %s" % (alpha,obs,
                                math.exp(sum(ppl)/float(len(ppl))),
                                obs / (time.time() - start_time),epoch,time_name)
                    OutText2(text,log_file)
                    
            params.epoch = epoch
            alpha = Alpha_Algo(alpha,params)
            
            if params.save_net:
                ckpt_file = dir_name + "model.ckpt"
                print(ckpt_file)
                save_path = m.saver.save(sess,ckpt_file)
                text = "Model saved in file: %s" % save_path
                OutText2(text,log_file)

            if params.bValid:
                st = v.st_zero
                count = 0
                offs = []
                ppl = []

                for i in range(params.batch_size):
                    offs.append((len(valid)//params.batch_size)*i)
                while count != -1:
                    c_loc,t_loc,i_loc,count = Load_LSTM(count,offs,params.batch_size,params.num_steps,valid,vocab,c_loc,t_loc,i_loc)
                    st,pw = sess.run([v.fin_state,v.ppl],{v.c: c_loc,v.t: t_loc,v.in_state: st})
                    for p in pw:
                        ppl.append(p)

                text = "  validation perplexity[%d]: %7.1f " % (epoch,math.exp(sum(ppl)/float(len(ppl))))
                OutText2(text,log_file)

            if params.bTest:
                st = v.st_zero
                count = 0
                offs = []
                ppl = []
                ppl_stop = []
                ppl_freq = []

                for i in range(params.batch_size):
                    offs.append((len(test)//params.batch_size)*i)
                while count != -1:
                    c_loc,t_loc,i_loc,count = Load_LSTM(count,offs,params.batch_size,params.num_steps,test,vocab,c_loc,t_loc,i_loc)
                    st,pw = sess.run([v.fin_state,v.ppl],{v.c: c_loc,v.t: t_loc,v.in_state: st})
                    for p in pw:
                        ppl.append(p)
                    for b in range(params.batch_size):
                        for s in range(params.num_steps):
                            index = b*params.num_steps+s
                            w = vocab[test[i_loc[b,s]]]
                            if w in stop_words:
                                ppl_stop.append(pw[index])
                            if words[w][1] > 10000:
                                ppl_freq.append(pw[index])

                text = "  test perplexity[%d]: %7.1f %7.1f %7.1f" % (epoch,math.exp(sum(ppl)/float(len(ppl))),
                                                                     math.exp(sum(ppl_stop)/float(len(ppl_stop))),
                                                                     math.exp(sum(ppl_freq)/float(len(ppl_freq))))
                OutText2(text,log_file)
                
            if params.bSaveEmbed:
                file_name = dir_name + 'embed.csv'
                fout = open(file_name,'wt')
                em,bs = sess.run([m.embed,m.biases])
                for v1 in words:
                    elem = words[v1]
                    wID = elem[0]
                    count = elem[1]
                    fout.write('%s\t' % v1)
                    if params.bBias:
                        fout.write('%d\t%f\t' % (count,bs[wID]))
                    else:
                        fout.write('%d\t%f\t' % (count,0))
                    for k in range (params.dim):
                        fout.write('%f\t' % em[wID][k])
                    fout.write('\n')
                fout.close()
                print('finished writing embeddings....')  
                
        if params.recall_net:
            recall_name = dir_name + "model.ckpt"
            recall_name = recall_name.replace('//','/')
            print('RECALL NAME: %s ' % recall_name)
            m.saver.restore(sess,recall_name)
        print("RECALLED>>>>>>>")
        
        if params.bTest:
            records = []
            st = v.st_zero
            count = 0
            offs = []
            ppl = []
            
            params.batch_size = 20
            params.num_steps = 5

            for i in range(params.batch_size):
                offs.append((len(test)//params.batch_size)*i)
            while count != -1:
                c_loc,t_loc,i_loc,count = Load_LSTM(count,offs,params.batch_size,params.num_steps,test,vocab,c_loc,t_loc,i_loc)
                st,pw = sess.run([v.fin_state,v.ppl],{v.c: c_loc,v.t: t_loc,v.in_state: st})
                for p in pw:
                    ppl.append(p)
                
                for b in range(params.batch_size):
                    for s in range(params.num_steps):
                        if i_loc[b,s] > -1:
                            index = b*params.num_steps+s
                            ref = i_loc[b,s]
                            w = vocab[test[i_loc[b,s]]]
                            lprob = pw[index]
                            lprob_stop = float(0.0)
                            if w in stop_words:
                                lprob_stop = pw[index]
                            lprob_freq = float(0.0)
                            if words[w][1] > 10000:
                                lprob_freq = pw[index]
                            records.append([ref,w,words[w][1],lprob,lprob_stop,lprob_freq])
                        
            print(' ')
            print('BEFORE SORTING:')
            print(' ')
            for i in range(100):
                print(records[i])
            
            records = sorted(records,key=lambda x: x[0])
            
            print(' ')
            print('AFTER SORTING:')
            print(' ')
            for i in range(100):
                print(records[i])
                
               
    print("\n",time_name,"\n")
    
            
if __name__ == "__main__":
    tf.app.run()

 