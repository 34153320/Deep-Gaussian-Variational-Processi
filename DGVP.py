# -*- coding: utf-8 -*-
"""
A module containing functionality for the DGVAE model (Tensorflow-enabled, v7).
"""

import os
import sys
import time
import gzip
import math
import collections

import numpy as np
import six.moves.cPickle as pickle
import tensorflow as tf

import Config
# improved algorithm focus on the inference layer and the recogntion layer
# two part of Gaussian will be generated:
# index 1 refers to residual, index 2 refers to the instantaneous signal. 

low_c = 1e-8
upp_c = 10000.0
# Base neural network
#def init_mlp(layer_sizes, std=.01, bias_init=0.):
#    params = {'w':[], 'b':[]}
#    for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
#        params['w'].append(tf.Variable(tf.random_normal([n_in, n_out], stddev=std)))
#        params['b'].append(tf.Variable(tf.mul(bias_init, tf.ones([n_out,]))))
#    return params

#def mlp(X, params):
#    h = [X]
#    for w,b in zip(params['w'][:-1], params['b'][:-1]):
#        h.append( tf.nn.relu( tf.matmul(h[-1], w) + b ) )
        #h.append( tf.nn.tanh( tf.matmul(h[-1], w) + b ) ) 
#    return tf.matmul(h[-1], params['w'][-1]) + params['b'][-1]
    

def _beta_fn(a, b):
        return tf.exp(tf.lgamma(a) + tf.lgamma(b)-tf.lgamma(a+b))
    
def _compute_kumar2beta_kld(a, b, alpha, beta):
        ab = tf.multiply(a, b)
        a_inv = tf.pow(a, -1)
        b_inv = tf.pow(b, -1)
        
        kl  = tf.multiply(tf.pow(1+ab, -1), _beta_fn(a_inv, b))
        for idx in range(10):
            kl += tf.multiply(tf.pow(idx+2.+ab, -1), _beta_fn(tf.multiply(idx+2., a_inv), b))
        kl = tf.multiply(tf.multiply(beta-1, b), kl)
        
        kl += tf.multiply(tf.div(a-alpha,tf.clip_by_value(a, low_c, upp_c)), -0.57721-tf.digamma(b)-b_inv)
   #     kl += tf.log(tf.clip_by_value(ab, 1e-10)) + tf.log(tf.clip_by_value(_beta_fn(alpha, beta), 1e-10))
        kl += tf.log(tf.clip_by_value(ab, low_c, upp_c)) + tf.log(tf.clip_by_value(_beta_fn(alpha, beta), low_c, upp_c))        
    
        kl += tf.div(-(b-1), tf.clip_by_value(b, low_c, upp_c))
        kl  = tf.reduce_sum(kl, -1)        
        return tf.reduce_sum(kl)    

def log_normal_pdf(x, mu, cov):
        d  = mu - x
        d2 = tf.multiply(-1., tf.multiply(d, d))
        s2 = 2.*cov
        return tf.reduce_sum(tf.div(d2,tf.clip_by_value(s2, low_c, upp_c)) - tf.log(2.506628*tf.sqrt(cov)), axis=-1, keepdims=True)


# DEEP Gaussian Variational Autoencoder Model for feature extraction
class DGVAE(object):

    @RTConfig.loadDefaultsFromFunctionConfiguration
    def __init__(self, sess, params, paramFile=None):
        """
        View the ef2conf.json file for the default values for params
        """
        self.sess = sess
        self.params = params
        self.paramFile = paramFile
        self.tWeights = collections.OrderedDict()
        self.window_ = self.params['window_size']
       # tf.random_normal = tf.contrib.layers.xavier_tf.random_normal(uniform=False)
       # tf.random_normal = tf.random_normal()
        stddev = 0.010001


      # create the generative networks:
      # transition function: # includes two parts : W1*z1+W2*z2+b_t
        DIM_HIDDEN = self.params['dim_hidden']
        DIM_STOCHASTIC = self.params['dim_stochastic']
        for l in range(self.params['transition_layers']):
              dim_input, dim_output = DIM_HIDDEN, DIM_HIDDEN
              if l==0:
                    dim_input = self.params['dim_stochastic']
              self.tWeights['p_trans_W_z_' + str(l)] = tf.Variable(tf.random_normal([dim_input, dim_output], stddev=stddev), 'p_trans_W_z_'+ str(l))
              self.tWeights['p_trans_b_z_' + str(l)] = tf.Variable(tf.random_normal([dim_output,], stddev=stddev), 'p_trans_b_z_'+ str(l))
              self.tWeights['p_trans_W_u_' + str(l)] = tf.Variable(tf.random_normal([dim_input, dim_output], stddev=stddev), 'p_trans_W_u_'+ str(l))
              self.tWeights['p_trans_b_u_' + str(l)] = tf.Variable(tf.random_normal([dim_output,], stddev=stddev), 'p_trans_b_u_'+str(l))
      #                     self.tWeights['p_trans_W_2_' + str(l)] = tf.Variable(tf.random_normal([dim_input, dim_output], stddev=stddev), 'p_trans_W_2_'+ str(l))
      #                     self.tWeights['p_trans_b_2_' + str(l)] = tf.Variable(tf.random_normal([dim_output,], stddev=stddev), 'p_trans_b_2_'+ str(l))
              if self.params['transition_type'] == 'gated':
                    self.tWeights['p_trans_gate_W_z_' + str(l)]  = tf.Variable(tf.random_normal([dim_input, dim_output], stddev=stddev), 'p_trans_gate_W_z_' + str(l))
                    self.tWeights['p_trans_gate_b_z_' + str(l)]  = tf.Variable(tf.random_normal([dim_output,], stddev=stddev), 'p_trans_gate_b_z_' + str(l))
                    self.tWeights['p_trans_gate_W_u_' + str(l)]  = tf.Variable(tf.random_normal([dim_input, dim_output], stddev=stddev), 'p_trans_gate_W_u_' + str(l))
                    self.tWeights['p_trans_gate_b_u_' + str(l)]  = tf.Variable(tf.random_normal([dim_output,], stddev=stddev), 'p_trans_gate_b_u_' + str(l))
        mu_cov_inp = self.params['dim_hidden']
        if self.params['transition_layers'] == 0:
             mu_cov_inp = self.params['dim_stochastic']
        if self.params['transition_type'] == 'gated':
             self.tWeights['p_trans_z_W']   = tf.Variable( tf.random_normal([self.params['dim_stochastic'],self.params['dim_stochastic']], stddev=stddev), 'p_trans_z_W')
             self.tWeights['p_trans_gate_W'] = tf.Variable( tf.random_normal([mu_cov_inp,self.params['dim_stochastic']], stddev=stddev), 'p_trans_gate_W')
        # k-mixtures of Gaussian components: z_prev + u_prev will be transited to z_next, u_prev no transition

        self.tWeights['p_trans_W_mu'] = tf.Variable(tf.random_normal([mu_cov_inp, self.params['dim_stochastic']], stddev= stddev), 'p_trans_W_mu' )
        self.tWeights['p_trans_b_mu'] = tf.Variable(tf.random_normal([self.params['dim_stochastic'], ], stddev= stddev), 'p_trans_b_mu' )
        self.tWeights['p_trans_W_cov'] = tf.Variable(tf.random_normal([mu_cov_inp, self.params['dim_stochastic']] , stddev= stddev), 'p_trans_W_cov')
        self.tWeights['p_trans_b_cov'] = tf.Variable(tf.random_normal([self.params['dim_stochastic'], ], stddev= stddev), 'p_trans_b_cov')
      #               self.tWeights['p_trans_W_mu']  = tf.Variable(tf.random_normal([mu_cov_inp,self.params['dim_stochastic']], stddev=stddev), 'p_trans_W_mu')
        #               self.tWeights['p_trans_b_mu']  = tf.Variable(tf.random_normal([self.params['dim_stochastic'],], stddev=stddev), 'p_trans_b_mu')
        #               self.tWeights['p_trans_W_cov']  = tf.Variable(tf.random_normal([mu_cov_inp,self.params['dim_stochastic']], stddev=stddev), 'p_trans_W_cov')
        #               self.tWeights['p_trans_b_cov']  = tf.Variable(tf.random_normal([self.params['dim_stochastic'],], stddev=stddev), 'p_trans_b_cov')

        # emission function:# includes two parts : W1*z1+W2*z2+b_t
        # z1 refers to the residual latent components
        # z2 refers to the instantaneous components
        for l in range(self.params['emission_layers']):    # emission function
                dim_input,dim_output  = DIM_HIDDEN, DIM_HIDDEN
                if l==0:
                    dim_input = self.params['dim_stochastic']
                self.tWeights['p_emis_W_z_'+str(l)] = tf.Variable(tf.random_normal([dim_input, dim_output], stddev=stddev), 'p_emis_W_z_'+str(l))
                self.tWeights['p_emis_b_z_'+str(l)] = tf.Variable(tf.random_normal([dim_output,], stddev=stddev), 'p_emis_b_z_'+str(l))
                self.tWeights['p_emis_W_u_'+str(l)] = tf.Variable(tf.random_normal([dim_input, dim_output], stddev=stddev), 'p_emis_W_u_'+str(l))
                self.tWeights['p_emis_b_u_'+str(l)] = tf.Variable(tf.random_normal([dim_output,], stddev=stddev), 'p_emis_b_u_'+str(l))
        dim_out = self.params['dim_observations']*2
        if self.params['metric'] == 'logli':
            dim_out = self.params['dim_observations']*2
        if self.params['metric'] == 'crossen':
            dim_out = self.params['dim_observations']
        dim_in  = self.params['dim_hidden']
        if self.params['emission_layers']==0:
                dim_in = self.params['dim_stochastic']
        self.tWeights['p_emis_W_out'] = tf.Variable(tf.random_normal([dim_in, dim_out], stddev=stddev), 'p_emis_W_out')
        self.tWeights['p_emis_b_out'] = tf.Variable(tf.random_normal([dim_out,], stddev=stddev), 'p_emis_b_out')
        #               self.tWeights['p_emis_W_out_2'] = tf.Variable(tf.random_normal([dim_in, dim_out], stddev=stddev), 'p_emis_W_out_2')
        #               self.tWeights['p_emis_b_out_2'] = tf.Variable(tf.random_normal([dim_out,], stddev=stddev), 'p_emis_b_out_2')

        # create the Inference network
        # Step 1: Params for initial embedding of input # MLP for the first layer
        DIM_INPUT  = self.params['dim_observations']
        RNN_SIZE   = self.params['rnn_size']
        DIM_HIDDEN = RNN_SIZE
       # DIM_STOC   = self.params['dim_stochastic']
        dim_input, dim_output= DIM_INPUT, RNN_SIZE
        self.tWeights['q_W_input_0'] = tf.Variable(tf.random_normal([dim_input, dim_output], stddev=stddev), 'q_W_input_0')
        self.tWeights['q_b_input_0'] = tf.Variable(tf.random_normal([dim_output,], stddev=stddev), 'q_b_input_0')

      #Step 2: create LSTM params   # it is not a typical lstm and mask h-as been added
        suffices_to_build = ['r']
        if self.params['inference_model'] == 'LR':
               suffices_to_build.append('l')
        RNN_SIZE = self.params['rnn_size']
        # orthogonalWeight
        W_ = tf.random_normal([RNN_SIZE, RNN_SIZE])
        q_, r_ = tf.linalg.qr(W_)
        for suffix in suffices_to_build:
                self.tWeights['W_lstm_' + suffix ] = tf.Variable(tf.random_normal([RNN_SIZE, RNN_SIZE*4], stddev=stddev), 'W_lstm_' + suffix)
                self.tWeights['b_lstm_' + suffix ] = tf.Variable(tf.random_normal([RNN_SIZE*4,], stddev=stddev), 'b_lstm_' + suffix)
                self.tWeights['U_lstm_' + suffix ] = tf.Variable(tf.concat([q_, q_, q_, q_], 1), 'U_lstm_' + suffix)

        #Step 3: Parameters for combiner function
        if self.params['use_generative_prior']=='approx':
            DIM_INPUT = self.params['dim_stochastic']
            self.tWeights['q_W_st'] = tf.Variable(tf.random_normal([DIM_INPUT, self.params['rnn_size']], stddev=stddev), 'q_W_st')
            self.tWeights['q_b_st'] = tf.Variable(tf.random_normal([self.params['rnn_size'],], stddev=stddev), 'q_b_st')
        self.tWeights['q_W_mu_z']  = tf.Variable(tf.random_normal([RNN_SIZE, self.params['dim_stochastic']], stddev=stddev), 'q_W_mu_z')
        self.tWeights['q_b_mu_z']  = tf.Variable(tf.random_normal([self.params['dim_stochastic'],], stddev=stddev), 'q_b_mu_z')
        self.tWeights['q_W_cov_z'] = tf.Variable(tf.random_normal([RNN_SIZE, self.params['dim_stochastic']], stddev=stddev), 'q_W_cov_z')
        self.tWeights['q_b_cov_z'] = tf.Variable(tf.random_normal([self.params['dim_stochastic'],], stddev=stddev), 'q_b_cov_z')

        self.tWeights['q_W_mu_u']  = tf.Variable(tf.random_normal([RNN_SIZE, self.params['dim_stochastic']], stddev=stddev), 'q_W_mu_u')
        self.tWeights['q_b_mu_u']  = tf.Variable(tf.random_normal([self.params['dim_stochastic'],], stddev=stddev), 'q_b_mu_u')
        self.tWeights['q_W_cov_u'] = tf.Variable(tf.random_normal([RNN_SIZE, self.params['dim_stochastic']], stddev=stddev), 'q_W_cov_u')
        self.tWeights['q_b_cov_u'] = tf.Variable(tf.random_normal([self.params['dim_stochastic'],], stddev=stddev), 'q_b_cov_u')

        #Step4: parameters for the kumarswamy distribution:
        #  self.tWeights['kumar_A']  = tf.Variable(tf.random_normal([RNN_SIZE, self.params['k_mixtures']-1], stddev=stddev))
        #  self.tWeights['kumar_B']  = tf.Variable(tf.random_normal([RNN_SIZE, self.params['k_mixtures']-1], stddev=stddev))
        #  self.tWeights['kumar_A_b'] = tf.Variable(tf.random_normal([self.params['k_mixtures']-1], stddev=stddev))
        #  self.tWeights['kumar_B_b'] = tf.Variable(tf.random_normal([self.params['k_mixtures']-1], stddev=stddev))

        #Step4_1: Parameters for Gaussian mixtures:
        #              self.tWeights['gaussian_W_1'] = tf.Variable(tf.random_normal([RNN_SIZE, self.params['hidden_gaussian']], stddev=stddev), 'gaussian_W_1')
        #              self.tWeights['gaussian_W_2'] = tf.Variable(tf.random_normal([self.params['hidden_gaussian'], self.params['k_mixtures']], stddev=stddev), 'gaussian_W_2')
        #              self.tWeights['gaussian_b_1'] = tf.Variable(tf.random_normal([self.params['hidden_gaussian'],], stddev= stddev), 'gaussian_b_1')
        #              self.tWeights['gaussian_b_2'] = tf.Variable(tf.random_normal([self.params['k_mixtures'],], stddev=stddev), 'gaussian_b_2')

        self.X = tf.placeholder(tf.float32, [self.params['batch_size'], self.params['window_size'], self.params['dim_observations']])
       # self.M = tf.placeholder(tf.float32, [self.params['batch_size'], self.params['window_size']])
      #  self.train_op, self.train_cost, self.traindict = self._build()
      # with tf.device('/device:GPU:0'):
      #       self.train_op, self.train_cost, self.traindict = self._build()
        self.train_op, self.train_cost, self.traindict, self.anneal= self._build()
        init = tf.global_variables_initializer()
        self.sess.run(init)

              

    def _applyNL(self, lin_out):
        if self.params['nonlinearity'] == 'relu':
            return tf.nn.leaky_relu(lin_out)
        elif self.params['nonlinearity'] == 'softplus':
            return tf.nn.softplus(lin_out)
        elif self.params['nonlinearity'] == 'elu':
            return tf.nn.elu(lin_out)
        elif self.params['nonlinearity'] == 'maxout':
            return tf.contrib.layers.maxout(lin_out, self.params['retain_units'])
        else:
            return tf.tanh(lin_out)

    def _LinearNL(self, W, b, inp):
        lin = tf.nn.bias_add(tf.tensordot(inp, W, axes=[[-1], [0]]),  b)
        return self._applyNL(lin)

    def _apply_mask(self, loss_tensor, mask_tensor):
        if len(loss_tensor.get_shape().as_list())==len(mask_tensor.get_shape().as_list()):
            return tf.multiply(loss_tensor, mask_tensor) # tf.dot 
        elif len(loss_tensor.get_shape().as_list()) == len(mask_tensor.get_shape().as_list())+1:
            return tf.multiply(loss_tensor, mask_tensor[..., None])

    def _nll_gaussian(self, mu, logcov, params = None ):
        nll = 0.5*(tf.log(2*np.pi)+logcov+tf.div(tf.square(self.X-mu),tf.exp(logcov)))
        if params is not None:
            self.params['real_mu'] = mu
            self.params['real_logcov'] = logcov
         # the input mu and cov are both working as variables during in the training procedure
         # self._apply_mask(nll, mask)
        return nll

    def _transition(self, z, u):  # input includes z_previous, and u_previous
#         if trans_params is None:                       # return mu1, cov1, mu2, cov2
#             trans_params = self.tWeights
        hid_z    = z
        hid_z_g  = z
        hid_u    = u
        hid_u_g  = u
#        self.sess.run(z.get_shape())
        for l in range(self.params['transition_layers']):
#            self.sess.run(hid.get_shape())
            hid_z= self._LinearNL(self.tWeights['p_trans_W_z_'+str(l)], self.tWeights['p_trans_b_z_'+str(l)], hid_z)
            hid_u= self._LinearNL(self.tWeights['p_trans_W_u_'+str(l)], self.tWeights['p_trans_b_u_'+str(l)], hid_u)
            if self.params['transition_type']=='gated': 
                hid_z_g = self._LinearNL(self.tWeights['p_trans_gate_W_z_'+str(l)], self.tWeights['p_trans_gate_b_z_'+str(l)],hid_z_g)
               # if l == 0: 
                    # self.sess.run(self.tWeights['p_trans_gate_W_u_'+str(l)].get_shape())
               #      self.sess.run(hid_u_g.get_shape())
                hid_u_g = self._LinearNL(self.tWeights['p_trans_gate_W_u_'+str(l)], self.tWeights['p_trans_gate_b_u_'+str(l)], hid_u_g)
               
       
        mu_prop_z= tf.nn.bias_add(tf.tensordot( hid_z + hid_u, self.tWeights['p_trans_W_mu'], axes= [[-1], [0]]), self.tWeights['p_trans_b_mu'])
        if self.params['transition_type'] =='gated':
            gate   = tf.sigmoid(tf.tensordot(hid_z_g + hid_u_g,  self.tWeights['p_trans_gate_W'], axes= [[-1], [0]]))   # combination of the z and u hidden
            mu_z_t     = tf.multiply(gate, mu_prop_z) + tf.multiply( 1-gate,tf.tensordot(z+u,self.tWeights['p_trans_z_W'], axes=[[-1], [0]]))
        else:
            mu_z_t = mu_prop_z
        cov_z_t    = tf.nn.softplus(tf.nn.bias_add(tf.tensordot(hid_z + hid_u, self.tWeights['p_trans_W_cov'], axes = [[-1], [0]]), self.tWeights['p_trans_b_cov']))

        return mu_z_t, cov_z_t

    def _emission(self, z, u):        # emission function took two inputs: z refers to the residual components, u refers to the instantaneous  
        # Input:  z [bs x T x dim  Output: hid [bs x T x dim]
        hid_z     = z
        hid_u     = u
        for l in range(self.params['emission_layers']):
            hid_z = self._LinearNL(self.tWeights['p_emis_W_z_'+str(l)], self.tWeights['p_emis_b_z_'+str(l)], hid_z)
            hid_u = self._LinearNL(self.tWeights['p_emis_W_u_'+str(l)], self.tWeights['p_emis_b_u_'+str(l)], hid_u)
     
        outp_z   = tf.nn.bias_add(tf.tensordot(hid_z, self.tWeights['p_emis_W_out'], axes=[[-1], [0]]), self.tWeights['p_emis_b_out'])
        outp_u   = tf.nn.bias_add(tf.tensordot(hid_u, self.tWeights['p_emis_W_out'], axes=[[-1], [0]]), self.tWeights['p_emis_b_out'])        

        return outp_z, outp_u

    # Negative ELBO [Evidence Lower Bound] 
    def _temporalKL(self, mu_q, cov_q,  mu_prior, cov_prior, p_weights= None):
     #KL(q_t||p_t) = 0.5*(log|sigmasq_p| -log|sigmasq_q|  -D + Tr(sigmasq_p^-1 sigmasq_q) + (mu_p-mu_q)^T sigmasq_p^-1 (mu_p-mu_q))
#        assert np.all(cov_q.tag.test_value>0.),'should be positive'
#        assert np.all(cov_prior.tag.test_value>0.),'should be positive'
        diff_mu = mu_prior-mu_q
        # tensorflow tensor data structure
        KL      = tf.log(cov_prior)-tf.log(cov_q) - 1. + tf.div(cov_q, cov_prior) + tf.div(tf.square(diff_mu), cov_prior)
#        KL_t    = 0.5*KL.sum(2)
        if p_weights  is not None:
             KL      = tf.multiply(KL, p_weights)
        KL_t    = 0.5*tf.reduce_sum(KL, 2)
     #   KLmasked  = tf.multiply(KL_t,mask)Structured Inference Networks for Nonlinear State Space Models
        return tf.reduce_sum(KL_t) 

    def _nll_crossen(self, Predict_x, Original_x ):
         nll = -tf.multiply(Original_x, tf.log(tf.sigmoid(Predict_x))) - tf.multiply(1-Original_x, tf.log(1-tf.sigmoid(Predict_x)))
        # nll = 0.5*(tf.log(2*np.pi)+logcov+tf.div(tf.square(self.X-mu),tf.exp(logcov)))
         return nll         

    def _neg_elbo(self, anneal  = 1., dropout_prob=0., additional = None):  
        
        z_q, mu_q_z, cov_q_z, u_q, mu_q_u, cov_q_u, mu_u_p, cov_u_p = self._q_z_x(inp_data=self.X,dropout_prob=dropout_prob, anneal= anneal)
        mu_trans, cov_trans= self._transition(z_q, u_q)   # return the transition results for z not for u
        mu_prior_z         = tf.concat([tf.zeros_like(tf.slice(mu_trans, [0, 0, 0], [-1, 1, -1])), mu_trans[:,:-1,:]], 1)
        cov_prior_z        = tf.concat([tf.ones_like(tf.slice(mu_trans, [0, 0, 0], [-1, 1, -1])), cov_trans[:,:-1,:]], 1)
        
        # calculate pi_mean
      #  p_z, p_u           = self._estimate_gmm(z,u)
        KL_z               = self._temporalKL(mu_q_z, cov_q_z, mu_prior_z, cov_prior_z) 
      #  KL_z               = self._temporalKL(mu_q_z, cov_q_z, mu_prior_z, cov_prior_z)

       # p_u                = tf.tensordot(tf)
       # KL_u               =  0.5*(tf.log(2*np.pi)+tf.log(cov_q_u)+tf.div(tf.square(p_u-mu_q_u),cov_q_u))
       
        KL_u               =  self._temporalKL(mu_q_u, cov_q_u, mu_u_p, cov_u_p)
      #  self.sess.run(KL_u)
      #  KL_u               = self._temporalKL(mu_q_u, cov_q_u, 0.,1.001) # standard Gaussian as prior
        KL                 = KL_z + KL_u
    #    KL                 = KL_z
#        self.sess.run(KL.get_shape())   
        # calculate the kumar_ divergence
     #   KLD_u  = _compute_kumar2beta_kld(tf.expand_dims(self.kumar_a[:,:,0], 2), tf.expand_dims(self.kumar_b[:,:,0], 2), \
     #                                   self.params['dirichlet_alpha'], 1 + self.params['dirichlet_alpha'])
     #   KL     = KL + KLD_u
        
 #      self.sess.run(KLD_u.get_shape())
        # calculate the mixture divergence
#        MIX_d = self._MixtureEntropy(pi_samples, z_q, mu_q_z, cov_q_z, u_q, mu_q_u, cov_q_u)
        
        # the prediction loss: z and u 
        hid_out_z, hid_out_u            = self._emission(z_q, u_q)  # the tensor structure same as previous
        hid_out = hid_out_z + hid_out_u
        params = {}
        if self.params['metric'] == 'logli':
               dim_obs        = self.params['dim_observations']
               mu_hid         = hid_out[:,:,:dim_obs]
               logcov_hid     = hid_out[:,:,dim_obs:dim_obs*2]
               nll_mat        = self._nll_gaussian(mu_hid, logcov_hid, params=params)
               nll            = tf.reduce_sum(nll_mat)
        elif self.params['metric'] == 'crossen':
               nll_mat       = self._nll_crossen(hid_out, self.X)
               nll           = tf.reduce_sum(nll_mat) 

        #Evaluate negative ELBO
 #       neg_elbo       = nll+anneal*(KL+MIX_d)
        neg_elbo       = nll+anneal*(KL)
        if additional is not None:
            additional['hid_out']= hid_out
            additional['nll_mat']= nll_mat
            additional['nll_batch']= tf.reduce_sum(nll_mat, [1, 2])
            additional['nll_feat'] =  tf.reduce_sum(nll_mat, [0, 2])
            additional['nll']    = nll
            additional['kl']     = KL
            additional['mu_q_z']   = mu_q_z
            additional['cov_q_z']  = cov_q_z
            additional['z_q']    = z_q
            additional['mu_t']   = mu_trans
            additional['cov_t']  = cov_trans

            for k in params:
                additional[k] = params[k]
        return neg_elbo
    
    
    def _estimate_gmm(self, hidden_state):
        # this net is used to infer the parameters: 
        # 1) for kumaraswamy model, the pi_means and pi_samples
        # 2) for general Gaussian mixture, the means and covariance
        
     #   input_     = tf.placeholder(tf.float32, [self.params['window_size'], self.params['dim_observations']])
       # input_     = hidden_slice
        input_     = hidden_state[0, :, :]
        alpha      = tf.cast([1.0], dtype=tf.float32)
        beta       = tf.cast([1.0], dtype=tf.float32)
        batch_     = hidden_state.get_shape().as_list()[0]
        hidden_output = []
        
        
        ln2piD     = tf.constant(np.log(2*np.pi)*self.params['dim_observations'], dtype=tf.float32)

        for batch_idx in range(batch_):
             for step in range(2):
                  input_ = tf.cast(hidden_state[batch_idx, :, :], dtype=tf.float32)                  

                  dim_means  = tf.reduce_mean(input_, 0)
                  dim_dis    = tf.squared_difference(input_, tf.expand_dims(dim_means, 0))
                  dim_vari   = tf.reduce_sum(dim_dis, 0)/tf.cast(tf.shape(input_)[0], dtype=tf.float32)
                  avg_vari   = tf.cast(tf.reduce_sum(dim_vari)/self.params['k_mixtures']/self.params['dim_observations'], dtype=tf.float32)
                  r_point_in = tf.squeeze(tf.multinomial(tf.ones([1, tf.shape(input_)[0]]), self.params['k_mixtures']))

                  means    = tf.cast(tf.gather(input_, r_point_in), dtype=tf.float32)
                  vari     = tf.cast(tf.ones([self.params['k_mixtures'], self.params['dim_observations']]),dtype=tf.float32)*avg_vari
                  weights  = tf.cast(tf.fill([self.params['k_mixtures']], 1./self.params['k_mixtures']), dtype=tf.float32)
                  
                  sq_distances = tf.squared_difference(tf.expand_dims(input_, 0), tf.expand_dims(means, 1))
                  sum_sq_dist_times_inv_var = tf.reduce_sum(sq_distances/tf.expand_dims(vari, 1), 2)
                  log_coefficients  = tf.expand_dims(ln2piD + tf.reduce_sum(tf.log(vari), 1), 1)
                  log_components  = -0.5*(log_coefficients + sum_sq_dist_times_inv_var)
                  log_weighted    = log_components + tf.expand_dims(tf.log(weights), 1)
                  log_shift       = tf.expand_dims(tf.reduce_max(log_weighted, 0), 0)
                  exp_log_shifted = tf.exp(log_weighted-log_shift)
                  exp_log_shifted_sum = tf.reduce_sum(exp_log_shifted, 0)
                  gamma  = exp_log_shifted /exp_log_shifted_sum


                  gamma_sum    = tf.reduce_sum(gamma, 1)
                  gamma_weight = gamma / tf.expand_dims(gamma_sum, 1)
                  means_        = tf.reduce_sum(tf.expand_dims(input_, 0) * tf.expand_dims(gamma_weight, 2), 1)
                  distances_    = tf.squared_difference(tf.expand_dims(input_, 0), tf.expand_dims(means_, 1))
                  vari_         = tf.reduce_sum(distances_ * tf.expand_dims(gamma_weight, 2), 1) 
                  weights_      = gamma_sum/tf.cast(tf.shape(input_)[0], dtype=tf.float32)

                  vari_     *= tf.expand_dims(gamma_sum, 1)
                  vari_     += (2.0*beta)
                  vari_     /= tf.expand_dims(gamma_sum + (2.0*(alpha+1.0)), 1)
                  
                  means   = means_
                  vari    = vari_
                  weights = weights_

             idx    = tf.cast(tf.argmin(weights), dtype = tf.int32)
             core_  = -0.5*tf.square(tf.nn.bias_add(hidden_state[batch_idx,:,:], means[idx, :]))/tf.expand_dims(vari[idx,:],0)
            # P_out  = weights[idx]*tf.exp(core_)/tf.sqrt(2*np.pi*self.params['dim_observations']*tf.expand_dims(vari[idx,:], 0))
             P_out  = weights[idx]*tf.exp(core_)/tf.sqrt(tf.expand_dims(vari[idx,:], 0))/tf.pow(tf.sqrt(2*np.pi), self.params['dim_observations'])
            # P_out  = tf.nn.softmax(P_out)
             hidden_output.append(tf.multiply(P_out, hidden_state[batch_idx, :, :]))
        
        mu_u_p   =   tf.nn.bias_add(tf.tensordot(hidden_output, self.tWeights['q_W_mu_u'], axes=[[-1], [0]]), self.tWeights['q_b_mu_u'])
        cov_u_p  =   tf.nn.softplus(tf.nn.bias_add(tf.tensordot(hidden_output, self.tWeights['q_W_cov_u'], axes=[[-1], [0]]), self.tWeights['q_b_cov_u']))
        
        return mu_u_p, cov_u_p
        # log_likelihood  = tf.reduce_sum(tf.log(exp_log_shifted_sum)) + tf.reduce_sum(log_shift)
        
       # train_step   = tf.group(means.assign(means_), vari.assign(vari_), weights.assign(weights))
         
      #  self.sess.run(local_variable_initializer(), feed_dict={input_: hidden_state[0, :, :]})

        
             
             # self.sess.run(re_normalize_weights)
#n_batch  = hidden_state.get_shape().as_list()[0j]
# for i in range(n_batch):
        #    dim_means  = tf.reduce_mean(hidden_state[i, :, :])
            
        
       # dim_means  = tf.reduce_mean(hidden_state)
        
       # hid_gau_1 = self._LinearNL(self.tWeights['gaussian_W_1'], self.tWeights['gaussian_b_1'], hidden_state)
       # hid_gau_2 = hidden_state-hid_gau_1
        
       # logits_hi = self._LinearNL(self.tWeights['gaussian_W_2'], self.tWeights['gaussian_b_2'], hid_gau_1) 
  
       # soft_prob = tf.nn.softmax(logits_hi) 
      #  p_z       = soft_prob[0]*hidden_state 
      #  p_u       = soft_prob[1]*hidden_state     
        
        #  esitmate mean_z, mean_u, cov_z, cov_u
        

        #b_inv = tf.pow(self.kumar_b, -1)
       # v_means          =   tf.multiply(self.kumar_b, _beta_fn(1.+a_inv, self.kumar_b))
       # uni_samples      =   tf.random_uniform(tf.shape(v_means), minval=1e-8, maxval=1-1e-8)
       # v_samples        =   tf.pow(1-tf.pow(uni_samples, b_inv), a_inv)
       
       # pi_means    =   self._compose_stick_segments(v_means)
       # pi_samples  =   self._compose_stick_segments(v_samples)
       
           
#        return mu_u_p, cov_u_p
        
    def _compose_stick_segments(self, v):
        segments = []
        # for 2D dimension matrix, for 3D tensor
        self.remaining_stick = [tf.ones((tf.shape(v)[0],tf.shape(v)[1], 1))]
        for i in range(self.params['k_mixtures']-1):
             curr_v = tf.expand_dims(v[:,:, i], 2)
             segments.append(tf.multiply(curr_v, self.remaining_stick[-1]))
             self.remaining_stick.append(tf.multiply(1-curr_v, self.remaining_stick[-1]))
        segments.append(self.remaining_stick[-1])
         
        return segments  
     
    def _MixtureEntropy(self, pi_samples, z, mu_z, cov_z, u, mu_u, cov_u):
      #  self.sess.run(pi_samples[0].get_shape())
        s_z = tf.multiply(pi_samples[0], tf.exp(log_normal_pdf(z, mu_z, cov_z)))
        s_u = tf.multiply(pi_samples[1], tf.exp(log_normal_pdf(u, mu_u, cov_u)))
        
        return -tf.log(s_z + s_u)
                    
            
    def _aggregateLSTM(self, hidden_state):
        # LSTM hidden layer [T x bs x dim]  z [bs x T x dim], mu [bs x T x dim], cov [bs x T x dim]
        # for ssm, the prior knowledge are introduced to make up the transitions, 
        def _pog(mu_1,cov_1, mu_2,cov_2):
             cov_f = tf.sqrt(tf.div(tf.multiply(cov_1, cov_2),(cov_1+cov_2)))
             mu_f  = tf.div(tf.multiply(mu_1, cov_2)+tf.multiply(mu_2, cov_1),cov_1+cov_2)
             return mu_f, cov_f
                                                                                
        def st_approx(args1, args2):
            z_prev = args1[0]
          #  mu = args1[1]
          #  cov = args1[2]
            h_t = args2[0]
            eps = args2[1]
 #           tParams    = collections.OrderedDict()
      #      tParams = self.tWeights
#            h_t, eps  =  tf.split(inputs, [self.dim, self.params['dim_stochastic']], -1)
#            noseq = self.non_seq
#            for p in noseq:
#                tParams[p.name] = p

            h_next     = tf.tanh(tf.nn.bias_add(tf.tensordot(z_prev, self.tWeights['q_W_st'], axes=[[-1], [0]]), self.tWeights['q_b_st']))
            h_next     = (1./2.)*(h_t+h_next)
            
            mu_t       = tf.nn.bias_add(tf.tensordot(h_next, self.tWeights['q_W_mu_z'], axes=[[-1], [0]]), self.tWeights['q_b_mu_z'])
            cov_t      = tf.nn.softplus(tf.nn.bias_add(tf.tensordot(h_next, self.tWeights['q_W_cov_z'], axes=[[-1], [0]]), self.tWeights['q_b_cov_z']))
            
            z_t        = mu_t+tf.multiply(tf.sqrt(cov_t), eps)
            return z_t, mu_t, cov_t
        
        def st_approx_u(args1, args2):  # it works for input components, so arbitrarily initialized by 
            u = args1[0]
          #  mu = args1[1]
          #  cov = args1[2]
            h_next = args2[0]
            eps    = args2[1]
            mu_u_t       = tf.nn.bias_add(tf.tensordot(h_next, self.tWeights['q_W_mu_u'], axes=[[-1], [0]]), self.tWeights['q_b_mu_u'])
            cov_u_t      = tf.nn.softplus(tf.nn.bias_add(tf.tensordot(h_next, self.tWeights['q_W_cov_u'], axes=[[-1], [0]]), self.tWeights['q_b_cov_u']))
            
            u_t        = mu_u_t+tf.multiply(tf.sqrt(cov_u_t), eps)
            return u_t, mu_u_t, cov_u_t
                                                                                
       # def st_true((z_prev, mu, cov), (h_t, eps)):
        def st_true(args1, args2):
            z_prev = args1[0]
            h_t    = args2[0]
            u_t    = args2[1]
            eps    = args2[2]
#             tParams = self.tWeights
            mu_trans, cov_trans = self._transition(z_prev, u_t) # previous input u_prev is introduced
            # the inro
            mu_t       = tf.nn.bias_add(tf.tensordot(h_t, self.tWeights['q_W_mu_z'], axes=[[-1], [0]]), self.tWeights['q_b_mu_z'])
            cov_t      = tf.nn.softplus(tf.nn.bias_add(tf.tensordot(h_t, self.tWeights['q_W_cov_z'], axes=[[-1], [0]]), self.tWeights['q_b_cov_z']))
            
            # code for mixing two components(priors and current inferences
            mu_f, cov_f= _pog(mu_trans, cov_trans, mu_t, cov_t)  # the introduced priors will not work for the 
            z_f        = mu_f + tf.multiply(tf.sqrt(cov_f), eps)
            return z_f, mu_f, cov_f


#        z0    = tf.random_normal([tf.shape(eps)[1], tf.shape(eps)[-1]])
#        z0    = tf.random_normal([hidden_state.get_shape()[1], self.params['dim_stochastic']])
        self.dim = hidden_state.get_shape().as_list()[2]
        eps   = tf.random_normal([hidden_state.get_shape().as_list()[0], hidden_state.get_shape().as_list()[1], self.params['dim_stochastic']])
        z0    = tf.random_normal([hidden_state.get_shape().as_list()[1], self.params['dim_stochastic']])
#         u0    = tf.random_normal([hidden_state.get_shape().as_list()[1], self.params['dim_stochastic']])
        # u used as the input element to be looped 
        if self.params['use_generative_prior'] == 'true':
#             self.non_seq = [self.tWeights[k] for k in ['q_W_mu','q_b_mu','q_W_cov','q_b_cov']]+[self.tWeights[k] for k in self.tWeights if '_trans_' in k]
            step_fxn = st_true
        else:
#             self.non_seq = [self.tWeights[k] for k in ['q_W_st', 'q_b_st','q_W_mu','q_b_mu','q_W_cov','q_b_cov']]
            step_fxn = st_approx
                                                                            
       # rval     = tf.scan(step_fxn, elems=[sequences=[hidden_state, eps], outputs_info=[z0, None,None], non_sequences=non_seq])
#        concat_tensor = tf.concat([hidden_state, eps], -1)
#        tf.random_normal  = tf.concat([])
       
         # output u 
        mu_u   =   tf.nn.bias_add(tf.tensordot(hidden_state, self.tWeights['q_W_mu_u'], axes=[[-1], [0]]), self.tWeights['q_b_mu_u'])
        cov_u  =   tf.nn.softplus(tf.nn.bias_add(tf.tensordot(hidden_state, self.tWeights['q_W_cov_u'], axes=[[-1], [0]]), self.tWeights['q_b_cov_u']))
        u      =   mu_u + tf.multiply(tf.sqrt(cov_u), eps)  
            
        rval_z =   tf.scan(step_fxn, elems=(hidden_state, u, eps), initializer= (z0, z0, z0)) # z components need recursive
           
     #  rval_u     = tf.scan(step_fxn=st_approx, elems=(hidden_state, eps), initializer= (u0, u0, u0))
     
        z, mu_z, cov_z  = tf.transpose(rval_z[0], [1, 0, 2]), tf.transpose(rval_z[1], [1, 0, 2]),  tf.transpose(rval_z[2], [1, 0, 2])
    
        u, mu_u, cov_u  = tf.transpose(u, [1, 0, 2]), tf.transpose(mu_u, [1, 0, 2]), tf. transpose(cov_u, [1, 0, 2]) 
#         u, mu_u, cov_u  = tf.transpose(rval_u[0], [1, 0, 2]), tf.transpose(rval_u[1], [1, 0, 2]),  tf.transpose(rval_u[2], [1, 0, 2])
        return z, mu_z, cov_z, u, mu_u, cov_u
    
    def _LSTM_layer(self, inp, suffix, dropout_prob=0., RNN_SIZE = None):
        def _slice(mat, n, dim):
#            self.sess.run(mat.get_shape())
            if len(mat.get_shape().as_list()) == 3:
                return mat[:,:, n*dim:(n+1)*dim]
            return mat[:, n*dim:(n+1)*dim]
#            return mat[:,:, n*dim:(n+1)*dim]

        def _lstm_layer(previous_states, inputs):
       # def _lstm_layer((h_, c_), inputs):
            h_  = previous_states[0]
            c_  = previous_states[1]
            lstm_U     = self.tWeights['U_lstm_'+suffix]
#            h_, c_   = tf.split(previous_states, 2, 0)
            x_       = inputs
           # t_m_      = self.tMask
            preact = tf.tensordot(h_, lstm_U, axes=[[-1], [0]])
            preact += x_
#            self.sess.run(preact.get_shape())
            i = tf.sigmoid(_slice(preact, 0, RNN_SIZE))
            f = tf.sigmoid(_slice(preact, 1, RNN_SIZE))
            o = tf.sigmoid(_slice(preact, 2, RNN_SIZE))
            c = tf.tanh(_slice(preact, 3, RNN_SIZE))
            # c and h are only updated if the current time 
            # step contains atleast an observed feature 
            # obs_t = t_m_[:,None]
            c_new = tf.multiply(f, c_) + tf.multiply(i, c)
#            c = tf.multiply(c_new, obs_t)+ tf.multiply(1-obs_t,c_)
            c = c_new
            h_new =tf.multiply(o,  tf.tanh(c))
#            h = tf.multiply(h_new, obs_t)+ tf.multiply(1-obs_t,h_)
            h = h_new
#            return tf.concat([h, c], 0)
            return h, c

     #   lstm_input = tf.tensordot(inp.transpose([1,0,2]), self.tWeights['W_lstm_'+suffix], axes=[[2], [0]])+ self.tWeights['b_lstm_'+suffix]
        lstm_input = tf.nn.bias_add(tf.tensordot(tf.transpose(inp, perm=[1,0,2]), self.tWeights['W_lstm_'+suffix], axes=[[-1], [0]]), self.tWeights['b_lstm_'+suffix])
#        self.sess.run(lstm_input.get_shape())
#        nsteps     = tf.get_shape(lstm_input)[0]
#        n_samples  = tf.get_shape(lstm_input)[1]
        nsteps     = lstm_input.get_shape().as_list()[0]
        n_samples  = lstm_input.get_shape().as_list()[1]
#        o_info = [tf.zeros([n_samples,RNN_SIZE]), tf.ones([n_samples,RNN_SIZE])]
        init_h = tf.zeros([n_samples, RNN_SIZE])
        init_c = tf.ones([n_samples, RNN_SIZE])
#        concat_tensor = tf.concat([init_h, init_c], 0)
#        n_seq      = self.tWeights['U_lstm_'+suffix] # it is a tensorflow tensor type

       # if temporalMask is None: 
       #     tMask      = tf.ones([nsteps, n_samples])
       # else:
       #     tMask      = tf.transpose(temporalMask, [1, 0])

        if suffix=='r':
            lstm_input = lstm_input[::-1]
      #      tMask      = tMask[::-1]
      #  self.tMask = tMask
#        inputs = tf.concat([lstm_input, tMask], 0)

       # rval, _= tf.scan(_lstm_layer, sequences=[lstm_input, tMask], outputs_info=o_info, non_sequences=n_seq)
#        elems = [lstm_input, tMask, tf.zeros([n_samples,RNN_SIZE]), tf.ones([n_samples,RNN_SIZE]), n_seq]
#        rval, _= tf.scan(_lstm_layer,elems=[lstm_input, tMask, o_info, n_seq])
#        self.sess.run(lstm_input.get_shape())
#        rval = tf.scan(_lstm_layer, elems=lstm_input, tf.random_normal=concat_tensor)
        lstm_output, _ = tf.scan(_lstm_layer, elems=lstm_input, initializer=(init_h, init_c))
#        lstm_output, _ =  tf.split(rval, 2, 0)
#        self.sess.run(lstm_output.get_shape())
        if suffix=='r':
            lstm_output = lstm_output[::-1]
                                                        
        return tf.nn.dropout(lstm_output, dropout_prob)

    def _q_z_x(self, inp_data=None,  dropout_prob = 0., anneal =1.):    # inference model
        # X: nbatch x time x dim_observations  Returns: z_q, mu_q  and cov_q (nbatch x time x dim_stochastic)
       # self.X  = tf.nn.l2_normalize(self.X)
        embedding         = self._LinearNL(self.tWeights['q_W_input_0'],self.tWeights['q_b_input_0'], inp_data)
#        self.sess.run(embedding.get_shape())
        h_r               = self._LSTM_layer(embedding, 'r', dropout_prob = dropout_prob, RNN_SIZE = self.params['rnn_size'])       
        if self.params['inference_model']=='LR':
            h_l           = self._LSTM_layer(embedding, 'l', dropout_prob = dropout_prob, RNN_SIZE = self.params['rnn_size'])
            hidden_state  = (h_r+h_l)/2.
        elif self.params['inference_model']=='R':
            hidden_state  = h_r
            
         # kumar output layer:
        z_q, mu_q_z, cov_q_z, u_q, mu_q_u, cov_q_u  = self._aggregateLSTM(hidden_state)
         # kumar output layer:
        hidden_state = tf.transpose(hidden_state, [1, 0, 2])
        mu_u_p, cov_u_p = self._estimate_gmm(hidden_state) # GMM based prediction distribution for u component
        #  p_u  = 0
        # self.kumar_a  = tf.nn.bias_add(tf.tensordot(hidden_state, self.tWeights['kumar_A'], axes=[[-1], [0]]), self.tWeights['kumar_A_b'])
        # self.kumar_b  = tf.nn.bias_add(tf.tensordot(hidden_state, self.tWeights['kumar_B'], axes=[[-1], [0]]), self.tWeights['kumar_B_b'])
        return z_q, mu_q_z, cov_q_z, u_q, mu_q_u, cov_q_u, mu_u_p, cov_u_p
               

    def _build(self):
        traindict  = {}
        learning_rate  = tf.train.exponential_decay(
                                   self.params['lr'],
                                   tf.train.get_global_step(),
                                   self.params['decay_step'],
                                   self.params['decay_rate'],
                                   staircase = True)
        self.tWeights['anneal'] = np.asarray(5.0)
        self.tWeights['update_ctr'] = np.asarray(1.)
        anneal = self.tWeights['anneal']
        iteration_t = self.tWeights['update_ctr']
        anneal_div = tf.constant(1000.)
#        model_params = [
#        for pvariable  in self.tWeights:
#             model_params.append(pvariable)
       # model_params = self.tWeights
        if 'anneal_rate' in self.params:
             anneal_div = tf.constant(self.params['anneal_rate'])
        iteration_t = tf.Variable(1.)
        increment_t = tf.assign(iteration_t, iteration_t +1.)
        anneal = tf.where(0.01+increment_t/anneal_div>1., 1., 0.01+increment_t/anneal_div)
      #  anneal_update = [(iteration_t, iteration_t+1), (anneal,tf.where(0.01+iteration_t/anneal_div>1, 1, 0.01+iteration_t/anneal_div))]
        anneal_sum = tf.reduce_sum(anneal) 

        #learning_rate = 0.0001
        model_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.92, beta2=0.999)
       # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_cost = self._neg_elbo(anneal = anneal, dropout_prob = self.params['rnn_dropout'], additional=traindict)
        train_cost = tf.reduce_mean(train_cost)
      # to create the regularizer globally and then feed into the optimizer
      #  for p in self.tWeights:
      #       if self.params['reg_spec'] in p.name:
      #            regularizer += tf.nn.l2_loss(p)

      #  regularizer = np.asarray(0.)
      #  for k in self.tWeights.values():
      #       for name, values in k.items():
      #           print(name)
      #       if '_' in k.name:
      #             regularizer += tf.nn.l2_loss(k)

        regularizer  = tf.add_n([tf.nn.l2_loss(v) for v in model_params if '_b' not in v.name])
        train_cost = train_cost + self.params['reg_value']*regularizer
        train_cost = tf.reduce_mean(train_cost)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
              train_op  = tf.contrib.layers.optimize_loss(loss=train_cost,
                                          global_step=tf.train.get_global_step(),
                                          learning_rate=learning_rate,
                                          optimizer=optimizer,
                                          variables=model_params)
       # train_op += anneal_update
#        init = tf.local_variables_tf.random_normal()
#        self.sess.run(init)
        return train_op, train_cost, traindict, anneal_sum

    def _shuffle_idx(self, len_): 
        
       # idxlist   = tf.range(len_)
        idxlist   = list(range(len_))
        batch_size = self.params['batch_size']
        tf.random_shuffle(idxlist)
        batch_list_0 = tf.split(idxlist[:int(len_/batch_size)*batch_size], int(len_/batch_size))
        batch_list   = batch_list_0[:-1]
        
        return batch_list                
                        
    def _train(self, dataset, shuffle=True,  dataset_eval = None):
        # check the pre-trained model:
        Flag  = self.params['Flag']
        batch_size = self.params['batch_size']
        end_epoch  = self.params['epochs']
        start_epoch= self.params['start_epoch']
        N          = dataset.shape[0]
       # T_d        = dataset[Flag]['tensor'].shape[1]
        epfreq     = 1
        Normalizer_ = self.params['window_size']* self.params['dim_observations']* self.params['batch_size']
        savefreq   = self.params['savefreq']
      #  idxlist    = list(range(N))   # range here is a generator object
   
        pre_train, global_step = self.load(self.params['save_dir'])
        if  pre_train:
            iter_num    = global_step
            start_epoch = global_step // int(N/batch_size)
        else:
            iter_num    =  0
            start_epoch =  0

       # len_       = tf.placeholder(tf.int32, shape=())
        batch_list =  self._shuffle_idx(N)    
       
#        init = tf.global_variables_tf.random_normal()
#        self.sess.run(init)  
       #  init = tf.initialize_all_variables()
       # self.sess.run(init)
       # print('Initialize model successfully...')                                                                
       # X = tf.placeholder(tf.float32, [batch_size, T_d, self.params['dim_observations']])

       # X = tf.placeholder(tf.float32, [batch_size, T_d, self.params['dim_observations']])
       # M = tf.placeholder(tf.float32, [batch_size, T_d])
       # with tf.device('/device:GPU:0'):
       #      train_op, train_cost, traindict = self._build(X, M)
#        train_op, train_cost, traindict = self._build(X, M)
        run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)       
        for epoch in range(start_epoch, end_epoch):
             if shuffle:
                  batch_ = self.sess.run(batch_list)
             for bnum, batch_idx in enumerate(batch_):
                  batch_idx = batch_[bnum]
                  [step_loss, batch_bound, evaluate, anneal_sum] = self.sess.run([self.train_op, self.train_cost, self.traindict, self.anneal], feed_dict={self.X:dataset[batch_idx]}, options=run_options)
               #   negCLL = evaluate['nll']
               #   KL     = evaluate['kl']
                  if epoch%epfreq==0 and bnum%5==0:
                     bval = batch_bound/float(Normalizer_)
                   #  print(('Bnum: %d, Batch Bound: %.4f')%(bnum, bval))
                     print(('batch_bound:%.4f, -veCLL:%.4f, KL:%.4f, anneal:% 4f')%(bval, evaluate['nll'], evaluate['kl'], anneal_sum))
                     sys.stdout.flush()
                 #    if bnum == 10:
                 #       self.sess.run(evaluate['mu_q_z'].get_shape())
                  iter_num += 1
             if savefreq is not None and epoch%savefreq==0:
                #  self.sess.run(evaluate['mu_q_z'].get_shape())
                  self.save(iter_num, self.params['save_dir'], model_name= 'DGVAE_feature')
                  if dataset_eval is not None:
                      tmpMap  = {}
                      bound   =  self._evaluateBound(dataset_eval)        
                      print(('Validation Bound:%.4f')%(bound))
         
    def _evaluateBound(self, dataset_eval):
        #evaluate ELBO values#
        bound  = 0
        N  = dataset_eval.shape[0]
        batch_size  = self.params['batch_size']
        evaldict    = {}
        Normalizer_ = self.params['batch_size']*self.params['window_size']* self.params['dim_observation']
        eval_cost   = self._neg_elbo(anneal = 1., dropout_prob=0., additional=evaldict)
        

        for bnum, st_idx in enumerate(range(0, N, batch_size)):
            end_idx    = min(st_idx + batch_size, N)
            idx_data   = np.arange(st_idx, end_idx)
            batch_bd   = self.sess.run(eval_cost, feed_dict={self.X:dataset_eval[idx_data]})
            bound     += batch_bd/Normalizer_
        
        return bound
                       
    def save(self, iter_num, savedir, model_name):
       saver  = tf.train.Saver()
       checkpoint_path = savedir
       if not os.path.exists(checkpoint_path):
          os.makedirs(checkpoint_path)
       print("Saving model ...")
       saver.save(self.sess,os.path.join(checkpoint_path, model_name),
                 global_step = iter_num)
                                                                                
                                    
    def load(self, savedir):
       print('reading checkpoint...')
       saver = tf.train.Saver()
       ckpt = tf.train.get_checkpoint_state(savedir)
       if ckpt and ckpt.model_checkpoint_path:
            path_ck = tf.train.latest_checkpoint(savedir)
            global_step = int(path_ck.split('/')[-1].split('-')[-1])
            saver.restore(self.sess, path_ck)
            return True, global_step
       else:
            return False, 0

    def create_feature_extractor(self, inp_data):
        z_q, _, _, u_q, _, _, _, _ = self._q_z_x(
            inp_data=inp_data, dropout_prob=self.params['rnn_dropout'],
            anneal=1.
        )
        _, feature_u = self._emission(z_q, u_q)
        return feature_u

    def _generate(self, test_data):
        testdict  = {}
        N         = test_data.shape[0]
        dur_      = self.params['window_size']
        if len(test_data.shape) >2:
            len_t = N
            label = 3
            data_o = tf.placeholder(tf.float32, shape=[1, test_data.shape[1], test_data.shape[2]])
        else:
            len_t = int(N/dur_)-1
            label = 2
            data_o  = tf.placeholder(tf.float32, shape=[1, dur_, test_data.shape[-1]])
        #test_cost  = self._neg_elbo(anneal = anneal, dropout_prob = self.params['rnn_dropout'], additional=testdict)
        z_q, _, _, u_q, _, _, _, _   =  self._q_z_x(inp_data=data_o,dropout_prob = self.params['rnn_dropout'], anneal =1.)
        feature_z, feature_u         = self._emission(z_q, u_q)   
        g_feature = []      
  
        start = time.time() 
        for idx in range(len_t):
             if label ==3:
                   f_recon   = self.sess.run(feature_u, feed_dict={data_o:test_data[idx:idx+1, :, :]})     
             elif label == 2:
                   f_recon   = self.sess.run(feature_u, feed_dict={data_o:test_data[idx*dur_:(idx+1)*dur_]}) 
             g_feature.append(f_recon[0])
        end   = time.time()
        print(end-start)
        print(N)
        return np.array(g_feature)
        #with open(self.params['save_dir']+'recon.pkl', 'wb') as f:
        #       pickle.dump(g_feature, f)
