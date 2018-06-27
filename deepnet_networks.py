"""
Collection of classes and functions relating to the network.

@authors: Leon Sütfeld, Flemming Brieger
"""

import numpy as np
import tensorflow as tf
import os
import pickle
import json


class NetSettings(object):


    """Class/ object containing all settings/ options relevant to the network.
    """

    def __init__(self, args):

        assert args['experiment_name'] is not None, 'experiment_name must be specified.'
        assert args['spec_name'] is not None, 'spec_name must be specified.'
        assert args['run'] is not None, 'run must be specified.'
        assert args['task'] is not None, 'task must be specified.'
        assert args['minibatch_size'] is not None, 'minibatch_size must be specified.'
        assert args['network'] is not None, 'network must be specified.'
        assert args['activation_function'] is not None, 'activation_function must be specified.'
        assert args['task'] in ['imagenet', 'cifar10', 'cifar100'], 'task must be \'imagenet\', \'cifar10\', or \'cifar100\'.'
        assert args['af_weights_pretrained'] is not None, 'af_weights_pretrained must be specified.'
        assert args['ABU_trainable'] is not None, 'ABU_trainable must be specified.'
        assert args['ABU_normalization'] is not None, 'ABU_normalization must be specified.'
        assert args['swish_beta_trainable'] is not None, 'swish_beta_trainable must be specified.'
        assert args['ABU_normalization'] in ['unrestricted', 'normalized','posnormed','absnormed','softmaxed'], 'requested setting for ABU_normalization unknown.'
        assert args['load_af_weights_from'] is not None, 'load_af_weights_from must be specified.'
        assert args['norm_blendw_at_init'] is not None, 'norm_blendw_at_init must be specified.'
        assert args['optimizer'] is not None, 'optimizer must be specified.'
        assert args['lr'] is not None, 'lr must be specified for training.'
        assert args['lr_schedule_type'] is not None, 'lr_schedule_type must be specified for training.'
        assert args['lr_decay'] is not None, 'lr_decay must be specified for training.'
        assert args['lr_lin_min'] is not None, 'lr_lin_min must be specified for training.'
        assert args['lr_lin_steps'] is not None, 'lr_lin_steps must be specified for training.'
        assert args['lr_step_ep'] is not None, 'lr_step_ep must be specified for training.'
        assert args['lr_step_multi'] is not None, 'lr_step_multi must be specified for training.'
        assert args['use_wd'] is not None, 'use_wd must be specified.'
        assert args['wd_lambda'] is not None, 'wd_lambda must be specified.'
        assert args['learn_layer_stats'] is not None, 'learn_layer_stats must be specified.'

        self.mode = args['mode']
        self.task = args['task']
        self.experiment_name = args['experiment_name']
        self.spec_name = args['spec_name']
        self.run = args['run']
        self.minibatch_size = args['minibatch_size']
        if self.task == 'cifar10':
            self.logit_dims = 10
            self.image_x = 32
            self.image_y = 32
            self.image_z = 3
        elif self.task == 'cifar100':
            self.logit_dims = 100
            self.image_x = 32
            self.image_y = 32
            self.image_z = 3
        self.network_spec = args['network']
        self.optimizer_choice = args['optimizer']
        self.lr = args['lr']
        self.lr_schedule_type = args['lr_schedule_type']
        self.lr_decay = args['lr_decay']
        self.lr_lin_min = args['lr_lin_min']
        self.lr_lin_steps = args['lr_lin_steps']
        self.lr_step_ep = args['lr_step_ep']
        self.lr_step_multi = args['lr_step_multi']
        self.use_wd = args['use_wd']
        self.wd_lambda = args['wd_lambda']
        self.activation_function = args['activation_function']
        self.learn_layer_stats = args['learn_layer_stats']
        self.ABU_trainable = args['ABU_trainable']
        self.ABU_normalization = args['ABU_normalization']
        self.swish_beta_trainable = args['swish_beta_trainable']
        self.af_weights_pretrained = args['af_weights_pretrained']
        self.af_weights_init_from_spec_name = args['load_af_weights_from']
        self.blending_weights_normalize_at_init = args['norm_blendw_at_init']

        self.print_overview()

    def print_overview(self):
        print('')
        print('###########################################')
        print('### NETWORK SETTINGS OVERVIEW #############')
        print('###########################################')
        print('')
        print(' - network spec: %s' %(self.network_spec))
        print(' - input image format: (%i,%i,%i)' %(self.image_x,self.image_y,self.image_z))
        print(' - output dims: %i' %(self.logit_dims))
        if self.mode in ['train','training','']:
            print(' - optimizer: %s' %(self.optimizer_choice))
            print(' - (initial) learning rate: %i' %(self.lr))
            print(' - multiply lr after epochs: %s' %(str(self.lr_step_ep)))
            print(' - multiply lr by: %s' %(str(self.lr_step_multi)))
            print(' - use weight decay: %s' %(str(self.use_wd)))
            print(' - weight decay lambda: %i' %(self.wd_lambda))
            print(' - activation function: %s' %(self.activation_function))
            print(' - pre-trained AF weights: %s' %(str(self.af_weights_pretrained)))
            print(' - ABU trainable: %s' %(str(self.ABU_trainable)))
            print(' - ABU normalization: %s' %(str(self.ABU_normalization)))
            print(' - normalize blending_weights at init: %s' %(str(self.blending_weights_normalize_at_init)))
            if self.activation_function == 'swish':
                print(' - swish beta trainable: %s' %(str(self.swish_beta_trainable)))


class Network(object):


    """Main network class. Contains the complete network, input to output.
    Options for architectures, optimizers, loss / evaluatio functions, etc.
    """

    # ======================== GENERAL NETWORK DEFINITION ======================

    def __init__(self, NetSettings, Paths, namescope=None, reuse=False):

        # SETTINGS / PATHS
        self.NetSettings = NetSettings
        self.Paths = Paths

        # INPUT
        self.X = tf.placeholder(tf.float32, [NetSettings.minibatch_size, 32,32,3], name='images')
        self.Y = tf.placeholder(tf.int64, [None], name='labels')
        self.lr = tf.placeholder(tf.float32, name='learning_rate')
        self.dropout_keep_prob = tf.placeholder(tf.float32, [None], name='dropout_keep_prob')
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.reuse = reuse
        self.Xp = self.X

        # CHOOSE NETWORK ARCHITECTURE
        if self.NetSettings.network_spec == 'smcn':
            self.logits = self.__smcn(namescope='smcn')
        elif self.NetSettings.network_spec == 'smcnLin':
            self.logits = self.__smcnS(namescope='smcnLin')
        elif self.NetSettings.network_spec == 'smcnDeep':
            self.logits = self.__smcn10(namescope='smcnDeep')
        elif self.NetSettings.network_spec == 'smcnBN':
            self.logits = self.__smcnBN(namescope='smcnBN')
        else:
            print('[ERROR] requested network spec unknown (%s)' %(self.NetSettings.network_spec))

        # OBJECTIVE / EVALUATION
        with tf.name_scope('objective'):
            self.xentropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
            if NetSettings.use_wd:
                self.l2 = [tf.nn.l2_loss(v) for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if 'weight' in v.name and not 'blending_weight' in v.name]
                self.weights_norm = tf.reduce_sum(input_tensor = NetSettings.wd_lambda*tf.stack( self.l2 ), name='weights_norm')
                self.loss = self.xentropy + self.weights_norm
            else:
                self.loss = self.xentropy
            self.top1 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(self.logits, self.Y, 1), tf.float32))
            self.top5 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(self.logits, self.Y, 5), tf.float32))
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('top1', self.top1)
            tf.summary.scalar('top5', self.top5)

        # OPTIMIZER
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            with tf.name_scope('optimizer'):
                if self.NetSettings.optimizer_choice == 'Adam':
                    self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
                elif self.NetSettings.optimizer_choice == 'SGD':
                    self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
                elif self.NetSettings.optimizer_choice == 'Momentum':
                    self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=0.9, use_nesterov=False)
                elif self.NetSettings.optimizer_choice == 'Nesterov':
                    self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=0.9, use_nesterov=True)
                self.minimize = self.optimizer.minimize(self.loss)
                varlist = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)]
                self.gradients = self.optimizer.compute_gradients(self.loss, var_list=varlist)
                self.update = self.optimizer.apply_gradients(grads_and_vars=self.gradients)

                for grad, var in self.gradients:
                    summary_label = var.name+'_gradient'
                    summary_label = summary_label.replace('/','_').replace(':','_')
                    self.__variable_summaries(grad, summary_label)

    # ========================== NETWORK ARCHITECTURES =========================

    # SIMPLE MODULAR CONV NET
    def __smcn(self, namescope=None):
        # input block
        self.state = self.__conv2d_layer(layer_input=self.Xp, W_shape=[5,5,3,64], bias_init=0.0, reuse=self.reuse, varscope=namescope+'/conv1')
        self.state = tf.nn.dropout(self.state, keep_prob=self.dropout_keep_prob[0], name=namescope+'/dropout1')
        self.state = self.__conv2d_layer(layer_input=self.state, W_shape=[3,3,64,64], reuse=self.reuse, varscope=namescope+'/conv2')
        # pooling layer
        self.state = tf.nn.max_pool(self.state, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME', name=namescope+'/pool2')
        # standard block
        self.state = self.__conv2d_layer(layer_input=self.state, W_shape=[1,1,64,64], reuse=self.reuse, varscope=namescope+'/conv3')
        self.state = tf.nn.dropout(self.state, keep_prob=self.dropout_keep_prob[0], name=namescope+'/dropout3')
        self.state = self.__conv2d_layer(layer_input=self.state, W_shape=[5,5,64,64], reuse=self.reuse, varscope=namescope+'/conv4')
        # pooling layer
        self.state = tf.nn.max_pool(self.state, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME', name=namescope+'/pool4')
        # output block
        self.state = self.__dense_layer(layer_input=self.state, W_shape=[384], reuse=self.reuse, varscope=namescope+'/dense5')
        self.state = tf.nn.dropout(self.state, keep_prob=self.dropout_keep_prob[0], name=namescope+'/dropout5')
        self.state = self.__dense_layer(layer_input=self.state, W_shape=[192], reuse=self.reuse, varscope=namescope+'/dense6')
        # output layer
        self.logits = self.__dense_layer(layer_input=self.state, W_shape=[self.NetSettings.logit_dims], skip_af=True, reuse=self.reuse, varscope=namescope+'/denseout')
        return self.logits

    # SIMPLE MODULAR CONV NET (MID-SIZED, 10 LAYER VERSION)
    def __smcn10(self, namescope=None):
        # input block
        self.state = self.__conv2d_layer(layer_input=self.Xp, W_shape=[5,5,3,64], bias_init=0.0, reuse=self.reuse, varscope=namescope+'/conv1')
        self.state = tf.nn.dropout(self.state, keep_prob=self.dropout_keep_prob[0], name=namescope+'/dropout1')
        self.state = self.__conv2d_layer(layer_input=self.state, W_shape=[3,3,64,64], reuse=self.reuse, varscope=namescope+'/conv2')
        # pooling layer
        self.state = tf.nn.max_pool(self.state, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME', name=namescope+'/pool2')
        # standard block
        self.state = self.__conv2d_layer(layer_input=self.state, W_shape=[1,1,64,64], reuse=self.reuse, varscope=namescope+'/conv3')
        self.state = tf.nn.dropout(self.state, keep_prob=self.dropout_keep_prob[0], name=namescope+'/dropout3')
        self.state = self.__conv2d_layer(layer_input=self.state, W_shape=[5,5,64,64], reuse=self.reuse, varscope=namescope+'/conv4')
        self.state = self.__conv2d_layer(layer_input=self.state, W_shape=[1,1,64,64], reuse=self.reuse, varscope=namescope+'/conv5')
        self.state = tf.nn.dropout(self.state, keep_prob=self.dropout_keep_prob[0], name=namescope+'/dropout3')
        self.state = self.__conv2d_layer(layer_input=self.state, W_shape=[5,5,64,64], reuse=self.reuse, varscope=namescope+'/conv6')
        self.state = self.__conv2d_layer(layer_input=self.state, W_shape=[1,1,64,64], reuse=self.reuse, varscope=namescope+'/conv7')
        self.state = tf.nn.dropout(self.state, keep_prob=self.dropout_keep_prob[0], name=namescope+'/dropout3')
        self.state = self.__conv2d_layer(layer_input=self.state, W_shape=[5,5,64,64], reuse=self.reuse, varscope=namescope+'/conv8')
        # pooling layer
        self.state = tf.nn.max_pool(self.state, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME', name=namescope+'/pool8')
        # output block
        self.state = self.__dense_layer(layer_input=self.state, W_shape=[384], reuse=self.reuse, varscope=namescope+'/dense9')
        self.state = tf.nn.dropout(self.state, keep_prob=self.dropout_keep_prob[0], name=namescope+'/dropout9')
        self.state = self.__dense_layer(layer_input=self.state, W_shape=[192], reuse=self.reuse, varscope=namescope+'/dense10')
        # output layer
        self.logits = self.__dense_layer(layer_input=self.state, W_shape=[self.NetSettings.logit_dims], skip_af=True, reuse=self.reuse, varscope=namescope+'/denseout')
        return self.logits

    # SIMPLE MODULAR CONV NET (SIMPLIFIED)
    def __smcnS(self, namescope=None):
        # input block
        self.state = self.__conv2d_layer(layer_input=self.Xp, W_shape=[5,5,3,64], bias_init=0.0, reuse=self.reuse, varscope=namescope+'/conv1')
        self.state = self.__conv2d_layer(layer_input=self.state, W_shape=[3,3,64,64], reuse=self.reuse, varscope=namescope+'/conv2')
        # pooling layer
        self.state = tf.nn.avg_pool(self.state, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name=namescope+'/avgpool2')
        # standard block
        self.state = self.__conv2d_layer(layer_input=self.state, W_shape=[1,1,64,64], reuse=self.reuse, varscope=namescope+'/conv3')
        self.state = self.__conv2d_layer(layer_input=self.state, W_shape=[5,5,64,64], reuse=self.reuse, varscope=namescope+'/conv4')
        # pooling layer
        self.state = tf.nn.avg_pool(self.state, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name=namescope+'/avgpool4')
        # output block
        self.state = self.__dense_layer(layer_input=self.state, W_shape=[384], reuse=self.reuse, varscope=namescope+'/dense5')
        self.state = self.__dense_layer(layer_input=self.state, W_shape=[192], reuse=self.reuse, varscope=namescope+'/dense6')
        # output layer
        self.logits = self.__dense_layer(layer_input=self.state, W_shape=[self.NetSettings.logit_dims], skip_af=True, reuse=self.reuse, varscope=namescope+'/denseout')
        return self.logits

    # SIMPLE MODULAR CONV NET (WITH BATCH NORMALIZATION)
    def __smcnBN(self, namescope=None):
        # input block
        self.state = self.__conv2d_layer(layer_input=self.Xp, W_shape=[5,5,3,64], bias_init=0.0, batch_norm=True, reuse=self.reuse, varscope=namescope+'/conv1')
        self.state = self.__conv2d_layer(layer_input=self.state, W_shape=[3,3,64,64], batch_norm=True, reuse=self.reuse, varscope=namescope+'/conv2')
        # pooling layer
        self.state = tf.nn.max_pool(self.state, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME', name=namescope+'/pool2')
        # standard block
        self.state = self.__conv2d_layer(layer_input=self.state, W_shape=[1,1,64,64], batch_norm=True, reuse=self.reuse, varscope=namescope+'/conv3')
        self.state = self.__conv2d_layer(layer_input=self.state, W_shape=[5,5,64,64], batch_norm=True, reuse=self.reuse, varscope=namescope+'/conv4')
        # pooling layer
        self.state = tf.nn.max_pool(self.state, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME', name=namescope+'/pool4')
        # output block
        self.state = self.__dense_layer(layer_input=self.state, W_shape=[384], batch_norm=True, reuse=self.reuse, varscope=namescope+'/dense5')
        self.state = self.__dense_layer(layer_input=self.state, W_shape=[192], batch_norm=True, reuse=self.reuse, varscope=namescope+'/dense6')
        # output layer
        self.logits = self.__dense_layer(layer_input=self.state, W_shape=[self.NetSettings.logit_dims], batch_norm=False, skip_af=True, reuse=self.reuse, varscope=namescope+'/denseout')
        return self.logits

    # ============================= NETWORK LAYERS =============================

    def __dense_layer(self, layer_input, W_shape, b_shape=[-1], batch_norm=False, bias_init=0.1, skip_af=False, reuse=False, varscope=None):
        with tf.variable_scope(varscope, reuse=reuse):
            flat_input = tf.layers.flatten(layer_input)
            input_dims = flat_input.get_shape().as_list()[1]
            W_shape = [input_dims, W_shape[0]]
            W = tf.get_variable('weights', W_shape, initializer=tf.truncated_normal_initializer(stddev=tf.sqrt(2./(W_shape[0]*W_shape[1]))))
            self.__variable_summaries(W, 'weights')
            iState = tf.matmul(flat_input, W)
            if b_shape == [-1]:
                b_shape = [W_shape[-1]]
            if b_shape != [0]:
                b = tf.get_variable('biases', b_shape, initializer=tf.constant_initializer(bias_init))
                self.__variable_summaries(b, 'biases')
                iState += b
            return self.__activate(iState, varscope, skip_af=skip_af, batch_norm=batch_norm)

    def __conv2d_layer(self, layer_input, W_shape, b_shape=[-1], strides=[1,1,1,1], padding='SAME', bias_init=0.1, batch_norm=False, skip_af=False, reuse=False, varscope=None):
        with tf.variable_scope(varscope, reuse=reuse):
            W_initializer = tf.truncated_normal_initializer(stddev=tf.sqrt(2./(W_shape[0]*W_shape[1]*W_shape[2])))
            W = tf.get_variable('weights', W_shape, initializer=W_initializer)
            self.__variable_summaries(W, 'weights')
            iState = tf.nn.conv2d(layer_input, W, strides, padding)
            if b_shape == [-1]:
                b_shape = [W_shape[-1]]
            if b_shape != [0]:
                b = tf.get_variable('biases', b_shape, initializer=tf.constant_initializer(bias_init))
                self.__variable_summaries(b, 'biases')
                iState += b
            return self.__activate(iState, varscope, skip_af=skip_af, batch_norm=batch_norm)

    # ============================== ACTIVATIONS ===============================

    def __activate(self, preact, layer_name, skip_af=False, batch_norm=False):
        """Contains the complete activation function logic. Adaptive scaling,
        ABUs, normalization, all in here.
        """
        assert not (self.NetSettings.learn_layer_stats and batch_norm), 'batch normalization and learned normalization prohibited to run in parralel.'
        assert self.NetSettings.ABU_normalization in ['unrestricted','normalized','posnormed','absnormed','softmaxed'], 'specified ABU normalization unknown.'

        # NORMALIZATION
        self.__variable_summaries(preact, 'preact', stats=['mean', 'std', 'hist'])
        if batch_norm:
            preact = tf.layers.batch_normalization(preact, name='batchnorm', training=True)
        if self.NetSettings.learn_layer_stats:
            alpha_initializer = tf.fill([1], 1.0)
            beta_initializer = tf.fill([1], 0.0)
            alpha = tf.get_variable("alpha", initializer=alpha_initializer)
            beta = tf.get_variable("beta", initializer=beta_initializer)
            self.__variable_summaries(alpha, 'alpha', stats=['mean'])
            self.__variable_summaries(beta, 'beta', stats=['mean'])
            preact = alpha * (preact + beta)

        if skip_af:
            activation = preact
        else:
            # ACTIVATION FUNCTIONS
            if self.NetSettings.activation_function == 'relu':
                activation = tf.nn.relu(preact)
            elif self.NetSettings.activation_function == 'elu':
                activation = tf.nn.elu(preact)
            elif self.NetSettings.activation_function == 'tanh':
                activation = tf.nn.tanh(preact)
            elif self.NetSettings.activation_function == 'swish':
                if self.NetSettings.af_weights_pretrained:
                    swish_beta_init = self.__get_predefined_af_weights(layer_name, w_type='swish_beta')
                else:
                    swish_beta_init = tf.fill([1], 1.0)
                swish_beta = tf.get_variable("swish_beta", trainable=self.NetSettings.swish_beta_trainable, initializer=swish_beta_init)
                self.__variable_summaries(swish_beta, 'swish_beta', stats=['mean'])
                activation = preact * tf.nn.sigmoid(swish_beta*preact)
            elif self.NetSettings.activation_function == 'id':
                activation = preact
            elif self.NetSettings.activation_function == 'selu':
                activation = tf.nn.selu(preact)
            elif self.NetSettings.activation_function == 'ABU':
                # INITIALIZE BLENDING WEIGHTS & SWISH BETA
                if self.NetSettings.af_weights_pretrained:
                    blending_weights_initializer = self.__get_predefined_af_weights(layer_name, w_type='blend')
                    swish_beta_init = self.__get_predefined_af_weights(layer_name, w_type='swish_beta')
                else:
                    blending_weights_initializer = tf.fill([5], 1/5.0)
                    swish_beta_init = tf.fill([1], 1.0)
                blending_weights_raw = tf.get_variable("blending_weights_raw", trainable=self.NetSettings.ABU_trainable, initializer=blending_weights_initializer)
                swish_beta = tf.get_variable("swish_beta", trainable=self.NetSettings.swish_beta_trainable, initializer=swish_beta_init)
                self.__variable_summaries(blending_weights_raw, 'blending_weights', stats=['mean', 'min', 'max'])
                self.__variable_summaries(swish_beta, 'swish_beta', stats=['mean'])
                # NORMALIZE
                if self.NetSettings.ABU_normalization == 'unrestricted':
                    blending_weights = blending_weights_raw
                elif self.NetSettings.ABU_normalization == 'normalized':
                    blending_weights = tf.divide(blending_weights_raw, tf.reduce_sum(blending_weights_raw, keep_dims=True))
                elif self.NetSettings.ABU_normalization == 'absnormed':
                    blending_weights = tf.divide(blending_weights_raw, tf.reduce_sum(tf.abs(blending_weights_raw), keep_dims=True))
                elif self.NetSettings.ABU_normalization == 'posnormed':
                    blending_weights = tf.divide(tf.clip_by_value(blending_weights_raw, 0.0001, 1000.0), tf.reduce_sum(tf.clip_by_value(blending_weights_raw, 0.0001, 1000.0), keep_dims=True))
                    blending_weights = tf.divide(blending_weights_raw, tf.reduce_sum(blending_weights_raw, keep_dims=True))
                elif self.NetSettings.ABU_normalization == 'softmaxed':
                    blending_weights = tf.exp(blending_weights_raw)
                    blending_weights = tf.divide(blending_weights, tf.reduce_sum(blending_weights, keep_dims=True))
                # ACTIVATE
                act_relu = tf.nn.relu(preact)
                act_elu = tf.nn.elu(preact)
                act_tanh = tf.nn.tanh(preact)
                act_swish = preact * tf.nn.sigmoid(swish_beta*preact)
                act_linu = preact
                if batch_norm:
                    act_relu = tf.layers.batch_normalization(act_relu, name='batchnorm_relu', training=True)
                    act_elu = tf.layers.batch_normalization(act_elu, name='batchnorm_elu', training=True)
                    act_tanh = tf.layers.batch_normalization(act_tanh, name='batchnorm_tanh', training=True)
                    act_swish = tf.layers.batch_normalization(act_swish, name='batchnorm_swish', training=True)
                    act_linu = tf.layers.batch_normalization(act_linu, name='batchnorm_linu', training=True)
                activation = tf.add_n([blending_weights[0] * act_relu,
                                       blending_weights[1] * act_elu,
                                       blending_weights[2] * act_tanh,
                                       blending_weights[3] * act_swish,
                                       blending_weights[4] * act_linu])

        self.__variable_summaries(activation, 'activation', stats=['mean', 'std', 'hist'])
        return activation

    def __get_predefined_af_weights(self, layer_name, w_type='blend'):
        """Loads af weights from a manually saved pickle dict file, converts
        to tensor, and returns that tensor.
        """
        # define which file to load the blending weights from
        blending_weights_files = [f for f in os.listdir(self.Paths.af_weight_dicts) if '.pkl' in f and 'run_'+str(self.NetSettings.run) in f and self.NetSettings.af_weights_init_from_spec_name in f]
        if len(blending_weights_files) > 0:
            file_num = 0
            # if there are multiple save files from different minibatches,
            # find largest mb in list
            if len(blending_weights_files) > 1:
                highest_mb_count = 0
                for i in range(len(blending_weights_files)):
                    mb_count = int(blending_weights_files[i].split('mb_')[-1].split('.')[0])
                    if mb_count > highest_mb_count:
                        highest_mb_count = mb_count
                        file_num = i
            blending_weights_file = blending_weights_files[file_num]
            af_weights_dict = pickle.load(open(self.Paths.af_weight_dicts+blending_weights_file,'rb'))
            print('[MESSAGE] (predefined) blending weights loaded from file "%s"' %(self.Paths.af_weight_dicts+blending_weights_file))
        else:
            raise ValueError('Could not find af weights file containing "%s" and "run_%i"' %(self.NetSettings.af_weights_init_from_spec_name, self.NetSettings.run))
        # extract from dict
        if w_type == 'blend':
            if layer_name+'/blending_weights:0' in af_weights_dict:
                layer_blending_weights = af_weights_dict[layer_name+'/blending_weights:0']
            elif layer_name+'/blending_weights_raw:0' in af_weights_dict:
                layer_blending_weights = af_weights_dict[layer_name+'/blending_weights_raw:0']
            else:
                raise ValueError('[ERROR] Could not find the correct set of weights in weight dict loaded from file.')
            if self.NetSettings.blending_weights_normalize_at_init:
                layer_blending_weights /= tf.reduce_sum(layer_blending_weights)
            print('[MESSAGE] (predefined) blending weights for layer',layer_name,':',layer_blending_weights)
            return tf.convert_to_tensor(layer_blending_weights)
        if w_type == 'swish_beta':
            print('[MESSAGE] (predefined) swish beta for layer',layer_name,':',af_weights_dict[layer_name+'/swish_beta:0'])
            return tf.convert_to_tensor(af_weights_dict[layer_name+'/swish_beta:0'])

    def get_af_weights_dict(self, sess):
        """Returns all current AF weights (scaling/ blending, Swish beta) as
        dict.
        """
        af_weights_dict = {}
        # if trainable blending weights available put in list
        if len([v for v in tf.trainable_variables() if 'blending_weights' in v.name]) > 0:
            for name in [v.name for v in tf.trainable_variables() if 'blending_weights' in v.name]:
                af_weights_dict[name] = list(sess.run(name))
        # if trainable swish betas available put in list
        if len([v for v in tf.trainable_variables() if 'swish_beta' in v.name]) > 0:
            for name in [v.name for v in tf.trainable_variables() if 'swish_beta' in v.name]:
                af_weights_dict[name] = list(sess.run(name))
        return af_weights_dict

    def save_af_weights(self, sess, mb_count, print_messages=False):
        """Saves current AF weights (scaling / blending weights, Swish beta)
        as pickle dict.
        """
        af_weights_dict_pkl = {}
        af_weights_dict_json = {}
        # if trainable blending weights available put in dicts
        if len([v for v in tf.trainable_variables() if 'blending_weights' in v.name]) > 0:
            for name in [v.name for v in tf.trainable_variables() if 'blending_weights' in v.name]:
                af_weights_dict_pkl[name] = list(sess.run(name))
                af_weights_dict_json[name] = str(list(sess.run(name)))
        # if trainable swish betas available put in dicts
        if len([v for v in tf.trainable_variables() if 'swish_beta' in v.name]) > 0:
            for name in [v.name for v in tf.trainable_variables() if 'swish_beta' in v.name]:
                af_weights_dict_pkl[name] = list(sess.run(name))
                af_weights_dict_json[name] = str(list(sess.run(name)))
        # save dicts in files
        if len(af_weights_dict_pkl.keys()) > 0:
            if not os.path.exists(self.Paths.af_weight_dicts):
                os.makedirs(self.Paths.af_weight_dicts)
            file_name = 'af_weights_'+self.NetSettings.spec_name+'_run_'+str(self.NetSettings.run)+'_mb_'+str(mb_count)
            pickle.dump(af_weights_dict_pkl, open(self.Paths.af_weight_dicts+file_name+'.pkl', 'wb'),protocol=3)
            json.dump(af_weights_dict_json, open(self.Paths.af_weight_dicts+file_name+'.json', 'w'), sort_keys=True, indent=4)
            if print_messages:
                print('[MESSAGE] file saved: %s (af weights)' %(self.Paths.af_weight_dicts+file_name+'.pkl'))
        else:
            if print_messages:
                print('[WARNING] no trainable variables "blending_weights" or "swish_beta" found - no af weights saved.')

    def save_all_weights(self, sess, mb_count, print_messages=False):
        """Saves all current weights (conv, dense, scaling/ blending, Swish
        beta) as pickle dict.
        """
        if len([v for v in tf.trainable_variables()]) > 0:
            # create dict of all trainable variables in network
            filter_dict = {}
            for name in [v.name for v in tf.trainable_variables()]:
                filter_dict[name] = sess.run(name)
            # save dict in pickle file
            if not os.path.exists(self.Paths.all_weight_dicts):
                os.makedirs(self.Paths.all_weight_dicts)
            filename = 'all_weights_'+self.NetSettings.spec_name+'_run_'+str(self.NetSettings.run)+'_mb_'+str(mb_count)+'.pkl'
            pickle.dump(filter_dict, open(self.Paths.all_weight_dicts+filename,'wb'), protocol=3)
            if print_messages:
                print('[MESSAGE] file saved: %s (all weights)' %(self.Paths.all_weight_dicts+filename))
        else:
            if print_messages:
                print('[WARNING] no trainable variables found - no weights saved.')

    def __variable_summaries(self, var, label, stats=['all']):
        """Attaches relevant statistics of variables to tf.summaries
        for TensorBoard visualization.
        """
        assert isinstance(label, str), 'label must be of type str.'
        with tf.name_scope(label):
            mean = tf.reduce_mean(var)
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            if 'mean' in stats or 'all' in stats:
                tf.summary.scalar('mean', mean)
            if 'std' in stats or 'all' in stats:
                tf.summary.scalar('stddev', stddev)
            if 'max' in stats or 'all' in stats:
                tf.summary.scalar('max', tf.reduce_max(var))
            if 'min' in stats or 'all' in stats:
                tf.summary.scalar('min', tf.reduce_min(var))
            if 'hist' in stats or 'all' in stats:
                tf.summary.histogram('histogram', var)

    # ========================= NOT IN USE / DEBUGGING =========================

    def __getLayerShape(self, state, print_shape=False):
        N = int(state.get_shape()[1]) # feature map X
        M = int(state.get_shape()[2]) # feature map Y
        L = int(state.get_shape()[3]) # feature map Z
        if print_shape:
            print("layer shape: ",N,M,L)
        return [N,M,L]

    def __print_all_trainable_var_names(self):
        if len([v for v in tf.trainable_variables()]) > 0:
            for name in [v.name for v in tf.trainable_variables()]:
                print(name)
