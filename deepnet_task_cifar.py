"""
Collection of classes and functions relating to the task (CIFAR10 / CIFAR100).

@authors: Leon Sütfeld, Flemming Brieger
"""

import os
import os.path
import numpy as np
import pickle
import time
import tensorflow as tf
from tensorflow.python.client import timeline
import scipy
import scipy.misc
import scipy.ndimage
import scipy.stats as st
import random
from random import sample
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.mlab as mlab
import json
import csv
from sklearn.utils import shuffle
import psutil
from tensorflow import SessionLog

# ##############################################################################
# ### TASK SETTINGS ############################################################
# ##############################################################################


class TaskSettings(object):


    """Class/ object containing all settings/ options relevant to the task.
    """

    def __init__(self, args):

        assert args['path_relative'] is not None, 'path_relative must be specified.'
        assert args['experiment_name'] is not None, 'experiment_name must be specified.'
        self.path_relative = args['path_relative']
        self.mode = args['mode']
        self.experiment_name = args['experiment_name']
        self.task_name = args['task']

        if self.mode != 'analysis':
            assert args['run'] is not None, 'run must be specified.'
            assert args['spec_name'] is not None, 'spec_name must be specified.'
            assert args['minibatch_size'] is not None, 'minibatch_size must be specified.'
            assert args['dropout_keep_probs_inference'] is not None, 'dropout_keep_probs_inference must be specified for training.'
            self.spec_name = args['spec_name']
            self.run = args['run']
            self.minibatch_size = args['minibatch_size']
            self.dropout_keep_probs_inference = args['dropout_keep_probs_inference']
            if self.mode in ['train','training','']:
                assert args['n_minibatches'] is not None, 'n_minibatches (runtime) must be specified for training.'
                assert args['epochs_between_checkpoints'] is not None, 'epochs_between_checkpoints must be specified for training.'
                assert args['safe_af_ws_n'] is not None, 'safe_af_ws_n must be specified for training.'
                assert args['safe_all_ws_n'] is not None, 'safe_all_ws_n must be specified for training.'
                assert args['create_val_set'] is not None, 'create_val_set must be specified for training.'
                assert args['val_set_fraction'] is not None, 'val_set_fraction must be specified for training.'
                assert args['dropout_keep_probs'] is not None, 'dropout_keep_probs must be specified for training.'
                assert args['lr'] is not None, 'lr must be specified for training.'
                assert args['lr_schedule_type'] is not None, 'lr_schedule_type must be specified for training.'
                assert args['lr_decay'] is not None, 'lr_decay must be specified for training.'
                assert args['lr_lin_min'] is not None, 'lr_lin_min must be specified for training.'
                assert args['lr_lin_steps'] is not None, 'lr_lin_steps must be specified for training.'
                assert args['lr_step_ep'] is not None, 'lr_step_ep must be specified for training.'
                assert args['lr_step_multi'] is not None, 'lr_step_multi must be specified for training.'
                assert args['preprocessing'] is not None, 'preprocessing must be specified for training.'
                assert args['walltime'] is not None, 'walltime must be specified for training.'
                assert args['create_checkpoints'] is not None, 'create_checkpoints must be specified for training.'
                assert args['save_af_weights_at_test_mb'] is not None, 'save_af_weights_at_test_mb must be specified for training.'
                assert args['save_all_weights_at_test_mb'] is not None, 'save_all_weights_at_test_mb must be specified for training.'
                assert args['create_lc_on_the_fly'] is not None, 'create_lc_on_the_fly must be specified for training.'
                self.pre_processing = args['preprocessing']
                self.n_minibatches = args['n_minibatches']
                self.create_val_set = args['create_val_set']
                self.val_set_fraction = args['val_set_fraction']
                self.val_to_val_mbs = 20
                self.walltime = args['walltime']
                self.restore_model = True
                self.dropout_keep_probs = args['dropout_keep_probs']
                self.lr = args['lr']
                self.lr_schedule_type = args['lr_schedule_type']
                self.lr_decay = args['lr_decay']
                self.lr_lin_min = args['lr_lin_min']
                self.lr_lin_steps = args['lr_lin_steps']
                self.lr_step_ep = args['lr_step_ep']
                self.lr_step_multi = args['lr_step_multi']
                # === FILE WRITING OPTIONS =====================================
                self.create_checkpoints = args['create_checkpoints']
                self.epochs_between_checkpoints = args['epochs_between_checkpoints']
                self.checkpoints = self.__get_checkpoints(50000, args['val_set_fraction'], args['minibatch_size'], args['n_minibatches'], args['epochs_between_checkpoints'])
                self.save_af_weights_at_minibatch = np.around(np.linspace(1, args['n_minibatches'], num=args['safe_af_ws_n'], endpoint=True)).tolist()
                self.save_all_weights_at_minibatch = np.around(np.linspace(1, args['n_minibatches'], num=args['safe_all_ws_n'], endpoint=True)).tolist()
                self.save_af_weights_at_test_mb = args['save_af_weights_at_test_mb']
                self.save_all_weights_at_test_mb = args['save_all_weights_at_test_mb']
                self.create_lc_on_the_fly = args['create_lc_on_the_fly']
                self.tracer_minibatches = [0,50,100]
                self.write_summary = True
                self.run_tracer = False # recommended: False (use for efficiency optimization)
                self.keep_saved_datasets_after_run_complete = False # recommended: False (disables auto-deletion of train / val set save files)
        self.__print_overview()

    def __get_checkpoints(self, dataset_size, val_set_fraction, mb_size, training_duration_in_mbs, epochs_between_checkpoints):
        """Calculates the checkpoints = global time steps during training
        where the model is saved (in mb). Returns list of ints containing
        mini-batch counts.
        """
        trainingset_size = int(np.floor(dataset_size*(1-val_set_fraction)))
        mbs_per_epoch = trainingset_size // mb_size
        mbs_per_checkpoint = mbs_per_epoch * epochs_between_checkpoints
        checkpoints = [0]
        while checkpoints[-1] < training_duration_in_mbs:
            checkpoints.append(checkpoints[-1]+mbs_per_checkpoint)
        checkpoints[-1] = training_duration_in_mbs
        return checkpoints

    def __print_overview(self):
        print('')
        print('###########################################')
        print('### TASK SETTINGS OVERVIEW ################')
        print('###########################################')
        print('')
        print(' - experiment name: "%s"' %(self.experiment_name))
        print(' - task name: "%s"' %(self.task_name))
        if self.mode != 'analysis':
            print(' - spec name: "%s"' %(self.spec_name))
            print(' - run: %i' %(self.run))
            if self.mode in ['train','training','']:
                print(' - pre-processing: %s' %(self.pre_processing))
                print(' - # of minibatches per run: %i' %(self.n_minibatches))
                print(' - minibatch size: %i' %(self.minibatch_size))


class Paths(object):


    """Class/ object containing all paths to read from or save to.
    """

    def __init__(self, TaskSettings):

        self.relative = TaskSettings.path_relative # path from scheduler
        # data locations
        self.train_batches = self.relative+'1_data_'+TaskSettings.task_name+'/train_batches/' # original data
        self.test_batches = self.relative+'1_data_'+TaskSettings.task_name+'/test_batches/' # original data
        self.train_set = self.relative+'1_data_'+TaskSettings.task_name+'/train_set/'
        self.test_set = self.relative+'1_data_'+TaskSettings.task_name+'/test_set/'
        self.train_set_ztrans = self.relative+'1_data_'+TaskSettings.task_name+'/train_set_ztrans/'
        self.test_set_ztrans = self.relative+'1_data_'+TaskSettings.task_name+'/test_set_ztrans/'
        self.train_set_gcn_zca = self.relative+'1_data_'+TaskSettings.task_name+'/train_set_gcn_zca/'
        self.test_set_gcn_zca = self.relative+'1_data_'+TaskSettings.task_name+'/test_set_gcn_zca/'
        # save paths (experiment level)
        self.experiment = self.relative+'3_output_cifar/'+str(TaskSettings.experiment_name)+'/'
        self.af_weight_dicts = self.experiment+'0_af_weights/' # corresponds to TaskSettings.save_af_weights
        self.all_weight_dicts = self.experiment+'0_all_weights/' # corresponds to TaskSettings.save_weights
        self.analysis = self.experiment+'0_analysis/' # path for analysis files, not used during training
        self.chrome_tls = self.experiment+'0_chrome_timelines/' # corresponds to TaskSettings.run_tracer
        # save paths (spec / run level)
        if TaskSettings.mode != 'analysis':
            self.summaries = self.experiment+'0_summaries/'+str(TaskSettings.spec_name)+'_'+str(TaskSettings.run)
            self.experiment_spec = self.experiment+str(TaskSettings.spec_name)+'/'
            self.experiment_spec_run = self.experiment_spec+'run_'+str(TaskSettings.run)+'/'
            # sub-paths (run level)
            self.info_files = self.experiment_spec_run
            self.recorder_files = self.experiment_spec_run
            self.incomplete_run_info = self.experiment_spec_run
            self.run_learning_curves = self.experiment_spec_run
            self.run_datasets = self.experiment_spec_run+'datasets/'
            self.models = self.experiment_spec_run+'models/' # corresponds to TaskSettings.save_models

# ##############################################################################
# ### DATA HANDLER #############################################################
# ##############################################################################


class TrainingHandler(object):


    """Object handling the training procedure, i.e., preparing training and
    validation sets. Also saving, restoring and deleting of checkpoint data
    sets.
    """

    def __init__(self, TaskSettings, Paths, args):

        self.TaskSettings = TaskSettings
        self.Paths = Paths
        self.val_mb_counter = 0
        self.train_mb_counter = 0
        self.__load_dataset(args)
        self.__split_training_validation()
        self.__shuffle_training_data_idcs()
        self.n_train_minibatches_per_epoch = int(np.floor((self.n_training_samples / TaskSettings.minibatch_size)))
        self.n_val_minibatches = int(self.n_validation_samples / TaskSettings.minibatch_size)

        self.__print_overview()

    def reset_val(self):
        self.val_mb_counter = 0

    def __load_dataset(self, args):
        """Load (pre-processed) training data set from file defined in "Paths".
        """
        if args['preprocessing'] in ['none', 'tf_ztrans']:
            path_train_set = self.Paths.train_set+self.TaskSettings.task_name+'_trainset.pkl'
            data_dict = pickle.load(open( path_train_set, 'rb'), encoding='bytes')
            self.dataset_images = data_dict['images']
            self.dataset_labels = data_dict['labels']
        if args['preprocessing'] == 'ztrans':
            path_train_set = self.Paths.train_set_ztrans+self.TaskSettings.task_name+'_trainset.pkl'
            data_dict = pickle.load(open( path_train_set, 'rb'), encoding='bytes')
            self.dataset_images = data_dict['images']
            self.dataset_labels = data_dict['labels']
        elif args['preprocessing'] == 'gcn_zca':
            path_train_set = self.Paths.train_set_gcn_zca+self.TaskSettings.task_name+'_trainset.pkl'
            data_dict = pickle.load(open( path_train_set, 'rb'), encoding='bytes')
            self.dataset_images = data_dict['images']
            self.dataset_labels = data_dict['labels']
        else:
            print('[ERROR] requested preprocessing type unknown (%s)' %(args['preprocessing']))

    def __split_training_validation(self):
        """Splits data set into training and validation set. Only call once at
        the beginning of a runself.
        """
        self.dataset_images, self.dataset_labels = shuffle(self.dataset_images, self.dataset_labels)
        self.n_total_samples = int(len(self.dataset_labels))
        self.n_training_samples = self.n_total_samples
        self.n_validation_samples = 0
        self.validation_images = []
        self.validation_labels = []
        self.training_images = self.dataset_images[:]
        self.training_labels = self.dataset_labels[:]
        if self.TaskSettings.create_val_set:
            # determine train/ val split: make n_validation_samples multiple of minibatch_size
            self.n_validation_samples = np.round(self.n_total_samples*self.TaskSettings.val_set_fraction)
            offset = self.n_validation_samples%self.TaskSettings.minibatch_size
            self.n_validation_samples -= offset
            if offset > 0.5*self.TaskSettings.minibatch_size: # add one minibatch to n_validation_samples if that gets it closer to n_total_samples*val_set_fraction
                self.n_validation_samples += self.TaskSettings.minibatch_size
            self.n_training_samples = int(self.n_total_samples - self.n_validation_samples)
            self.n_validation_samples = int(self.n_validation_samples)
            # split dataset
            self.validation_images = self.dataset_images[:self.n_validation_samples]
            self.validation_labels = self.dataset_labels[:self.n_validation_samples]
            self.training_images = self.dataset_images[self.n_validation_samples:]
            self.training_labels = self.dataset_labels[self.n_validation_samples:]

    def __shuffle_training_data_idcs(self):
        self.training_data_idcs = shuffle(list(range(self.n_training_samples)))

    def get_next_train_minibatch(self):
        """Returns the next mini-batch from the training set.
        """
        if self.train_mb_counter % self.n_train_minibatches_per_epoch == 0:
            self.__shuffle_training_data_idcs()
        start_idx = int(self.TaskSettings.minibatch_size*(self.train_mb_counter % self.n_train_minibatches_per_epoch))
        end_idx = int(self.TaskSettings.minibatch_size*((self.train_mb_counter % self.n_train_minibatches_per_epoch)+1))
        mb_idcs = self.training_data_idcs[start_idx:end_idx]
        next_mb_images = [self.training_images[i] for i in mb_idcs]
        next_mb_labels = [self.training_labels[i] for i in mb_idcs]
        self.train_mb_counter += 1
        return next_mb_images, next_mb_labels

    def get_next_val_minibatch(self):
        """Returns the next mini-batch from the validation set.
        """
        start_idx = int(self.TaskSettings.minibatch_size*self.val_mb_counter)
        end_idx = int(self.TaskSettings.minibatch_size*(self.val_mb_counter+1))
        mb_idcs = list(range(start_idx,end_idx))
        next_mb_images = [self.validation_images[i] for i in mb_idcs]
        next_mb_labels = [self.validation_labels[i] for i in mb_idcs]
        self.val_mb_counter += 1
        return next_mb_images, next_mb_labels

    def save_run_datasets(self, print_messages=False):
        """Saves the current spec_name/run's datasets to a file to allow for
        an uncontaminated resume after restart of run. Should only be called
        once at the beginning of a run.
        """
        datasets_dict = { 't_img': self.training_images,
                          't_lab': self.training_labels,
                          'v_img': self.validation_images,
                          'v_lab': self.validation_labels }
        if not os.path.exists(self.Paths.run_datasets):
            os.makedirs(self.Paths.run_datasets)
        file_path = self.Paths.run_datasets+'datasets_'+str(self.TaskSettings.spec_name)+'_run_'+str(self.TaskSettings.run)+'.pkl'
        if os.path.exists(file_path):
            if print_messages:
                print('[MESSAGE] no datasets saved for current spec/run as file already existed: %s'%(file_path))
                print('[MESSAGE] instead restoring datasets from saved file.')
            self.restore_run_datasets()
        else:
            pickle.dump(datasets_dict, open(file_path,'wb'), protocol=3)
            if print_messages:
                print('[MESSAGE] file saved: %s (datasets for current spec/run)'%(file_path))

    def restore_run_datasets(self, print_messages=False):
        """Loads the current spec_name/run's datasets from a file to make
        sure the validation set is uncontaminated after restart of run.
        """
        file_path = self.Paths.run_datasets+'datasets_'+str(self.TaskSettings.spec_name)+'_run_'+str(self.TaskSettings.run)+'.pkl'
        if os.path.exists(file_path):
            datasets_dict = pickle.load(open( file_path, 'rb'), encoding='bytes')
            self.training_images = datasets_dict['t_img']
            self.training_labels = datasets_dict['t_lab']
            self.validation_images = datasets_dict['v_img']
            self.validation_labels = datasets_dict['v_lab']
            if print_messages:
                print('[MESSAGE] spec/run dataset restored from file: %s' %(file_path))
        else:
            raise IOError('\n[ERROR] Couldn\'t restore datasets, stopping run to avoid contamination of validation set.\n')

    def delete_run_datasets(self):
        """Deletes the run datasets at the end of a completed run to save
        storage space.
        """
        file_path = self.Paths.run_datasets+'datasets_'+str(self.TaskSettings.spec_name)+'_run_'+str(self.TaskSettings.run)+'.pkl'
        if os.path.exists(file_path) and not self.TaskSettings.keep_saved_datasets_after_run_complete:
            os.remove(file_path)
            print('[MESSAGE] spec/run dataset deleted to save disk space')
        else:
            print('[MESSAGE] call to delete spec/run dataset had no effect: no dataset file found or TaskSettings.keep_saved_datasets_after_run_complete==True')

    def __print_overview(self):
        print('')
        print('###########################################')
        print('### TRAINING & VALIDATION SET OVERVIEW ####')
        print('###########################################')
        print('')
        print(' - desired split: %.4f (t) / %.4f (v)' %(1.0-self.TaskSettings.val_set_fraction, self.TaskSettings.val_set_fraction))
        print(' - actual split: %.4f (t) / %.4f (v)' %(self.n_training_samples/self.n_total_samples, self.n_validation_samples/self.n_total_samples))
        print(' - # total samples: ' + str(self.n_total_samples))
        print(' - # training samples: ' + str(self.n_training_samples))
        print(' - # validation samples: ' + str(self.n_validation_samples))
        print(' - # validation minibatches: ' + str(self.n_val_minibatches))
        print(' - # minibatches per epoch: ' + str(self.n_train_minibatches_per_epoch))


class TestHandler(object):


    """Object handling the test procedure, i.e., preparing and passing on
    the test set.
    """

    def __init__(self, TaskSettings, Paths, args):

        self.TaskSettings = TaskSettings
        self.Paths = Paths
        self.__load_test_data(args)
        self.n_test_samples = int(len(self.test_images))
        self.n_test_minibatches = int(np.floor(self.n_test_samples/TaskSettings.minibatch_size))
        self.test_mb_counter = 0
        self.__print_overview()

    def __load_test_data(self, args):
        """Load (pre-processed) test data set from file defined in "Paths".
        """
        if args['preprocessing'] in ['none', 'tf_ztrans']:
            path_test_set = self.Paths.test_set+self.TaskSettings.task_name+'_testset.pkl'
            data_dict = pickle.load(open( path_test_set, 'rb'), encoding='bytes')
            self.test_images = data_dict['images']
            self.test_labels = data_dict['labels']
        elif args['preprocessing'] == 'ztrans':
            path_test_set = self.Paths.test_set_ztrans+self.TaskSettings.task_name+'_testset.pkl'
            data_dict = pickle.load(open( path_test_set, 'rb'), encoding='bytes')
            self.test_images = data_dict['images']
            self.test_labels = data_dict['labels']
        elif args['preprocessing'] == 'gcn_zca':
            path_test_set = self.Paths.test_set_gcn_zca+self.TaskSettings.task_name+'_testset.pkl'
            data_dict = pickle.load(open( path_test_set, 'rb'), encoding='bytes')
            self.test_images = data_dict['images']
            self.test_labels = data_dict['labels']
        else:
            print('[ERROR] requested preprocessing type unknown (%s)' %(args['preprocessing']))

    def get_next_test_minibatch(self):
        """Returns the next mini-batch from the test set.
        """
        start_idx = int(self.TaskSettings.minibatch_size*self.test_mb_counter)
        end_idx = int(self.TaskSettings.minibatch_size*(self.test_mb_counter+1))
        mb_idcs = list(range(start_idx,end_idx))
        next_mb_images = [self.test_images[i] for i in mb_idcs]
        next_mb_labels = [self.test_labels[i] for i in mb_idcs]
        self.test_mb_counter += 1
        return next_mb_images, next_mb_labels

    def reset_test(self):
        self.test_mb_counter = 0

    def __print_overview(self):
        print('')
        print('###########################################')
        print('############ TEST SET OVERVIEW ############')
        print('###########################################')
        print('')
        print(' - # test samples: %i (using %i)'%(self.n_test_samples, self.n_test_samples - (self.n_test_samples % self.TaskSettings.minibatch_size)))
        print(' - # test minibatches: '+str(self.n_test_minibatches))

# ##############################################################################
# ### SUPORT CLASSES ###########################################################
# ##############################################################################


class Recorder(object):


    """Keeps track of current state of the training procedure over restored
    runs. Also logs performance figures.
    """

    def __init__(self, TaskSettings, TrainingHandler, Paths):

        self.TaskSettings = TaskSettings
        self.Paths = Paths
        # train
        self.train_loss_hist = []
        self.train_top1_hist = []
        self.train_mb_n_hist = []
        # val
        self.val_loss_hist = []
        self.val_top1_hist = []
        self.val_apc_hist = []
        self.val_af_weights_hist = []
        self.val_mb_n_hist = []
        # test
        self.test_loss_hist = []
        self.test_top1_hist = []
        self.test_apc_hist = []
        self.test_mb_n_hist = []
        # split & time keeping
        self.checkpoints = self.TaskSettings.checkpoints
        self.completed_ckpt_list = []
        self.completed_ckpt_mbs = []
        self.completed_ckpt_epochs = []
        # counter
        self.mb_count_total = 0 # current mb
        self.ep_count_total = 0    # current ep
        self.mbs_per_epoch = TrainingHandler.n_train_minibatches_per_epoch
        self.training_completed = False
        self.test_completed = False

    def feed_train_performance(self, loss, top1):
        """Accessible from the script, to feed the recorder a new set of
        training performance data.
        """
        self.train_loss_hist.append(loss)
        self.train_top1_hist.append(top1)
        self.train_mb_n_hist.append(self.mb_count_total)

    def feed_val_performance(self, loss, top1, apc, af_weights_dict={}):
        """Accessible from the script, to feed the recorder a new set of
        training performance data, and current AF weigths.
        """
        self.val_loss_hist.append(loss)
        self.val_top1_hist.append(top1)
        self.val_apc_hist.append(apc)
        self.val_af_weights_hist.append(af_weights_dict)
        self.val_mb_n_hist.append(self.mb_count_total)

    def feed_test_performance(self, loss, top1, apc, mb=-1):
        """Accessible from the script, to feed the recorder a new set of
        test performance data.
        """
        self.test_loss_hist.append(loss)
        self.test_top1_hist.append(top1)
        self.test_apc_hist.append(apc)
        if mb == -1:
            mb = self.mb_count_total
        self.test_mb_n_hist.append(mb)
        self.test_completed = True

    def mb_plus_one(self):
        """Count global step in recorder. Call once per training step.
        """
        self.mb_count_total += 1
        self.ep_count_total = 1 + (self.mb_count_total-1) // self.mbs_per_epoch

    def mark_end_of_session(self):
        """Marks end of session, because of walltime or because run finished.
        """
        if self.mb_count_total == self.TaskSettings.n_minibatches:
            self.training_completed = True
        else:
            with open(self.Paths.incomplete_run_info+'run_incomplete_mb_'+str(self.mb_count_total)+'.txt', "w+") as text_file:
                print('run stopped incomplete. currently stopped after minibatch '+str(self.mb_count_total), file=text_file)
            print('[MESSAGE] incomplete run file written.')

    def get_running_average(self, measure, window_length=50):
        """Returns running average (sliding window) of performance values.
        """
        assert measure in ['t-loss','t-acc','v-loss','v-acc'], 'requested performance measure unknown.'
        if measure == 't-loss':
            p_measure = self.train_loss_hist
        if measure == 't-acc':
            p_measure = self.train_top1_hist
        if measure == 'v-loss':
            p_measure = self.val_loss_hist
        if measure == 'v-acc':
            p_measure = self.val_top1_hist
        if window_length > len(p_measure):
            window_length = len(p_measure)
        return np.mean(np.array(p_measure)[-window_length:])

    def save_as_dict(self, print_messages=True):
        """Saves the whole recorder in a dict/ pickle file.
        """
        # create dict
        recorder_dict = {}
        # names
        recorder_dict['task_settings'] = self.TaskSettings
        recorder_dict['paths'] = self.Paths
        recorder_dict['experiment_name'] = self.TaskSettings.experiment_name
        recorder_dict['spec_name'] = self.TaskSettings.spec_name
        recorder_dict['run'] = self.TaskSettings.run
        # training performance
        recorder_dict['train_loss_hist'] = self.train_loss_hist
        recorder_dict['train_top1_hist'] = self.train_top1_hist
        recorder_dict['train_mb_n_hist'] = self.train_mb_n_hist
        # validation performance
        recorder_dict['val_loss_hist'] = self.val_loss_hist
        recorder_dict['val_top1_hist'] = self.val_top1_hist
        recorder_dict['val_apc_hist'] = self.val_apc_hist
        recorder_dict['val_af_weights_hist'] = self.val_af_weights_hist
        recorder_dict['val_mb_n_hist'] = self.val_mb_n_hist
        # test performance
        recorder_dict['test_loss'] = self.test_loss_hist
        recorder_dict['test_top1'] = self.test_top1_hist
        recorder_dict['test_apc'] = self.test_apc_hist
        recorder_dict['test_mb_n_hist'] = self.test_mb_n_hist
        # splits
        recorder_dict['self.checkpoints'] = self.checkpoints
        recorder_dict['completed_ckpt_list'] = self.completed_ckpt_list
        recorder_dict['completed_ckpt_mbs'] = self.completed_ckpt_mbs
        recorder_dict['completed_ckpt_epochs'] = self.completed_ckpt_epochs
        # counter
        recorder_dict['mb_count_total'] = self.mb_count_total
        recorder_dict['ep_count_total'] = self.ep_count_total
        recorder_dict['mbs_per_epoch'] = self.mbs_per_epoch
        recorder_dict['training_completed'] = self.training_completed
        recorder_dict['test_completed'] = self.test_completed
        # save dict
        if not os.path.exists(self.Paths.recorder_files):
            os.makedirs(self.Paths.recorder_files)
        savepath = self.Paths.recorder_files
        filename = 'record_'+str(self.TaskSettings.spec_name)+'_run_'+str(self.TaskSettings.run)+'.pkl'
        pickle.dump(recorder_dict, open(savepath+filename,'wb'), protocol=3)
        if print_messages:
            print('================================================================================================================================================================================================================')
            print('[MESSAGE] recorder dict saved after minibatch %i: %s'%(self.mb_count_total, savepath+filename))

    def restore_from_dict(self, Timer, mb_to_restore):
        """Overwrites the values in the recorder with values from a save file.
        """
        # restore dict
        restore_dict_filename = self.Paths.recorder_files+'record_'+str(self.TaskSettings.spec_name)+'_run_'+str(self.TaskSettings.run)+'.pkl'
        if os.path.exists(restore_dict_filename):
            recorder_dict = pickle.load( open( restore_dict_filename, 'rb' ) )
            # names
            if not self.TaskSettings.experiment_name == recorder_dict['experiment_name']:
                print('[WARNING] experiment name in restored performance file and current TaskSettings don\'t match up (restored: %s).'%(recorder_dict['experiment_name']))
            if not self.TaskSettings.spec_name == recorder_dict['spec_name']:
                print('[WARNING] spec name in restored performance file and current TaskSettings don\'t match up (restored: %s).'%(recorder_dict['spec_name']))
            if not self.TaskSettings.run == recorder_dict['run']:
                print('[WARNING] run in restored performance file and current TaskSettings don\'t match up (restored: %s).'%(recorder_dict['run']))
            self.TaskSettings = recorder_dict['task_settings']
            self.Paths = recorder_dict['paths']
            # training performance
            self.train_loss_hist = recorder_dict['train_loss_hist']
            self.train_top1_hist = recorder_dict['train_top1_hist']
            self.train_mb_n_hist = recorder_dict['train_mb_n_hist']
            # validation performance
            self.val_loss_hist = recorder_dict['val_loss_hist']
            self.val_top1_hist = recorder_dict['val_top1_hist']
            self.val_apc_hist = recorder_dict['val_apc_hist'] # accuracy per class
            self.val_af_weights_hist = recorder_dict['val_af_weights_hist']
            self.val_mb_n_hist = recorder_dict['val_mb_n_hist']
            # test performance
            self.test_loss_hist = recorder_dict['test_loss']
            self.test_top1_hist = recorder_dict['test_top1']
            self.test_apc_hist = recorder_dict['test_apc']
            self.test_mb_n_hist = recorder_dict['test_mb_n_hist']
            # splits
            self.checkpoints = recorder_dict['self.checkpoints']
            self.completed_ckpt_list = recorder_dict['completed_ckpt_list']
            self.completed_ckpt_mbs = recorder_dict['completed_ckpt_mbs']
            self.completed_ckpt_epochs = recorder_dict['completed_ckpt_epochs']
            # counter
            self.mb_count_total = recorder_dict['mb_count_total']
            self.ep_count_total = recorder_dict['ep_count_total']
            self.mbs_per_epoch = recorder_dict['mbs_per_epoch']
            self.training_completed = recorder_dict['training_completed']
            self.test_completed = recorder_dict['test_completed']
            self.__set_record_back_to_mb(mb_to_restore)
            # set Timer
            Timer.set_ep_count_at_session_start(self.ep_count_total)
            # return
            print('================================================================================================================================================================================================================')
            print('[MESSAGE] performance recorder restored from file: %s' %(restore_dict_filename))
            return True
        return False

    def __set_record_back_to_mb(self, mb_to_restore):
        """Deletes all entries to the recorder made after the requested mb.
        """
        if len(self.train_mb_n_hist) > 0:
            idx_train_hist = np.argmin(np.abs(np.array(self.train_mb_n_hist) - mb_to_restore))+1
            self.train_mb_n_hist = self.train_mb_n_hist[:idx_train_hist]
            self.train_loss_hist = self.train_loss_hist[:idx_train_hist]
            self.train_top1_hist = self.train_top1_hist[:idx_train_hist]
        # validation performance
        if len(self.val_mb_n_hist) > 0:
            idx_val_hist = np.argmin(np.abs(np.array(self.val_mb_n_hist) - mb_to_restore))+1
            self.val_loss_hist = self.val_loss_hist[:idx_val_hist]
            self.val_top1_hist = self.val_top1_hist[:idx_val_hist]
            self.val_apc_hist = self.val_apc_hist[:idx_val_hist] # accuracy per class
            self.val_af_weights_hist = self.val_af_weights_hist[:idx_val_hist]
            self.val_mb_n_hist = self.val_mb_n_hist[:idx_val_hist]
        # test performance
        if len(self.test_mb_n_hist) > 0:
            idx_test_hist = np.argmin(np.abs(np.array(self.test_mb_n_hist) - mb_to_restore))+1
            self.test_loss_hist = self.test_loss_hist[:idx_test_hist]
            self.test_top1_hist = self.test_top1_hist[:idx_test_hist]
            self.test_apc_hist = self.test_apc_hist[:idx_test_hist]
            self.test_mb_n_hist = self.test_mb_n_hist[:idx_test_hist]
        # splits
        idx_checkpoints = np.argmin(np.abs(np.array(self.checkpoints) - mb_to_restore))+1
        self.completed_ckpt_list = self.completed_ckpt_list[:idx_checkpoints]
        self.completed_ckpt_mbs = self.completed_ckpt_mbs[:idx_checkpoints]
        self.completed_ckpt_epochs = self.completed_ckpt_epochs[:idx_checkpoints]
        # counter
        self.mb_count_total = mb_to_restore
        self.ep_count_total = 1 + (self.mb_count_total-1) // self.mbs_per_epoch
        # print debug
        assert self.mb_count_total == self.train_mb_n_hist[-1], 'something wrong here. %i %i' %(self.mb_count_total, self.train_mb_n_hist[-1])


class SessionTimer(object):


    """Keeps track of run times. Responsible for remaining time estimations
    in print statements.
    """

    def __init__(self, Paths):

        self.session_start_time = 0
        self.session_end_time = 0
        self.session_duration = 0
        self.mb_list = []
        self.mb_duration_list = []
        self.mb_list_val = []
        self.val_duration_list = []
        self.laptime_start = 0
        self.laptime_store = []
        self.session_shut_down = False
        self.last_checkpoint_time = time.time()
        self.ep_count_at_session_start = 0
        # delete any incomplete run files
        if os.path.exists(Paths.incomplete_run_info):
            incomplete_run_files = [f for f in os.listdir(Paths.incomplete_run_info) if ('run_incomplete' in f)]
            for fname in incomplete_run_files:
                os.remove(Paths.incomplete_run_info+fname)

    def set_ep_count_at_session_start(self, ep_count):
        """Called when recorder is reloaded. Needed for remaining time
        predictions.
        """
        self.ep_count_at_session_start = ep_count

    def feed_mb_duration(self, mb, mb_duration):
        """Pushes the duration of the last mini batch to the Timer object.
        Pushed values are stored in the corresponding lists.
        """
        self.mb_list.append(mb)
        self.mb_duration_list.append(mb_duration)

    def feed_val_duration(self, val_duration):
        """Pushes the duration of the last validation to the Timer object.
        Pushed values are stored in the corresponding lists.
        """
        self.val_duration_list.append(val_duration)

    def set_session_start_time(self):
        self.session_start_time = time.time()

    def set_session_end_time(self):
        now = time.time()
        self.session_end_time = now
        self.session_duration = now-self.session_start_time

    def get_mean_mb_duration(self, window_length=500):
        if window_length > len(self.mb_duration_list):
            window_length = len(self.mb_duration_list)
        if (window_length == -1):
            return np.mean(np.array(self.mb_duration_list))
        else:
            return np.mean(np.array(self.mb_duration_list)[-window_length:])

    def get_mean_val_duration(self, window_length=25):
        if window_length > len(self.val_duration_list):
            window_length = len(self.val_duration_list)
        if (window_length == -1):
            return np.mean(np.array(self.val_duration_list))
        else:
            return np.mean(np.array(self.val_duration_list)[-window_length:])

    def laptime(self):
        """Keeps and returns times between calls of this function.
        """
        now = time.time()
        latest_laptime = now-self.laptime_start
        self.laptime_start = now
        self.laptime_store.append(latest_laptime)
        return latest_laptime

    def set_checkpoint_time(self):
        """Called when checkpoint is saved. Used to estimate if next
        checkpoint can be finished in time before wall time hits.
        """
        self.last_checkpoint_time = time.time()

    def end_session(self):
        self.session_shut_down = True

# ##############################################################################
# ### MAIN FUNCTIONS ###########################################################
# ##############################################################################

def train(TaskSettings, Paths, Network, TrainingHandler, TestHandler, Timer, Rec, args):
    """Contains the complete training procedure.
    """

    # CREATE RUN FOLDER
    if not os.path.exists(Paths.experiment_spec_run):
        os.makedirs(Paths.experiment_spec_run)
    print('')

    # SESSION CONFIG AND START
    Timer.set_session_start_time()
    Timer.laptime()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config = config) as sess:
    # tf.set_random_seed(1)

        # INITIALIZATION OF VARIABLES/ GRAPH, SAVER, SUMMARY WRITER
        saver = tf.train.Saver(max_to_keep=100)
        sess.run(tf.global_variables_initializer()) # initialize all variables (after graph construction and session start)
        merged_summary_op = tf.summary.merge_all()
        if TaskSettings.write_summary:
            summary_writer = tf.summary.FileWriter(Paths.summaries, sess.graph)
        n_minibatches_remaining = TaskSettings.n_minibatches

        # RESTORE
        if TaskSettings.restore_model:
            model_restored = False
            # create model weights folder if it does not exist yet
            if not os.path.exists(Paths.models):
                os.makedirs(Paths.models)
            # make list of files in model folder
            files_in_models_folder = [f for f in os.listdir(Paths.models) if (os.path.isfile(os.path.join(Paths.models, f)) and f.startswith('model'))]
            files_in_models_folder = sorted(files_in_models_folder, key=str.lower)
            # find model save file with the highest number (of minibatches) in weight folder
            highest_mb_in_filelist = -1
            for fnum in range(len(files_in_models_folder)):
                if files_in_models_folder[fnum].split('.')[-1].startswith('data'):
                    if int(files_in_models_folder[fnum].split('-')[1].split('.')[0]) > highest_mb_in_filelist:
                        highest_mb_in_filelist = int(files_in_models_folder[fnum].split('-')[1].split('.')[0])
                        restore_meta_filename = files_in_models_folder[fnum].split('.')[0]+'.meta'
                        restore_data_filename = files_in_models_folder[fnum].split('.')[0]
            if highest_mb_in_filelist > -1: # if a saved model file was found
                # restore weights, counter, and performance history (recorder)
                n_minibatches_remaining = TaskSettings.n_minibatches - highest_mb_in_filelist
                TrainingHandler.restore_run_datasets()
                Rec.restore_from_dict(Timer, highest_mb_in_filelist)
                saver.restore(sess, Paths.models+restore_data_filename)
                # marking this as a new session in the summary writer to avoid overlapping plots after model restore
                if TaskSettings.write_summary:
                    summary_writer.add_session_log(SessionLog(status=SessionLog.START), global_step=Rec.mb_count_total)
                model_restored = True
                # print notification
                print('================================================================================================================================================================================================================')
                print('[MESSAGE] model restored from file "' + restore_data_filename + '"')

        # CHECK IF RUN FOLDER EXISTS, ABORT RUN IF IT DOES
        if not TaskSettings.restore_model or not model_restored:
            print('================================================================================================================================================================================================================')
            if os.path.exists(Paths.experiment_spec_run):
                run_info_files = [f for f in os.listdir(Paths.experiment_spec_run) if 'run_info' in f]
                if len(run_info_files) > 0:
                    raise IOError('\n[ERROR] Training aborted to prevent accidental overwriting. A run info file for "%s", run %i already exists, but no model was found to restore from or restore_model was set to False. Delete run folder manually to start fresh.'%(TaskSettings.spec_name, TaskSettings.run))
                else:
                    print('[MESSAGE] no pre-existing folder found for experiment "%s", spec "%s", run %i. starting fresh run.' %(TaskSettings.experiment_name, TaskSettings.spec_name, TaskSettings.run))
            else:
                print('[MESSAGE] no pre-existing folder found for experiment "%s", spec "%s", run %i. starting fresh run.' %(TaskSettings.experiment_name, TaskSettings.spec_name, TaskSettings.run))

        print('================================================================================================================================================================================================================')
        print('=== TRAINING STARTED ===========================================================================================================================================================================================')
        print('================================================================================================================================================================================================================')

        # save overview of the run as a txt file
        __args_to_txt(args, Paths, training_complete_info=str(Rec.training_completed), test_complete_info=str(Rec.test_completed))

        if Rec.mb_count_total == 0:
            if TaskSettings.create_val_set:
                validate(TaskSettings, sess, Network, TrainingHandler, Timer, Rec)
            __save_model_checkpoint(TaskSettings, TrainingHandler, Paths, Network, sess, saver, Rec, recorder=True, tf_model=True, dataset=True, print_messsages=True)

        while n_minibatches_remaining > 0 and Timer.session_shut_down == False:

            # MB START
            time_mb_start = time.time()
            Rec.mb_plus_one()
            imageBatch, labelBatch = TrainingHandler.get_next_train_minibatch()

            # SESSION RUN
            current_lr = __lr_scheduler(TaskSettings, Rec.mb_count_total)
            input_dict = {Network.X: imageBatch, Network.Y: labelBatch, Network.lr: current_lr, Network.dropout_keep_prob: TaskSettings.dropout_keep_probs}
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            # session run call for summaries
            if TaskSettings.write_summary and (Rec.mb_count_total % 20 == 0):
                if TaskSettings.run_tracer and Rec.mb_count_total in TaskSettings.tracer_minibatches:
                    summary = sess.run(merged_summary_op, feed_dict = input_dict, options=run_options, run_metadata=run_metadata)
                else:
                    summary = sess.run(merged_summary_op, feed_dict = input_dict)
                summary_writer.add_summary(summary, Rec.mb_count_total)
            # main session run call for training
            if TaskSettings.run_tracer and Rec.mb_count_total in TaskSettings.tracer_minibatches:
                _, loss, top1 = sess.run([Network.update, Network.loss, Network.top1], feed_dict = input_dict, options=run_options, run_metadata=run_metadata)
            else:
                _, loss, top1 = sess.run([Network.update, Network.loss, Network.top1], feed_dict = input_dict)

            # WRITE TRACER FILE
            if TaskSettings.run_tracer and Rec.mb_count_total in TaskSettings.tracer_minibatches:
                tl = timeline.Timeline(run_metadata.step_stats)
                ctf = tl.generate_chrome_trace_format()
                if not os.path.exists(Paths.chrome_tls):
                    os.makedirs(Paths.chrome_tls)
                with open(Paths.chrome_tls+'timeline_mb'+str(Rec.mb_count_total)+'.json', 'w') as f:
                    f.write(ctf)

            # TIME KEEPING
            mb_duration = time.time()-time_mb_start
            Timer.feed_mb_duration(Rec.mb_count_total, mb_duration)
            t_remaining_minibatches = ((TaskSettings.n_minibatches-Rec.mb_count_total)*Timer.get_mean_mb_duration(window_length=100))/60.
            t_remaining = t_remaining_minibatches
            if TaskSettings.create_val_set:
                t_remaining_validations = ((TaskSettings.n_minibatches-Rec.mb_count_total)*Timer.get_mean_val_duration(window_length=100))/60./TaskSettings.val_to_val_mbs
                t_remaining += t_remaining_validations

            # FEED TRAINING PERFORMANCE TO RECORDER
            Rec.feed_train_performance(loss, top1)

            # RUN VALIDATION AND PRINT
            if TaskSettings.create_val_set:
                if Rec.mb_count_total % TaskSettings.val_to_val_mbs == 0 or Rec.mb_count_total == TaskSettings.n_minibatches:
                    validate(TaskSettings, sess, Network, TrainingHandler, Timer, Rec)
                    t_total = (time.time()-Timer.session_start_time)/60.
                    pid = os.getpid()
                    py = psutil.Process(pid)
                    memoryUse = py.memory_info()[0]/2.**30
                    print('['+str(TaskSettings.spec_name)+'/'+str(TaskSettings.run).zfill(2)+'] mb '+str(Rec.mb_count_total).zfill(len(str(TaskSettings.n_minibatches)))+'/'+str(TaskSettings.n_minibatches)+
                          ' | lr %0.5f' %(current_lr) +
                          ' | acc(t) %05.3f [%05.3f]' %(Rec.train_top1_hist[-1], Rec.get_running_average(measure='t-acc', window_length=100)) +
                          ' | acc(v) %05.3f [%05.3f]' %(Rec.val_top1_hist[-1], Rec.get_running_average(measure='v-acc', window_length=10)) +
                          ' | t_mb %05.3f s' %(Timer.mb_duration_list[-1]) +
                          ' | t_v %05.3f s' %(Timer.val_duration_list[-1]) +
                          ' | t_tot %05.2f m' %(t_total) +
                          ' | t_rem %05.2f m' %(t_remaining) +
                          ' | wt_rem %05.2f m' %(TaskSettings.walltime-t_total) +
                          ' | mem %02.2f GB' %(memoryUse) )
            else:
                if (Rec.mb_count_total)%TaskSettings.val_to_val_mbs == 0 or Rec.mb_count_total == TaskSettings.n_minibatches:
                    t_total = (time.time()-Timer.session_start_time)/60.
                    pid = os.getpid()
                    py = psutil.Process(pid)
                    memoryUse = py.memory_info()[0]/2.**30
                    print('['+str(TaskSettings.spec_name)+'/'+str(TaskSettings.run).zfill(2)+'] mb '+str(Rec.mb_count_total).zfill(len(str(TaskSettings.n_minibatches)))+'/'+str(TaskSettings.n_minibatches)+
                          ' | lr %0.5f' %(current_lr) +
                          ' | acc(t) %05.3f [%05.3f]' %(Rec.train_top1_hist[-1], Rec.get_running_average(measure='t-acc', window_length=100)) +
                          ' | t_mb %05.3f s' %(Timer.mb_duration_list[-1]) +
                          ' | t_tot %05.2f m' %(t_total) +
                          ' | t_rem %05.2f m' %(t_remaining) +
                          ' | wt_rem %05.2f m' %(TaskSettings.walltime-t_total) +
                          ' | mem %02.2f GB' %(memoryUse) )

            # SAVE (AF) WEIGHTS INTERMITTENTLY IF REQUESTED
            if Rec.mb_count_total in TaskSettings.save_af_weights_at_minibatch:
                Network.save_af_weights(sess, Rec.mb_count_total)
            if Rec.mb_count_total in TaskSettings.save_all_weights_at_minibatch:
                Network.save_all_weights(sess, Rec.mb_count_total)

            # SAVE MODEL & DATASETS AT CHECKPOINTS
            if Rec.mb_count_total in TaskSettings.checkpoints or Rec.mb_count_total == TaskSettings.n_minibatches:
                __save_model_checkpoint(TaskSettings, TrainingHandler, Paths, Network, sess, saver, Rec, recorder=True, tf_model=True, dataset=True, print_messsages=True)
                if TaskSettings.create_lc_on_the_fly:
                    visualize_performance(TaskSettings, Paths)

            # minibatch finished
            n_minibatches_remaining -= 1
            assert n_minibatches_remaining + Rec.mb_count_total == TaskSettings.n_minibatches, '[BUG IN CODE] minibatch counting not aligned. (remaining %i, count_total %i, n_mb %s)' %(n_minibatches_remaining, Rec.mb_count_total, TaskSettings.n_minibatches)

            # CHECK IF SESSION (=SPLIT) CAN BE COMPLETED BEFORE WALLTIME
            end_session_now = False
            if Rec.mb_count_total in TaskSettings.checkpoints:
                if TaskSettings.create_checkpoints and TaskSettings.walltime > 0:
                    t_total_this_session = (time.time()-Timer.session_start_time)/60.
                    epochs_in_session_so_far = Rec.ep_count_total-Timer.ep_count_at_session_start
                    checkpoints_in_session_so_far = epochs_in_session_so_far // TaskSettings.epochs_between_checkpoints
                    Timer.set_checkpoint_time()
                    t_since_last_checkpoint = (time.time() - Timer.last_checkpoint_time)/60.
                    t_remaining_until_walltime = TaskSettings.walltime - t_total_this_session
                    t_est_for_next_checkpoint = t_since_last_checkpoint
                    if checkpoints_in_session_so_far > 0:
                        t_per_checkpoint_mean = t_total_this_session / checkpoints_in_session_so_far
                        t_est_for_next_checkpoint = np.maximum(t_per_checkpoint_mean, t_since_last_checkpoint)
                    if t_est_for_next_checkpoint*2.0+1 > t_remaining_until_walltime:
                        end_session_now = True

            # CHECK IF TRAINING IS COMPLETED
            if Rec.mb_count_total == TaskSettings.n_minibatches or end_session_now:
                Rec.mark_end_of_session() # created incomplete_run file or sets Rec.training_completed to True
                Rec.save_as_dict()
                __args_to_txt(args, Paths, training_complete_info=str(Rec.training_completed), test_complete_info=str(Rec.test_completed))
                Timer.set_session_end_time()
                Timer.end_session()

        # if model 60000 is loaded but Rec.training_complted == False
        if Rec.mb_count_total == TaskSettings.n_minibatches:
            Rec.mark_end_of_session() # sets Rec.training_completed to True if applicable
            Rec.save_as_dict()
            __args_to_txt(args, Paths, training_complete_info=str(Rec.training_completed), test_complete_info=str(Rec.test_completed))

    # AFTER TRAINING COMPLETION: SAVE MODEL WEIGHTS AND PERFORMANCE DICT
    if Rec.training_completed and not Rec.test_completed:
        print('================================================================================================================================================================================================================')
        print('=== TRAINING FINISHED -- TOTAL TIME: %04.2f MIN ================================================================================================================================================================' %(Timer.session_duration/60.))
        print('================================================================================================================================================================================================================')

        # RUN EVALUATION PROCEDURE:
        # (1) smooth validation learning curve & evaluate smoothed curve at model checkpoints, extract minibatch number for checkpoint with highest (smoothed) performance
        # (2) perform test with model from this minibatch number & save results
        # (3) delete saved datasets and all models but the tested one
        if len(Rec.val_top1_hist) > 1:
            early_stopping_mb = __early_stopping_minibatch(Rec.val_top1_hist, Rec.val_mb_n_hist, Rec.checkpoints, Paths)
            test_top1, test_loss = test(TaskSettings, Paths, Network, TestHandler, Rec, model_mb=early_stopping_mb)
            __delete_all_models_but_one(Paths, early_stopping_mb)
        else:
            print('[WARNING] no validation performance record available, performing test with model after %i minibatches (full run).' %(TaskSettings.n_minibatches))
            test_top1, test_loss = test(TaskSettings, Paths, Network, TestHandler, Rec, model_mb=TaskSettings.n_minibatches)
            __delete_all_models_but_one(Paths, TaskSettings.n_minibatches)
        Rec.save_as_dict()
        __args_to_txt(args, Paths, training_complete_info=str(Rec.training_completed), test_complete_info=str(Rec.test_completed))
        TrainingHandler.delete_run_datasets()

        print('================================================================================================================================================================================================================')
        print('=== TEST FINISHED -- ACCURACY: %.3f ============================================================================================================================================================================' %(test_top1))
        print('================================================================================================================================================================================================================')

    print('')

def __delete_all_models_but_one(Paths, keep_model_mb):
    """Deletes saved TF models, except for one (keep_model_mb).
    Called only after training is completed. Keeps best model (according to
    post-hoc early stopping), or last model (if no post-hoc early stopping
    is performed).
    """
    # list all files in dir and delete files that don't contain keep_model_mb
    files_in_dir = [f for f in os.listdir(Paths.models) if ('model' in f or 'checkpoint' in f)]
    kill_list = []
    for filename in files_in_dir:
        if not 'model' in filename:
            os.remove(Paths.models+filename)
        else:
            if not str(keep_model_mb) in filename.split('.')[0]:
                os.remove(Paths.models+filename)

def __early_stopping_minibatch(val_data, val_mb, checkpoints, Paths, save_plot=True):
    """Smoothes vector of validation accuracies (val_data) and evaluates
    it at the checkpoints. Returns mb of checkpoint with highest smoothed
    validation accuracy.
    """
    # calculate size of smoothing window & smoothe data
    val_steps_total = len(val_data)
    mb_per_val = val_mb[-1] // val_steps_total
    smoothing_window_mb = np.minimum(2000, np.maximum(val_mb[-1]//10, 500))
    smoothing_window = smoothing_window_mb // mb_per_val
    smooth_val_data = __smooth(val_data, smoothing_window, 3)
    # get intercepts of checkpoints and smooth_val_data
    available_val_data = []
    for i in range(len(checkpoints)):
        val_data_idx_closest_to_save_spot = np.argmin(np.abs(np.array(val_mb)-checkpoints[i]))
        available_val_data.append(smooth_val_data[val_data_idx_closest_to_save_spot])
    # get max of available val data
    max_available_val_data = np.amax(available_val_data)
    minibatch_max_available_val_data = checkpoints[np.argmax(available_val_data)]
    # plot
    if save_plot:
        plt.figure(figsize=(12,8))
        plt.plot(np.array([0, val_mb[-1]]), np.array([np.max(smooth_val_data), np.max(smooth_val_data)]), linestyle='--', linewidth=1, color='black', alpha=0.8)
        for i in range(len(checkpoints)):
            if i == 0:
                plt.plot([checkpoints[i],checkpoints[i]],[0,1], color='blue', linewidth=0.5, label='model save points')
            else:
                plt.plot([checkpoints[i],checkpoints[i]],[0,1], color='blue', linewidth=0.5)
        plt.plot(val_mb, val_data, linewidth=0.5, color='green', alpha=0.5, label='raw validation data')
        plt.plot(val_mb, smooth_val_data, linewidth=2, color='red', label='smoothed validation data')
        plt.plot(checkpoints, available_val_data, linewidth=0.3, color='black', marker='x', markersize=4, alpha=1.0, label='estimated performance at save points')
        plt.plot([minibatch_max_available_val_data], [max_available_val_data], marker='*', markersize=8, color='orange', alpha=1.0, label='max(estimated performance at save point)')
        plt.grid(False)
        plt.ylim([max_available_val_data*0.9,max_available_val_data*1.05])
        plt.legend(loc='lower right', prop={'size': 11})
        plt.tight_layout()
        plt.savefig(Paths.experiment_spec_run+'early_stopping_analysis.png', dpi=300)
    early_stopping_mb = minibatch_max_available_val_data
    return early_stopping_mb

def __smooth(y, smoothing_window, times):
    """Smoothes a vector y with a sliding window extending to both sides
    of the current value.
    """
    for t in range(times):
        smooth = []
        for i in range(len(y)):
            window_start = np.clip(i-(smoothing_window//2), 0, i)
            window_end = np.clip(i+(smoothing_window//2), 0, len(y))
            smooth.append(np.mean(y[window_start:window_end]))
        y = smooth
    return smooth

def __save_model_checkpoint(TaskSettings, TrainingHandler, Paths, Network, sess, saver, Rec, all_weights_dict=False, af_weights_dict=False, recorder=False, tf_model=False, dataset=False, delete_previous=False, print_messsages=False):
    """Takes care of all saves performed at a checkpoint, i.e., TF model,
    dataset (train & val sets), recorder, manually saved weight dicts.
    """
    # model
    if tf_model:
        if not os.path.exists(Paths.models):
            os.makedirs(Paths.models)
        saver.save(sess, Paths.models+'model', global_step=Rec.mb_count_total, write_meta_graph=True) # MEMORY LEAK HAPPENING HERE. ANY IDEAS?
        if print_messsages:
            print('================================================================================================================================================================================================================')
            print('[MESSAGE] model saved after minibatch %i: %s' %(Rec.mb_count_total, Paths.models+'model'))
    # dataset
    if dataset:
        TrainingHandler.save_run_datasets()
    # rec
    if recorder:
        Rec.save_as_dict()
    # all weights
    if all_weights_dict:
        Network.save_all_weights(sess, Rec.mb_count_total)
    # af weights
    if af_weights_dict:
        Network.save_af_weights(sess, Rec.mb_count_total)
    # delete previous
    if delete_previous:
        __delete_previous_savefiles(TaskSettings, Paths, Rec, ['all_weights','af_weights','models'])

def __delete_previous_savefiles(TaskSettings, Paths, Rec, which_files, print_messsages=False):
    """Deletes all but the newest savefiles of models and manually saved
    weights.
    """
    # filenames must be manually defined to match the saved filenames
    current_mb = Rec.mb_count_total
    current_run = TaskSettings.run
    del_list = []
    # all weights
    if 'all_weights' in which_files:
        directory = Paths.all_weight_dicts
        if os.path.isdir(directory):
            files_in_dir = [f for f in os.listdir(directory) if 'all_weights_' in f]
            for f in files_in_dir:
                file_mb = int(f.split('.')[0].split('_')[-1])
                if file_mb < current_mb:
                    del_list.append(directory+f)
    # af weights
    if 'af_weights' in which_files:
        directory = Paths.af_weight_dicts
        if os.path.isdir(directory):
            files_in_dir = [f for f in os.listdir(directory) if 'af_weights_' in f]
            for f in files_in_dir:
                file_mb = int(f.split('.')[0].split('_')[-1])
                file_run = int(f.split('run_')[1].split('_')[0])
                if file_mb < current_mb and file_run == current_run:
                    del_list.append(directory+f)
    # model
    if 'models' in which_files:
        directory = Paths.models
        if os.path.isdir(directory):
            files_in_dir = [f for f in os.listdir(directory) if 'model' in f]
            for f in files_in_dir:
                file_mb = int(f.split('.')[0].split('-')[-1])
                if file_mb < current_mb:
                    del_list.append(directory+f)
    # delete
    for del_file in del_list:
        os.remove(del_file)
        if print_messsages:
            print('[MESSAGE] file deleted: %s'%(del_file))

def validate(TaskSettings, sess, Network, TrainingHandler, Timer, Rec, print_val_apc=False):
    """Contains the complete validation procedure. Requires a running session,
    call from train().
    """
    # VALIDATION START
    time_val_start = time.time()
    TrainingHandler.reset_val()
    # MINIBATCH HANDLING
    loss_store = []
    top1_store = []
    if TaskSettings.task_name == 'cifar10':
        n_classes_in_task = 10
    elif TaskSettings.task_name == 'cifar100':
        n_classes_in_task = 100
    val_confusion_matrix = np.zeros((n_classes_in_task,n_classes_in_task))
    val_count_vector = np.zeros((n_classes_in_task,1))
    while TrainingHandler.val_mb_counter < TrainingHandler.n_val_minibatches:
        # LOAD VARIABLES & RUN SESSION
        val_imageBatch, val_labelBatch = TrainingHandler.get_next_val_minibatch()
        loss, top1, logits = sess.run([Network.loss, Network.top1, Network.logits], feed_dict = { Network.X: val_imageBatch, Network.Y: val_labelBatch, Network.lr: 0., Network.dropout_keep_prob: TaskSettings.dropout_keep_probs_inference})
        # STORE PERFORMANCE
        loss_store.append(loss)
        top1_store.append(top1)
        max_logits = np.argmax(logits, axis=1)
        for entry in range(len(val_labelBatch)):
            val_confusion_matrix[int(val_labelBatch[entry]), max_logits[entry]] += 1.
            val_count_vector[int(val_labelBatch[entry])] += 1.
    # GET MEAN PERFORMANCE OVER VALIDATION MINIBATCHES
    val_loss = np.mean(loss_store)
    val_top1 = np.mean(top1_store)
    val_apc = np.zeros((n_classes_in_task,))
    for i in range(n_classes_in_task):
        val_apc[i] = np.array(val_confusion_matrix[i,i]/val_count_vector[i])[0]
    if print_val_apc and TaskSettings.task_name == 'cifar10':
        print('[MESSAGE] accuracy per class (v): {1: %.3f |' %val_apc[0] + ' 2: %.3f |' %val_apc[1] + ' 3: %.3f |' %val_apc[2] + ' 4: %.3f |' %val_apc[3] + ' 5: %.3f |' %val_apc[4] +
                                                ' 6: %.3f |' %val_apc[5] + ' 7: %.3f |' %val_apc[6] + ' 8: %.3f |' %val_apc[7] + ' 9: %.3f |' %val_apc[8] + ' 10: %.3f}' %val_apc[9])
    # GET AF WEIGHTS
    af_weights_dict = Network.get_af_weights_dict(sess)
    # STORE RESULTS
    Rec.feed_val_performance(val_loss, val_top1, val_apc, af_weights_dict)
    Timer.feed_val_duration(time.time()-time_val_start)

def __test_in_session(TaskSettings, sess, Network, TestHandler, Rec, print_test_apc=False):
    """Contains the complete test procedure. Requires a running session,
    call from train().
    """
    # TEST START
    TestHandler.reset_test()
    # MINIBATCH HANDLING
    loss_store = []
    top1_store = []
    if TaskSettings.task_name == 'cifar10':
        n_classes_in_task = 10
    elif TaskSettings.task_name == 'cifar100':
        n_classes_in_task = 100
    test_confusion_matrix = np.zeros((n_classes_in_task,n_classes_in_task))
    test_count_vector = np.zeros((n_classes_in_task,1))
    while TestHandler.test_mb_counter < TestHandler.n_test_minibatches:
        # LOAD VARIABLES & RUN SESSION
        test_imageBatch, test_labelBatch = TestHandler.get_next_test_minibatch()
        loss, top1, logits = sess.run([Network.loss, Network.top1, Network.logits], feed_dict = { Network.X: test_imageBatch, Network.Y: test_labelBatch, Network.lr: 0., Network.dropout_keep_prob: TaskSettings.dropout_keep_probs_inference})
        # STORE PERFORMANCE
        loss_store.append(loss)
        top1_store.append(top1)
        max_logits = np.argmax(logits, axis=1)
        for entry in range(len(test_labelBatch)):
            test_confusion_matrix[int(test_labelBatch[entry]), max_logits[entry]] += 1.
            test_count_vector[int(test_labelBatch[entry])] += 1.
    # GET MEAN PERFORMANCE OVER VALIDATION MINIBATCHES
    test_loss = np.mean(loss_store)
    test_top1 = np.mean(top1_store)
    test_apc = np.zeros((n_classes_in_task,))
    for i in range(n_classes_in_task):
        test_apc[i] = np.array(test_confusion_matrix[i,i]/test_count_vector[i])[0]
    if print_test_apc and TaskSettings.task_name == 'cifar10':
        print('[MESSAGE] accuracy per class (test): {1: %.3f |' %test_apc[0] + ' 2: %.3f |' %test_apc[1] + ' 3: %.3f |' %test_apc[2] + ' 4: %.3f |' %test_apc[3] + ' 5: %.3f |' %test_apc[4] +
                                                   ' 6: %.3f |' %test_apc[5] + ' 7: %.3f |' %test_apc[6] + ' 8: %.3f |' %test_apc[7] + ' 9: %.3f |' %test_apc[8] + ' 10: %.3f}' %test_apc[9])
    # STORE RESULTS
    Rec.feed_test_performance(test_loss, test_top1, test_apc)
    # RETURN
    return test_top1, test_loss

def test(TaskSettings, Paths, Network, TestHandler, Rec, model_mb=-1, print_results=False, print_messages=False):
    """Contains the complete test procedure. Creates its own session,
    do not call from train().
    """
    # SESSION CONFIG AND START
    time_test_start = time.time()
    TestHandler.reset_test()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config = config) as sess:

        # INITIALIZATION OF VARIABLES & SAVER
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer()) # initialize all variables (after graph construction and session start)

        # RESTORE
        # make list of files in model weights folder
        if not os.path.exists(Paths.models):
            os.makedirs(Paths.models)
        files_in_models_folder = [f for f in os.listdir(Paths.models) if (os.path.isfile(os.path.join(Paths.models, f)) and f.startswith('model'))]
        files_in_models_folder = sorted(files_in_models_folder, key=str.lower)
        restore_data_filename = 'none'
        # find model save file with the highest number (of mbs) in weight folder
        if model_mb == -1:
            highest_mb_in_filelist = -1
            for fname in files_in_models_folder:
                if fname.split('.')[-1].startswith('data') and int(fname.split('-')[1].split('.')[0]) > highest_mb_in_filelist:
                    highest_mb_in_filelist = int(fname.split('-')[1].split('.')[0])
                    restore_data_filename = fname.split('.')[0]
        # or if a certain model_mb was requested, find that one
        else:
            for fname in files_in_models_folder:
                if fname.split('.')[-1].startswith('data') and int(fname.split('-')[1].split('.')[0]) == model_mb:
                    restore_data_filename = fname.split('.')[0]
        # if no model meeting the requirements was found
        if restore_data_filename == 'none':
            raise IOError('\n[ERROR] Test aborted. Couldn\'t find a model of the requested name to test (%s / %s / run %i).\n'%(TaskSettings.experiment_name,TaskSettings.spec_name,TaskSettings.run))
        # restore weights, counter, and performance history (recorder)
        saver.restore(sess, Paths.models+restore_data_filename)
        # print notification
        print('================================================================================================================================================================================================================')
        print('[MESSAGE] model restored from file "' + restore_data_filename + '"')

        # MAIN
        if print_messages:
            print('================================================================================================================================================================================================================')
            print('=== TEST STARTED ===============================================================================================================================================================================================')
            print('================================================================================================================================================================================================================')
        # create stores for multi-batch processing
        loss_store = []
        top1_store = []
        if TaskSettings.task_name == 'cifar10':
            n_classes_in_task = 10
        elif TaskSettings.task_name == 'cifar100':
            n_classes_in_task = 100
        test_confusion_matrix = np.zeros((n_classes_in_task,n_classes_in_task))
        test_count_vector = np.zeros((n_classes_in_task,1))
        while TestHandler.test_mb_counter < TestHandler.n_test_minibatches:
            # LOAD DATA & RUN SESSION
            test_imageBatch, test_labelBatch = TestHandler.get_next_test_minibatch()
            loss, top1, logits = sess.run([Network.loss, Network.top1, Network.logits], feed_dict = { Network.X: test_imageBatch, Network.Y: test_labelBatch, Network.dropout_keep_prob: TaskSettings.dropout_keep_probs_inference}) # why was this set to 0.5?
            # STORE PERFORMANCE
            loss_store.append(loss)
            top1_store.append(top1)
            # ACCURACY PER CLASS CALCULATIONS
            max_logits = np.argmax(logits, axis=1)
            for entry in range(len(test_labelBatch)):
                test_confusion_matrix[int(test_labelBatch[entry]), max_logits[entry]] += 1.
                test_count_vector[int(test_labelBatch[entry])] += 1.
        # GET MEAN PERFORMANCE OVER MINIBATCHES
        test_loss = np.mean(loss_store)
        test_top1 = np.mean(top1_store)
        test_apc = np.zeros((n_classes_in_task,))
        for i in range(n_classes_in_task):
            test_apc[i] = np.array(test_confusion_matrix[i,i] / test_count_vector[i])[0]
        # STORE RESULTS AND PRINT TEST OVERVIEW
        if print_results and TaskSettings.task_name == 'cifar10':
            print('================================================================================================================================================================================================================')
            print('['+str(TaskSettings.spec_name)+'] test' +
                  ' | l(t): %.3f' %test_loss +
                  ' | acc(t): %.3f' %test_top1 +
                  ' | apc(t): { 1: %.3f |' %test_apc[0] + ' 2: %.3f |' %test_apc[1] + ' 3: %.3f |' %test_apc[2] + ' 4: %.3f |' %test_apc[3] + ' 5: %.3f |' %test_apc[4] +
                              ' 6: %.3f |' %test_apc[5] + ' 7: %.3f |' %test_apc[6] + ' 8: %.3f |' %test_apc[7] + ' 9: %.3f |' %test_apc[8] + ' 10: %.3f }' %test_apc[9] +
                  ' | t_test %05.2f' %(time.time()-time_test_start))
        if print_messages:
            print('================================================================================================================================================================================================================')
            print('=== TEST FINISHED ==============================================================================================================================================================================================')
            print('================================================================================================================================================================================================================')

        # SAVE (AF) WEIGHTS IF REQUESTED
        if TaskSettings.save_af_weights_at_test_mb:
            Network.save_af_weights(sess, model_mb)
        if TaskSettings.save_all_weights_at_test_mb:
            Network.save_all_weights(sess, model_mb)

    # STORE RESULTS
    Rec.feed_test_performance(test_loss, test_top1, test_apc, mb=model_mb)
    # RETURN
    return test_top1, test_loss

# ##############################################################################
# ### AUXILIARY FUNCTIONS ######################################################
# ##############################################################################

def analysis(TaskSettings, Paths, make_plot=True, make_hrtf=True):
    """Main analysis function. Analyzes the whole experiment (all specs).
    Call separately from all training and testing procedures.
    """
    experiment_name = Paths.experiment.split('/')[-2]
    # get spec spec names from their respective folder names
    spec_list = [f for f in os.listdir(Paths.experiment) if (os.path.isdir(os.path.join(Paths.experiment,f)) and not f.startswith('0_'))]
    # put the pieces together and call __spec_analysis() for each spec of the experiment
    spec_name_list, n_runs_list, mb_list, test_earlys_mb_mean, test_earlys_mb_std = [], [], [], [], []
    median_run_per_spec, best_run_per_spec, mean_run_per_spec, worst_run_per_spec, std_per_spec = [], [], [], [], []
    t_min_per_spec, t_max_per_spec, t_median_per_spec, t_mean_per_spec, t_var_per_spec, t_std_per_spec, t_standard_error = [], [], [], [], [], [], []
    v_min_per_spec, v_max_per_spec, v_median_per_spec, v_mean_per_spec, v_var_per_spec, v_std_per_spec, v_standard_error = [], [], [], [], [], [], []
    test_min_per_spec, test_max_per_spec, test_median_per_spec, test_mean_per_spec, test_var_per_spec, test_std_per_spec, test_standard_error = [], [], [], [], [], [], []
    print('')
    print('================================================================================================================================================================================================================')
    spec_list_filtered = [] # will only contain specs that actually have completed runs
    for spec_name in spec_list:
        spec_path = Paths.experiment+spec_name+'/'
        spec_perf_dict = __spec_analysis(TaskSettings, Paths, spec_name, spec_path, make_plot=True)
        # general info about spec
        if spec_perf_dict:
            spec_list_filtered.append(spec_name)
            spec_name_list.append(spec_perf_dict['spec_name'])
            n_runs_list.append(spec_perf_dict['n_runs'])
            test_earlys_mb_mean.append(spec_perf_dict['test_mb_n_mean'])
            test_earlys_mb_std.append(spec_perf_dict['test_mb_n_std'])
            # full runs for plotting
            mb_list.append(spec_perf_dict['v_mb'])
            median_run_per_spec.append(spec_perf_dict['v_top1_median_run'])
            best_run_per_spec.append(spec_perf_dict['v_top1_himax_run'])
            mean_run_per_spec.append(spec_perf_dict['v_top1_mean_run'])
            worst_run_per_spec.append(spec_perf_dict['v_top1_lomax_run'])
            std_per_spec.append(spec_perf_dict['v_top1_std_run'])
            # performance info for text output
            t_min_per_spec.append(spec_perf_dict['t_top1_run_max_min'])
            t_max_per_spec.append(spec_perf_dict['t_top1_run_max_max'])
            t_mean_per_spec.append(spec_perf_dict['t_top1_run_max_mean'])
            t_median_per_spec.append(spec_perf_dict['t_top1_run_max_median'])
            t_var_per_spec.append(spec_perf_dict['t_top1_run_max_var'])
            t_std_per_spec.append(spec_perf_dict['t_top1_run_max_std'])
            t_standard_error.append(spec_perf_dict['t_top1_standard_error'])
            v_min_per_spec.append(spec_perf_dict['v_top1_run_max_min'])
            v_max_per_spec.append(spec_perf_dict['v_top1_run_max_max'])
            v_mean_per_spec.append(spec_perf_dict['v_top1_run_max_mean'])
            v_median_per_spec.append(spec_perf_dict['v_top1_run_max_median'])
            v_var_per_spec.append(spec_perf_dict['v_top1_run_max_var'])
            v_std_per_spec.append(spec_perf_dict['v_top1_run_max_std'])
            v_standard_error.append(spec_perf_dict['v_top1_standard_error'])
            test_min_per_spec.append(spec_perf_dict['test_top1_min'])
            test_max_per_spec.append(spec_perf_dict['test_top1_max'])
            test_mean_per_spec.append(spec_perf_dict['test_top1_mean'])
            test_median_per_spec.append(spec_perf_dict['test_top1_median'])
            test_var_per_spec.append(spec_perf_dict['test_top1_var'])
            test_std_per_spec.append(spec_perf_dict['test_top1_std'])
            test_standard_error.append(spec_perf_dict['test_top1_standard_error'])
        print('[MESSAGE] spec analysis complete:', spec_name)
    print('================================================================================================================================================================================================================')

    # BIG FINAL PLOT
    if make_plot and len(mb_list[0])>0:
        n_mb_total = int(np.max(mb_list[0]))
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(1,1,1)
        cmap = matplotlib.cm.get_cmap('nipy_spectral')#('Dark2') ('nipy_spectral')
        color_list = ['black','blue','green','red']
        # layer 1
        for spec in range(len(spec_list_filtered)):
            # ax.plot(mb_list[spec], best_run_per_spec[spec], linewidth=1.5, color=cmap(spec/len(spec_list)), alpha=0.15)
            ax.plot(np.array([0,300]), np.array([np.max(mean_run_per_spec[spec]),np.max(mean_run_per_spec[spec])]), linewidth=1.5, linestyle='-', color=cmap(spec/len(spec_list_filtered)), alpha=0.8)
        # layer 2
        for spec in range(len(spec_list_filtered)):
            ax.plot(mb_list[spec], mean_run_per_spec[spec], linewidth=2.0, color=cmap(spec/len(spec_list_filtered)), label='[%i / m %.4f / v %.6f] %s' %(n_runs_list[spec], 100*test_mean_per_spec[spec], 100*test_var_per_spec[spec], spec_list_filtered[spec]), alpha=0.8)
        # settings
        ax.set_ylim(0.1,1.)
        ax.set_xlim(0.,float(n_mb_total))
        ax.set_xticks(np.arange(0, float(n_mb_total)+.1, float(n_mb_total)/10.))
        ax.set_xticks(np.arange(0, float(n_mb_total)+.1, float(n_mb_total)/40.), minor=True)
        ax.set_yticks(np.arange(0.1, 1.01, .1))
        ax.set_yticks(np.arange(0.1, 1.01, .05), minor=True)
        ax.grid(which='major', linewidth=0.5, linestyle='dotted', color='black', alpha=0.3)
        ax.grid(which='minor', linewidth=0.5, linestyle='dotted', color='black', alpha=0.3)
        ax.set_aspect(float(n_mb_total)+.1/0.901)
        ax.set_title('performance analysis (validation accuracy): %s' %(experiment_name))
        ax.legend(loc=4)
        plt.tight_layout()
        # save
        savepath = Paths.analysis
        plot_filename = 'PA_'+str(experiment_name)+'.png'
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        plt.savefig(savepath+plot_filename, dpi = 120, transparent=False, bbox_inches='tight')
        plt.close()
        print('[MESSAGE] file saved: %s (performance analysis plot for experiment "%s")' %(savepath+plot_filename, experiment_name))

    # HUMAN READABLE TEXT FILE:
    if make_hrtf:
        savepath = Paths.analysis
        hrtf_filename = 'main_performance_analysis.csv'
        with open(savepath+hrtf_filename, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([experiment_name,
                             'n_runs',
                             'test acc [mean %]',
                             'test acc [stderr %]',
                             'test acc [std %]',
                             'test acc [var %]',
                             'test acc [median %]',
                             'test acc [max %]',
                             'test acc [min %]',
                             'val acc [mean %]',
                             'val acc [stderr %]',
                             'val acc [std %]',
                             'val acc [var %]',
                             'val acc [median %]',
                             'val acc [max %]',
                             'val acc [min %]',
                             'train acc [mean %]',
                             'train acc [stderr %]',
                             'train acc [std %]',
                             'train acc [var %]',
                             'train acc [median %]',
                             'train acc [max %]',
                             'train acc [min %]',
                             'test / e.s. mb [mean]',
                             'test / e.s. mb [std]'])
            for spec in range(len(spec_name_list)):
                writer.writerow([spec_name_list[spec],
                                 n_runs_list[spec],
                                 test_mean_per_spec[spec]*100,
                                 test_standard_error[spec]*100,
                                 test_std_per_spec[spec]*100,
                                 test_var_per_spec[spec]*100,
                                 test_median_per_spec[spec]*100,
                                 test_max_per_spec[spec]*100,
                                 test_min_per_spec[spec]*100,
                                 v_mean_per_spec[spec]*100,
                                 v_standard_error[spec]*100,
                                 v_std_per_spec[spec]*100,
                                 v_var_per_spec[spec]*100,
                                 v_median_per_spec[spec]*100,
                                 v_max_per_spec[spec]*100,
                                 v_min_per_spec[spec]*100,
                                 t_mean_per_spec[spec]*100,
                                 t_standard_error[spec]*100,
                                 t_std_per_spec[spec]*100,
                                 t_var_per_spec[spec]*100,
                                 t_median_per_spec[spec]*100,
                                 t_max_per_spec[spec]*100,
                                 t_min_per_spec[spec]*100,
                                 test_earlys_mb_mean[spec],
                                 test_earlys_mb_std[spec]])
        print('[MESSAGE] file saved: %s (performance analysis csv for experiment "%s")' %(savepath+hrtf_filename, experiment_name))

def __spec_analysis(TaskSettings, Paths, spec_name, spec_path, axis_2='af_weights', make_plot=False, return_loss=False):
    """Performes analysis of one spec. Call only from analysis().
    """
    assert (axis_2 in ['af_weights', 'loss']) or axis_2 is None, 'axis_2 needs to be defined as None, \'af_weights\', or \'loss\'.'
    analysis_savepath = Paths.analysis
    # get list of all performance files (pickle dicts) within a spec
    rec_files_list = []
    if os.path.isdir(spec_path):
        run_folder_list = [f for f in os.listdir(spec_path) if (os.path.isdir(os.path.join(spec_path, f)) and ('run_' in f))]
        for run_folder in run_folder_list:
            run_dir = os.path.join(spec_path, run_folder)+'/'
            relative_path_perf_file = [f for f in os.listdir(run_dir) if (('.pkl' in f) and ('record' in f)) ]
            if len(relative_path_perf_file) > 1:
                print('[WARNING] more than one record file found in %s. Using first file (%s).' %(run_dir, relative_path_perf_file[0]))
            if len(relative_path_perf_file) > 0:
                rec_files_list.append(os.path.join(run_dir, relative_path_perf_file[0]))

    # prepare extraction from dicts
    afw_hist_store, run_number_store = [], []
    train_mb_n_hist_store, train_top1_hist_store, train_loss_hist_store = [], [], []
    val_mb_n_hist_store, val_top1_hist_store, val_loss_hist_store = [], [], []
    test_mb_n_hist_store, test_top1_store, test_loss_store = [], [], []

    # criteria for exclusion: incomplete runs. this section makes a list of all completed runs
    n_val_samples_list = []
    for rec_file in rec_files_list:
        rec_dict = pickle.load( open( rec_file, "rb" ) )
        run_number = int(rec_file.split('/')[-2].split('run_')[-1])
        n_val_samples = len(rec_dict['val_mb_n_hist'])
        n_val_samples_list.append(n_val_samples)
    if len(n_val_samples_list) > 0:
        run_length = np.max(n_val_samples_list)
    complete_runs = []
    for rec_file in rec_files_list:
        if os.path.getsize(rec_file) > 0:
            rec_dict = pickle.load( open( rec_file, "rb" ) )
            run_number = int(rec_file.split('/')[-2].split('run_')[-1])
            if len(rec_dict['test_top1']) > 0: # put 'if len(rec_dict['val_mb_n_hist']) == run_length:' for runs without test
                complete_runs.append(rec_file)
            else:
                print('[WARNING] spec %s run %i found to be incomplete (no test accuracy data found), excluded from analysis.'%(spec_name, run_number))
    if len(complete_runs) == 0:
        print('[WARNING] No complete run found for spec %s' %(spec_name))
        return {}

    # extract data from files
    for rec_file in complete_runs:
        rec_dict = pickle.load( open( rec_file, "rb" ) )
        run_number = int(rec_file.split('/')[-2].split('run_')[-1])
        n_val_samples = len(rec_dict['val_mb_n_hist'])
        run_val_mean = np.mean(rec_dict['val_top1_hist'])
        test_t1 = rec_dict['test_top1'][0]
        # exclude bad runs
        if (n_val_samples > 0) and (run_val_mean > .95 or run_val_mean < .3):
            print('[WARNING] bad run detected and excluded from analysis (based on validation performance): %s, run %i' %(spec_name, run_number))
        if test_t1 and test_t1 > 0. and test_t1 < .3:
            print('[WARNING] bad run detected and excluded from analysis (based on test performance): %s, run %i' %(spec_name, run_number))
        else:
            train_mb_n_hist_store.append(np.array(rec_dict['train_mb_n_hist']))
            train_top1_hist_store.append(np.array(rec_dict['train_top1_hist']))
            train_loss_hist_store.append(np.array(rec_dict['train_loss_hist']))
            test_mb_n_hist_store.append(rec_dict['test_mb_n_hist'])
            test_loss_store.append(rec_dict['test_loss'])
            test_top1_store.append(rec_dict['test_top1'])
            run_number_store.append(run_number)
            # special treatment for validation data: remove double mb counts from initial validation after restore
            vmb_hist = rec_dict['val_mb_n_hist']
            vt1_hist = rec_dict['val_top1_hist']
            vlo_hist = rec_dict['val_loss_hist']
            vmb_hist, vt1_hist, vlo_hist = __remove_double_logs(vmb_hist, vt1_hist, vlo_hist)
            val_mb_n_hist_store.append(np.array(vmb_hist))
            val_top1_hist_store.append(np.array(vt1_hist))
            val_loss_hist_store.append(np.array(vlo_hist))

    # if more than one run was done in this spec, build spec performance summary dict
    v_run_max_list = []
    spec_perf_dict = {}
    if len(run_number_store) > 0:
        n_runs = len(run_number_store)

        # get train statistics
        train_mb_n = np.array(train_mb_n_hist_store)
        train_top1 = np.array(train_top1_hist_store)
        train_loss = np.array(train_loss_hist_store)
        t_mb = train_mb_n[0,:]
        t_median_run, t_himax_run, t_lomax_run, _, _, t_mean_run, t_std_run, t_var_run, t_run_max_list, _ = __get_statistics(train_top1)
        t_loss_median_run, _, _, t_loss_himin_run, t_loss_lomin_run, t_loss_mean_run, t_loss_std_run, t_loss_var_run, _, t_loss_run_min_list = __get_statistics(train_loss)

        # get val statistics
        val_mb_n = np.array(val_mb_n_hist_store)
        val_top1 = np.array(val_top1_hist_store)
        val_loss = np.array(val_loss_hist_store)
        v_mb = val_mb_n[0,:]
        v_median_run, v_himax_run, v_lomax_run, _, _, v_mean_run, v_std_run, v_var_run, v_run_max_list, _ = __get_statistics(val_top1)
        v_loss_median_run, _, _, v_loss_himin_run, v_loss_lomin_run, v_loss_mean_run, v_loss_std_run, v_loss_var_run, _, v_loss_run_min_list = __get_statistics(val_loss)

        # put data into dict
        spec_perf_dict = { 'spec_name': spec_name,
                            'n_runs': n_runs,
                            # training loss main
                            't_mb': t_mb,
                            't_loss_run_min_list': t_loss_run_min_list,
                            't_loss_run_min_median': np.median(t_loss_run_min_list),
                            't_loss_run_min_max': np.max(t_loss_run_min_list),
                            't_loss_run_min_min': np.min(t_loss_run_min_list),
                            't_loss_run_min_mean': np.mean(t_loss_run_min_list),
                            't_loss_run_min_var': np.var(t_loss_run_min_list),
                            't_loss_run_min_std': np.std(t_loss_run_min_list),
                            # training loss full runs for plotting
                            't_loss_median_run': t_loss_median_run,
                            't_loss_himin_run': t_loss_himin_run,
                            't_loss_lomin_run': t_loss_lomin_run,
                            't_loss_mean_run': t_loss_mean_run,
                            't_loss_std_run': t_loss_std_run,
                            't_loss_var_run': t_loss_var_run,
                            # training acc main
                            't_top1_run_max_list': t_run_max_list,
                            't_top1_run_max_median': np.median(t_run_max_list),
                            't_top1_run_max_max': np.max(t_run_max_list),
                            't_top1_run_max_min': np.min(t_run_max_list),
                            't_top1_run_max_mean': np.mean(t_run_max_list),
                            't_top1_run_max_var': np.var(t_run_max_list),
                            't_top1_run_max_std': np.std(t_run_max_list),
                            't_top1_standard_error': np.std(t_run_max_list)/np.sqrt(n_runs), # standard error of the mean (SEM)
                            # training acc full runs for plotting
                            't_top1_median_run': t_median_run,
                            't_top1_himax_run': t_himax_run,
                            't_top1_lomax_run': t_lomax_run,
                            't_top1_mean_run': t_mean_run,
                            't_top1_std_run': t_std_run,
                            't_top1_var_run': t_var_run,
                            't_top1_standard_error': t_std_run/np.sqrt(n_runs), # standard error of the mean (SEM)

                            # val loss main
                            'v_mb': v_mb, # training minibatch numbers corresponding to validation runs
                            'v_loss_run_min_list': v_loss_run_min_list,
                            'v_loss_run_min_median': np.median(v_loss_run_min_list),
                            'v_loss_run_min_max': np.max(v_loss_run_min_list),
                            'v_loss_run_min_min': np.min(v_loss_run_min_list),
                            'v_loss_run_min_mean': np.mean(v_loss_run_min_list),
                            'v_loss_run_min_std': np.std(v_loss_run_min_list),
                            'v_loss_run_min_var': np.var(v_loss_run_min_list),
                            # val loss full runs for plotting
                            'v_loss_median_run': v_loss_median_run,
                            'v_loss_himin_run': v_loss_himin_run,
                            'v_loss_lomin_run': v_loss_lomin_run,
                            'v_loss_mean_run': v_loss_mean_run,
                            'v_loss_std_run': v_loss_std_run,
                            'v_loss_var_run': v_loss_var_run,
                            # var acc main
                            'v_top1_run_max_list': v_run_max_list,     # all runs' max values
                            'v_top1_run_max_median': np.median(v_run_max_list),
                            'v_top1_run_max_max': np.max(v_run_max_list),
                            'v_top1_run_max_min': np.min(v_run_max_list),
                            'v_top1_run_max_mean': np.mean(v_run_max_list),
                            'v_top1_run_max_var': np.var(v_run_max_list),
                            'v_top1_run_max_std': np.std(v_run_max_list),
                            'v_top1_standard_error': np.std(v_run_max_list)/np.sqrt(n_runs), # standard error of the mean (SEM)
                            # val acc full runs for plotting
                            'v_top1_median_run': v_median_run, # complete median run
                            'v_top1_himax_run': v_himax_run, # complete best run (highest max)
                            'v_top1_lomax_run': v_lomax_run, # complete worst run (lowest max)
                            'v_top1_mean_run': v_mean_run, # complete mean run
                            'v_top1_var_run': v_var_run, # complete var around mean run
                            'v_top1_std_run': v_std_run, # complete std around mean run

                            # test loss main
                            'test_loss_list': test_loss_store,
                            'test_loss_median': np.median(test_loss_store),
                            'test_loss_max': np.max(test_loss_store),
                            'test_loss_min': np.min(test_loss_store),
                            'test_loss_mean': np.mean(test_loss_store),
                            'test_loss_var': np.var(test_loss_store),
                            'test_loss_std': np.std(test_loss_store),
                            # test acc main
                            'test_top1_list': test_top1_store,
                            'test_top1_median': np.median(test_top1_store),
                            'test_top1_max': np.max(test_top1_store),
                            'test_top1_min': np.min(test_top1_store),
                            'test_top1_mean': np.mean(test_top1_store),
                            'test_top1_var': np.var(test_top1_store),
                            'test_top1_std': np.std(test_top1_store),
                            'test_top1_standard_error': np.std(test_top1_store)/np.sqrt(n_runs), # standard error of the mean (SEM)

                            # early stopping minibatches
                            'test_mb_n_hist': test_mb_n_hist_store,
                            'test_mb_n_min': np.amin(test_mb_n_hist_store),
                            'test_mb_n_max': np.amax(test_mb_n_hist_store),
                            'test_mb_n_mean': np.mean(test_mb_n_hist_store),
                            'test_mb_n_median': np.median(test_mb_n_hist_store),
                            'test_mb_n_std': np.std(test_mb_n_hist_store) }

        # save dict as pickle file
        if not os.path.exists(analysis_savepath):
            os.makedirs(analysis_savepath)
        filename = 'performance_summary_'+spec_name+'.pkl'
        filepath = analysis_savepath + filename
        pickle.dump(spec_perf_dict, open(filepath,'wb'), protocol=3)
        print('[MESSAGE] file saved: %s (performance summary dict for "%s")' %(filepath, spec_name))

        # for the current spec: put all runs' val learning curves in one plot
        n_mb_total = int(np.max(t_mb))
        if make_plot:
            fig = plt.figure(figsize=(10,10))
            ax = fig.add_subplot(1,1,1)
            for r in range(len(train_mb_n_hist_store)):
                colors = [ cm.plasma(x) for x in np.linspace(0.01, 0.99, len(train_mb_n_hist_store))  ]
                ax.plot(val_mb_n_hist_store[r], val_top1_hist_store[r], linewidth=0.5, color=colors[r], label='run '+str(run_number_store[r]), alpha=0.5)
            ax.plot(np.array([0,len(t_mb)]), np.array([np.max(v_himax_run),np.max(v_himax_run)]), linewidth=1., linestyle='--', color='black', alpha=0.5, label='highest max')
            # settings ax
            ax.set_ylim(0.1,1.)
            ax.set_xlim(0.,float(n_mb_total))
            ax.set_xticks(np.arange(0, float(n_mb_total)+.1, float(n_mb_total)/10.))
            ax.set_xticks(np.arange(0, float(n_mb_total)+.1, float(n_mb_total)/40.), minor=True)
            ax.set_yticks(np.arange(0.1, 1.01, .1))
            ax.set_yticks(np.arange(0.1, 1.01, .05), minor=True)
            ax.grid(which='major', linewidth=0.5, linestyle='dotted', color='black', alpha=0.3)
            ax.grid(which='minor', linewidth=0.5, linestyle='dotted', color='black', alpha=0.3)
            ax.set_aspect(float(n_mb_total)+.1/0.901)
            ax.set_title('all runs: %s (%i runs)' %(spec_name, n_runs))
            # ax.legend(loc='lower left', prop={'size': 11})
            # save
            plt.tight_layout()
            filename = 'all_runs_'+str(spec_name)+'.png'
            if not os.path.exists(analysis_savepath):
                os.makedirs(analysis_savepath)
            plt.savefig(analysis_savepath+filename, dpi = 120, transparent=False, bbox_inches='tight')
            plt.close()
            print('[MESSAGE] file saved: %s (performance analysis plot for spec "%s")' %(spec_name, analysis_savepath+filename))

    # if validation was done and more than one run was done in this spec, make spec plot
    if len(v_run_max_list) > 1:
        n_mb_total = int(np.max(t_mb))
        if make_plot:
            fig = plt.figure(figsize=(10,10))
            ax = fig.add_subplot(1,1,1)
            if axis_2 == 'loss':
                ax2 = ax.twinx()
                ax2_ylim = [0.,1.]
                ax2.plot(t_mb, t_loss_mean_run, linewidth=1., color='red', label='training acc', alpha=0.2)
                ax2_ylim = [0.,np.max(t_loss_mean_run)]
                ax2.set_ylim(ax2_ylim[0],ax2_ylim[1])
            ax.plot(t_mb, t_mean_run, linewidth=1., color='green', label='training acc', alpha=0.4)
            # best and worst run
            ax.fill_between(v_mb, v_himax_run, v_lomax_run, linewidth=0., color='black', alpha=0.15)
            ax.plot(np.array([0,n_mb_total]), np.array([np.max(v_himax_run),np.max(v_himax_run)]), linewidth=1., linestyle='--', color='black', alpha=0.6)
            ax.plot(np.array([0,n_mb_total]), np.array([np.max(v_lomax_run),np.max(v_lomax_run)]), linewidth=1., linestyle='--', color='black', alpha=0.6)
            ax.plot(v_mb, v_himax_run, linewidth=1., linestyle='-', color='black', label='best run val acc (max = %.3f)'%(np.max(v_himax_run)), alpha=0.6)
            ax.plot(v_mb, v_lomax_run, linewidth=1., linestyle='-', color='black', label='worst run val acc (max = %.3f)'%(np.max(v_lomax_run)), alpha=0.6)
            # mean run
            ax.plot(np.array([0,len(t_mb)]), np.array([np.max(v_mean_run),np.max(v_mean_run)]), linewidth=1., linestyle='--', color='blue', alpha=0.8)
            ax.plot(v_mb, v_mean_run, linewidth=2., color='blue', label='mean val acc (max = %.3f)'%(np.max(v_mean_run)), alpha=0.8)
            # settings ax
            ax.set_ylim(0.1,1.)
            ax.set_xlim(0.,float(n_mb_total))
            ax.set_xticks(np.arange(0, float(n_mb_total)+.1, float(n_mb_total)/10.))
            ax.set_xticks(np.arange(0, float(n_mb_total)+.1, float(n_mb_total)/40.), minor=True)
            ax.set_yticks(np.arange(0.1, 1.01, .1))
            ax.set_yticks(np.arange(0.1, 1.01, .05), minor=True)
            ax.grid(which='major', linewidth=0.5, linestyle='dotted', color='black', alpha=0.3)
            ax.grid(which='minor', linewidth=0.5, linestyle='dotted', color='black', alpha=0.3)
            ax.set_aspect(float(n_mb_total)+.1/0.901)
            ax.set_title('performance analysis: %s (%i runs)' %(spec_name, n_runs))
            ax.legend(loc='lower left', prop={'size': 11})
            # save
            plt.tight_layout()
            filename = 'performance_analysis_'+str(spec_name)+'.png'
            if not os.path.exists(analysis_savepath):
                os.makedirs(analysis_savepath)
            plt.savefig(analysis_savepath+filename, dpi = 120, transparent=False, bbox_inches='tight')
            plt.close()
            print('[MESSAGE] file saved: %s (performance analysis plot for spec "%s")' %(spec_name, analysis_savepath+filename))

    # return spec_perf_dict
    return spec_perf_dict

def __remove_double_logs(vmb_hist, vt1_hist, vlo_hist):
    """Removes doubles from list of validation performance values (cleanup
    after mishaps in data storage). Called only from __spec_analysis().
    """
    vmb_new, vt1_new, vlo_new = [], [], []
    for i in range(len(vmb_hist)):
        if not vmb_hist[i] in vmb_new:
            vmb_new.append(vmb_hist[i])
            vt1_new.append(vt1_hist[i])
            vlo_new.append(vlo_hist[i])
    return vmb_new, vt1_new, vlo_new

def __get_statistics(data_array):
    """Gets a all full runs of a spec (train / val) and returns stats.
    Do not use this for single values as in test().
    """
    assert len(data_array.shape) == 2, 'data_array must be 2-dimensional: number of runs * number of performance measurement per run'
    if data_array.shape[1] > 0:
        # find himax and lomax run's indices
        run_max_list = []
        run_min_list = []
        for i in range(data_array.shape[0]):
            run = data_array[i,:]
            run_max_list.append(np.max(run))
            run_min_list.append(np.min(run))
        himax_run_idx = np.argmax(np.array(run_max_list))
        lomax_run_idx = np.argmin(np.array(run_max_list))
        himin_run_idx = np.argmax(np.array(run_min_list))
        lomin_run_idx = np.argmin(np.array(run_min_list))
        median_run_idx = np.argsort(np.array(run_min_list))[len(np.array(run_min_list))//2]
        # isolate himax, mean and lomax run
        himax_run = data_array[himax_run_idx,:]
        lomax_run = data_array[lomax_run_idx,:]
        himin_run = data_array[himin_run_idx,:]
        lomin_run = data_array[lomin_run_idx,:]
        median_run = data_array[median_run_idx,:]
        mean_run = np.mean(data_array, axis=0)
        std_run = np.std(data_array, axis=0)
        var_run = np.var(data_array, axis=0)
        # return many things
        return median_run, himax_run, lomax_run, himin_run, lomin_run, mean_run, std_run, var_run, run_max_list, run_min_list
    else:
        return 0, 0, 0, 0, 0, 0, 0, 0, [0], [0]

def visualize_performance(TaskSettings, Paths):
    """Creates learning curves plot from performance recorder files. Can be
    called whenever there's an existing recorder save file.
    """
    # load
    filename = Paths.recorder_files+'record_'+str(TaskSettings.spec_name)+'_run_'+str(TaskSettings.run)+'.pkl'
    performance_dict = pickle.load( open( filename, "rb" ) )
    n_mb_total = int(np.max(performance_dict['train_mb_n_hist']))
    # plot
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1,1,1)
    ax.plot(np.array([0,n_mb_total]), np.array([1.0,1.0]), linewidth=3., color='black', label='100%', alpha=0.5)
    ax.plot(np.array(performance_dict['train_mb_n_hist']), performance_dict['train_top1_hist'], linewidth=1., color='green', label='accuracy train', alpha=0.3) # linewidth=2.,
    ax.plot(np.array([0,n_mb_total]), np.array([np.max(performance_dict['val_top1_hist']),np.max(performance_dict['val_top1_hist'])]), linewidth=1.0, color='blue', label='max val acc (%.3f)'%(np.max(performance_dict['val_top1_hist'])), alpha=0.5)
    ax.plot(np.array(performance_dict['val_mb_n_hist']), performance_dict['val_top1_hist'], linewidth=2., color='blue', label='accuracy val', alpha=0.5)
    ax.plot(np.array([0,n_mb_total]), np.array([0.1,0.1]), linewidth=3., color='red', label='chance level', alpha=0.5)
    ax.set_ylim(0.,1.01)
    ax.set_xlim(0.,float(n_mb_total))
    ax.set_xticks(np.arange(0, n_mb_total, n_mb_total//6))
    ax.set_yticks(np.arange(0., 1.1, .1))
    ax.set_yticks(np.arange(0., 1.1, .02), minor=True)
    ax.grid(which='major', alpha=0.6)
    ax.grid(which='minor', alpha=0.1)
    ax.set_aspect(float(n_mb_total))
    ax.set_title('accuracy over epochs ('+str(TaskSettings.spec_name)+', run_'+str(TaskSettings.run)+')')
    ax.legend(loc=4)
    plt.tight_layout()
    # save
    savepath = Paths.run_learning_curves
    filename = 'learning_curves_'+str(TaskSettings.spec_name)+'_run_'+str(TaskSettings.run)+'.png'
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    plt.savefig(savepath+filename, dpi = 120, transparent=False, bbox_inches='tight')
    plt.close('all')
    print('================================================================================================================================================================================================================')
    print('[MESSAGE] performance figure saved: %s' %(savepath+filename))
    print('================================================================================================================================================================================================================')

def __lr_scheduler(TaskSettings, current_mb):
    """Returns the current learning rate depending on lr schedule, lr settings,
    and current mini batch (training progress).
    """
    if TaskSettings.lr_schedule_type == 'constant':
        return TaskSettings.lr
    if TaskSettings.lr_schedule_type == 'decay':
        lr = TaskSettings.lr * (1. / (1. + TaskSettings.lr_decay * current_mb))
        return lr
    if TaskSettings.lr_schedule_type == 'linear':
        if current_mb < TaskSettings.lr_lin_steps:
            return np.linspace(TaskSettings.lr, TaskSettings.lr_lin_min, num=TaskSettings.lr_lin_steps)[current_mb] #start_lr=0.04, stop_lr=0.00004
        else:
            return TaskSettings.lr_lin_min
    if TaskSettings.lr_schedule_type == 'step':
        mb_per_ep = 50000*(1-TaskSettings.val_set_fraction)//TaskSettings.minibatch_size
        lr_step_mb = np.array(TaskSettings.lr_step_ep) * mb_per_ep
        step_mbs_relative = lr_step_mb - current_mb
        for i in range(len(step_mbs_relative)):
            if step_mbs_relative[i] > 0:
                if i == 0: return TaskSettings.lr
                else: return TaskSettings.lr * TaskSettings.lr_step_multi[i-1]
        return TaskSettings.lr * TaskSettings.lr_step_multi[-1]

def __args_to_txt(args, Paths, training_complete_info='', test_complete_info=''):
    """Writes human-readable text file containing run info (settings) and
    info if the training and testing have been completed for this run.
    """
    # prepare
    experiment_name = args['experiment_name']
    spec_name = args['spec_name']
    run = args['run']
    network = args['network']
    task = args['task']
    mode = args['mode']
    # write file
    if not os.path.exists(Paths.experiment_spec_run):
        os.makedirs(Paths.experiment_spec_run)
    filename = "run_info_"+str(experiment_name)+"_"+str(spec_name)+"_run_"+str(run)+".txt"
    with open(Paths.experiment_spec_run+filename, "w+") as text_file:
        print("{:>35}".format('RUN SETTINGS:'), file=text_file)
        print("", file=text_file)
        print("{:>35} {:<30}".format('experiment_name:',experiment_name), file=text_file)
        print("{:>35} {:<30}".format('spec_name:', spec_name), file=text_file)
        print("{:>35} {:<30}".format('run:', run), file=text_file)
        if len(training_complete_info) > 0:
            print("", file=text_file)
            print("{:>35} {:<30}".format('training complete:', training_complete_info), file=text_file)
        if len(test_complete_info) > 0:
            print("{:>35} {:<30}".format('test complete:', test_complete_info), file=text_file)
        print("", file=text_file)
        print("{:>35} {:<30}".format('network:', network), file=text_file)
        print("{:>35} {:<30}".format('task:', task), file=text_file)
        print("{:>35} {:<30}".format('mode:', mode), file=text_file)
        print("", file=text_file)
        for key in args.keys():
            if args[key] is not None and key not in ['experiment_name','spec_name','run','network','task','mode']:
                print("{:>35} {:<30}".format(key+':', str(args[key])), file=text_file)
