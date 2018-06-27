"""
Main script. Handling the individual modules of the framework, i.e., managing
the tasks passed on by the supervisor.

@authors: Leon Sütfeld, Flemming Brieger
"""

import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import deepnet_networks as net
import deepnet_task_cifar as task

# ##############################################################################
# ################################### MAIN #####################################
# ##############################################################################

if __name__ == '__main__':

    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'n', 'f', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    # parse input arguments
    parser = argparse.ArgumentParser(description='Run network session.')
    parser.add_argument('-path_relative', type=str, help='path from scheduler to main folder. put "./" if they are identical')
    parser.add_argument('-experiment_name', type=str)
    parser.add_argument('-spec_name', type=str)
    parser.add_argument('-run', type=int)
    parser.add_argument('-task', type=str, help='cifar10, cifar100')
    parser.add_argument('-preprocessing', type=str, help='none, ztrans, gcn_zca')
    parser.add_argument('-network', type=str, help='smcn, ...')
    parser.add_argument('-mode', type=str, help='(optional) training, analysis, test')
    parser.add_argument('-n_minibatches', type=int)
    parser.add_argument('-minibatch_size', type=int)
    parser.add_argument('-dropout_keep_probs', type=float, nargs='*', help='keep_probs for all dropout layers')
    parser.add_argument('-dropout_keep_probs_inference', type=float, nargs='*', help='keep_probs for all dropout layers during inference (usually all 1.0)')
    parser.add_argument('-optimizer', type=str)
    parser.add_argument('-lr', type=float, help='main or start learning rate, active in all lr_schedule_types')
    parser.add_argument('-lr_schedule_type', type=str, help='constant, linear, step or decay')
    parser.add_argument('-lr_decay', type=float, help='decay rate for lr, only active when lr_schedule_type==decay')
    parser.add_argument('-lr_lin_min', type=float, help='final lr in linear decay, only active when lr_schedule_type==linear')
    parser.add_argument('-lr_lin_steps', type=int, help='mb at which final lr is reached, only active when lr_schedule_type==linear')
    parser.add_argument('-lr_step_ep', type=int, nargs='*', help='epochs with steps in learning rate scheduler, only active when lr_schedule_type==step')
    parser.add_argument('-lr_step_multi', type=float, nargs='*', help='multiplicative factors applied to lr after corresponding step was reached, only active when lr_schedule_type==step')
    parser.add_argument('-use_wd', type=str2bool, help='weight decay')
    parser.add_argument('-wd_lambda', type=float, help='weight decay lambda')
    parser.add_argument('-create_val_set', type=str2bool, help='enables validation')
    parser.add_argument('-val_set_fraction', type=float, help='fraction of training set used for validation')
    parser.add_argument('-learn_layer_stats', type=str2bool, help='switch on learning of layer stats')
    parser.add_argument('-activation_function', type=str)
    parser.add_argument('-af_weights_pretrained', type=str2bool, help='load pre-trained AF weights (blending weights / swish beta)')
    parser.add_argument('-load_af_weights_from', type=str, help='define the spec name from which to load pre-trained weights (requires that spec to have completed a run with the same run number and saved the weights)')
    parser.add_argument('-norm_blendw_at_init', type=str2bool, help='normalize weights once at init')
    parser.add_argument('-safe_af_ws_n', type=int, help='number of times the af weights are saved during the run')
    parser.add_argument('-safe_all_ws_n', type=int, help='number of times all weights are saved during the run')
    parser.add_argument('-ABU_trainable', type=str2bool, help='enables adaptive blend / AF weights')
    parser.add_argument('-ABU_normalization',  type=str, help='\'unrestricted\', \'normalized\', \'posnormed\', \'absnormed\', \'softmaxed\'')
    parser.add_argument('-swish_beta_trainable', type=str2bool)
    parser.add_argument('-walltime', type=float, help='walltime in minutes (max length of a split), usually 89')
    parser.add_argument('-create_checkpoints', type=str2bool)
    parser.add_argument('-epochs_between_checkpoints', type=int)
    parser.add_argument('-save_af_weights_at_test_mb', type=str2bool)
    parser.add_argument('-save_all_weights_at_test_mb', type=str2bool)
    parser.add_argument('-create_lc_on_the_fly', type=str2bool, help='create learning curves on the fly, i.e., at checkpoints and at the end of the run')
    args = vars(parser.parse_args())

    # training
    if args['mode'] in ['training', 'train', '']:
        NetSettings = net.NetSettings(args)
        TaskSettings = task.TaskSettings(args)
        Paths = task.Paths(TaskSettings)
        TrainingHandler = task.TrainingHandler(TaskSettings, Paths, args)
        TestHandler = task.TestHandler(TaskSettings, Paths, args)
        Rec = task.Recorder(TaskSettings, TrainingHandler, Paths)
        Timer = task.SessionTimer(Paths)
        Network = net.Network(NetSettings, Paths, namescope='Network')
        task.train(TaskSettings, Paths, Network, TrainingHandler, TestHandler, Timer, Rec, args)

    # analysis
    if args['mode'] in ['analysis']:
        TaskSettings = task.TaskSettings(args)
        Paths = task.Paths(TaskSettings)
        task.analysis(TaskSettings, Paths)

    # testing: needs to be updates
    if args['mode'] in ['test', 'testing']:
        NetSettings = net.NetSettings(args)
        TaskSettings = task.TaskSettings(args)
        Paths = task.Paths(TaskSettings)
        TestHandler = task.TestHandler(TaskSettings, Paths)
        Timer = task.SessionTimer(Paths)
        Network = net.Network(NetSettings, Paths, namescope='Network')
        task.test(TaskSettings, Paths, Network, TestHandler)
