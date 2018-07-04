"""
Script to manage the schedule / experiments. Controls repetitive script calls
and paricular parameter settings.

@authors: Leon Sütfeld, Flemming Brieger
"""

import time
import os
import sys
import subprocess

for run in range(1):
    os.system("nvidia-smi")
    command = "python3 "                        + 'deepnet_main.py' + \
              " -path_relative "                + './' + \
              " -experiment_name "              + 'ABU_flex_test' + \
              " -spec_name "                    + 'ABU-swish-tanh-id' + \
              " -run "                          + '2' + \
              " -task="                         + 'cifar10' + \
              " -preprocessing="                + 'ztrans' +\
              " -network="                      + 'smcn' + \
              " -mode "                         + 'training' + \
              " -n_minibatches "                + '3000' + \
              " -minibatch_size "               + '256' + \
              " -dropout_keep_probs "           + '0.5' + \
              " -dropout_keep_probs_inference " + '1.0' + \
              " -optimizer "                    + 'Adam' + \
              " -lr "                           + '0.001' + \
              " -lr_schedule_type "             + 'constant' + \
              " -lr_decay "                     + '0.00004' + \
              " -lr_lin_min "                   + '0.0004' + \
              " -lr_lin_steps "                 + '3000' + \
              " -lr_step_ep "                   + '200 250 300' + \
              " -lr_step_multi "                + '0.1 0.01 0.001' + \
              " -use_wd "                       + 'False' + \
              " -wd_lambda "                    + '0.01' + \
              " -create_val_set "               + 'True' + \
              " -val_set_fraction "             + '0.05' + \
              " -learn_layer_stats "            + 'True' +\
              " -activation_function "          + 'ABU-swish-tanh-id' +\
              " -af_weights_pretrained "        + 'False' + \
              " -load_af_weights_from "         + 'none' + \
              " -norm_blendw_at_init "          + 'False' + \
              " -safe_af_ws_n "                 + '10' + \
              " -safe_all_ws_n "                + '2' + \
              " -ABU_trainable "                + 'True' + \
              " -ABU_normalization "            + 'unrestricted' + \
              " -swish_beta_trainable "         + 'True' + \
              " -walltime "                     + '1000.0' + \
              " -create_checkpoints "           + 'True' + \
              " -epochs_between_checkpoints "   + '3' + \
              " -save_af_weights_at_test_mb "   + 'True' + \
              " -save_all_weights_at_test_mb "  + 'True' + \
              " -create_lc_on_the_fly "         + 'True'

    subprocess.run(command, shell=True)
