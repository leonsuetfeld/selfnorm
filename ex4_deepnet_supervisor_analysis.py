"""
Script to manage schedule / experiments. Controls repetitive script calls
and particular parameter settings.

@authors: Leon Sütfeld, Flemming Brieger
"""

import time
import os
import sys
import subprocess

"""
################################################################################
### RULES TO CONSTRUCT "AF_SET" ################################################
################################################################################

- must start with the number of AFs, then "_", then the set name
- must contain "swish" if swish AF is used at all
- must contain "scaled" if the scaled version of the respective AF(s) is to be used
- set name is the AFs name (in lowercase) or the name of the AF blend set, e.g., "blendSF7" or "blend9_swish"
- "elu" is called "jelu" ("just elu") to differentiate it from relu and selu
- "lu" (linear unit) is called "linu" for the same reason
- "swish" alone is called "jswish" ("just swish")
- [EXAMPLE] af_set='1_relu'
- [EXAMPLE] af_set='1_jswish'
- [EXAMPLE] af_set='1_jelu_scaled'
- [EXAMPLE] af_set='9_blend9_siwsh_scaled'

################################################################################
################################################################################
################################################################################
"""

os.system("nvidia-smi")

command = "python3 "                + 'deepnet_main.py' + \
          " -path_relative "        + './' + \
          " -experiment_name="      + 'ex4_smcn-bn_ABU_soft' + \
          " -mode="                 + 'analysis' + \
          " -task="                 + 'cifar100'
subprocess.run(command, shell=True)
