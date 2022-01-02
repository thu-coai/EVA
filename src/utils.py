# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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

"""Utilities for logging and serialization"""

import os
import random
import numpy as np
import torch

from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
import mpu
import deepspeed


def print_rank_0(message):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)


def print_args(args):
    """Print arguments."""

    print('arguments:', flush=True)
    for arg in vars(args):
        dots = '.' * (29 - len(arg))
        print('  {} {} {}'.format(arg, dots, getattr(args, arg)), flush=True)


def save_rank_0(args, message):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            with open(args.log_file, "a") as f:
                f.write(message + "\n")
                f.flush()
    else:
        with open(args.log_file, "a") as f:
            f.write(message + "\n")
            f.flush()


def set_deepspeed_activation_checkpointing(args, num_checkpoints):
    deepspeed.checkpointing.configure(mpu, deepspeed_config=args.deepspeed_config, num_checkpoints=num_checkpoints)
    mpu.checkpoint = deepspeed.checkpointing.checkpoint
    mpu.get_cuda_rng_tracker = deepspeed.checkpointing.get_cuda_rng_tracker
    mpu.model_parallel_cuda_manual_seed = deepspeed.checkpointing.model_parallel_cuda_manual_seed


def initialize_distributed(args):
    """Initialize torch.distributed."""

    # Manually set the device ids.
    device = args.rank % torch.cuda.device_count()
    if args.local_rank is not None:
        device = args.local_rank
    torch.cuda.set_device(device)
    # Call the init process
    init_method = 'tcp://'
    master_ip = os.getenv('MASTER_ADDR', 'localhost')
    master_port = os.getenv('MASTER_PORT', '6000')
    init_method += master_ip + ':' + master_port
    deepspeed.init_distributed()

    # Set the model-parallel / data-parallel communicators.
    mpu.initialize_model_parallel(args.model_parallel_size)


def set_random_seed(seed):
    """Set random seed for reproducability."""

    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        mpu.model_parallel_cuda_manual_seed(seed)


def ensure_directory_exists(filename):
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_checkpoint_tracker_filename(checkpoints_path):
    return os.path.join(checkpoints_path, 'latest_checkpointed_iteration.txt')


def save_checkpoint(iteration, model, optimizer,
                    lr_scheduler, args):
    """Save a model checkpoint."""
    if args.deepspeed:
        save_ds_checkpoint(iteration, model, args)

    torch.distributed.barrier()
    # Update the latest iteration
    if torch.distributed.get_rank() == 0:
        tracker_filename = get_checkpoint_tracker_filename(args.save)
        with open(tracker_filename, 'w') as f:
            f.write(str(iteration))

    torch.distributed.barrier()


def save_ds_checkpoint(iteration, model, args):
    """Save a model checkpoint."""

    sd = {}
    sd['iteration'] = iteration
    # rng states.
    if not args.no_save_rng:
        sd['random_rng_state'] = random.getstate()
        sd['np_rng_state'] = np.random.get_state()
        sd['torch_rng_state'] = torch.get_rng_state()
        sd['cuda_rng_state'] = torch.cuda.get_rng_state()
        sd['rng_tracker_states'] = mpu.get_cuda_rng_tracker().get_states()
        
    model.save_checkpoint(args.save, str(iteration), client_state = sd, save_zero=False)


def get_checkpoint_iteration(args):
    tracker_filename = get_checkpoint_tracker_filename(args.load)
    if not os.path.isfile(tracker_filename):
        print_rank_0('WARNING: could not find the metadata file {} '.format(tracker_filename))
        print_rank_0('    will not load any checkpoints and will start from RANDOM')
        return 0, False
    
    iteration = 0
    with open(tracker_filename, 'r') as f:
        metastring = f.read().strip()
        try:
            iteration = int(metastring)
        except ValueError:
            print_rank_0('ERROR: Invalid metadata file {}. Exiting'.format(tracker_filename))
            exit()

    assert iteration > 0, 'error parsing metadata file {}'.format(tracker_filename)
    
    return iteration, True


def load_checkpoint(args, model, optimizer=None, lr_scheduler=None):
    """Load a model checkpoint."""

    iteration, success = get_checkpoint_iteration(args)

    if not success:
        return 0

    checkpoint_name, sd = model.load_checkpoint(args.load, iteration, load_optimizer_states=args.load_optimizer_states, load_lr_scheduler_states=args.load_lr_scheduler_states, load_module_strict=(not args.no_load_strict))

    if checkpoint_name is None:
        if mpu.get_data_parallel_rank() == 0:
            print("Unable to load checkpoint.")
        return iteration

    try:
        iteration = sd['iteration']
    except KeyError:
        try: # Backward compatible with older checkpoints
            iteration = sd['total_iters']
        except KeyError:
            print_rank_0('A metadata file exists but Unable to load iteration '
                            ' from checkpoint {}, exiting'.format(checkpoint_name))
            exit()

    torch.distributed.barrier()
    if mpu.get_data_parallel_rank() == 0:
        print('  successfully loaded {}'.format(checkpoint_name))

    return iteration
