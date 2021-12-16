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

"""argparser configuration"""

import argparse
import os
import torch
import deepspeed


def add_model_config_args(parser: argparse.ArgumentParser):
    """Model arguments"""

    group = parser.add_argument_group('model', 'model configuration')

    group.add_argument('--model-config', type=str)
    group.add_argument('--model-parallel-size', type=int, default=1)
    group.add_argument("--ranker-config", type=str)
    group.add_argument('--attention-dropout', type=float, default=0.1,
                       help='dropout probability for attention weights')
    group.add_argument('--vocab-size', type=int, default=30522,
                       help='vocab size to use for non-character-level '
                       'tokenization. This value will only be used when '
                       'creating a tokenizer')
    group.add_argument('--make-vocab-size-divisible-by', type=int, default=128,
                       help='Pad the vocab size to be divisible by this value.'
                       'This is added for computational efficieny reasons.')
    group.add_argument('--cpu-optimizer', action='store_true',
                                   help='Run optimizer on CPU')
    group.add_argument('--cpu_torch_adam', action='store_true',
                                   help='Use Torch Adam as optimizer on CPU.')

    return parser


def add_fp16_config_args(parser: argparse.ArgumentParser):
    """Mixed precision arguments."""

    group = parser.add_argument_group('fp16', 'fp16 configurations')

    group.add_argument('--fp16', action='store_true',
                       help='Run model in fp16 mode')

    return parser


def add_training_args(parser: argparse.ArgumentParser):
    """Training arguments."""

    group = parser.add_argument_group('train', 'training configurations')

    group.add_argument('--batch-size', type=int, default=4,
                       help='Data Loader batch size')
    group.add_argument('--weight-decay', type=float, default=0.01,
                       help='weight decay coefficient for L2 regularization')
    group.add_argument('--checkpoint-activations', action='store_true',
                       help='checkpoint activation to allow for training '
                       'with larger models and sequences')
    group.add_argument('--checkpoint-num-layers', type=int, default=1,
                       help='chunk size (number of layers) for checkpointing')
    group.add_argument('--deepspeed-activation-checkpointing', action='store_true',
                       help='uses activation checkpointing from deepspeed')
    group.add_argument('--clip-grad', type=float, default=1.0,
                       help='gradient clipping')

    group.add_argument('--seed', type=int, default=1234,
                       help='random seed')
    # Batch prodecuer arguments
    group.add_argument('--reset-position-ids', action='store_true',
                       help='Reset posistion ids after end-of-document token.')
    group.add_argument('--reset-attention-mask', action='store_true',
                       help='Reset self attention maske after '
                       'end-of-document token.')

    # Learning rate.
    group.add_argument('--lr-decay-iters', type=int, default=None,
                       help='number of iterations to decay LR over,'
                       ' If None defaults to `--train-iters`*`--epochs`')
    group.add_argument('--lr-decay-style', type=str, default='linear',
                       choices=['constant', 'linear', 'cosine', 'exponential', 'noam'],
                       help='learning rate decay function')
    group.add_argument('--lr', type=float, default=1.0e-4,
                       help='initial learning rate')
    group.add_argument('--warmup', type=float, default=0.01,
                       help='percentage of data to warmup on (.01 = 1% of all '
                       'training iters). Default 0.01')
    # model checkpointing
    group.add_argument('--load', type=str, default=None,
                       help='Path to a directory containing a model checkpoint.')
    group.add_argument("--ranker-load", type=str, default=None,
                       help='Path to a directory containing a ranker model checkpoint')
    group.add_argument('--no-load-optim', action='store_true',
                       help='Do not load optimizer when loading checkpoint.')
    group.add_argument('--no-load-rng', action='store_true',
                       help='Do not load rng state when loading checkpoint.')
    # distributed training args
    group.add_argument('--distributed-backend', default='nccl',
                       help='which backend to use for distributed '
                       'training. One of [gloo, nccl]')

    group.add_argument('--local_rank', type=int, default=None,
                       help='local rank passed from distributed launcher')

    return parser


def add_evaluation_args(parser: argparse.ArgumentParser):
    """Evaluation arguments."""

    group = parser.add_argument_group('validation', 'validation configurations')

    group.add_argument('--rerank', action="store_true")
    group.add_argument('--rerank-num', type=int, default=1)
    group.add_argument("--human-rules", action="store_true")
    group.add_argument('--eval-batch-size', type=int, default=None,
                       help='Data Loader batch size for evaluation datasets.'
                       'Defaults to `--batch-size`')

    return parser


def add_text_generate_args(parser: argparse.ArgumentParser):
    """Text generate arguments."""

    group = parser.add_argument_group('Text generation', 'configurations')
    group.add_argument("--temperature", type=float, default=0.9)
    group.add_argument("--top_p", type=float, default=0.9)
    group.add_argument("--top_k", type=int, default=0)
    group.add_argument('--max-length', type=int, default=64)
    group.add_argument('--min-length', type=int, default=2)
    group.add_argument('--num-beams', type=int, default=1)
    group.add_argument('--no-repeat-ngram-size', type=int, default=3)
    group.add_argument('--eval-ratio', type=float, default=1.0)
    group.add_argument("--out-seq-length", type=int, default=256)
    group.add_argument("--repetition-penalty", type=float, default=1.2)
    group.add_argument("--early-stopping", action="store_true")
    group.add_argument("--length-penalty", type=float, default=1.8)
    group.add_argument("--rule_path", type=str)
    return parser


def add_data_args(parser: argparse.ArgumentParser):
    """Train/valid/test data arguments."""

    group = parser.add_argument_group('data', 'data configurations')
    group.add_argument('--data-path', type=str, default=None,
                        help='Path to combined dataset to split.')
    group.add_argument('--tokenizer-path', type=str, default=None)
    group.add_argument('--num-workers', type=int, default=2,
                       help="""Number of workers to use for dataloading""")
    group.add_argument('--seq-length', type=int, default=512)
    group.add_argument('--enc-seq-length', type=int, default=512,
                       help="Maximum sequence length to process")
    group.add_argument('--dec-seq-length', type=int, default=512,
                       help="Maximum sequence length to process")

    return parser

def get_args():
    """Parse all the args."""

    parser = argparse.ArgumentParser(description='PyTorch BERT Model')
    parser = add_model_config_args(parser)
    parser = add_fp16_config_args(parser)
    parser = add_training_args(parser)
    parser = add_evaluation_args(parser)
    parser = add_text_generate_args(parser)
    parser = add_data_args(parser)

    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()

    args.cuda = torch.cuda.is_available()

    args.rank = int(os.getenv('RANK', '0'))
    args.world_size = int(os.getenv("WORLD_SIZE", '1'))

    if os.getenv('OMPI_COMM_WORLD_LOCAL_RANK'):
        # We are using (OpenMPI) mpirun for launching distributed data parallel processes
        local_rank = int(os.getenv('OMPI_COMM_WORLD_LOCAL_RANK'))
        local_size = int(os.getenv('OMPI_COMM_WORLD_LOCAL_SIZE'))

        # Possibly running with Slurm
        num_nodes = int(os.getenv('SLURM_JOB_NUM_NODES', '1'))
        nodeid = int(os.getenv('SLURM_NODEID', '0'))

        args.local_rank = local_rank
        args.rank = nodeid*local_size + local_rank
        args.world_size = num_nodes*local_size

    args.model_parallel_size = min(args.model_parallel_size, args.world_size)
    if args.rank == 0:
        print('using world size: {} and model-parallel size: {} '.format(
            args.world_size, args.model_parallel_size))

    args.dynamic_loss_scale = True
    if args.rank == 0:
        print(' > using dynamic loss scaling')

    # The args fp32_* or fp16_* meant to be active when the
    # args fp16 is set. So the default behaviour should all
    # be false.
    if not args.fp16:
        args.fp32_embedding = False
        args.fp32_tokentypes = False
        args.fp32_layernorm = False

    return args
