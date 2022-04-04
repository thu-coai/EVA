# coding=utf-8

"""argparser configuration"""

import argparse
import os
import torch
import deepspeed


def add_model_config_args(parser: argparse.ArgumentParser):
    """Model arguments"""

    group = parser.add_argument_group("model", "model configuration")

    group.add_argument("--model-config", type=str, 
                        help="The path to the model configuration file.")
    group.add_argument("--model-parallel-size", type=int, default=1, 
                        help="The segment number of model parallism.")

    return parser


def add_fp16_config_args(parser: argparse.ArgumentParser):
    """Mixed precision arguments."""

    group = parser.add_argument_group("fp16", "fp16 configurations")

    group.add_argument("--fp16", action="store_true", 
                        help="Run model in fp16 mode.")

    return parser


def add_training_args(parser: argparse.ArgumentParser):
    """Training arguments."""

    group = parser.add_argument_group("train", "training configurations")

    group.add_argument("--do-train", action="store_true", 
                        help="Do model training.")
    group.add_argument("--do-valid", action="store_true", 
                        help="Do model validation while training.")
    group.add_argument("--do-eval", action="store_true", 
                        help="Do model evaluation/inference.")

    group.add_argument("--train-ratio", type=float, default=1, 
                        help="The ratio of the training dataset used.")
    group.add_argument("--valid-ratio", type=float, default=1, 
                        help="The ratio of the validation dataset used.")
    group.add_argument("--test-ratio", type=float, default=1, 
                        help="The ratio of the evaluation dataset used.")

    group.add_argument("--batch-size", type=int, default=4, 
                        help="Training batch size.")
    group.add_argument("--gradient-accumulation-steps", type=int, default=1, 
                        help="Gradient accumation steps.")
    group.add_argument("--train-iters", type=int, default=-1, 
                        help="Total iterations for training. If set to -1, the iteration number depends on `--epochs`.")
    group.add_argument("--epochs", type=int, default=3, 
                        help="Total epochs for training. If set to -1. the epoch number depends on `--iterations`.")
    group.add_argument("--weight-decay", type=float, default=0.01, 
                        help="Weight decay coefficient for L2 regularization.")
    group.add_argument("--checkpoint-activations", action="store_true",
                        help="Checkpoint activation to allow for training with larger models and sequences.")
    group.add_argument("--checkpoint-num-layers", type=int, default=1,
                        help="Chunk size (number of layers) for checkpointing.")
    group.add_argument("--deepspeed-activation-checkpointing", action="store_true",
                        help="Use activation checkpointing from deepspeed.")
    group.add_argument("--clip-grad", type=float, default=1.0,
                        help="Do gradient clipping.")

    group.add_argument("--seed", type=int, default=422,
                        help="Set the random seed.")

    # Learning rate.
    group.add_argument("--lr-decay-style", type=str, default="linear",
                        choices=["constant", "linear", "cosine", "exponential", "noam"],
                        help="Learning rate decay function.")
    group.add_argument("--lr", type=float, default=1.0e-4,
                        help="Initial learning rate.")
    group.add_argument("--warmup", type=float, default=0.01,
                        help="Percentage of training steps for warmup.")
    
    # model checkpointing
    group.add_argument("--load", type=str, default=None,
                       help="Path to a directory containing a model checkpoint.")
    group.add_argument("--load_optimizer_states", action="store_true", 
                        help="Load the optimizer states from the checkpoint.")
    group.add_argument("--load_lr_scheduler_states", action="store_true",
                        help="Load the learning rate scheduler states from the checkpoint.")
    group.add_argument("--no_load_strict", action="store_true",
                        help="Strictly check if the weights in the checkpoints match that defined in the model.")

    group.add_argument("--save", type=str, default=None,
                        help="The path to save the model.")
    group.add_argument("--save-interval", type=int, default=10,
                        help="The step interval to save the model.")

    # logging
    group.add_argument("--log-file", type=str, default=None,
                        help="The path to save the log file.")
    group.add_argument("--log-interval", type=int, default=5,
                        help="The step interval to do logging.")

    # distributed training args
    group.add_argument("--distributed-backend", default="nccl",
                       help="Which backend to use for distributed training. One of [gloo, nccl].")

    group.add_argument("--local_rank", type=int, default=None,
                       help="Local rank passed from distributed launcher.")

    return parser


def add_evaluation_args(parser: argparse.ArgumentParser):
    """Evaluation arguments."""

    group = parser.add_argument_group("validation", "validation configurations")

    group.add_argument("--eval-batch-size", type=int, default=None,
                       help="Data Loader batch size for evaluation datasets. Defaults to `--batch-size`.")
    group.add_argument("--eval-interval", type=int, default=10,
                        help="The steo interval to do validation.")
    group.add_argument("--eval-generation", action="store_true",
                        help="Do the evaluation of generation.")

    return parser


def add_text_generate_args(parser: argparse.ArgumentParser):
    """Text generate arguments."""

    group = parser.add_argument_group("Text generation", "configurations")
    group.add_argument("--temperature", type=float, default=0.9,
                        help="The temperature of sampling.")
    group.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p sampling.")
    group.add_argument("--top_k", type=int, default=0,
                        help="Top-k sampling.")
    group.add_argument("--max-generation-length", type=int, default=128,
                        help="The maximum sequence length to generate.")
    group.add_argument("--min-generation-length", type=int, default=2,
                        help="The minimum sequence length to generate.")
    group.add_argument("--num-beams", type=int, default=1,
                        help="The beam number of beam search.")
    group.add_argument("--no-repeat-ngram-size", type=int, default=3,
                        help="The n-gram whose length is less than this option will appear at most once in the whole dialog.")
    group.add_argument("--repetition-penalty", type=float, default=1.2,
                        help="Repetition penalty, to prevent repeated words.")
    group.add_argument("--early-stopping", action="store_true",
                        help="Early-stopping while generating.")
    group.add_argument("--length-penalty", type=float, default=1.8,
                        help="Length penalty, to prevent short generation.")
    group.add_argument("--rule-path", type=str, default=None,
                        help="The directory that contains hand-written rules.")
    return parser


def add_data_args(parser: argparse.ArgumentParser):
    """Train/valid/test data arguments."""

    group = parser.add_argument_group("data", "data configurations")
    group.add_argument("--data-path", type=str, default=None,
                        help="Path to the directory that contains train/valid/test.txt.")
    group.add_argument("--cache-path", type=str, default=None, 
                        help="Path to cache the data as .pkl.")
    group.add_argument("--tokenizer-path", type=str, default=None,
                        help="The directory of the vocabulary.")
    group.add_argument("--data-ext", type=str, default=".txt",
                        help="The extention name of the data files.")
    group.add_argument("--num-workers", type=int, default=2,
                       help="Number of workers to use for dataloading.")
    group.add_argument("--enc-seq-length", type=int, default=512,
                       help="Maximum encoder sequence length to process.")
    group.add_argument("--dec-seq-length", type=int, default=512,
                       help="Maximum decoder sequence length to process.")

    return parser

def get_args():
    """Parse all the args."""

    parser = argparse.ArgumentParser(description="PyTorch BERT Model")
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

    args.rank = int(os.getenv("RANK", "0"))
    args.world_size = int(os.getenv("WORLD_SIZE", "1"))
    args.local_rank = int(os.getenv("LOCAL_RANK", "0"))

    args.model_parallel_size = min(args.model_parallel_size, args.world_size)
    if args.rank == 0:
        print("using world size: {} and model-parallel size: {} ".format(
            args.world_size, args.model_parallel_size))

    args.dynamic_loss_scale = True
    if args.rank == 0:
        print(" > using dynamic loss scaling")

    return args
