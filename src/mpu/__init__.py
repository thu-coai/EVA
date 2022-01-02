# coding=utf-8

"""Model parallel utility interface."""

from .cross_entropy import vocab_parallel_cross_entropy

from .grads import clip_grad_norm

from .initialize import destroy_model_parallel
from .initialize import get_data_parallel_group
from .initialize import get_data_parallel_rank
from .initialize import get_data_parallel_world_size
from .initialize import get_model_parallel_group
from .initialize import get_model_parallel_rank
from .initialize import get_model_parallel_src_rank
from .initialize import get_model_parallel_world_size
from .initialize import initialize_model_parallel
from .initialize import model_parallel_is_initialized

from .layers import ColumnParallelLinear
from .layers import ParallelEmbedding
from .layers import RowParallelLinear
from .layers import VocabParallelEmbedding

from .mappings import copy_to_model_parallel_region
from .mappings import gather_from_model_parallel_region
from .mappings import reduce_from_model_parallel_region
from .mappings import scatter_to_model_parallel_region

from .random import checkpoint
from .random import partition_activations_in_checkpoint
from .random import get_cuda_rng_tracker
from .random import model_parallel_cuda_manual_seed

from .transformer_enc_dec import ParallelTransformer
