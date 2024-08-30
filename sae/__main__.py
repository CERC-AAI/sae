import os
from dataclasses import dataclass
from multiprocessing import cpu_count

import torch
import torch.distributed as dist
from datasets import Dataset, load_from_disk
from simple_parsing import field, parse
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from .data import chunk_and_tokenize, MemmapDataset
from .trainer import SaeTrainer, TrainConfig


@dataclass
class RunConfig(TrainConfig):
    model: str = field(default="EleutherAI/pythia-160m")
    """Name of the model to train."""

    dataset: str = field(default="test.hf")
    """Path to the dataset to use for training."""

    cache_dir: str = field(default=None)
    """Directory to use for caching the model."""

    ctx_len: int = 2048
    """Context length to use for training."""

    load_in_8bit: bool = False
    """Load the model in 8-bit mode."""

    max_examples: int | None = None
    """Maximum number of examples to use for training."""

    resume: bool = False
    """Whether to try resuming from the checkpoint present at `run_name`."""

    seed: int = 42
    """Random seed for shuffling the dataset."""

    data_preprocessing_num_proc: int = field(
        default_factory=lambda: cpu_count() // 2,
    )
    """Number of processes to use for preprocessing data"""


def load_artifacts(args: RunConfig, rank: int) -> tuple[PreTrainedModel, Dataset | MemmapDataset]:
    os.environ['HF_DATASETS_OFFLINE'] = "1"

    if args.load_in_8bit:
        dtype = torch.float16
    elif torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = "auto"

    print('loading model...')
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map={"": f"cuda:{rank}"},
        torch_dtype=dtype,
        cache_dir=args.cache_dir
    )
    print('model loaded')

    print('loading datasets...')
    dataset = load_from_disk(args.dataset)
    dataset = dataset['train']
    print('datasets loaded')

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dataset = chunk_and_tokenize(
        dataset,
        tokenizer,
        max_seq_len=args.ctx_len,
        num_proc=args.data_preprocessing_num_proc,
    )

    if args.max_examples:
        dataset = dataset.select(range(args.max_examples))

    return model, dataset


def run():
    local_rank = os.environ.get("LOCAL_RANK")
    ddp = local_rank is not None
    rank = int(local_rank) if ddp else 0

    if ddp:
        torch.cuda.set_device(int(local_rank))
        dist.init_process_group("nccl")

        if rank == 0:
            print(f"Using DDP across {dist.get_world_size()} GPUs.")

    args = parse(RunConfig)

    if not ddp or rank == 0:
        model, dataset = load_artifacts(args, rank)
    if ddp:
        dist.barrier()
        if rank != 0:
            model, dataset = load_artifacts(args, rank)
        dataset = dataset.shard(dist.get_world_size(), rank)

    print(f"Training on '{args.dataset}'")
    print(f"Storing model weights in {model.dtype}")

    trainer = SaeTrainer(args, dataset, model)
    if args.resume:
        trainer.load_state(args.run_name or "sae-ckpts")

    trainer.fit()


if __name__ == "__main__":
    run()
