import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

from PIL import Image
from data import get_wds_dataset
import numpy as np
import logging
import random
import math
import sys
import copy

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

from open_clip import create_model_and_transforms, trace_model, get_tokenizer, create_loss
from training.data import *
from training.distributed import is_master, init_distributed_device, broadcast_object
from training.logger import setup_logging
from training.params import parse_args
from training.scheduler import cosine_lr, const_lr, const_lr_cooldown
from training.train import train_one_epoch, evaluate
from training.file_utils import pt_load, check_exists, start_sync_process, remote_sync
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import DataLoader
import webdataset as wds


def setup_logging(log_file, level, include_host=False):
    if include_host:
        import socket
        hostname = socket.gethostname()
        formatter = logging.Formatter(
            f'%(asctime)s |  {hostname} | %(levelname)s | %(message)s', datefmt='%Y-%m-%d,%H:%M:%S')
    else:
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d,%H:%M:%S')

    logging.root.setLevel(level)
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        logger.setLevel(level)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logging.root.addHandler(stream_handler)

    if log_file:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setFormatter(formatter)
        logging.root.addHandler(file_handler)


def is_global_master(args):
    return args.rank == 0


def is_local_master(args):
    return args.local_rank == 0


def is_master(args, local=False):
    return is_local_master(args) if local else is_global_master(args)


def is_using_horovod():
    # NOTE w/ horovod run, OMPI vars should be set, but w/ SLURM PMI vars will be set
    # Differentiating between horovod and DDP use via SLURM may not be possible, so horovod arg still required...
    ompi_vars = ["OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_SIZE"]
    pmi_vars = ["PMI_RANK", "PMI_SIZE"]
    if all([var in os.environ for var in ompi_vars]) or all([var in os.environ for var in pmi_vars]):
        return True
    else:
        return False


def is_using_distributed():
    if 'WORLD_SIZE' in os.environ:
        return int(os.environ['WORLD_SIZE']) > 1
    if 'SLURM_NTASKS' in os.environ:
        return int(os.environ['SLURM_NTASKS']) > 1
    return False


def world_info_from_env():
    local_rank = 0
    for v in ('LOCAL_RANK', 'MPI_LOCALRANKID', 'SLURM_LOCALID', 'OMPI_COMM_WORLD_LOCAL_RANK'):
        if v in os.environ:
            local_rank = int(os.environ[v])
            break
    global_rank = 0
    for v in ('RANK', 'PMI_RANK', 'SLURM_PROCID', 'OMPI_COMM_WORLD_RANK'):
        if v in os.environ:
            global_rank = int(os.environ[v])
            break
    world_size = 1
    for v in ('WORLD_SIZE', 'PMI_SIZE', 'SLURM_NTASKS', 'OMPI_COMM_WORLD_SIZE'):
        if v in os.environ:
            world_size = int(os.environ[v])
            break

    return local_rank, global_rank, world_size


def init_distributed_device(args):
    # Distributed training = training on more than one GPU.
    # Works in both single and multi-node scenarios.
    args.distributed = False
    args.world_size = 1
    args.rank = 0  # global rank
    args.local_rank = 0
    if args.horovod:
        assert hvd is not None, "Horovod is not installed"
        hvd.init()
        args.local_rank = int(hvd.local_rank())
        args.rank = hvd.rank()
        args.world_size = hvd.size()
        args.distributed = True
        os.environ['LOCAL_RANK'] = str(args.local_rank)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
    elif is_using_distributed():
        if 'SLURM_PROCID' in os.environ:
            # DDP via SLURM
            args.local_rank, args.rank, args.world_size = world_info_from_env()
            # SLURM var -> torch.distributed vars in case needed
            os.environ['LOCAL_RANK'] = str(args.local_rank)
            os.environ['RANK'] = str(args.rank)
            os.environ['WORLD_SIZE'] = str(args.world_size)
            torch.distributed.init_process_group(
                backend=args.dist_backend,
                init_method=args.dist_url,
                world_size=args.world_size,
                rank=args.rank,
            )
        else:
            # DDP via torchrun, torch.distributed.launch
            args.local_rank, _, _ = world_info_from_env()
            torch.distributed.init_process_group(
                backend=args.dist_backend,
                init_method=args.dist_url)
            args.world_size = torch.distributed.get_world_size()
            args.rank = torch.distributed.get_rank()
        args.distributed = True

    if torch.cuda.is_available():
        if args.distributed and not args.no_set_device_rank:
            device = 'cuda:%d' % args.local_rank
        else:
            device = 'cuda:0'
        torch.cuda.set_device(device)
    else:
        device = 'cpu'
    args.device = device
    device = torch.device(device)
    return device


def sort_by_key(input_json_file, output_json_file, key):
    with open(input_json_file, 'r') as input_file:
        original = json.load(input_file)

    sorted_items = sorted(original, key=lambda item: item[f'{key}'])

    with open(output_json_file, 'w') as output_file:
        json.dump(sorted_items, output_file)


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def traversal(model, tokenizer, dataloader, args):
    embeddings = {}
    items = []
    for i, batch in enumerate(dataloader):
        # items = []
        percent_complete = 100.0 * i / dataloader.num_batches
        save_list = [dataloader.num_batches // i for i in [16, 8, 4, 2]]
        if is_master(args):
            logging.info(f"[Rewriting Batch: {i}/{dataloader.num_batches} ({percent_complete:.0f}%)]")
        key, image, text = batch

        for k in range(len(key)):
            items.append({'__key__': key[k], 'raw': text[k]})

        # exit(-1)
        if (i == 1 or i in save_list) and len(items) > 0:
            with open(f'/data1/datasets/yfcc15m/train-mllm-a/mllm-a-raw-samples_{len(items)}.json', 'w') as output_file:
                logging.info(f"[Dumping to json: {len(items)} items]")
                json.dump(items, output_file)

    print(len(items))
    with open(f'/data1/datasets/yfcc15m/train-mllm-a/mllm-a-raw-samples_{len(items)}.json', 'w') as output_file:
        logging.info(f"[Dumping to json: {len(items)} items]")
        json.dump(items, output_file)
    sort_by_key(f'/data1/datasets/yfcc15m/train-mllm-a/mllm-a-raw-samples_{len(items)}.json', f'/data1/datasets/yfcc15m/train-mllm-a/mllm-a-raw-samples_{len(items)}-sorted.json', '__key__')
    return 


def main(args):
    args = parse_args(args)

    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # fully initialize distributed device environment
    device = init_distributed_device(args)

    random_seed(args.seed, 0)

    args.log_path = None
    args.log_level = logging.INFO
    setup_logging(args.log_path, args.log_level)

    if args.distributed:
        logging.info(
            f'Running in distributed mode with multiple processes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    else:
        logging.info(f'Running with a single process. Device {args.device}.')
    
    # model = AutoModelForCausalLM.from_pretrained(args.model, load_in_8bit=args.load_in_8bit, trust_remote_code=True).to(device).eval()
    # tokenizer = AutoTokenizer.from_pretrained(args.model, add_bos_token=True, add_eos_token=False)

    model, preprocess_train, preprocess_val = create_model_and_transforms(
        args.model,
        args.pretrained,
        precision=args.precision,
        device=device,
        jit=args.torchscript,
        force_quick_gelu=args.force_quick_gelu,
        force_custom_text=args.force_custom_text,
        force_patch_dropout=args.force_patch_dropout,
        force_image_size=args.force_image_size,
        image_mean=args.image_mean,
        image_std=args.image_std,
        image_interpolation=args.image_interpolation,
        image_resize_mode=args.image_resize_mode,  # only effective for inference
        aug_cfg=args.aug_cfg,
        pretrained_image=args.pretrained_image,
        output_dict=True,
    )   
    tokenizer = get_tokenizer(args.model)

    if args.distributed and not args.horovod:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device])

    if is_master(args):
        logging.info("Model:")
        logging.info(f"{str(model)}")
        logging.info("Params:")
    

    is_train = False
    num_samples = args.train_num_samples
    input_shards = args.train_data
    pipeline = [wds.SimpleShardList(input_shards)]
    pipeline.extend([
        wds.split_by_worker,
        # at this point, we have an iterator over the shards assigned to each worker
        wds.tarfile_to_samples(handler=log_and_continue),
    ])
    pipeline.extend([
        wds.select(filter_no_caption_or_no_image),
        wds.decode("pilrgb", handler=log_and_continue),
        wds.rename(key="__key__", image="jpg;png;jpeg;webp", text="txt"),
        wds.map_dict(key=lambda key: key, image=preprocess_train, text=lambda text: text),
        wds.to_tuple("key", "image", "text"),
        wds.batched(args.batch_size, partial=not is_train)
    ])

    dataset = wds.DataPipeline(*pipeline)
    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )
    num_batches = math.ceil(num_samples / args.batch_size)
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    logging.info("Start traversal")
    traversal(model=model, tokenizer=tokenizer, dataloader=dataloader, args=args)

    logging.info("Finish traversal")
    logging.info("Done!")


if __name__ == "__main__":
    main(sys.argv[1:])
