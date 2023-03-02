
import logging
import multiprocessing
import resource
from typing import Dict, Union, Callable

import monai
import psutil
import torch
import pandas as pd

import os

logger = logging.getLogger(__name__)

USE_AMP = monai.utils.get_torch_version_tuple() >= (1, 6)  # type: ignore


def num_workers() -> int:
    """Get max supported workers -2 for multiprocessing"""

    n_workers = multiprocessing.cpu_count() - 2  # leave two workers so machine can still respond

    # check if we will run into OOM errors because of too many workers
    # In most projects 2-4GB/Worker seems to be save
    available_ram_in_gb = psutil.virtual_memory()[0] / 1021**3
    max_workers = int(available_ram_in_gb // 4)
    if max_workers < n_workers:
        n_workers = max_workers

    # now check for max number of open files allowed on system
    soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_CORE, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
    # giving each worker at least 216 open processes should allow them to run smoothly
    max_workers = soft_limit // 216

    if max_workers < n_workers:

        logger.info(
            "Number of allowed open files is to small, "
            "which might lead to problems with multiprocessing"
            "Current limits are:\n"
            f"\t soft_limit: {soft_limit}\n"
            f"\t hard_limit: {hard_limit}\n"
            "try increasing the limits to at least {216*n_workers}."
            "See https://superuser.com/questions/1200539/cannot-increase-open-file-limit-past"
            "-4096-ubuntu for more details.\n"
            "Will use torch.multiprocessing.set_sharing_strategy('file_system') as a workarround."
        )
        n_workers = min(32, n_workers)
        torch.multiprocessing.set_sharing_strategy("file_system")
    logger.info(f"using number of workers: {n_workers}")

    return n_workers

def get_datalist(split: str) -> Dict[str, str]: 

    if split in ["train", "valid", "test"]: 
        df = pd.read_csv(f"/sc-projects/sc-proj-dha/Niere/segmentator_{split}.csv")
    else: 
        raise ValueError("Split must be `train`, `valid` or `test`")
    
    df["image"] = [f"/sc-projects/sc-proj-dha/{fn}" for fn in df.image]
    df["label"] = [f"/sc-projects/sc-proj-dha/{fn}" for fn in df.label]
    data_list = df.to_dict("records")
    return data_list

IMAGE_FILES = [".nii", ".nii.gz", ".nrrd", ".dcm"]

def parse_data_for_inference(fn_or_dir: str = None) -> Union[None, Dict]:
    """Convert filepath to data_dict"""

    if not fn_or_dir: 
        return 

    if os.path.isfile(fn_or_dir): 
        data_dict = [{
            "image": fn_or_dir
        }]
    
    elif os.path.isdir(fn_or_dir):
        files = [fn for fn in os.listdir(fn_or_dir) if any([fn.endswith(ext) for ext in IMAGE_FILES])]
        data_dict = [{"image": os.path.join(fn_or_dir, fn)} for fn in files]
        
    else:
        raise FileNotFoundError(fn_or_dir)

    return data_dict



def get_meta_dict(image_key) -> Callable: 

    def _inner(batch) -> list:
        """Get dict of metadata from engine. Needed as `batch_transform` for MetricSaver to also save filenames"""
        key = image_key[0] if isinstance(image_key, list) else image_key
        return [item[key].meta for item in batch]

    return _inner


def adapt_filename(x): 
    x.meta['filename_or_obj'] = x.meta['filename_or_obj'].replace('/image', '')
    return x
