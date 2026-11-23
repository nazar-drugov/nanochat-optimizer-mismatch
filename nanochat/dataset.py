"""
The base/pretraining dataset is a set of parquet files.
This file contains utilities for:
- a small registry of named pretraining datasets (where they live online and on disk)
- iterating over the parquet files and yielding documents from it
- download the files on demand if they are not on disk

Every dataset is a directory of nanochat-format parquet shards: a 'text' column,
files sorted by name, and the LAST file is the validation split (all earlier
files are train). Any directory following that convention (e.g. the fine-tuning
data written by scripts/prepare_finetune_data.py) can be passed as `data_dir`.

For details of how the pretraining datasets were prepared, see
`dev/repackage_data_reference.py`.

CLI:
    python -m nanochat.dataset --dataset fineweb-edu-100b -n 60   # 60 train shards + the val shard
"""

import os
import argparse
import time
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool

import requests
import pyarrow.parquet as pq

from nanochat.common import get_base_dir

# -----------------------------------------------------------------------------
# Registry of pretraining datasets

@dataclass(frozen=True)
class DatasetSpec:
    name: str
    base_url: str   # where shards are hosted; shard i is at f"{base_url}/{index_to_filename(i)}"
    max_shard: int  # index of the last shard, which is the validation shard
    dirname: str    # local directory name under get_base_dir()
    description: str = ""

DATASETS = {
    "fineweb-edu-100b": DatasetSpec(
        name="fineweb-edu-100b",
        base_url="https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main",
        max_shard=1822,  # shard_01822.parquet is the last one (val); ~94MB/shard, ~171GB total
        dirname="base_data_fineweb_edu_100b",
        description="FineWeb-Edu 100B-token sample (nanochat's dataset before commit 324e69c)",
    ),
    "climbmix": DatasetSpec(
        name="climbmix",
        base_url="https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main",
        max_shard=6542,  # shard_06542.parquet is the last one (val)
        dirname="base_data_climbmix",
        description="NVIDIA ClimbMix 400B-token sample (upstream nanochat default since 324e69c)",
    ),
}
DEFAULT_DATASET = "fineweb-edu-100b"  # team decision: pretrain on FineWeb-Edu

# Old nanochat (pre-324e69c) stored FineWeb-Edu-100B shards here. It is never used implicitly.
LEGACY_DIRNAME = "base_data"

index_to_filename = lambda index: f"shard_{index:05d}.parquet" # format of the filenames


def get_dataset_spec(dataset_name=None):
    """Return the DatasetSpec for a registered dataset name (None -> DEFAULT_DATASET)."""
    name = DEFAULT_DATASET if dataset_name is None else dataset_name
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}; choose one of {sorted(DATASETS)}")
    return DATASETS[name]


def get_data_dir(dataset_name=None):
    """Local shard directory of a registered dataset (None -> DEFAULT_DATASET). Not created."""
    return os.path.join(get_base_dir(), get_dataset_spec(dataset_name).dirname)


def resolve_data_dir(dataset_name=None, data_dir=None):
    """Directory to train from: an explicit data_dir wins, else the named
    (or default) registered dataset's directory."""
    if data_dir is not None:
        return os.path.abspath(os.path.expanduser(data_dir))
    return get_data_dir(dataset_name)


def _legacy_hint(data_dir):
    """A hint string if the user has the old pre-ClimbMix FineWeb-Edu directory lying around."""
    legacy_dir = os.path.join(get_base_dir(), LEGACY_DIRNAME)
    fineweb_dir = get_data_dir("fineweb-edu-100b")
    if os.path.abspath(data_dir) != os.path.abspath(fineweb_dir) or not os.path.isdir(legacy_dir):
        return ""
    return (
        f"\nNote: found legacy directory {legacy_dir} (old nanochat FineWeb-Edu-100B shards, same "
        f"files as {get_dataset_spec('fineweb-edu-100b').base_url}). It is NOT used automatically. "
        f"If you want to reuse it, verify it and then e.g. `ln -s {legacy_dir} {fineweb_dir}` "
        f"(note: it may not contain the val shard {index_to_filename(get_dataset_spec('fineweb-edu-100b').max_shard)})."
    )

# -----------------------------------------------------------------------------
# These functions are useful utilities to other modules, can/should be imported

def list_parquet_files(data_dir=None):
    """
    Return sorted full paths of all parquet files in data_dir.
    data_dir=None means the default dataset's directory. By convention the last file is val.
    """
    data_dir = get_data_dir() if data_dir is None else data_dir
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Dataset directory not found: {data_dir}\n"
            f"Download pretraining shards with e.g. `python -m nanochat.dataset --dataset {DEFAULT_DATASET} -n 170`."
            + _legacy_hint(data_dir)
        )
    parquet_files = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith('.parquet') and not f.endswith('.tmp')
    ])
    parquet_paths = [os.path.join(data_dir, f) for f in parquet_files]
    return parquet_paths


def list_split_files(split, data_dir=None):
    """Parquet paths of one split: 'train' = all but the last file, 'val' = the last file."""
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    parquet_paths = list_parquet_files(data_dir=data_dir)
    if len(parquet_paths) == 0:
        raise FileNotFoundError(f"No parquet files in {get_data_dir() if data_dir is None else data_dir}")
    return parquet_paths[:-1] if split == "train" else parquet_paths[-1:]


def parquets_iter_batched(split, start=0, step=1, data_dir=None):
    """
    Iterate through the dataset, in batches of underlying row_groups for efficiency.
    - split can be "train" or "val". the last parquet file will be val.
    - start/step are useful for skipping rows in DDP. e.g. start=rank, step=world_size
    - data_dir: directory of parquet shards (None = default dataset's directory)
    """
    for filepath in list_split_files(split, data_dir=data_dir):
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(start, pf.num_row_groups, step):
            rg = pf.read_row_group(rg_idx, columns=['text'])
            texts = rg.column('text').to_pylist()
            yield texts

# -----------------------------------------------------------------------------
def download_single_file(index, dataset_name=DEFAULT_DATASET):
    """ Downloads a single shard index of a registered dataset, with some backoff """
    spec = get_dataset_spec(dataset_name)
    data_dir = get_data_dir(dataset_name)

    # Construct the local filepath for this file and skip if it already exists
    filename = index_to_filename(index)
    filepath = os.path.join(data_dir, filename)
    if os.path.exists(filepath):
        print(f"Skipping {filepath} (already exists)")
        return True

    # Construct the remote URL for this file
    url = f"{spec.base_url}/{filename}"
    print(f"Downloading {filename}...")

    # Download with retries
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            # Write to temporary file first
            temp_path = filepath + f".tmp"
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                    if chunk:
                        f.write(chunk)
            # Move temp file to final location
            os.rename(temp_path, filepath)
            print(f"Successfully downloaded {filename}")
            return True

        except (requests.RequestException, IOError) as e:
            print(f"Attempt {attempt}/{max_attempts} failed for {filename}: {e}")
            # Clean up any partial files
            for path in [filepath + f".tmp", filepath]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except:
                        pass
            # Try a few times with exponential backoff: 2^attempt seconds
            if attempt < max_attempts:
                wait_time = 2 ** attempt
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"Failed to download {filename} after {max_attempts} attempts")
                return False

    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download pretraining dataset shards")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET, choices=sorted(DATASETS), help=f"Which pretraining dataset (default: {DEFAULT_DATASET})")
    parser.add_argument("-n", "--num-files", type=int, default=-1, help="Number of train shards to download (default: -1), -1 = disable")
    parser.add_argument("-w", "--num-workers", type=int, default=4, help="Number of parallel download workers (default: 4)")
    args = parser.parse_args()

    spec = get_dataset_spec(args.dataset)
    data_dir = get_data_dir(args.dataset)
    hint = _legacy_hint(data_dir)
    if hint and not os.path.exists(data_dir):
        print(hint.strip())
        print()

    # Prepare the output directory
    os.makedirs(data_dir, exist_ok=True)

    # The way this works is that the user specifies the number of train shards to download via the -n flag.
    # In addition to that, the validation shard is *always* downloaded and is pinned to be the last shard.
    num_train_shards = spec.max_shard if args.num_files == -1 else min(args.num_files, spec.max_shard)
    ids_to_download = list(range(num_train_shards))
    ids_to_download.append(spec.max_shard) # always download the validation shard

    # Download the shards
    print(f"Dataset: {spec.name} ({spec.description})")
    print(f"Downloading {len(ids_to_download)} shards using {args.num_workers} workers...")
    print(f"Target directory: {data_dir}")
    print()
    with Pool(processes=args.num_workers) as pool:
        results = pool.map(partial(download_single_file, dataset_name=args.dataset), ids_to_download)

    # Report results
    successful = sum(1 for success in results if success)
    print(f"Done! Downloaded: {successful}/{len(ids_to_download)} shards to {data_dir}")
