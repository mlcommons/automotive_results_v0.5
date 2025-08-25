#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Script to preprocess data for BEVFormer benchmark."""

# NOTE: MLCommons provide preprocessed pkl files for cal/val set in FP32
#       We use them and reformat them to other precisions and storage format (numpy)

import argparse
import os
import sys
import pickle
import numpy as np
import torch

from typing import Dict, List, Tuple
from pathlib import Path
from multiprocessing import Pool

from code.common import logging


DATASET_TYPE_PREFIX = {
    "val": ("val_3d", "val"),
    "cal": ("calib_3d", "calib")
}
MAPPING_FILENAME = {
    "val": "/work/data_maps/nuscenes/val_map.txt",
    "cal": "/work/data_maps/nuscenes/cal_map.txt"
}
TENSOR_CATEGORIES = {
    "val": ['image', 'use_prev_bev', 'can_bus', 'lidar2img'],
    "cal": ['image', 'prev_bev', 'use_prev_bev', 'can_bus', 'lidar2img']
}

def file_loader(file: Path) -> np.ndarray:
    """
    Could be reading raw images, can bus etc to populate the data, but
    here, we use preprocessed pkl files from MLCommons
    """
    try:
        with open(file, 'rb') as f:
            content= pickle.load(f)
    except Exception as e:
        logging.error(f"Error loading {file}: {e}")
        raise e
    return content

def data_loader(file: Path) -> Tuple[int, np.ndarray]:
    """Load the dataset from the mapping file."""
    np_array_dict = file_loader(file)
    np_array_list = list(np_array_dict.items())
    return np_array_list

def data_dumper(filepath: Path, np_array: np.ndarray, overwrite: bool=False) -> None:
    """Dump the data to the output directory."""
    if not overwrite and filepath.exists():
        raise FileExistsError(f"File {filepath} already exists")
    filepath.parent.mkdir(parents=True, exist_ok=True)
    np.save(filepath, np_array, allow_pickle=False)

def populate_file_info(dataset_dir: str, dataset_type: str, preprocessed_data_dir: str, overwrite: bool=False) -> List[Path]:
    """Load the dataset from the mapping file."""
    mapping_file_path = Path(MAPPING_FILENAME[dataset_type])
    if not mapping_file_path.exists():
        raise FileNotFoundError(f"Mapping file {mapping_file_path} does not exist")

    file_list = []
    length_list = []
    with open(mapping_file_path, 'r') as f:
        for line in f:
            if line.startswith("#"):
                continue
            start_idx, length = line.strip().split(",")
            start_idx = int(start_idx)
            length = int(length)
            length_list.append(length)
            for idx in range(start_idx, start_idx + length):
                subdir, filename_prefix = DATASET_TYPE_PREFIX[dataset_type]
                pkl_file_path = Path(dataset_dir) / subdir / f"{filename_prefix}_{idx}.pkl"
                if not pkl_file_path.exists():
                    raise FileNotFoundError(f"PKL file {pkl_file_path} does not exist")
                file_list.append((idx, pkl_file_path, preprocessed_data_dir, dataset_type, overwrite))
    return file_list, length_list

def reformatter(value, target_precision: str="unknown", scale: float=0.0):
    """Return quantized tensor of input FP32 tensor."""
    if target_precision == "fp32":
        return value
    elif target_precision == "fp8":
        # NOTE: only explicit quantization for FP8, need to add scale (zero point = 0)
        #       https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-quantized-types.html
        # NOTE: symmetric scale:
        #       scale = max(abs(data_range_max), abs(data_range_min)) * 2 / (quantization_range_max - quantization_range_min)
        #       data_range_max/min would be derived using the calibration set
        #       Quantization (zero_point = 0.0):
        #       val_fp32 = scale * (val_quantized - zero_point)
        def to_float8(x, scale: float):
            # fp8 format, fix to E4M3 for now
            fp8_type=torch.float8_e4m3fn
            q_range_min, q_range_max = -448.0, 448.0
            # apply scale assuming zero point is 0.0 and clip to range
            x_scl_sat = np.clip(x / scale, q_range_min, q_range_max)
            fp8_x = x_scl_sat.to(fp8_type)
            # Convert FP8 tensor to numpy format
            # Since numpy doesn't support FP8, we store as uint8 (raw bytes)
            fp8_x_numpy = fp8_x.view(torch.uint8).numpy().astype(dtype=np.uint8, order='C')
            return fp8_x_numpy
        if scale not in (0.0, None,):
            return to_float8(value, scale)
        else:
            return value
    elif target_precision == "fp16":
        return value.astype(dtype=np.float16, order='C')
    elif target_precision == "int8":
        def to_int8(x, scale: float):
            # int8 format, symmetic scale
            int8_type=np.int8
            q_range_min, q_range_max = -128.0, 127.0
            # apply scale assuming zero point is 0.0 and clip to range
            x_scl_sat = np.clip(x / scale, q_range_min, q_range_max)
            int8_x = x_scl_sat.astype(dtype=int8_type, order='C')
            return int8_x
        if scale not in (0.0, None,):
            return to_int8(value, scale)
        else:
            return value
    elif target_precision == "bool":
        return value.astype(dtype='bool', order='C')
    else:
        raise ValueError(f"Unsupported target quantization precision: {target_precision}")

def load_split_dump(idx: int, pkl_file_path: Path, preprocessed_data_dir: str, dataset_type: str,
                    overwrite: bool=False, target_precision: dict={}, scale: dict={}) -> int:
    """Load the pkl file and split it into multiple np arrays."""
    np_array = data_loader(pkl_file_path)
    target_dir_path = Path(preprocessed_data_dir) / dataset_type / target_precision.get("image", "fp32")
    for category, value in np_array:
        # MLCommons preprocessed data inconsistency: val set uses img instead of image
        if category == 'img' and dataset_type == 'val':
            category = 'image'
        if category not in TENSOR_CATEGORIES[dataset_type]:
            raise ValueError(f"Invalid tensor category: {category}")
        # Only quantize or change format for val set; cal set will keep using fp32
        if dataset_type == "val":
            if category in target_precision:
                value = reformatter(value, target_precision[category], scale.get(category, 0.0))
        my_target_dir_path = target_dir_path / category
        target_file_path = my_target_dir_path / f"{DATASET_TYPE_PREFIX[dataset_type][1]}_{idx}.npy"
        data_dumper(target_file_path, value, overwrite)
        if idx % 100 == 0:
            logging.info(f"Dumped {target_file_path}")
    return idx

def preprocess_nuscenes(dataset_dir: str, 
                        preprocessed_data_dir: str, 
                        overwrite: bool=False, 
                        target_precision: dict={},
                        scale: dict={},
                        dataset_type: str="val",
                        workers: int=8
                        ) -> None:
    """Preprocess the raw images for inference."""

    logging.info(f"Loading and converting {dataset_type} set...")
    # populate the file list for loading
    file_list, length_list = populate_file_info(dataset_dir, dataset_type, preprocessed_data_dir, overwrite)
    total_num_files = sum(length_list)
    with Pool(workers) as pool:
        done_indices = pool.starmap(load_split_dump, [f + (target_precision, scale) for f in file_list])
    if len(set(done_indices)) != total_num_files:
        raise ValueError(f"Total number of finished jobs {len(set(done_indices))} does not match total number of files {total_num_files}")
    
    # FIXME: Do I need scene lengths?
    # logging.info(f"Converted {total_num_files} files")
    # scene_lengths_file_path = Path(preprocessed_data_dir) / "scene_lengths" / "scene_lengths.npy"
    # data_dumper(scene_lengths_file_path, np.array(length_list), overwrite)
    # logging.info(f"Dumped scene lengths to {scene_lengths_file_path}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", "-d",
        help="Directory containing the input images.",
        default="build/data/nuScenes/preprocessed"
    )
    parser.add_argument(
        "--preprocessed_data_dir", "-o",
        help="Output directory for the preprocessed data.",
        default="build/preprocessed_data/nuscenes"
    )
    parser.add_argument(
        "--target_precision", "-t",
        help="Target precision for quantization; key:value pair of category:precision, comma separated list",
        default="image:int8,use_prev_bev:bool,can_bus:fp16,lidar2img:fp16"
    )
    parser.add_argument(
        "--scale", "-s",
        help="Scale for fp8/int8 quantization; key:value pair of category:scale, comma separated list",
        default="image:0.020782470703125"
    )
    parser.add_argument(
        "--overwrite", "-f",
        help="Overwrite existing files.",
        action="store_true"
    )
    parser.add_argument(
        "--workers",
        help="Number of workers to use for preprocessing.",
        default=8
    )
    parser.add_argument(
        "--dataset_type",
        help="Dataset type to preprocess.",
        default="val",
        choices=["val", "cal"]
    )

    args = parser.parse_args()
    args.scale = {k: float(v) for k, v in map(lambda x: x.split(":"), args.scale.split(","))}
    args.target_precision = {k: v for k, v in map(lambda x: x.split(":"), args.target_precision.split(","))}
    # Force cal set to use fp32; no quantization or format change
    if args.dataset_type == "cal":
        logging.info("Force using fp32 for cal set; no quantization or format change")
        args.target_precision = {k: "fp32" for k in TENSOR_CATEGORIES['cal']}
        args.scale = {}

    return args


def main():
    """
    <dataset>
     └── nuScenes
    And the output directory will have the following structure:
    <preprocessed_data>
     └── nuscenes
    """

    args = parse_args()
    data_dir = args.data_dir
    preprocessed_data_dir = args.preprocessed_data_dir
    overwrite = args.overwrite
    target_precision = args.target_precision
    img_target_precision = target_precision.get("img", "int8")
    workers = int(args.workers)
    dataset_type = args.dataset_type
    scale = args.scale
    logging.info(f"Preprocessing {dataset_type} set")
    logging.info(f"Using target precision: {target_precision}, with scale: {scale}")
        
    # Now, actually preprocess the input input data
    logging.info("Loading and preprocessing input data. This might take a while...")
    preprocess_nuscenes(data_dir, 
                        preprocessed_data_dir, 
                        overwrite, 
                        target_precision,
                        scale,
                        dataset_type,
                        workers)

    logging.info("Preprocessing done.")


if __name__ == '__main__':
    main()
