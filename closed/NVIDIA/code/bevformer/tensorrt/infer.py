#! /usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from nvmitten.nvidia.runner import EngineRunner

import os
import json
import subprocess
import re
import torch
import tensorrt as trt
import numpy as np
from pathlib import Path
from tqdm import tqdm

import code.common.arguments as common_args
from code.common import logging
from code.common.constants import TRT_LOGGER
from code.plugin import load_trt_plugin_by_network

G_NUSCENES_VAL_SET_PATH = Path("data_maps")/ "nuscenes" / "val_map.txt"
G_NUSCENES_INPUT_CATEGORIES = ["image", "use_prev_bev", "can_bus", "lidar2img"]
G_ACCURACY_DUMP_FILE = Path("accuracy_results.json").absolute()
G_TOTAL_NUSCENES_VAL_SAMPLES = 6019

class BevFormerTrtRunner:
    def __init__(self, 
                 engine_file: str, 
                 batch_size: int = 1, 
                 precision: str = "fp32",
                 onnx_path: str = "build/models/bevformer_tiny/bevformer_tiny.onnx", 
                 engine_build: bool = False, 
                 output_file: str = "build/tmp/bevformer_trt.out",
                 verbose: bool = False,
                 ):
        """
        Test the accuracy using the onnx file and TensorRT runtime.
        The tester is able to build the engine from onnx.
        """
        # tracker
        self.inference_count = 0
        self.prev_bev = torch.zeros(2500, 1, 256, dtype=torch.float32).numpy()
        self.prev_frame_info = {"scene_token": None, "prev_pos": None, "prev_angle": None}

        # params
        self.engine_file = engine_file
        self.batch_size = batch_size
        self.precision = precision
        self.onnx_path = onnx_path
        self.engine_build = engine_build
        self.output_file = output_file
        self.verbose = verbose

        # plugin and logger
        self.logger = TRT_LOGGER  # Use the global singleton, which is required by TRT.
        self.logger.min_severity = trt.Logger.VERBOSE if self.verbose else trt.Logger.INFO
        load_trt_plugin_by_network("bevformer")
        trt.init_libnvinfer_plugins(self.logger, "")

        if self.onnx_path and self.engine_build:
            print(f"Creating engines from onnx: {self.onnx_path}")
            self.create_trt_engine()
        else:
            if not Path(self.engine_file).is_file():
                raise RuntimeError(f"Cannot find engine file {self.engine_file}. Please provide either the onnx file or engine file.")

        # runner from engine
        self.runner = EngineRunner.from_file(self.engine_file, verbose=self.verbose)

        # input related
        self.input_img_name = self.runner.engine.get_tensor_name(0)
        self.input_prev_bev_name = self.runner.engine.get_tensor_name(1)
        self.input_use_prev_bev_name = self.runner.engine.get_tensor_name(2)
        self.input_can_bus_name = self.runner.engine.get_tensor_name(3)
        self.input_lidar2img_name = self.runner.engine.get_tensor_name(4)
        
        self.input_dtype = self.runner.engine.get_tensor_dtype(self.input_img_name)
        self.input_format = self.runner.engine.get_tensor_format(self.input_img_name)
        if self.input_dtype == trt.DataType.FLOAT:
            format_string = "fp32"
        elif self.input_dtype == trt.DataType.HALF:
            format_string = "fp16"
        elif self.input_dtype == trt.DataType.INT8:
            format_string = "int8"
        else:
            raise ValueError(f"Unsupported input dtype: {self.input_dtype}")
        self.input_dir = Path(os.getenv("PREPROCESSED_DATA_DIR", "build/preprocessed_data")) / "nuscenes" / "val" / format_string

    def create_trt_engine(self):
        def apply_flag(self, flag):
            """Apply a TRT builder flag."""
            self.builder_config.flags = (self.builder_config.flags) | (1 << int(flag))

        def clear_flag(self, flag):
            """Clear a TRT builder flag."""
            self.builder_config.flags = (self.builder_config.flags) & ~(1 << int(flag))

        raise NotImplementedError(f"NOT IMPLEMENTED: create_trt_engine")

    def run_single(self, input_tensor: dict) -> tuple[int, np.ndarray, np.ndarray]:
        self.inference_count += 1

        # run inference
        inputs = [
            input_tensor[self.input_img_name],
            self.prev_bev,
            input_tensor[self.input_use_prev_bev_name],
            input_tensor[self.input_can_bus_name],
            input_tensor[self.input_lidar2img_name],
        ]

        outputs = self.runner(inputs, batch_size=self.batch_size)

        bev_embed = outputs[0]
        outputs_classes = outputs[1]
        outputs_coords = outputs[2]

        # prev_bev still in CUDA device
        # self.prev_bev = self.runner.outputs[0]
        self.prev_bev = bev_embed

        return (input_tensor["index"], outputs_classes, outputs_coords)

def run_bevformer(engine_file: str, 
                  num_input_scenes: int, 
                  batch_size: int = 1, 
                  verbose: bool = False,
                  accuracy: bool = True,
                  block_size: int = 512,
                  ):
    """
    A local runner to test Engine accuracy on a (sub)set of input samples.
    This will run accuracy tests *WITHOUT* Loadgen.
    """
    if verbose:
        logging.info("Running BEVFormer accuracy test with:")
        logging.info("    engine_file: {:}".format(engine_file))
        logging.info("    num scenes : {:}".format(num_input_scenes))
        logging.info("    batch_size : {:}".format(batch_size))

    runner = BevFormerTrtRunner(engine_file=engine_file,
                                batch_size=batch_size,
                                verbose=verbose)
    
    outputs = []
    # build dict based on number of input scenes from nuscenes val map file
    input_sample_indices = build_input_samples_from_scenes(num_input_scenes)

    # run inference in blocks
    logging.info(f"Running inferences...")
    for i in range(0, len(input_sample_indices), block_size):
        logging.info(f"Loading {len(input_sample_indices[i:i+block_size])} input tensors in block {i//block_size}")
        input_tensors = load_input_tensors(runner, input_sample_indices[i:i+block_size])
        logging.info(f"Running {len(input_tensors)} inferences in block {i//block_size}")
        for input_tensor in tqdm(input_tensors):
            outputs.append(runner.run_single(input_tensor))
        logging.info(f"Finished running {len(input_tensors)} inferences in block {i//block_size};"
                     f"total {len(outputs)} / {G_TOTAL_NUSCENES_VAL_SAMPLES} inferences done")
        # free memory
        del input_tensors

    # free memory
    del input_sample_indices
    
    # keep the number of outputs
    num_outputs = len(outputs)

    if accuracy:
        dump_accuracy_results(outputs)
        # free memory
        del outputs

        # if all samples ran, calculate accuracy
        if num_outputs == G_TOTAL_NUSCENES_VAL_SAMPLES:
            calculate_and_print_accuracy()
        else:
            logging.warning(f"Didn't run the whole set, skipping accuracy calculation")

    logging.info(f"Ran total {num_outputs} inferences for {num_input_scenes} scenes")
    
    return

def build_input_samples_from_scenes(num_input_scenes: int) -> list[int]:
    """
    Build the input scenes based on the number of input scenes.
    """
    with open(G_NUSCENES_VAL_SET_PATH) as f:
        val = f.readlines()
    val = [line.strip() for line in val if not line.startswith("#")]
    input_scenes = val[:num_input_scenes]
    input_scenes = [list(map(int, s.split(','))) for s in input_scenes]
    input_samples = []
    for start, length in input_scenes:
        input_samples.extend(range(start, start + length))
    return input_samples

def load_input_tensors(runner: BevFormerTrtRunner, input_sample_indices: list[int]) -> list[dict]:
    """
    Load the input tensors for the given input sample index.
    """
    input_tensors = []
    logging.debug(f"Loading {len(input_sample_indices)} input tensors")
    for input_sample_index in tqdm(input_sample_indices):
        # find input tensors from the index
        input_dict = {
            runner.input_img_name: runner.input_dir / "image" / f"val_{input_sample_index}.npy",
            runner.input_use_prev_bev_name: runner.input_dir / "use_prev_bev" / f"val_{input_sample_index}.npy",
            runner.input_can_bus_name: runner.input_dir / "can_bus" / f"val_{input_sample_index}.npy",
            runner.input_lidar2img_name: runner.input_dir / "lidar2img" / f"val_{input_sample_index}.npy",
        }

        # load input tensors
        input_tensors.append({k: np.load(v) for k, v in input_dict.items()})
        input_tensors[-1]["index"] = input_sample_index

    logging.debug(f"Loaded {len(input_tensors)} input tensors")
    return input_tensors

def dump_accuracy_results(outputs: list, json_file: str = G_ACCURACY_DUMP_FILE):
    """
    Calculate the accuracy of the output.
    """
    print(f"Total number of outputs: {len(outputs)}")
    final_results = []
    for output in outputs:
        final_results.append({
            "qsl_idx": output[0],
            "data": torch.stack([torch.tensor(o) for o in output[1:]]).numpy().tobytes(order="C").hex()
        })
    
    with open(json_file, "w") as f:
        json.dump(final_results, f, sort_keys=True, indent=4)
    logging.info(f"Results dumped to {G_ACCURACY_DUMP_FILE}")

    return

def calculate_and_print_accuracy(accuracy_file: str = G_ACCURACY_DUMP_FILE):
    """
    Calculate the accuracy of the output.
    """

    # reference mAP from https://github.com/mlcommons/mlperf_automotive/tree/master/automotive/camera-3d-detection
    reference_mAP = 0.2683556 

    # running accuracy script and fetch mAP
    logging.info(f"Running accuracy script to calculate the accuracy")

    nuscenes_dir = Path(os.getenv("MLPERF_DATA_DIR", "/work/build/data")) / "nuScenes"
    accuracy_script = Path("/work/build/automotive/automotive/camera-3d-detection/accuracy_nuscenes_cpu.py")
    accuracy_cmd = ["python",
                    f"{accuracy_script}",
                    f"--mlperf-accuracy-file={accuracy_file}",
                    f"--nuscenes-dir={nuscenes_dir}",
                    ]
    try:
        accuracy_result = subprocess.run(accuracy_cmd, check=True, capture_output=True)
        map_pattern = r"'pts_bbox_NuScenes/mAP': (0.\d+)"
        acc = float(re.search(map_pattern, accuracy_result.stdout.decode("utf-8")).group(1))
        logging.info(f"Accuracy [mAP]: {acc}, reference: {reference_mAP}, % of ref: {round(acc / reference_mAP, 3)}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running accuracy script: {e}")
        logging.error(f"Error output: {e.stderr.decode('utf-8')}")
        raise e


def main():
    args = common_args.parse_args(common_args.ACCURACY_ARGS)
    logging.info("Running accuracy test...")
    run_bevformer(engine_file=args["engine_file"],
                  num_input_scenes=args["num_scenes"],
                  batch_size=args["batch_size"],
                  verbose=args["verbose"],
                  accuracy=True,
                  )

if __name__ == "__main__":
    main()
