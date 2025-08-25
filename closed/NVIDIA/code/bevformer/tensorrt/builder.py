#!/usr/bin/env python3
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

from __future__ import annotations
from importlib import import_module
from os import PathLike
from pathlib import Path
from typing import Optional, Dict
import subprocess
import tensorrt as trt
import onnx
import re
import bz2

from code.common import dict_get
from code.common.fields import Fields
from code.common.mitten_compat import ArgDiscarder
from code.common.systems.system_list import DETECTED_SYSTEM
from code.common import logging


BevFormerCalibrator = import_module("code.bevformer.tensorrt.calibrator").BevFormerCalibrator
BevFormerGraphSurgeon = import_module("code.bevformer.tensorrt.bevformer_graphsurgeon").BevFormerGraphSurgeon

class BevFormerEngineBuilder:
    """BevFormer end to end base builder class.
    """

    def __init__(self,
                 *args,
                 system_id: str = "Thor-X",
                 benchmark: str = "bevformer",
                 scenario: str = "SimpleStream",
                 gpu_batch_size: int = 1,
                 input_dtype: str = "fp16",
                 accuracy_target: str = "k_99",
                 harness_type: str = "custom",
                 power_setting: str = "MaxP",
                 config_ver: str = "default",
                 model_path: str = "build/models/bevformer_tiny/bevformer_tiny.onnx",
                 optimized_model_path: str = "build/optimized/bevformer_tiny.onnx",
                 plugin_file: str = "build/plugins/BevFormerPlugin/libbev_plugins.so",
                 component: str = None,
                 **kwargs):
        self.config_ver = config_ver
        self.model_path = model_path
        self.optimized_model_path = optimized_model_path
        self.plugin_file = plugin_file
        self.component = component
        self.engine_dir = Path(f"/work/build/engines/{system_id}/{benchmark}/{scenario}")
        self.engine_dir.mkdir(parents=True, exist_ok=True)
        self.engine_file = self.engine_dir / f"{benchmark}-{scenario}-gpu-b{gpu_batch_size}-{input_dtype}.{harness_type}_{accuracy_target}_{power_setting}.plan"

    def build_engines(self):
        logging.info(f"Optimizing network from {self.model_path} to {self.optimized_model_path}")
        self.optimize_network()
        logging.info(f"Building engine from {self.optimized_model_path} to {self.engine_file}")
        self.build_engine_with_trtexec()
        logging.info(f"Engine built successfully")

    def optimize_network(self):
        bevformer_gs = BevFormerGraphSurgeon(self.model_path,
                                             self.optimized_model_path,
                                             self.plugin_file,
                                             force_rebuild=False)
        bevformer_gs.run_graphsurgeon()

        return self.optimized_model_path

    def build_engine_with_trtexec(self):
        # just check for the files for now for dependency
        optimized_model_path = Path(self.optimized_model_path)
        assert optimized_model_path.is_file(), f"Optimized model file {optimized_model_path} does not exist"
        plugin_file_path = Path(self.plugin_file)
        assert plugin_file_path.is_file(), f"Plugin file {plugin_file_path} does not exist"

        # using trtexec, build the engine
        model_onnx = self.optimized_model_path
        plugin_file = self.plugin_file
        engine_file = self.engine_file
        timing_cache_bz2_file = Path("/work/code/bevformer/tensorrt/timing.cache.bz2")
        timing_cache_file = Path("/work/code/bevformer/tensorrt/timing.cache")

        try:
            logging.info("Decompressing timing cache file for faster build")
            with bz2.open(timing_cache_bz2_file, 'rb') as bz2f: 
                decompressed_data = bz2f.read()
            logging.info("Writing decompressed timing cache")
            with open(timing_cache_file, 'wb') as outf:
                outf.write(decompressed_data)
            print(f"File '{timing_cache_bz2_file}' successfully decompressed to '{timing_cache_file}'.")
        except FileNotFoundError:
            print(f"Error: The file '{timing_cache_bz2_file}' was not found.")
        except Exception as e:
            print(f"An error occurred during decompression: {e}")

        # Build engine using trtexec
        cmds = [
            "/usr/local/tensorrt/bin/trtexec",
            f"--onnx={model_onnx}",
            f"--saveEngine={engine_file}",
            f"--plugins={plugin_file}",
            "--useCudaGraph",
            "--builderOptimizationLevel=5",
            "--tilingOptimizationLevel=3",
            "--maxTactics=2000",
            "--maxAuxStreams=0",
            "--noTF32",
            "--stronglyTyped",
            "--sparsity=disable",
            "--inputIOFormats=int8:chw,fp16:chw,bool:chw,fp16:chw,fp16:chw",
            "--outputIOFormats=fp16:chw,fp32:chw,fp32:chw",
            "--iterations=100",
        ]

        if timing_cache_file.exists():
            cmds.append(f"--timingCacheFile={timing_cache_file}")
        
        result = subprocess.run(cmds, capture_output=True, text=True, check=True)
        logging.info(f"trtexec result: {result.stdout}")
        
        if result.returncode != 0:
            raise RuntimeError(f"trtexec failed with return code {result.returncode}: {result.stderr}")

class BEVFormer(BevFormerEngineBuilder):
    def __init__(self, args):
        system_id = args.get("system_id", "Thor-X")
        benchmark = args.get("benchmark").name.lower()
        scenario = args.get("scenario").name
        batch_size = args.get("gpu_batch_size", dict()).get(benchmark, 1)
        input_dtype = args.get("input_dtype").lower()
        workload_setting = args.get("workload_setting")
        accuracy = workload_setting.accuracy_target.name.lower()
        harness = workload_setting.harness_type.name.lower()
        power = workload_setting.power_setting.name
        config_ver = "default"
        model_path = "build/models/bevformer_tiny/bevformer_tiny.onnx"
        optimized_model_path = "build/optimized/bevformer_tiny.onnx"
        plugin_file = "build/plugins/BevFormerPlugin/libbev_plugins.so"
        component = None
        super().__init__(system_id=system_id,
                         benchmark=benchmark,
                         scenario=scenario,
                         gpu_batch_size=batch_size,
                         input_dtype=input_dtype,
                         accuracy_target=accuracy,
                         harness_type=harness,
                         power_setting=power,
                         config_ver=config_ver,
                         model_path=model_path,
                         optimized_model_path=optimized_model_path,
                         plugin_file=plugin_file,
                         component=component,
                         *args)

    def build_engines(self):
        super().build_engines()
