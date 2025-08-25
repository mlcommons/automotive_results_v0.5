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

from nvmitten.nvidia.cupy import CUDARTWrapper as cudart

import tensorrt as trt
import os
import numpy as np

class BevFormerCalibrator(trt.IInt8EntropyCalibrator2):
    """calibrator for the BevFormer benchmark."""

    def __init__(self, calib_batch_size=1, calib_max_batches=500, force_calibration=False,
                 cache_file="code/bevformer/tensorrt/calibrator.cache",
                 tensor_dirs="build/preprocessed_data/nuscenes/cal/fp32/image,"
                             "build/preprocessed_data/nuscenes/cal/fp32/prev_bev,"
                             "build/preprocessed_data/nuscenes/cal/fp32/use_prev_bev,"
                             "build/preprocessed_data/nuscenes/cal/fp32/can_bus,"
                             "build/preprocessed_data/nuscenes/cal/fp32/lidar2img",
                 calib_data_map="data_maps/nuscenes/cal_file_map.txt"):
        # Whenever you specify a custom constructor for a TensorRT class,
        # you MUST call the constructor of the parent explicitly.

        self.calib_batch_size = calib_batch_size
        self.calib_max_batches = calib_max_batches
        self.force_calibration = force_calibration
        self.current_idx = 0
        self.cache_file = cache_file

        # Calibration should be done in x86 or GraceHopper/GraceBlackwell host where proper ONNX runtime is available
        # Please refer to the README.md for more details.

        print("Current system cannot perform calibration due to lack of ONNX runtime.")
        print("Please refer to the README.md for more details.")
        exit(1)

    def get_batch_size(self):
        return self.calib_batch_size

    def get_batch(self, names):
        """
        Acquire a single batch

        Arguments:
        names (string): names of the engine bindings from TensorRT. Useful to understand the order of inputs.
        """
        # When we're out of batches, we return either [] or None.
        # This signals to TensorRT that there is no calibration data remaining.
        return None

    def read_calibration_cache(self):
        return self.cache

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)

    def clear_cache(self):
        self.cache = None

    def __del__(self):
        cudart.cudaFree(self.device_input)
