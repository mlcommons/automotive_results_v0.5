# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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


import os
import sys
import numpy as np
import argparse

from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, required=True)
args = parser.parse_args()

Path(args.output_dir).mkdir(parents=True, exist_ok=True)

scratch_path = os.environ.get("MLPERF_SCRATCH_PATH", None)
if scratch_path is None:
    raise ValueError("MLPERF_SCRATCH_PATH is not set")
    exit(-1)

mnt_mlperf_automotive_data = Path(scratch_path)
keys = ["can_bus", "lidar2img", "use_prev_bev", "prev_bev", "image"]
cal_dir = mnt_mlperf_automotive_data / "preprocessed_data" / "nuscenes" / "cal" / "fp32"

# fp8
print("Preparing fp8 calibration data")
with open(Path(__file__).parent / "calib_order.txt", "r") as f:
    indices = [int(line.strip().split()[1]) for line in f.readlines()]

merged_data = {}
for key in keys:
    merged_data[key] = []
    
    for idx in indices:
        obj_dir = cal_dir / key / f"calib_{idx}.npy"
        print(f"Loading {obj_dir}")
        obj = np.load(obj_dir, allow_pickle=True)
        if key == "use_prev_bev":
            item = obj.reshape(1)
        else:
            item = obj
        merged_data[key].append(item)

for k in keys:
    merged_data[k] = np.concatenate(merged_data[k], axis=0)
    print(f"{k}: {merged_data[k].shape}, {merged_data[k].dtype}")
np.savez(os.path.join(args.output_dir, "mlcommon_calibration_data_fp8.npz"), **merged_data)

print("Preparing int8 calibration data")
# int8
merged_data = {}
for key in keys:
    merged_data[key] = []
    
    files = sorted(list((cal_dir / key).glob("*.npy")))
    for f in files:
        print(f"Loading {f}")
        obj = np.load(f, allow_pickle=True)
        if key == "use_prev_bev":
            item = obj.reshape(1)
        else:
            item = obj
        merged_data[key].append(item)

for k in keys:
    merged_data[k] = np.concatenate(merged_data[k], axis=0)
    print(f"{k}: {merged_data[k].shape}, {merged_data[k].dtype}")
np.savez(os.path.join(args.output_dir, "mlcommon_calibration_data_int8.npz"), **merged_data)
