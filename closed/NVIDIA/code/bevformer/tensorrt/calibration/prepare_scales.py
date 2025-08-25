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


import onnx
import onnx_graphsurgeon as gs
import json
import os
import tempfile
import numpy as np

from modelopt.onnx.quantization import quantize

scratch_path = os.environ.get("MLPERF_SCRATCH_PATH", None)
if scratch_path is None:
    raise ValueError("MLPERF_SCRATCH_PATH is not set")
    exit(-1)

mlcommon_onnx = os.path.join(scratch_path, "models", "bevformer_tiny", "bevformer_tiny.onnx")
# temp_dir = tempfile.mkdtemp(dir="/tmp/", prefix="mlcommon_bevformer_")
temp_dir = "/work/build/calibration"

output_int8_dir = os.path.join(temp_dir, "mlcommons_bevformer.tmp.int8.onnx")

def _to_data(path):
    print(f"Loading calibration data from {path}")
    calibration_data_npz = np.load(path, allow_pickle=True)
    calibration_data = {key: calibration_data_npz[key] for key in calibration_data_npz.files}
    calibration_data["image"] = calibration_data["image"].reshape(-1, 6, 3, 480, 800)
    for k in calibration_data.keys():
        print(f"{k}: {calibration_data[k].shape}, {calibration_data[k].dtype}")
    return calibration_data

def _attach_postfix(node, postfix):
    node.name = node.name + postfix
    for output in node.outputs:
        output.name = output.name + postfix
    if node.op == "QuantizeLinear" or node.op == "DequantizeLinear":
        for input_tensor in node.inputs[1:]:
            input_tensor.name = input_tensor.name + postfix

quantize(
    onnx_path=mlcommon_onnx,
    quantize_mode="int8",
    calibration_data=_to_data("/work/build/calibration/mlcommon_calibration_data_int8.npz"),
    calibration_method="entropy",
    calibration_eps=["trt"],
    op_types_to_quantize=["Conv", "Add"],
    nodes_to_exclude=None,
    nodes_to_quantize=['/img_backbone/conv1/Conv', '/img_backbone/layer1/layer1.0/.*'],
    high_precision_dtype="fp16",
    mha_accumulation_dtype="fp16",
    disable_mha_qdq=True,
    # dq_only=True,
    output_path=output_int8_dir,
    simplify=True,
    # verbose=True,
)

output_fp8_dir = os.path.join(temp_dir, "mlcommons_bevformer.tmp.fp8.onnx")
quantize(
    onnx_path=mlcommon_onnx,
    quantize_mode="fp8",
    calibration_data=_to_data("/work/build/calibration/mlcommon_calibration_data.npz"),
    calibration_method="max",
    calibration_eps=["trt"],
    op_types_to_quantize=["Conv"],
    # trt_plugins=args.plugin,
    # trt_plugins_precision=[
    #     "RotatePlugin:fp16", 
    #     "MultiScaleDeformableAttentionPlugin:fp16",
    # ],
    nodes_to_exclude=None,
    nodes_to_quantize=None,
    high_precision_dtype="fp16",
    mha_accumulation_dtype="fp16",
    disable_mha_qdq=True,
    dq_only=False,
    output_path=output_fp8_dir,
    simplify=True,
    # verbose=True,
)

scales = {}
model_i8 = gs.import_onnx(onnx.load(output_int8_dir))
model_f8 = gs.import_onnx(onnx.load(output_fp8_dir))

lut_i8 = {n.name: n for n in model_i8.nodes}
lut_f8 = {}
for n in model_f8.nodes:
    lut_f8[n.name] = n

image = model_f8.tensors()["image"]
image.shape = [6, 3, 480, 800]
image.dtype = np.int8

conv_act_pool = [
    # lut_i8["/Squeeze_output_0_QuantizeLinear"],
    lut_i8["/Squeeze_output_0_DequantizeLinear"],
    lut_i8["onnx::Conv_11308_DequantizeLinear"],
    lut_i8["/img_backbone/conv1/Conv"],
    lut_i8["/img_backbone/relu/Relu"],
    lut_i8["/img_backbone/maxpool/MaxPool"],
    lut_i8["/img_backbone/maxpool/MaxPool_output_0_QuantizeLinear"],
    lut_i8["/img_backbone/maxpool/MaxPool_output_0_DequantizeLinear"],
]
for n in conv_act_pool:
    _attach_postfix(n, ".int8")

conv_act_pool[0].inputs[0] = image
y_dq = conv_act_pool[-1].outputs[0]

lut_f8["/img_backbone/maxpool/MaxPool_output_0_QuantizeLinear"].inputs[0] = y_dq
lut_f8["/img_backbone/maxpool/MaxPool"].outputs.clear()
model_f8.nodes.extend(conv_act_pool)
model_f8.toposort().cleanup()

mixed_path = os.path.join(temp_dir, "mlcommons_bevformer.tmp.fp8.mixed.onnx")
onnx.save(gs.export_onnx(model_f8), mixed_path)

model = gs.import_onnx(onnx.load(mixed_path))
for node in model.nodes:
    if node.op not in ["QuantizeLinear", "DequantizeLinear"]:
        continue
    scale = node.inputs[1].values
    if scale.size != 1:
        continue
    print(node.name, scale)
    scales[node.name] = float(scale)
    
with open("/work/build/calibration/scale_cache.json", "w") as f:
    json.dump(scales, f, indent=2)
