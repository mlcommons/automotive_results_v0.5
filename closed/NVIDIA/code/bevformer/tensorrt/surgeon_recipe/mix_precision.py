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

import onnx
import onnx.version
import onnx_graphsurgeon as gs
import numpy as np
import tensorrt as trt
from pathlib import Path
import json

def _attach_postfix(node, postfix):
    node.name = node.name + postfix
    for output in node.outputs:
        output.name = output.name + postfix
    if node.op == "QuantizeLinear" or node.op == "DequantizeLinear":
        for input_tensor in node.inputs[1:]:
            input_tensor.name = input_tensor.name + postfix

def mix_precision(model, model_i8, model_f8):
    lut = {n.name: n for n in model.nodes}
    lut_i8 = {n.name: n for n in model_i8.nodes}
    lut_f8 = {}
    for n in model_f8.nodes:
        lut_f8[n.name] = n

    image = model.tensors()["image"]
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

    lut["/img_neck/fpn_convs.0/conv/Conv"].outputs.clear()
    lut["/pts_bbox_head/transformer/Add_8"].inputs[0] = lut_f8["/img_neck/fpn_convs.0/conv/Conv"].outputs[0]

    model.nodes.extend(conv_act_pool)
    model.nodes.extend([v for _, v in lut_f8.items()])
    model.toposort().cleanup()

    with open(Path(__file__).parent / "scale_cache.json", "r") as f:
        scales = json.load(f)

    for node in model.nodes:
        if node.name in scales:
            # make sure we use cached scales
            node.inputs[1].values = np.array(scales[node.name], dtype=np.float16)

    return model
