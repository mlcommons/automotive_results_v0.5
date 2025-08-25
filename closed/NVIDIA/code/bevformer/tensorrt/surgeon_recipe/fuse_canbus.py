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
from pprint import pprint
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
import ctypes
import tensorrt as trt

def fuse_canbus(src_model, input_name):
    lut = {n.name: n for n in src_model.nodes}
    can_bus = src_model.tensors()[input_name]

    xy = gs.Variable(
        name="/pts_bbox_head/transformer/encoder/Preprocess_canbus_out_xy",
        dtype=np.float16,
        shape=(2,),
    )
    plugin = gs.Node(
        op="FuseCanbus_TRT", 
        name="/pts_bbox_head/transformer/encoder/Preprocess_canbus", 
        inputs=[can_bus], 
        domain="custom_op",
        outputs=[xy])
    src_model.nodes.extend([plugin])

    msda_tsa = [f"/pts_bbox_head/transformer/encoder/layers.{i}/attentions.0/MultiScaleDeformableAttentionPlugin" for i in range(3)]
    for name in msda_tsa:
        msda = lut[name]
        msda.inputs[3] = xy

    lut["/pts_bbox_head/transformer/encoder/layers.0/attentions.0/Unsqueeze_1"].outputs.clear()
    src_model.toposort().cleanup()
    return src_model
