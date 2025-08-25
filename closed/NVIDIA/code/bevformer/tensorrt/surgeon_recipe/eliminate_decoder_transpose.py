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

import onnx_graphsurgeon as gs
import numpy as np

def _remove(node):
    in_tensor = node.inputs[0]
    next_node = node.outputs[0].outputs[0]
    next_node.inputs[0] = in_tensor
    node.outputs.clear()

def _adjust_B(node):
    if node.op == "Add":
        i0 = node.inputs[0]
        i1 = node.inputs[1]

        if isinstance(i1, gs.Constant):
            B = i1
        else:
            B = i0
        B_shape = B.values.shape
        if B_shape[0] == 900 and B_shape[1] == 1:
            print(f"adjust Add: {node.name}")
            B.values = B.values.reshape(1, 900, -1)
        else:
            print(f"won't adjust {node.name}, B.shape: {B_shape}")
    elif node.op == "Reshape":
        shape = node.inputs[1].values
        if shape[0] == 900 and shape[1] == 1:
            print(f"adjust Reshape: {node.name}")
            node.inputs[1].values = np.array([1, 900, -1], dtype=np.int64)
        else:
            print(f"won't adjust {node.name}, shape: {shape}")

def eliminate_decoder_transpose(src_model):
    print("eliminate_decoder_transpose")

    lut = {n.name: n for n in src_model.nodes}
    for k, v in lut.items():
        if v.op != "Transpose":
            continue
        s0 = v.inputs[0].shape
        o0 = v.outputs[0].shape
        f0 = s0[0] == 900 and s0[1] == 1
        f1 = o0[0] == 900 and o0[1] == 1
        if len(v.attrs["perm"]) < 3:
            continue
        f2 = v.attrs["perm"][0] == 1 and v.attrs["perm"][1] == 0 and v.attrs["perm"][2] == 2
        if (f0 or f1) and f2:
            print(f"remove {v.name}")
            _remove(v)

    to_adjust = ["/pts_bbox_head/transformer/decoder/layers.0/attentions.1/Add_3"]
    for i in range(1, 6):
        to_adjust.extend([
            f"/pts_bbox_head/transformer/decoder/layers.{i}/attentions.0/Add",
            f"/pts_bbox_head/transformer/decoder/layers.{i}/attentions.0/attn/Reshape_4",
            f"/pts_bbox_head/transformer/decoder/layers.{i}/attentions.1/Add",
        ])
    
    for n in to_adjust:
        node = lut[n]
        print(f"adjust {node.name}")
        _adjust_B(node)

    src_model.toposort().cleanup()
    print("end of eliminate_decoder_transpose")
    return src_model
