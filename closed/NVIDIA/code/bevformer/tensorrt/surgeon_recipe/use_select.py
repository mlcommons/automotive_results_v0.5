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

def use_select(src_model):
    print("use_select")
    lut = {n.name: n for n in src_model.nodes}

    flag = src_model.tensors()["use_prev_bev"] # lut["graph_input_cast2"]
    flag.dtype = bool

    # /pts_bbox_head/transformer/encoder/layers.0/attentions.0/MultiScaleDeformableAttentionPlugin
    # case 1
    prefix = "/pts_bbox_head/transformer/encoder/layers.0/attentions.0/"
    prev_bev = lut["/pts_bbox_head/transformer/Reshape_8"]
    embed = lut["/pts_bbox_head/transformer/Add_7"]

    select_out = gs.Variable(
        name=prefix + "/select_output",
        dtype=np.float16,
        shape=[1, 2500, 256],
    )
    select = gs.Node(
        op="Select_TRT",
        name=prefix + "/select",
        inputs=[prev_bev.outputs[0], embed.outputs[0], flag],
        outputs=[select_out],
    )

    src_model.nodes.append(select)
    feat_concat_out = gs.Variable(
        name=prefix + "feat_concat_output",
        dtype=np.float16,
        shape=[2, 2500, 256],
    )
    feat_concat = gs.Node(
        op="Concat",
        name=prefix + "feat_concat",
        attrs={"axis": 0},
        inputs=[select_out, embed.outputs[0]],
        outputs=[feat_concat_out],
    )
    src_model.nodes.append(feat_concat)

    feat = lut["/pts_bbox_head/transformer/encoder/layers.0/attentions.0/value_proj/MatMul"]
    feat.inputs[0] = feat_concat_out
    off = lut["/pts_bbox_head/transformer/encoder/layers.0/attentions.0/Concat"]
    off.inputs[0] = select_out

    for i in range(1, 3):
        prefix = f"/pts_bbox_head/transformer/encoder/layers.{i}/attentions.0/"
        embed_new = lut[f"/pts_bbox_head/transformer/encoder/layers.{i-1}/norms.2/LayerNormalization"]

        select0_out = gs.Variable(
            name=prefix + "/select0_output",
            dtype=np.float16,
            shape=[1, 2500, 256],
        )
        select0 = gs.Node(
            op="Select_TRT",
            name=prefix + "/select0",
            inputs=[prev_bev.outputs[0], embed_new.outputs[0], flag],
            outputs=[select0_out],
        )

        select1_out = gs.Variable(
            name=prefix + "/select1_output",
            dtype=np.float16,
            shape=[1, 2500, 256],
        )
        select1 = gs.Node(
            op="Select_TRT",
            name=prefix + "/select1",
            inputs=[embed.outputs[0], embed_new.outputs[0], flag],
            outputs=[select1_out],
        )
        src_model.nodes.append(select0)
        src_model.nodes.append(select1)

        concat = lut[f"/pts_bbox_head/transformer/encoder/layers.{i}/attentions.0/Concat"]
        concat.inputs[0] = select0_out

        feat_concat_out = gs.Variable(
            name=prefix + "/feat_concat_output",
            dtype=np.float16,
            shape=[2, 2500, 256],
        )
        feat_concat = gs.Node(
            op="Concat",
            name=prefix + "/feat_concat",
            attrs={"axis": 0},
            inputs=[select0_out, select1_out],
            outputs=[feat_concat_out],
        )
        src_model.nodes.append(feat_concat)
        feat = lut[f"/pts_bbox_head/transformer/encoder/layers.{i}/attentions.0/value_proj/MatMul"]
        feat.inputs[0] = feat_concat_out

    # lut["graph_input_cast2"].attrs["to"] = bool
    
    lut["/pts_bbox_head/transformer/encoder/Transpose_3"].outputs.clear()
    lut["/pts_bbox_head/transformer/encoder/layers.0/attentions.0/Add"].inputs[0] = embed.outputs[0]
    lut["/pts_bbox_head/transformer/encoder/layers.0/attentions.0/Add_3"].inputs[1] = embed.outputs[0]
    src_model.toposort().cleanup()
    print("end of use_select")
    return src_model
