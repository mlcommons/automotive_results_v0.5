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

import numpy as np
import onnx
import onnx_graphsurgeon as gs

from .fuse_msda import MSDA_MODE

def modify_duplicate_batch(model, names):
    br, mr, offset, weight, msda = names
    lut = {}
    for n in model.nodes:
        if n.name in names:
            lut[n.name] = n
    print(list(lut.keys()))

    node_mr = lut[mr]
    node_mr.inputs[0] = lut[br].inputs[0]
    node_mr.inputs[1] = gs.Constant(
        name=node_mr.inputs[1].name,
        values=np.array([1, 2500, 256], dtype=np.int64)
    )
    node_mr.outputs[0].shape = [1, 2500, 256]

    lut[br].outputs.clear()

    node_msda = lut[msda]

    visited = set()
    q = [node_mr]
    while len(q) > 0:
        next_node = q.pop(0)

        if next_node.name == msda:
            continue

        if next_node.op == "Reshape":
            # next_node.inputs[1].inputs[0].op
            if isinstance(next_node.inputs[1], gs.Constant):
                old_shape = next_node.inputs[1].values
            elif next_node.inputs[1].inputs[0].op == "Constant":
                old_shape = next_node.inputs[1].inputs[0].attrs["value"].values
            old_shape[0] = 1
            next_node.inputs[1] = gs.Constant(
                name=next_node.inputs[1].name,
                values=old_shape
            )
        else:
            for o in next_node.outputs:
                o.shape[0] = 1
        print(next_node.name)
        print(next_node.outputs[0].shape)

        for o in next_node.outputs:
            for no in o.outputs:
                if no.name not in visited:
                    q.append(no)
                    visited.add(no.name)
    
    node_msda.inputs[4] = lut[offset].outputs[0]
    node_msda.inputs[5] = lut[weight].outputs[0]
    node_msda.attrs["Mode"] = MSDA_MODE.kFusedScaStage2.value
    return model

def eliminate_scatter_before_msda(src_model):
    print("eliminate_scatter_before_msda")
    src_model = modify_duplicate_batch(src_model, [
        "/pts_bbox_head/transformer/encoder/layers.0/attentions.1/Reshape_1",
        "/pts_bbox_head/transformer/encoder/layers.0/attentions.1/Reshape_8",
        "/pts_bbox_head/transformer/encoder/layers.0/attentions.1/deformable_attention/sampling_offsets/Add",
        "/pts_bbox_head/transformer/encoder/layers.0/attentions.1/deformable_attention/attention_weights/Add",
        "/pts_bbox_head/transformer/encoder/layers.0/attentions.1/deformable_attention/MultiScaleDeformableAttentionPlugin",
    ])

    src_model = modify_duplicate_batch(src_model, [
        "/pts_bbox_head/transformer/encoder/layers.1/attentions.1/Reshape_1",
        "/pts_bbox_head/transformer/encoder/layers.1/attentions.1/Reshape_7",
        "/pts_bbox_head/transformer/encoder/layers.1/attentions.1/deformable_attention/sampling_offsets/Add",
        "/pts_bbox_head/transformer/encoder/layers.1/attentions.1/deformable_attention/attention_weights/Add",
        "/pts_bbox_head/transformer/encoder/layers.1/attentions.1/deformable_attention/MultiScaleDeformableAttentionPlugin",
    ])

    src_model = modify_duplicate_batch(src_model, [
        "/pts_bbox_head/transformer/encoder/layers.2/attentions.1/Reshape_1",
        "/pts_bbox_head/transformer/encoder/layers.2/attentions.1/Reshape_7",
        "/pts_bbox_head/transformer/encoder/layers.2/attentions.1/deformable_attention/sampling_offsets/Add",
        "/pts_bbox_head/transformer/encoder/layers.2/attentions.1/deformable_attention/attention_weights/Add",
        "/pts_bbox_head/transformer/encoder/layers.2/attentions.1/deformable_attention/MultiScaleDeformableAttentionPlugin",
    ])

    src_model.toposort().cleanup()
    print("end of eliminate_scatter_before_msda")
    return src_model
