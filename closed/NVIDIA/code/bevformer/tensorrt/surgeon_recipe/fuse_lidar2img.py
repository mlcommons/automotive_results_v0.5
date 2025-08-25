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

def fuse_lidar2img(src_model, input_name):
    lut = {n.name: n for n in src_model.nodes}

    lidar2img = src_model.tensors()[input_name]
    # to replace -> /pts_bbox_head/transformer/encoder/layers.0/attentions.1/Reshape_9_output_0
    reference_points = gs.Variable(
        name="/pts_bbox_head/transformer/encoder/Preprocess_lidar2img_out_reference_points",
        dtype=np.float16,
        shape=(6, 2500, 4, 2),
    )
    # to replace -> /pts_bbox_head/transformer/encoder/layers.0/attentions.1/Cast_1_output_0
    mask = gs.Variable(
        name="/pts_bbox_head/transformer/encoder/Preprocess_lidar2img_out_mask",
        dtype=np.float16,
        shape=(1, 6, 2500, 1),
    )

    plugin = gs.Node(
        op="FuseLidar2Img_TRT", 
        name="/pts_bbox_head/transformer/encoder/Preprocess_lidar2img", 
        inputs=[lidar2img], 
        domain="custom_op",
        outputs=[reference_points, mask])

    for i in range(3):
        key = f"/pts_bbox_head/transformer/encoder/layers.{i}/attentions.1/deformable_attention/Unsqueeze_2"
        if key in lut:
            unsqueeze_2 = lut[key]
            old_input = unsqueeze_2.inputs[0].inputs[0]
            old_input.outputs.clear()
            unsqueeze_2.inputs[0] = reference_points

    # reference_points output
    old_ref = lut["/pts_bbox_head/transformer/encoder/layers.0/attentions.1/Reshape_9"]
    old_ref.outputs.clear()

    sca_names = [
        "/pts_bbox_head/transformer/encoder/layers.0/attentions.1/deformable_attention/MultiScaleDeformableAttentionPlugin",
        "/pts_bbox_head/transformer/encoder/layers.1/attentions.1/deformable_attention/MultiScaleDeformableAttentionPlugin",
        "/pts_bbox_head/transformer/encoder/layers.2/attentions.1/deformable_attention/MultiScaleDeformableAttentionPlugin",
    ]
    for sca_name in sca_names:
        sca = lut[sca_name]
        sca.inputs[5] = mask

    cast_1 = lut["/pts_bbox_head/transformer/encoder/layers.0/attentions.1/Cast_1"]
    cast_1.outputs.clear()

    src_model.nodes.extend([plugin])
    src_model.toposort().cleanup()
    return src_model
