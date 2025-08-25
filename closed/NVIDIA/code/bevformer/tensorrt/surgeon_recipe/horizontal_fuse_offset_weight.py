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

from .fuse_msda import MSDA_MODE

def horizontal_fuse_offset_weight(src_model: gs.Graph):
    lut = {n.name: n for n in src_model.nodes}

    def _search_matmul_add(node):
        matmul_node = None
        add_node = None
        current_node = node
        while matmul_node is None or add_node is None:
            if current_node.op == "MatMul":
                matmul_node = current_node
            elif current_node.op == "Add":
                add_node = current_node
            
            for input in current_node.inputs:
                if isinstance(input, gs.Constant):
                    continue
                current_node = input.inputs[0]
                break
        return matmul_node, add_node
    
    # TSA
    # /pts_bbox_head/transformer/encoder/layers.[0~2]/attentions.0/MultiScaleDeformableAttentionPlugin
    def fuse_tsa(lut):
        for i in range(3):
            node_name = f"/pts_bbox_head/transformer/encoder/layers.{i}/attentions.0/MultiScaleDeformableAttentionPlugin"
            msda = lut[node_name]
            msda.attrs["Mode"] = MSDA_MODE.kFusedTsaStage3.value
            offset_matmul, offset_add = _search_matmul_add(msda.inputs[4].inputs[0])        
            weight_matmul, weight_add = _search_matmul_add(msda.inputs[5].inputs[0])
            print(offset_matmul.name, offset_add.name)
            print(weight_matmul.name, weight_add.name)
            offset_matmul.inputs[1].values = np.concatenate([offset_matmul.inputs[1].values, weight_matmul.inputs[1].values], axis=-1)
            offset_matmul.name = f"/pts_bbox_head/transformer/encoder/layers.{i}/attentions.0/fused_offset_weight/MatMul"
            offset_add.inputs[0].values = np.concatenate([offset_add.inputs[0].values, weight_add.inputs[0].values], axis=-1)
            offset_add.name = f"/pts_bbox_head/transformer/encoder/layers.{i}/attentions.0/fused_offset_weight/Add"
            msda.inputs[4] = offset_add.outputs[0]
            msda.inputs.pop(5)

    fuse_tsa(lut)

    # SCA
    # /pts_bbox_head/transformer/encoder/layers.[0~2]/attentions.1/MultiScaleDeformableAttentionPlugin
    def fuse_sca(lut):
        for i in range(3):
            node_name = f"/pts_bbox_head/transformer/encoder/layers.{i}/attentions.1/deformable_attention/MultiScaleDeformableAttentionPlugin"
            msda = lut[node_name]
            msda.attrs["Mode"] = MSDA_MODE.kFusedScaStage3.value
            offset_matmul, offset_add = _search_matmul_add(msda.inputs[4].inputs[0])
            weight_matmul, weight_add = _search_matmul_add(msda.inputs[5].inputs[0])
            print(offset_matmul.name, offset_add.name)
            print(weight_matmul.name, weight_add.name)
            offset_matmul.inputs[1].values = np.concatenate([offset_matmul.inputs[1].values, weight_matmul.inputs[1].values], axis=-1)
            offset_matmul.name = f"/pts_bbox_head/transformer/encoder/layers.{i}/attentions.1/deformable_attention/fused_offset_weight/MatMul"
            offset_add.inputs[0].values = np.concatenate([offset_add.inputs[0].values, weight_add.inputs[0].values], axis=-1)
            offset_add.name = f"/pts_bbox_head/transformer/encoder/layers.{i}/attentions.1/deformable_attention/fused_offset_weight/Add"
            msda.inputs[4] = offset_add.outputs[0]
            msda.inputs.pop(5)
            # # handle div
            # div = msda.outputs[0].outputs[0]
            # print(div.name)
            out_proj = msda.outputs[0].outputs[0]     
            print(out_proj.name)
            out_proj.inputs[0] = msda.outputs[0] # fuse div

    fuse_sca(lut)

    # Decoder MSDA
    # /pts_bbox_head/transformer/decoder/layers.[0~5]/attentions.1/MultiScaleDeformableAttentionPlugin
    def fuse_decoder_msda(lut):
        # skip the first msda in decoder?
        for i in range(1, 6):
            node_name = f"/pts_bbox_head/transformer/decoder/layers.{i}/attentions.1/MultiScaleDeformableAttentionPlugin"
            msda = lut[node_name]
            msda.attrs["Mode"] = MSDA_MODE.kFusedDecoderStage3.value
            offset_matmul, offset_add = _search_matmul_add(msda.inputs[4].inputs[0])
            weight_matmul, weight_add = _search_matmul_add(msda.inputs[5].inputs[0])
            print(offset_matmul.name, offset_add.name)
            print(weight_matmul.name, weight_add.name)
            offset_matmul.inputs[1].values = np.concatenate([offset_matmul.inputs[1].values, weight_matmul.inputs[1].values], axis=-1)
            offset_matmul.name = f"/pts_bbox_head/transformer/decoder/layers.{i}/attentions.1/fused_offset_weight/MatMul"
            offset_add.inputs[0].values = np.concatenate([offset_add.inputs[0].values, weight_add.inputs[0].values], axis=-1)
            offset_add.name = f"/pts_bbox_head/transformer/decoder/layers.{i}/attentions.1/fused_offset_weight/Add"
            msda.inputs[4] = offset_add.outputs[0]
            msda.inputs.pop(5)
            
    fuse_decoder_msda(lut)

    src_model.toposort().cleanup()
    return src_model
