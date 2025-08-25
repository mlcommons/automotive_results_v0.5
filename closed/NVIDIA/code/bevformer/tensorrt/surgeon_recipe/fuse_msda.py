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

from enum import Enum
import onnx_graphsurgeon as gs
import numpy as np

class MSDA_MODE(Enum):
    kVanila = 0
    kFusedDiv = 1
    kFusedReduceMean = 2
    kFusedTsaStage2 = 3
    kFusedScaStage2 = 4
    # stage3, fp16
    kFusedTsaStage3 = 5
    kFusedScaStage3 = 6
    kFusedDecoderStage3 = 7

SCA = [f"/pts_bbox_head/transformer/encoder/layers.{i}/attentions.1/deformable_attention/MultiScaleDeformableAttentionPlugin" for i in range(3)]
TSA = [f"/pts_bbox_head/transformer/encoder/layers.{i}/attentions.0/MultiScaleDeformableAttentionPlugin" for i in range(3)]
# DEC = []

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

def fuse_div_msda(model):
    print("fuse_div_msda")
    for n in model.nodes:
        if n.op == "MultiScaleDeformableAttentionPlugin":
            n.attrs["Mode"] = 1
            # observe input[3]
            loc = n.inputs[3].inputs[0]

            if loc.op == "Reshape":
                # sca
                add = loc.inputs[0].inputs[0]
                div = add.inputs[1].inputs[0].inputs[0].inputs[0]
                print(f"sca, {add.name}, {div.name}")
            elif loc.op == "Add":
                # tsa / msda
                add = loc
                div = add.inputs[1].inputs[0]
                print(f"tsa/msda, {add.name}, {div.name}")
            inputs_before = n.inputs
            n.inputs = [
                inputs_before[0],  # value
                inputs_before[1],  # spatial_shapes
                inputs_before[2],  # level_start_index
                add.inputs[0],     # ref
                div.inputs[0],     # offset
                inputs_before[4],  # weight
            ]
            add.outputs.clear()
            div.outputs.clear()

    model.cleanup().toposort()
    print("end of fuse_div_msda")
    return model

def fuse_reduce_mean_sca_msda(model):
    print("fuse_reduce_mean_sca_msda")
    for msda in model.nodes:
        f0 = msda.op == "MultiScaleDeformableAttentionPlugin"
        f1 = msda.name in SCA
        if f0 and f1:
            print(msda.name)
            # msda -> reshape -> mul -> reducemean
            reshape = msda.outputs[0].outputs[0]
            mul = reshape.outputs[0].outputs[0]
            reducemean = mul.outputs[0].outputs[0]
            
            msda.inputs.append(mul.inputs[1])
            msda.attrs["Mode"] = MSDA_MODE.kFusedScaStage3.value
            msda.outputs = [reducemean.outputs[0]]
            
            reducemean.outputs.clear()

            # handle div
            div = msda.outputs[0].outputs[0]
            print(div.name)
            out_proj = div.outputs[0].outputs[0]        
            print(out_proj.name)
            out_proj.inputs[0] = msda.outputs[0] # fuse div
            print(f"fused msda: {msda.name}, mul: {mul.name}, reducemean: {reducemean.name}, div: {div.name}")

    model.toposort().cleanup()
    print("end of fuse_reduce_mean_sca_msda")
    return model

def fuse_softmax_sca_msda(model):
    layer_names = [(
        "/pts_bbox_head/transformer/encoder/layers.0/attentions.1/deformable_attention/MultiScaleDeformableAttentionPlugin",
        "/pts_bbox_head/transformer/encoder/layers.0/attentions.1/deformable_attention/Reshape_1",
        "/pts_bbox_head/transformer/encoder/layers.0/attentions.1/deformable_attention/Reshape_2",
    ), (
        "/pts_bbox_head/transformer/encoder/layers.1/attentions.1/deformable_attention/MultiScaleDeformableAttentionPlugin",
        "/pts_bbox_head/transformer/encoder/layers.1/attentions.1/deformable_attention/Reshape_1",
        "/pts_bbox_head/transformer/encoder/layers.1/attentions.1/deformable_attention/Reshape_2",
    ), (
        "/pts_bbox_head/transformer/encoder/layers.2/attentions.1/deformable_attention/MultiScaleDeformableAttentionPlugin",
        "/pts_bbox_head/transformer/encoder/layers.2/attentions.1/deformable_attention/Reshape_1",
        "/pts_bbox_head/transformer/encoder/layers.2/attentions.1/deformable_attention/Reshape_2",
    )]

    for msda, offset, weight in layer_names:
        lut = {}
        for n in model.nodes:
            if n.name in (msda, offset, weight):
                lut[n.name] = n
        
        node_msda = lut[msda]
        node_msda.inputs[4] = lut[offset].outputs[0]
        node_msda.inputs[5] = lut[weight].outputs[0]
        node_msda.attrs["Mode"] = MSDA_MODE.kFusedScaStage2.value

    model.toposort().cleanup()
    return model

def fuse_softmax_tsa_msda(model):
    print("fusing softmax in Temporal-Self-Attention MSDA")
    layer_names = [(
        "/pts_bbox_head/transformer/encoder/layers.0/attentions.0/MultiScaleDeformableAttentionPlugin",
        "/pts_bbox_head/transformer/encoder/layers.0/attentions.0/Reshape_1",
        "/pts_bbox_head/transformer/encoder/layers.0/attentions.0/Reshape_2",
        "/pts_bbox_head/transformer/encoder/layers.0/attentions.0/Transpose_6",
    ), (
        "/pts_bbox_head/transformer/encoder/layers.1/attentions.0/MultiScaleDeformableAttentionPlugin",
        "/pts_bbox_head/transformer/encoder/layers.1/attentions.0/Reshape_1",
        "/pts_bbox_head/transformer/encoder/layers.1/attentions.0/Reshape_2",
        "/pts_bbox_head/transformer/encoder/layers.1/attentions.0/Transpose_6",
    ), (
        "/pts_bbox_head/transformer/encoder/layers.2/attentions.0/MultiScaleDeformableAttentionPlugin",
        "/pts_bbox_head/transformer/encoder/layers.2/attentions.0/Reshape_1",
        "/pts_bbox_head/transformer/encoder/layers.2/attentions.0/Reshape_2",
        "/pts_bbox_head/transformer/encoder/layers.2/attentions.0/Transpose_6",
    )]

    for msda, offset, weight, out in layer_names:
        lut = {}
        for n in model.nodes:
            if n.name in (msda, offset, weight, out):
                lut[n.name] = n
        
        node_msda = lut[msda]
        node_msda.inputs[4] = lut[offset].outputs[0]
        node_msda.inputs[5] = lut[weight].outputs[0]
        node_msda.attrs["Mode"] = MSDA_MODE.kFusedTsaStage2.value
        after_msda = lut[out].outputs[0].outputs[0]
        print(after_msda.name)
        after_msda.inputs[0] = node_msda.outputs[0]

    model.toposort().cleanup()
    print("end of softmax in Temporal-Self-Attention MSDA")
    return model

def fuse_decoder_msda(model):
    print("fusing Decoder MSDA")
    lut = {n.name: n for n in model.nodes}
    
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

    model.toposort().cleanup()
    print("end of fusing Decoder MSDA")
    return model
