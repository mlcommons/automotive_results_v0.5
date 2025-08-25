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

def adjust_feature(src_model):
    lut = {n.name: n for n in src_model.nodes}
    conv = lut["/img_neck/fpn_convs.0/conv/Conv"]

    add_8 = lut["/pts_bbox_head/transformer/Add_8"]
    add_9 = lut["/pts_bbox_head/transformer/Add_9"]
    add_B = add_8.inputs[1].values + add_9.inputs[1].values
    add_B = add_B.reshape(6, 256, 1, 1)
    add_8.inputs[0] = conv.outputs[0]
    add_8.inputs[1].values = add_B
    reshape_7 = lut["/pts_bbox_head/transformer/encoder/layers.0/attentions.1/Reshape_7"]
    reshape_7.inputs[0] = add_8.outputs[0]
    reshape_7.inputs[1].values = np.array([6, 256, 375], dtype=np.int64)
    matmul = reshape_7.outputs[0].outputs[0]
    matmul.name = "/pts_bbox_head/transformer/encoder/layers.0-2/attentions.1/deformable_attention/value_proj/MatMul"

    transpose_1 = lut["/pts_bbox_head/transformer/encoder/layers.0/attentions.1/Transpose_1"]
    transpose_1.inputs[0] = reshape_7.outputs[0]
    transpose_1.attrs["perm"] = [0, 2, 1]  # 6, 256, 375 -> 6, 375, 256
    matmul.inputs[0] = transpose_1.outputs[0]
    
    value_proj_adds = [lut[f"/pts_bbox_head/transformer/encoder/layers.{i}/attentions.1/deformable_attention/value_proj/Add"] for i in range(3)]
    value_proj_add_As = [add.inputs[0].values for add in value_proj_adds]
    value_proj_add_A = np.concatenate(value_proj_add_As, axis=0)
    value_proj_add_Ac = gs.Constant(f"/pts_bbox_head/transformer/encoder/layers.0/attentions.1/deformable_attention/value_proj/Add_A", values=value_proj_add_A)
    value_proj_add_O = gs.Variable(name="/pts_bbox_head/transformer/encoder/layers.0-2/attentions.1/deformable_attention/value_proj/Add_output", dtype=matmul.outputs[0].dtype)
    value_proj_add_node = gs.Node(
        op="Add",
        name="/pts_bbox_head/transformer/encoder/layers.0-2/attentions.1/deformable_attention/value_proj/Add",
        inputs=[value_proj_add_Ac, matmul.outputs[0]],
        outputs=[value_proj_add_O]
    )
    for i in range(3):
        msda = lut[f"/pts_bbox_head/transformer/encoder/layers.{i}/attentions.1/deformable_attention/MultiScaleDeformableAttentionPlugin"]
        msda.inputs[0] = value_proj_add_O
        msda.attrs["iLayer"] = i
        reshape = lut[f"/pts_bbox_head/transformer/encoder/layers.{i}/attentions.1/deformable_attention/Reshape"]
        reshape.outputs.clear()

    src_model.nodes.append(value_proj_add_node)
    src_model.toposort().cleanup()

    return src_model

def adjust_decoder_block0_reg_branch(src_model):
    lut = {n.name: n for n in src_model.nodes}
    matmul = lut["/pts_bbox_head/transformer/decoder/reg_branches.0/reg_branches.0.4/MatMul"]
    Wc = gs.Constant(f"/pts_bbox_head/transformer/decoder/reg_branches.0/reg_branches.0.4/reg_branches_xyz_W", values=matmul.inputs[1].values[:, [0, 1, 4]])
    matmul.inputs[1] = Wc
    add = lut["/pts_bbox_head/transformer/decoder/reg_branches.0/reg_branches.0.4/Add"]
    Bc = gs.Constant(f"/pts_bbox_head/transformer/decoder/reg_branches.0/reg_branches.0.4/reg_branches_xyz_B", values=add.inputs[0].values[[0, 1, 4]])
    add.inputs[0] = Bc

    extra_add_xy = lut["/pts_bbox_head/transformer/decoder/Add"]
    extra_add_z = lut["/pts_bbox_head/transformer/decoder/Add_1"]
    extra_add_xyz = np.concatenate([extra_add_xy.inputs[1].values, extra_add_z.inputs[1].values], axis=-1)
    extra_add_xy.inputs[0] = add.outputs[0]
    extra_add_xy.inputs[1].values = extra_add_xyz

    lut["/pts_bbox_head/transformer/decoder/Sigmoid"].inputs[0] = extra_add_xy.outputs[0]
    lut["/pts_bbox_head/transformer/decoder/layer_0/Add"].inputs[1] = extra_add_xy.outputs[0]

    lut["/pts_bbox_head/transformer/decoder/ScatterND_1"].outputs.clear()

    src_model.toposort().cleanup()
    return src_model

def adjust_add_7(src_model):
    lut = {n.name: n for n in src_model.nodes}
    add_7 = lut["/pts_bbox_head/transformer/Add_7"]
    add_7.inputs[0].values = add_7.inputs[0].values.transpose(1, 0, 2)
    src_model.toposort().cleanup()
    return src_model

def compose(src_model):
    src_model = adjust_feature(src_model)
    src_model = adjust_decoder_block0_reg_branch(src_model)
    src_model = adjust_add_7(src_model)
    return src_model

def cast(src_model, name, dtype):
    prev_node = src_model.tensors()[name].inputs[0]
    cast_out = prev_node.outputs[0]
    cast_in = gs.Variable(name=f"{prev_node.name}_output_before_cast", dtype=cast_out.dtype)
    prev_node.outputs[0] = cast_in
    cast_node = gs.Node(
        op="Cast",
        name=f"{name}_cast",
        inputs=[cast_in],
        outputs=[cast_out],
    )
    cast_node.attrs["to"] = dtype
    cast_out.dtype = dtype
    src_model.nodes.append(cast_node)
    return src_model

def swap_sigmoid_slice(src_model):
    src_model = gs.import_onnx(gs.export_onnx(src_model))
    lut = {n.name: n for n in src_model.nodes}

    for i in range(1, 6):
        msda = lut[f"/pts_bbox_head/transformer/decoder/layers.{i}/attentions.1/MultiScaleDeformableAttentionPlugin"]
        ref = msda.inputs[3]
        unsqueeze = ref.inputs[0]
        slice = unsqueeze.inputs[0].inputs[0]
        slice_out = slice.outputs[0]
        sigmoid = slice.inputs[0].inputs[0]
        sigmoid_out = sigmoid.outputs[0]
        if i == 1:
            sigmoid_in = lut["/pts_bbox_head/transformer/decoder/Add"].outputs[0]
        else:
            sigmoid_in = sigmoid.inputs[0]
        print(f"i={i}, {unsqueeze.name}, {slice.name}, {sigmoid.name}, {sigmoid_in.name}, {sigmoid_out.name}, {slice_out.name}")
        unsqueeze.inputs[0] = sigmoid_out
        sigmoid.inputs[0] = slice_out
        slice.inputs[0] = sigmoid_in

    src_model.toposort().cleanup()
    return src_model
