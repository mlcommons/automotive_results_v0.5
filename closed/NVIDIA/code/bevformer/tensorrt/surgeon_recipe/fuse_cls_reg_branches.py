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

from pathlib import Path
import onnx
import onnx.version
import onnx_graphsurgeon as gs
import numpy as np
import tensorrt as trt

# create updated subgraph
cls_branch = [
    "/pts_bbox_head/cls_branches/cls_branches.0/MatMul",
    "/pts_bbox_head/cls_branches/cls_branches.0/Add",
    "/pts_bbox_head/cls_branches/cls_branches.1/LayerNormalization",
    "/pts_bbox_head/cls_branches/cls_branches.2/Relu",
    "/pts_bbox_head/cls_branches/cls_branches.3/MatMul",
    "/pts_bbox_head/cls_branches/cls_branches.3/Add",
    "/pts_bbox_head/cls_branches/cls_branches.4/LayerNormalization",
    "/pts_bbox_head/cls_branches/cls_branches.5/Relu",
    "/pts_bbox_head/cls_branches/cls_branches.6/MatMul",
    "/pts_bbox_head/cls_branches/cls_branches.6/Add",
]

reg_branch = [
    "/pts_bbox_head/reg_branches/reg_branches.0/MatMul",
    "/pts_bbox_head/reg_branches/reg_branches.0/Add",
    "/pts_bbox_head/reg_branches/reg_branches.1/Relu",
    "/pts_bbox_head/reg_branches/reg_branches.2/MatMul",
    "/pts_bbox_head/reg_branches/reg_branches.2/Add",
    "/pts_bbox_head/reg_branches/reg_branches.3/Relu",
    "/pts_bbox_head/reg_branches/reg_branches.4/MatMul",
    "/pts_bbox_head/reg_branches/reg_branches.4/Add",
]

# reference_points
# 5 inter_reference_points
reg_refine_branch = {
    "/pts_bbox_head/reg_branches/refine/Concat",
    "/pts_bbox_head/reg_branches/refine/Unsqueeze",
    "/pts_bbox_head/reg_branches/refine/Slice_xyz",
    "/pts_bbox_head/reg_branches/refine/Add",
    "/pts_bbox_head/reg_branches/refine/Sigmoid",
    "/pts_bbox_head/reg_branches/refine/Mul",  # 102.375, 102.375, 8
    "/pts_bbox_head/reg_branches/refine/Add",  # -51.1875, -51.1875, -5
    "/pts_bbox_head/reg_branches/refine/Slice_3-5",
    "/pts_bbox_head/reg_branches/refine/Slice_5-10",
    "/pts_bbox_head/reg_branches/refine/Slice_0-2",
    "/pts_bbox_head/reg_branches/refine/Slice_2-3",
    "/pts_bbox_head/reg_branches/refine/Concat_1",
}

def build_matmul(name, input, W):
    Wc = gs.Constant(name + "_W", values=W)
    out = gs.Variable(name + "_output", dtype=input.dtype)
    node = gs.Node(
        op="MatMul", 
        name=name,
        inputs=[input, Wc], outputs=[out])
    return node, out

def build_mul(name, input, W):
    if isinstance(W, np.ndarray):
        Wc = gs.Constant(name + "_B", values=W)
        out = gs.Variable(name + "_output", dtype=input.dtype)
        node = gs.Node(op="Mul", name=name, inputs=[input, Wc], outputs=[out])
        return node, out
    else:
        out = gs.Variable(name + "_output", dtype=input.dtype)
        node = gs.Node(op="Mul", name=name, inputs=[input, W], outputs=[out])
    return node, out

def build_add(name, input, B):
    if isinstance(B, np.ndarray):
        Bc = gs.Constant(name + "_B", values=B)
        out = gs.Variable(name + "_output", dtype=input.dtype)
        node = gs.Node(op="Add", name=name, inputs=[input, Bc], outputs=[out])
        return node, out
    else:
        out = gs.Variable(name + "_output", dtype=input.dtype)
        node = gs.Node(op="Add", name=name, inputs=[input, B], outputs=[out])
        return node, out

def build_layernorm(name, input, gamma, beta, eps):
    gamma_c = gs.Constant(name + "_gamma", values=gamma)
    beta_c = gs.Constant(name + "_beta", values=beta)
    out = gs.Variable(name + "_output", dtype=input.dtype)
    node = gs.Node(op="LayerNormalization", name=name, inputs=[input, gamma_c, beta_c], outputs=[out])
    node.attrs["epsilon"] = eps
    node.attrs["axis"] = -1
    return node, out

def build_relu(name, input):
    out = gs.Variable(name + "_output", dtype=input.dtype)
    node = gs.Node(op="Relu", name=name, inputs=[input], outputs=[out])
    return node, out

def build_sigmoid(name, input):
    out = gs.Variable(name + "_output", dtype=input.dtype)
    node = gs.Node(op="Sigmoid", name=name, inputs=[input], outputs=[out])
    return node, out

def build_node_with_siblings(name, current_feat, siblings):
    nodes = []
    if "MatMul" in name:
        Ws = np.concatenate([s.inputs[1].values.reshape(1, 1, 256, -1) for s in siblings], axis=0)
        node, current_feat = build_matmul(name, current_feat, Ws)
        nodes.append(node)
    elif "Add" in name:
        Bs = np.concatenate([s.inputs[0].values.reshape(1, 1, 1, -1) for s in siblings], axis=0)
        node, current_feat = build_add(name, current_feat, Bs)
        nodes.append(node)
    elif "LayerNormalization" in name:
        gamma_s = np.concatenate([s.inputs[1].values.reshape(1, 1, 1, -1) for s in siblings], axis=0)
        beta_s = np.concatenate([s.inputs[2].values.reshape(1, 1, 1, -1) for s in siblings], axis=0)
        node, current_feat = build_layernorm(name, current_feat, gamma_s, beta_s, 1e-5)
        nodes.append(node)
    elif "Relu" in name:
        node, current_feat = build_relu(name, current_feat)
        nodes.append(node)
    return nodes, current_feat

def build_slice(name, input, starts, ends):
    starts_c = gs.Constant(name + "_starts", values=np.array(starts, dtype=np.int64))
    ends_c = gs.Constant(name + "_ends", values=np.array(ends, dtype=np.int64))
    axis_c = gs.Constant(name + "_axis", values=np.array([3], dtype=np.int64))
    step_c = gs.Constant(name + "_step", values=np.array([1], dtype=np.int64))
    out = gs.Variable(name + "_output", dtype=input.dtype)
    node = gs.Node(op="Slice", name=name, inputs=[input, starts_c, ends_c, axis_c, step_c], outputs=[out])
    return node, out

def build_concat(name, inputs, axis):
    out = gs.Variable(name + "_output", dtype=inputs[0].dtype)
    node = gs.Node(op="Concat", name=name, inputs=inputs, outputs=[out])
    node.attrs["axis"] = axis
    return node, out

reference_points_data = np.load(str(Path(__file__).parent / "_pts_bbox_head_transformer_decoder_Concat_19_const.npy"))

def fuse_cls_reg_branches(src_model):
    lut = {n.name: n for n in src_model.nodes}

    feature_prev = lut["/pts_bbox_head/transformer/decoder/Concat_18"].outputs[0]
    feature = gs.Variable(name="/pts_bbox_head/transformer/decoder/feature", dtype=feature_prev.dtype)
    feature_shape = gs.Constant(
        name="/pts_bbox_head/transformer/decoder/feature_shape", 
        values=np.array([6, 1, 900, 256], dtype=np.int64))
    feature_reshape = gs.Node(
        op="Reshape", 
        name="/pts_bbox_head/transformer/decoder/feature_reshape",
        inputs=[feature_prev, feature_shape],
        outputs=[feature])

    fused_nodes = [feature_reshape]
    current_feat = feature
    for nc in cls_branch:
        # /pts_bbox_head/cls_branches/cls_branches.0/MatMul <- /pts_bbox_head/cls_branches.[0~5]/cls_branches.[0~5].0/MatMul
        name_sections = nc.split("/")        
        siblings = []
        for i in range(6):
            sec_3 = name_sections[3].split(".")[0] + f".{i}." + name_sections[3].split(".")[1]
            sib_name = "/".join([name_sections[0], name_sections[1], name_sections[2] + f".{i}", sec_3, name_sections[4]])
            siblings.append(lut[sib_name])
        print("Fuse:")
        print(f"  from: {[s.name for s in siblings]}")
        print(f"  into: {nc}")
        nodes, current_feat = build_node_with_siblings(nc, current_feat, siblings)
        fused_nodes.extend(nodes)

    last_siblings = siblings
    for s in last_siblings:
        s.outputs.clear()
    print([n.name for n in fused_nodes])
    # lut["graph_output_cast1"].inputs[0] = current_feat
    current_feat.name = "11293"
    src_model.outputs[1] = current_feat

    current_feat = feature
    
    for nc in reg_branch:
        name_sections = nc.split("/")        
        siblings = []
        for i in range(6):
            sec_3 = name_sections[3].split(".")[0] + f".{i}." + name_sections[3].split(".")[1]
            sib_name = "/".join([name_sections[0], name_sections[1], name_sections[2] + f".{i}", sec_3, name_sections[4]])
            siblings.append(lut[sib_name])
        print("Fuse:")
        print(f"  from: {[s.name for s in siblings]}")
        print(f"  into: {nc}")
        nodes, current_feat = build_node_with_siblings(nc, current_feat, siblings)
        if nc in ["/pts_bbox_head/reg_branches/reg_branches.4/MatMul", "/pts_bbox_head/reg_branches/reg_branches.4/Add"]:
            shuffle_indices = [0, 1, 4, 2, 3, 5, 6, 7, 8, 9]
            nodes[0].inputs[1].values = nodes[0].inputs[1].values[:, :, :, shuffle_indices]
    
        fused_nodes.extend(nodes)
    last_siblings = siblings
    for s in last_siblings:
        s.outputs.clear()
    
    reg_refine_output = current_feat
    reference_points = gs.Constant("/pts_bbox_head/transformer/decoder/reference_points_const", values=reference_points_data)

    add_names = [
        "/pts_bbox_head/transformer/decoder/Add", 
        "/pts_bbox_head/transformer/decoder/layer_0/Add",
        "/pts_bbox_head/transformer/decoder/layer_1/Add",
        "/pts_bbox_head/transformer/decoder/layer_2/Add",
        "/pts_bbox_head/transformer/decoder/layer_3/Add",
    ]
    add_tensors = [reference_points]
    for name in add_names:
        add_tensors.append(lut[name].outputs[0])
    concat_19 = lut["/pts_bbox_head/transformer/decoder/Concat_19"]
    concat_19.inputs = add_tensors
    unsqueeze_19_out = gs.Variable(
        name="/pts_bbox_head/transformer/decoder/Unsqueeze_19_output", 
        dtype=concat_19.outputs[0].dtype)
    unsqueeze_19_axes = gs.Constant(
        name="/pts_bbox_head/transformer/decoder/Unsqueeze_19_axes",
        values=np.array([1], dtype=np.int64))
    unsqueeze_19 = gs.Node(        
        op="Unsqueeze", 
        name="/pts_bbox_head/transformer/decoder/Unsqueeze_19",
        inputs=[concat_19.outputs[0], unsqueeze_19_axes], 
        outputs=[unsqueeze_19_out])

    slice_xyz, slice_xyz_out = build_slice("/pts_bbox_head/reg_branches/refine/Slice_xyz", reg_refine_output, [0], [3])
    add_xyz, add_xyz_out = build_add("/pts_bbox_head/reg_branches/refine/Add", slice_xyz_out, unsqueeze_19_out)
    sigmoid_xyz, sigmoid_xyz_out = build_sigmoid("/pts_bbox_head/reg_branches/refine/Sigmoid", add_xyz_out)
    mul_xyz, mul_xyz_out = build_mul("/pts_bbox_head/reg_branches/refine/Mul", sigmoid_xyz_out, np.array([102.375, 102.375, 8], dtype=np.float16))
    add_1_xyz, add_1_xyz_out = build_add("/pts_bbox_head/reg_branches/refine/Add_1", mul_xyz_out, np.array([-51.1875, -51.1875, -5], dtype=np.float16))
    slice_3_5, slice_3_5_out = build_slice("/pts_bbox_head/reg_branches/refine/Slice_3-5", reg_refine_output, [3], [5])
    slice_5_10, slice_5_10_out = build_slice("/pts_bbox_head/reg_branches/refine/Slice_5-10", reg_refine_output, [5], [10])
    slice_0_2, slice_0_2_out = build_slice("/pts_bbox_head/reg_branches/refine/Slice_0-2", add_1_xyz_out, [0], [2])
    slice_2_3, slice_2_3_out = build_slice("/pts_bbox_head/reg_branches/refine/Slice_2-3", add_1_xyz_out, [2], [3])
    concat_1, concat_1_out = build_concat("/pts_bbox_head/reg_branches/refine/Concat_1", [slice_0_2_out, slice_3_5_out, slice_2_3_out, slice_5_10_out], 3)
    fused_nodes.extend([unsqueeze_19, slice_xyz, add_xyz, sigmoid_xyz, mul_xyz, add_1_xyz, slice_3_5, slice_5_10, slice_0_2, slice_2_3, concat_1])

    # cast2 = lut["graph_output_cast2"]
    # cast2.inputs[0] = concat_1_out
    concat_1_out.name = "11306"
    src_model.outputs[2] = concat_1_out
    lut["/pts_bbox_head/Concat_109"].outputs.clear()
    
    src_model.nodes.extend(fused_nodes)
    src_model.toposort().cleanup()
    return src_model
