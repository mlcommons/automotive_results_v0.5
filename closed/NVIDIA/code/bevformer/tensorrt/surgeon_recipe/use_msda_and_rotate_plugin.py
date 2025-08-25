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

import argparse
import os
import re
import sys
import ctypes

import numpy as np
import onnx
import onnx_graphsurgeon as gs

from .util import load_ort_supported_model

class MsdaSubgraph(object):
    def __init__(self, prefix):
        self.prefix = prefix
        self.fused_transpose = True
        # value, sampling_locations, weight, out
        self.io_nodes = {}
        self.out_candidates = [None, None]
        self.nodes = []

    def decide_out(self):
        if self.fused_transpose:
            self.io_nodes["out"] = self.out_candidates[1]
        else:
            self.io_nodes["out"] = self.out_candidates[0]

    def build_msda(self, prefix, lut):
        value_shape = lut[self.io_nodes["value"]].inputs[0].shape
        if value_shape[1] == 2500:
            spatial_shape = np.array([[50, 50]], dtype=np.int32)
        elif value_shape[1] == 15 * 25:
            spatial_shape = np.array([[15, 25]], dtype=np.int32)
        else:
            raise ValueError(f"Invalid value shape: {value_shape}")

        inputs = [
            lut[self.io_nodes["value"]].inputs[0],
            gs.Constant(prefix + "/msda_spatial_shape", spatial_shape),
            gs.Constant(prefix + "/msda_spatial_indices", np.array([0], dtype=np.int32)),
            lut[self.io_nodes["sampling_locations"]].inputs[0],
            lut[self.io_nodes["weight"]].inputs[0]
        ]
        outputs = [lut[self.io_nodes["out"]].outputs[0]]
        msda_node = gs.Node(
            op="MultiScaleDeformableAttentionPlugin", name=prefix + "/MultiScaleDeformableAttentionPlugin",
            inputs=inputs,
            outputs=outputs,
            domain="custom_op")
        
        if self.fused_transpose:
            # looking for the transpose node after out
            next_node = lut[self.io_nodes["out"]].outputs[0].outputs[0]
            if next_node.op == "Transpose":
                print(f"recover {next_node.name}.perm back to [1, 2, 0]")
                next_node.attrs["perm"] = [1, 2, 0]
        lut[self.io_nodes["out"]].outputs.clear()
        return msda_node

def build_rotate(lut):
    # angle: /pts_bbox_head/transformer/Neg
    # feature: /pts_bbox_head/transformer/Reshape_1
    rotate_name = "/pts_bbox_head/transformer/RotatePlugin"
    inputs = [
        lut["/pts_bbox_head/transformer/Reshape_1"].outputs[0],
        lut["/pts_bbox_head/transformer/Neg"].inputs[0],
        gs.Constant(rotate_name + "_center", np.array([100, 100], dtype=np.float32)),]
    lut["/pts_bbox_head/transformer/Reshape_1"].inputs[0] = lut["/pts_bbox_head/transformer/Gather_1"].inputs[0]
    lut["/pts_bbox_head/transformer/Gather_1"].outputs.clear()

    lut["/pts_bbox_head/transformer/Reshape_1"].inputs[1].values = np.array([50, 50, -1], dtype=np.int32)
    
    # /pts_bbox_head/transformer/Squeeze_2
    out_name = "/pts_bbox_head/transformer/Transpose_3"
    outputs = [lut[out_name].outputs[0],]
    rotate_node = gs.Node(
        op="RotatePlugin", name=rotate_name,
        attrs={"interpolation": 1},
        inputs=inputs,
        outputs=outputs,
        domain="custom_op")
    lut[out_name].outputs.clear()

    lut["/pts_bbox_head/transformer/Reshape_8"].inputs[1] = gs.Constant(name="/pts_bbox_head/transformer/Reshape_8_shape", values=np.array([1, 2500, 256], dtype=np.int64))
    lut["/pts_bbox_head/transformer/encoder/Unsqueeze_40"].inputs[0] = lut["/pts_bbox_head/transformer/Reshape_8"].outputs[0]
    return rotate_node

def optimize_use_msda_and_rotate_plugin(mlcommon_onnx_path, trt_plugin_path):
    shaped_model, _, _ = load_ort_supported_model(mlcommon_onnx_path, trt_plugin_path)
    shaped_model = gs.import_onnx(shaped_model)

    onnx_model = onnx.load(mlcommon_onnx_path)
    stack_lut = {}
    msda_table = {}

    for n in onnx_model.graph.node:
        # /pts_bbox_head/transformer/encoder/layers.0/attentions.0/GridSample
        stack = n.doc_string.split("\n")
        stack_lut[n.name] = stack

        prefix = "/".join(n.name.split("/")[:-1])
        if prefix not in msda_table:
            msda_table[prefix] = MsdaSubgraph(prefix)

        current_msda_subgraph = msda_table[prefix]

        for line in stack:
            if "multi_scale_deform_attn" in line:
                match = re.search(r'multi_scale_deform_attn\.py\((\d+)\)', line)
                if match:
                    line_number = match.group(1)
                # print(f"msda -> {msda_path}:{line_number}, {n.name}, {n.op_type}")
                lineno = int(line_number)
                current_msda_subgraph.nodes.append((lineno, n))

                if lineno == 149 and n.op_type == "Transpose":
                    current_msda_subgraph.fused_transpose = False
                    current_msda_subgraph.out_candidates[0] = n.name
                elif lineno == 146 and n.op_type == "Reshape":
                    current_msda_subgraph.out_candidates[1] = n.name
                else:
                    pass

                if lineno == 117 and n.op_type == "Slice":
                    current_msda_subgraph.io_nodes["value"] = n.name
                elif lineno == 119 and n.op_type == "Mul":
                    current_msda_subgraph.io_nodes["sampling_locations"] = n.name
                elif lineno == 144 and n.op_type == "Transpose":
                    current_msda_subgraph.io_nodes["weight"] = n.name
                break

    lut = {}
    for n in shaped_model.nodes:
        lut[n.name] = n

    for prefix, msda_subgraph in msda_table.items():
        if len(msda_subgraph.nodes) == 0:
            continue
        msda_subgraph.decide_out()

        print(f"{prefix}:")
        print(f"  fused_transpose: {msda_subgraph.fused_transpose}")
        shaped_model.nodes.append(msda_subgraph.build_msda(prefix, lut))

    shaped_model.nodes.append(build_rotate(lut))
    shaped_model.cleanup().toposort()

    return shaped_model
