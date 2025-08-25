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

import re
import onnx
import onnx_graphsurgeon as gs

def build_layernorm(prefix, ln_in, ln_out, ln_scale, ln_bias):
    domain = None
    node = gs.Node(
        op="LayerNormalization",
        name=prefix + "/LayerNormalization",
        attrs={"epsilon": 1e-5, "axis": -1},
        inputs=[
            ln_in.inputs[0],
            ln_scale.inputs[1],
            ln_bias.inputs[1],
        ],
        outputs=[ln_out.outputs[0]],
        domain=domain,
    )
    ln_out.outputs.clear()
    return node

class LayerNormConverter(object):
    stack_lut = None

    @classmethod
    def get_stack(cls, node_name, mlcommon_onnx=None):
        if LayerNormConverter.stack_lut is None:
            LayerNormConverter.stack_lut = {}
            model = onnx.load(mlcommon_onnx)
            for n in model.graph.node:
                stack = n.doc_string.split("\n")
                LayerNormConverter.stack_lut[n.name] = stack
        return LayerNormConverter.stack_lut.get(node_name, [])

    def __init__(self, onnx_or_path):
        if isinstance(onnx_or_path, str):
            self.onnx_ = onnx.load(onnx_or_path)
        else:
            self.onnx_ = onnx_or_path
        self.model = gs.import_onnx(self.onnx_)
        self._build_table()

    def _build_table(self):
        ln_table = {}
        lut = {}
        for n in self.model.nodes:
            lut[n.name] = n

            stack = LayerNormConverter.get_stack(n.name)
            prefix = "/".join(n.name.split("/")[:-1])
            if prefix not in ln_table:
                ln_table[prefix] = []

            for line in stack:
                if "layer_norm" in line:
                    match = re.search(r'functional\.py\((\d+)\)', line)
                    if match:
                        line_number = match.group(1)
                    lineno = int(line_number)
                    ln_table[prefix].append((lineno, n))
                    break

        self.table = ln_table
        self.lut = lut
    
    def _convert_layernorm(self):        
        for prefix, ln_subgraph in self.table.items():
            if len(ln_subgraph) == 0:
                continue
            print(f"{prefix}")
            #  in: ReduceMean
            # out: Add_1
            # for lineno, n in ln_subgraph:
            #     print(f"  {lineno}, {n.name}, {n.op_type}")
            # print(self.lut[f"{prefix}/ReduceMean"].attrs["axes"])
            node = build_layernorm(
                prefix, 
                self.lut[f"{prefix}/ReduceMean"], 
                self.lut[f"{prefix}/Add_1"],
                self.lut[f"{prefix}/Mul"],
                self.lut[f"{prefix}/Add_1"],
            )
            self.model.nodes.append(node)
        self.model.toposort().cleanup()
        return self

def use_layernorm(mlcommon_onnx_path, model):
    _ = LayerNormConverter.get_stack("dummy_name", mlcommon_onnx=mlcommon_onnx_path)
    cvt = LayerNormConverter(model)
    cvt._convert_layernorm()
    return cvt.model
