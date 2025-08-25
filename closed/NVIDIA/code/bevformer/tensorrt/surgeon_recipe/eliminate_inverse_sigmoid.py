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

def eliminate_inverse_sigmoid(src_model: gs.Graph):
    print("eliminate_inverse_sigmoid")
    keys = []
    lut = {n.name: n for n in src_model.nodes}

    for n in src_model.nodes:
        if n.op == "Sigmoid" and "decoder" in n.name:
            keys.append(n.name)

    reg_branches = [
        "/pts_bbox_head/transformer/decoder/reg_branches.1/reg_branches.1.4/Add",
        "/pts_bbox_head/transformer/decoder/reg_branches.2/reg_branches.2.4/Add",
        "/pts_bbox_head/transformer/decoder/reg_branches.3/reg_branches.3.4/Add",
        "/pts_bbox_head/transformer/decoder/reg_branches.4/reg_branches.4.4/Add",
        "/pts_bbox_head/transformer/decoder/reg_branches.5/reg_branches.5.4/Add",
    ]
    
    for i in range(5):
        curr_sigmoid = lut[keys[i]]
        next_sigmoid = lut[keys[i + 1]]
        # curr_sigmoid - unsqueeze -> reference_points - , we keep this branch
        #              \ slice(0, 2) - inverse_sigmoid - add - scatter - next_sigmoid
        #              \ slice(4, 5) - inverse_sigmoid - add - scatter /
        #           reg_branch --------------------------/
        # as inverse_sigmoid(sigmoid(x)) = x, we can eliminate the inverse_sigmoid

        # looking for:
        # 1. the reg_branch in between
        t_in = curr_sigmoid.inputs[0]
        # reg.matmul, reg.add
        reg_add = lut[reg_branches[i]]
        reg_matmul = reg_add.inputs[1].inputs[0]
        # as the output of matmul and add is sliced, we can fuse the slice indice into the matmul
        # process W
        W = reg_matmul.inputs[1].values[:, [0, 1, 4]]
        Wc = gs.Constant(f"/pts_bbox_head/transformer/decoder/layers.{i}/reg_branches_xyz_W", values=W)
        reg_matmul.inputs[1] = Wc
        # process bias
        B = reg_add.inputs[0].values[[0, 1, 4]]
        Bc = gs.Constant(f"/pts_bbox_head/transformer/decoder/layers.{i}/reg_branches_xyz_B", values=B)
        reg_add.inputs[0] = Bc

        # as we can prune the unused channels, we can directly use the output of reg_add
        # otherwise, we need to slice the output of reg_add
        xyz_out = reg_add.outputs[0]
        t_out = gs.Variable(
            name=f"/pts_bbox_head/transformer/decoder/layer_{i}/Add_2_output_0", 
            dtype=t_in.dtype, 
            shape=t_in.shape)
        n_add = gs.Node(op="Add", 
                        name=f"/pts_bbox_head/transformer/decoder/layer_{i}/Add",
                        inputs=[xyz_out, t_in],
                        outputs=[t_out])
        src_model.nodes.append(n_add)
        scatter_before_next = next_sigmoid.inputs[0].inputs[0]
        scatter_before_next.outputs.clear()
        next_sigmoid.inputs[0] = t_out

    src_model.toposort().cleanup()
    
    print("end of eliminate_inverse_sigmoid")
    return src_model
