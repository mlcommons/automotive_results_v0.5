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

def eliminate_gather_reshape(src_model):
    print("eliminate_gather_reshape")

    lut = {n.name: n for n in src_model.nodes}

    for k in lut:
        n = lut[k]
        if n.op == "Gather":
            # cond 1: gather follows by a reshape
            next_node = n.outputs[0].outputs[0]
            cond1 = next_node.op == "Reshape"
            # cond 2: gather.axis == 0, and gather.indices = 0
            axis = n.attrs["axis"]
            indices = int(n.inputs[1].values)
            cond2 = axis == 0 and indices == 0
            # cond 3: reshape's output has the same shape as gather's input
            shape_reshape = next_node.outputs[0].shape
            shape_gather = n.inputs[0].shape
            cond3 = all(shape_reshape[i] == shape_gather[i] for i in range(3))
            if cond1 and cond2 and cond3:
                print(n.name, next_node.name)
                out = next_node.outputs[0].outputs[0]
                out.inputs[0] = n.inputs[0]
                next_node.outputs.clear()

    src_model.toposort().cleanup()
    print("end of eliminate_gather_reshape")
    return src_model
