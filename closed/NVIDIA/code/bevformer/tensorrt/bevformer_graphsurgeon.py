#!/usr/bin/env python3
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
import tempfile
import argparse
import os
import re
import sys
import ctypes

import numpy as np
import onnx

from pathlib import Path
from onnxsim import simplify
import onnx_graphsurgeon as gs
gs.logger.G_LOGGER.severity = 50  # critical

from onnxconverter_common import float16

from .surgeon_recipe.use_msda_and_rotate_plugin import optimize_use_msda_and_rotate_plugin
from .surgeon_recipe.fuse_msda import fuse_div_msda, fuse_reduce_mean_sca_msda, fuse_softmax_sca_msda, fuse_softmax_tsa_msda, fuse_decoder_msda
from .surgeon_recipe.horizontal_fuse_offset_weight import horizontal_fuse_offset_weight
from .surgeon_recipe.eliminate_scatter_before_msda import eliminate_scatter_before_msda

from .surgeon_recipe.use_layernorm import use_layernorm
from .surgeon_recipe.use_select import use_select
from .surgeon_recipe.fuse_lidar2img import fuse_lidar2img
from .surgeon_recipe.fuse_canbus import fuse_canbus

from .surgeon_recipe.eliminate_inverse_sigmoid import eliminate_inverse_sigmoid
from .surgeon_recipe.eliminate_gather_reshape import eliminate_gather_reshape
from .surgeon_recipe.misc import compose, cast, swap_sigmoid_slice

from .surgeon_recipe.fuse_cls_reg_branches import fuse_cls_reg_branches
from .surgeon_recipe.eliminate_decoder_transpose import eliminate_decoder_transpose
from .surgeon_recipe.mix_precision import mix_precision

from .surgeon_recipe.util import load_ort_supported_model

from modelopt.onnx.quantization import quantize

from code.common import logging


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="build/models/bevformer_tiny/bevformer_tiny.onnx", help="Input ONNX file")
    parser.add_argument("--output", type=str, default="build/optimized/bevformer_tiny.onnx", help="Output ONNX file")
    parser.add_argument("--plugin", type=str, default="build/plugins/BevFormerPlugin/libbev_plugins.so", help="Plugin file")
    parser.add_argument("--force_rebuild", action="store_true", help="Force rebuild")

    args = parser.parse_args()
    return args

class BevFormerGraphSurgeon:
    """
    BevFormerGraphSurgeon is a class that performs graph optimization on the BevFormer model with NVIDIA ModelOpt.
    """

    def __init__(self, 
                 input_onnx: str = "build/models/bevformer_tiny/bevformer_tiny.onnx", 
                 output_file: str = "build/optimized/bevformer_tiny.onnx", 
                 plugin_file: str = "build/plugins/BevFormerPlugin/libbev_plugins.so", 
                 force_rebuild: bool = False):
        self.input_onnx = input_onnx
        self.output_file = output_file
        self.output_dir = Path(output_file).parent
        self.plugin_file = plugin_file
        self.force_rebuild = force_rebuild

    def run_graphsurgeon(self):
        if not self.force_rebuild:
            if Path(self.output_file).exists():
                logging.info(f"Output file {self.output_file} already exists, skipping graphsurgeon")
                return

        mlcommon_onnx = self.input_onnx
        logging.info(f"Force rebuilding: {self.force_rebuild}")
        logging.info(f"Loading from : {mlcommon_onnx}")

        model = optimize_use_msda_and_rotate_plugin(mlcommon_onnx, self.plugin_file)
        temp_dir = Path(tempfile.mkdtemp(dir="/tmp/", prefix="mlcommon_bevformer_"))
        onnx.save(gs.export_onnx(model), os.path.join(temp_dir, "mlcommons_bevformer.tmp.onnx"))

        quantize(
            onnx_path=mlcommon_onnx,
            quantize_mode="int8",
            calibration_method="entropy",
            calibration_eps=["trt"],
            op_types_to_quantize=["Conv", "Add"],
            nodes_to_exclude=None,
            nodes_to_quantize=['/img_backbone/conv1/Conv', '/img_backbone/layer1/layer1.0/.*'],
            high_precision_dtype="fp16",
            mha_accumulation_dtype="fp16",
            disable_mha_qdq=True,
            output_path=temp_dir / "mlcommons_bevformer.tmp.int8.onnx",
            simplify=True,
        )

        quantize(
            onnx_path=mlcommon_onnx,
            quantize_mode="fp8",
            calibration_method="max",
            calibration_eps=["trt"],
            op_types_to_quantize=["Conv"],
            nodes_to_exclude=None,
            nodes_to_quantize=None,
            high_precision_dtype="fp16",
            mha_accumulation_dtype="fp16",
            disable_mha_qdq=True,
            dq_only=False,
            output_path=temp_dir / "mlcommons_bevformer.tmp.fp8.onnx",
            simplify=True,
        )

        model = fuse_div_msda(model)
        model = fuse_reduce_mean_sca_msda(model)
        model = eliminate_scatter_before_msda(model)
        model = fuse_softmax_tsa_msda(model)
        model = horizontal_fuse_offset_weight(model)
        model = fuse_lidar2img(model, "lidar2img")
        model = fuse_canbus(model, "can_bus")

        model_sim, _ = simplify(gs.export_onnx(model))

        model = use_layernorm(mlcommon_onnx, model_sim)
        model = eliminate_inverse_sigmoid(model)
        model = eliminate_gather_reshape(model)
        model = use_select(model)
        model = compose(model)
        model = fuse_cls_reg_branches(model)
        model = eliminate_decoder_transpose(model)
        model = swap_sigmoid_slice(model)

        onnx.save(gs.export_onnx(model), temp_dir / "optimized.onnx")
        shaped_model, _, _ = load_ort_supported_model(temp_dir / "optimized.onnx", self.plugin_file)
        shaped_model = float16.convert_float_to_float16(shaped_model, check_fp16_ready=False)

        model_i8 = gs.import_onnx(onnx.load(temp_dir / "mlcommons_bevformer.tmp.int8.onnx"))
        model_f8 = gs.import_onnx(onnx.load(temp_dir / "mlcommons_bevformer.tmp.fp8.onnx"))

        final_model = mix_precision(gs.import_onnx(shaped_model), model_i8, model_f8)

        final_model = cast(final_model, "11293", np.float32)
        final_model = cast(final_model, "11306", np.float32)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        onnx.save(gs.export_onnx(final_model), self.output_file)
        lut = {n.name: n for n in final_model.nodes}
        dq_name = "/Squeeze_output_0_DequantizeLinear.int8"
        logging.info(f"DQ Node '{dq_name}' has scale: {lut[dq_name].inputs[1].values} for input: image in INT8 when preprocessing.")
        logging.info(f"Final output onnx: {self.output_file}")

    def run(self):
        self.run_graphsurgeon()

def main(input_onnx, output_dir, plugin_dir, force_rebuild=True):
    bevformer_graphsurgeon = BevFormerGraphSurgeon(input_onnx, output_dir, plugin_dir, force_rebuild)
    bevformer_graphsurgeon.run()

if __name__ == "__main__":
    args = parse_args()
    main(args.input, args.output, args.plugin)
