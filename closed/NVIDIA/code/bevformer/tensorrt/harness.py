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

import os
from typing import Dict

import code.common.arguments as common_args
from code.common import logging, dict_get, run_command, args_to_string
from code.common.harness import BaseBenchmarkHarness
from code.common.constants import Benchmark
from code import ModuleLocation


class BEVFormerHarness(BaseBenchmarkHarness):
    """BEVFormer harness."""

    def __init__(self, args: Dict, benchmark: Benchmark) -> None:
        super().__init__(args, benchmark)
        self.flag_builder_custom_args = common_args.LOADGEN_ARGS +\
            common_args.LWIS_ARGS +\
            common_args.SHARED_ARGS
        
    def _get_harness_executable(self) -> str:
        """Return path to BEVFormer harness binary."""
        return "./build/bin/harness_bevformer"

    def _get_harness_mitten_workload(self):
        return ModuleLocation("code.harness.harness_bevformer.lwis_bevformer_ops", "LwisBEVFormerWorkload")

    def _build_custom_flags(self, flag_dict: Dict) -> str:
        pass

        flag_dict["scenario"] = self.scenario.valstr()
        flag_dict["model"] = self.name
        argstr = args_to_string(flag_dict)
        return argstr
