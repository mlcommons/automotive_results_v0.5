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

import code.common.arguments as common_args
from code.common import args_to_string
from code.common.constants import Benchmark, Scenario
from code.common.harness import BaseBenchmarkHarness
from code import ModuleLocation


response_postprocess_map = {}


class LWISHarness(BaseBenchmarkHarness):

    def __init__(self, args, benchmark):
        super().__init__(args, benchmark)

        self.use_jemalloc = (Scenario.Server == self.scenario)
        self.flag_builder_custom_args = common_args.LOADGEN_ARGS + common_args.LWIS_ARGS + common_args.SHARED_ARGS + common_args.GBS_ARGS

    def _get_harness_executable(self):
        return "./build/bin/harness_default"

    def _get_harness_mitten_workload(self):
        return ModuleLocation("code.common.lwis_ops", "LwisWorkload")

    def _build_custom_flags(self, flag_dict):
        flag_dict["scenario"] = self.scenario.valstr()
        flag_dict["model"] = self.name
        argstr = args_to_string(flag_dict) + " --scenario " + self.scenario.valstr() + " --model " + self.name

        if self.name in response_postprocess_map:
            flag_dict["response_postprocess"] = response_postprocess_map[self.name]
            argstr += " --response_postprocess " + response_postprocess_map[self.name]

        return argstr
