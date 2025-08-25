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
import os
import sys
from importlib import import_module
from pathlib import Path
import numpy as np
import tensorrt as trt

from nvmitten.nvidia.accelerator import GPU

import code.common.arguments as common_args
from code.common import logging, dict_get, run_command, args_to_string
from code.common.constants import Benchmark, Scenario, G_HIGH_ACC_ENABLED_BENCHMARKS, G_MLCOMMONS_REPO_PATH, VERSION
from code.common.log_parser import from_loadgen_by_keys, get_loadgen_log_keys
from code.common.submission import generate_measurements_entry
from code.common.fix_sys_path import ScopedRestrictedImport
from code.common.systems.system_list import DETECTED_SYSTEM
from code.common.utils import get_e2e_batch_size, ScopeWrap
from code.plugin import get_trt_plugin_paths_by_network
from code import get_cls

_new_path = [os.path.join(G_MLCOMMONS_REPO_PATH, "tools", "submission")] + sys.path
with ScopedRestrictedImport(_new_path):
    submission_checker = import_module("submission_checker")
    _version_str = VERSION if VERSION in submission_checker.MODEL_CONFIG else "v0.5"
    benchmark_qsl_size_map = submission_checker.MODEL_CONFIG[_version_str]["performance-sample-count"].copy()


    # Check for query constraints documented in https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc#scenarios
    _min_queries = submission_checker.MODEL_CONFIG[_version_str]["min-queries"].copy()
    # https://github.com/mlcommons/mlperf_automotive/issues/56
    QUERY_METRIC_CONSTRAINTS = {
        "SingleStream": ("effective_min_query_count", _min_queries["bevformer"].get("SingleStream", 6636)),
        "ConstantStream": ("effective_min_query_count", _min_queries["bevformer"].get("ConstantStream", 100000)),
    }


class BaseBenchmarkHarness:
    """Base class for benchmark harnesses."""

    def __init__(self, args, benchmark):
        self.args = args
        self.benchmark = benchmark
        self.name = self.benchmark.valstr()
        self.workload_setting = dict_get(args, "workload_setting", default=None)
        self.verbose = dict_get(args, "verbose", default=None)
        self.verbose_nvtx = dict_get(args, "verbose_nvtx", default=None)
        self.verbose_glog = dict_get(args, "verbose_glog", default=None)
        if self.verbose:
            logging.info(f"===== Harness arguments for {self.name} =====")
            for key in args:
                logging.info("{:}={:}".format(key, args[key]))

        self.system_id = args["system_id"]
        self.scenario = args["scenario"]
        self.config_ver = args["config_ver"]
        self.engine_dir = "./build/engines/{:}/{:}/{:}".format(self.system_id, self.name, self.scenario.valstr())
        self.default_engine_dir = True
        if "engine_dir" in args:
            self.engine_dir = args["engine_dir"]
            self.default_engine_dir = False
        self.precision = args["precision"]
        self.has_gpu = dict_get(args, "gpu_batch_size", default=None) is not None
        if self.has_gpu:
            gpu_e2e_batch_size = get_e2e_batch_size(self.args["gpu_batch_size"])
            self.args["gpu_batch_size"] = gpu_e2e_batch_size

        # Engine not needed if we are only generating measurements/ entries
        self.skip_file_checks = dict_get(self.args, "skip_file_checks", False)
        self.engine_file = self._get_engine_fpath("gpu", self.args["gpu_batch_size"])

        # Capture harness executable
        self.python_executable = Benchmark.PYTHON_PATH(self.benchmark)
        self.executable = self._get_harness_executable()
        self.check_file_exists(self.executable)

        self.mitten_workload = self._get_harness_mitten_workload()

        self.env_vars = os.environ.copy()
        self.flag_builder_custom_args = []

    def _get_harness_executable(self):
        raise NotImplementedError("BaseBenchmarkHarness cannot be called directly")

    def _get_harness_mitten_workload(self):
        return None

    def _construct_terminal_command(self, argstr):
        return f"{self.executable} {argstr}"

    def _build_custom_flags(self, flag_dict):
        """
        Handles any custom flags to insert into flag_dict. Can return either a flag_dict, or a converted arg string.
        """
        return flag_dict

    def _get_engine_fpath(self, device_type, batch_size):
        return f"{self.engine_dir}/{self.name}-{self.scenario.valstr()}-{device_type}-b{batch_size}-{self.precision}.{self.config_ver}.plan"


    def get_system_name(self, add_trt=True):
        override_system_name = dict_get(self.args, "system_name", default=None)
        if override_system_name not in {None, ""}:
            return override_system_name

        system_name = self.system_id

        if add_trt:
            system_name = f"{system_name}_TRT"

        return system_name

    def _get_submission_benchmark_name(self):
        full_benchmark_name = self.name
        # NOTE: automotive submission checker doesn't like the accuracy ("-99") suffix
        # accuracy_target = dict_get(self.args, "accuracy_level", "99%")
        #prefix = accuracy_target.replace("%", "")
        #return f"{full_benchmark_name}-{prefix}"
        return full_benchmark_name

    def get_full_log_dir(self):
        return os.path.join(self.args["log_dir"], self.get_system_name(), self._get_submission_benchmark_name(),
                            self.scenario.valstr())

    def check_file_exists(self, f):
        """Check if file exists. Complain if configured to do so."""

        if not os.path.isfile(f):
            if self.skip_file_checks:
                print(f"Note: File {f} does not exist. Attempting to continue regardless, as hard file checks are disabled.")
                return False
            else:
                raise RuntimeError("File {:} does not exist.".format(f))
        return True

    def build_default_flags(self):
        flag_dict = {}
        flag_dict["verbose"] = self.verbose
        flag_dict["verbose_nvtx"] = self.verbose_nvtx
        flag_dict["v"] = self.verbose_glog

        # Handle plugins
        plugins = get_trt_plugin_paths_by_network(self.name, self.args)
        if len(plugins) > 0:
            logging.info(f"The harness will load {len(plugins)} plugins: {plugins}")
            flag_dict["plugins"] = ",".join(plugins)

        # Handle engine file
        flag_dict["engine_file"] = self.engine_file

        # Generate flags for logfile names.
        log_dir = self.get_full_log_dir()
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        flag_dict["logfile_outdir"] = log_dir
        flag_dict["logfile_prefix"] = "mlperf_log_"

        # Handle performance sample count
        perf_sample_count = dict_get(self.args, "performance_sample_count", None)
        perf_sample_count_override = dict_get(self.args, "performance_sample_count_override", None)
        if perf_sample_count_override is not None:
            flag_dict["performance_sample_count"] = perf_sample_count_override
        elif perf_sample_count is not None:
            flag_dict["performance_sample_count"] = perf_sample_count
        elif benchmark_qsl_size_map[self._get_submission_benchmark_name()] > 0:
            flag_dict["performance_sample_count"] = benchmark_qsl_size_map[self._get_submission_benchmark_name()]
        else:
            flag_dict["performance_sample_count"] = self.args["gpu_batch_size"] # FIXME: jsuh: automotive will use batch size 1 for multi-dim inputs; what should we do for else?

        # Handle custom arguments
        for arg in self.flag_builder_custom_args:
            val = dict_get(self.args, arg, None)
            if val is not None:
                flag_dict[arg] = val

        return flag_dict

    def build_scenario_specific_flags(self):
        """Return flags specific to current scenario."""

        flag_dict = {}

        scenario_keys = common_args.getScenarioMetricArgs(self.scenario)

        for arg in scenario_keys:
            val = dict_get(self.args, arg, None)
            if val is None:
                raise ValueError("Missing required key {:}".format(arg))
            flag_dict[arg] = val

        # Handle RUN_ARGS
        for arg in scenario_keys:
            val = dict_get(self.args, arg, None)
            if val is not None:
                flag_dict[arg] = val

        return flag_dict

    def build_non_custom_flags(self):
        """Returns the flag_dict for all flags excluding custom ones.
        """
        flag_dict = self.build_default_flags()
        flag_dict.update(self.build_scenario_specific_flags())

        # Special handing of --test_run for short runs during testing. Use min_duration=60000 and min_query_count=1.
        if flag_dict.get("test_run", False):
            if "min_duration" not in flag_dict:
                flag_dict["min_duration"] = 60000
            if "min_query_count" not in flag_dict:
                flag_dict["min_query_count"] = 1
            flag_dict["test_run"] = None
        if "min_duration" in flag_dict:
            logging.info(f"min_duration is overwritten to {flag_dict['min_duration']}.")
        if "min_query_count" in flag_dict:
            logging.info(f"min_query_cnt is overwritten to {flag_dict['min_query_count']}.")

        return flag_dict

    def prepend_ld_preload(self, so_path):
        if "LD_PRELOAD" in self.env_vars:
            self.env_vars["LD_PRELOAD"] = ":".join([so_path, self.env_vars["LD_PRELOAD"]])
        else:
            self.env_vars["LD_PRELOAD"] = so_path

        logging.info("Updated LD_PRELOAD: " + self.env_vars["LD_PRELOAD"])

    def run_harness(self, flag_dict=None, skip_generate_measurements=False, use_py_harness=False):
        if flag_dict is None:
            flag_dict = self.build_non_custom_flags()

        if not skip_generate_measurements:
            # Generates the entries in the `measurements/` directory, and updates flag_dict accordingly
            generate_measurements_entry(
                self.get_system_name(),
                self.name,
                self._get_submission_benchmark_name(),
                self.scenario,
                self.args["input_dtype"],
                self.args["precision"],
                flag_dict)
        argstr = self._build_custom_flags(flag_dict)
        if type(argstr) is dict:
            argstr = args_to_string(flag_dict)

        if use_py_harness and self.benchmark in [
            Benchmark.BEVFormer,
        ]:
            logging.info(f"Using pybind11 for this workload: {self.benchmark} {self.scenario} {self.system_id} ...")
            workload = get_cls(self.mitten_workload)(flag_dict)
            workload.run()
        else:
            if use_py_harness:
                logging.warning("This workload does not support the new Python harness.")
            logging.info("Using harness launch command...")
            output = run_command(self._construct_terminal_command(argstr),
                                    get_output=True,
                                    custom_env=self.env_vars)

        # Return harness result.
        scenario_key = get_loadgen_log_keys(self.scenario, self.benchmark)
        query_metric_key = QUERY_METRIC_CONSTRAINTS[self.scenario.valstr()][0]
        loadgen_query_keys = ["result_validity", scenario_key, "early_stopping_met", query_metric_key, "effective_min_duration_ms"]

        results = from_loadgen_by_keys(os.path.join(self.args["log_dir"],
                                                    self.get_system_name(),
                                                    self._get_submission_benchmark_name(),
                                                    self.scenario.valstr()),
                                       loadgen_query_keys)
        test_mode = flag_dict.get("test_mode", "PerformanceOnly")
        satisfies_query_constraint = float(results.get(query_metric_key, "0.0")) \
            >= QUERY_METRIC_CONSTRAINTS[self.scenario.valstr()][1]
        results.update({"system_name": self.get_system_name(),
                        "benchmark_short": self.benchmark.valstr(),
                        "benchmark_full": self._get_submission_benchmark_name(),
                        "scenario": self.scenario.valstr(),
                        "test_mode": test_mode,
                        "tensorrt_version": trt.__version__,
                        "detected_system": DETECTED_SYSTEM.summary_description(),
                        "scenario_key": scenario_key,
                        "satisfies_query_constraint": satisfies_query_constraint
                        })

        return results
