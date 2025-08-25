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
import argparse

from code.common.constants import Scenario


#: Global args for all scripts
arguments_dict = {
    # Common arguments
    "gpu_batch_size": {
        "help": "GPU batch size to use for the engine.",
        "type": int,
    },
    "batch_size": {
        "help": "Batch size to use for the engine.",
        "type": int,
    },
    "verbose": {
        "help": "Use verbose output",
        "action": "store_true",
    },
    "verbose_nvtx": {
        "help": "Turn ProfilingVerbosity to kDETAILED so that layer detail is printed in NVTX.",
        "action": "store_true",
    },
    "verbose_glog": {
        "help": "glog verbosity level",
        "type": int,
    },
    "no_child_process": {
        "help": "Do not generate engines on child process. Do it on current process instead.",
        "action": "store_true"
    },
    "workspace_size": {
        "help": "The maximum size of temporary workspace that any layer in the network can use in TRT",
        "type": int,
        "default": None
    },

    # Power measurements
    "power": {
        "help": "Select if you would like to measure power",
        "action": "store_true"
    },
    "power_limit": {
        "help": "Set power upper limit to the specified value.",
        "type": int,
        "default": None
    },
    # Dataset location
    "data_dir": {
        "help": "Directory containing unprocessed datasets",
        "default": os.environ.get("DATA_DIR", "build/data"),
    },
    "preprocessed_data_dir": {
        "help": "Directory containing preprocessed datasets",
        "default": os.environ.get("PREPROCESSED_DATA_DIR", "build/preprocessed_data"),
    },

    # Arguments related to precision
    "precision": {
        "help": "Precision. Default: int8",
        "choices": ["fp32", "fp16", "bf16", "int8", "fp8", "fp4", None],
        # None needs to be set as default since passthrough arguments will
        # contain a default value and override configs. Specifying None as the
        # default will cause it to not be inserted into passthrough / override
        # arguments.
        "default": None,
    },
    "input_dtype": {
        "help": "Input datatype. Choices: fp32, int8.",
        "choices": ["fp32", "fp16", "bf16", "int8", "fp8", "fp4", None],
        "default": None
    },
    "input_format": {
        "help": "Input format (layout). Choices: linear, chw4, dhwc8, cdhw32",
        "choices": ["linear", "chw4", "dhwc8", "cdhw32", None],
        "default": None
    },
    # Arguments related to quantization calibration
    "force_calibration": {
        "help": "Run quantization calibration even if the cache exists. (Only used for quantized models)",
        "action": "store_true",
    },
    "calib_batch_size": {
        "help": "Batch size for calibration.",
        "type": int
    },
    "calib_max_batches": {
        "help": "Number of batches to run for calibration.",
        "type": int
    },
    "cache_file": {
        "help": "Path to calibration cache.",
        "default": None,
    },
    "calib_data_map": {
        "help": "Path to the data map of the calibration set.",
        "default": None,
    },

    # Benchmark configuration arguments
    "scenario": {
        "help": "Name for the scenario. Used to generate engine name.",
    },
    "model_path": {
        "help": "Path to the model (weights) file.",
    },
    "active_sms": {
        "help": "Control the percentage of active SMs while generating engines.",
        "type": int
    },

    # Profiler selection
    "profile": {
        "help": "[INTERNAL ONLY] Select if you would like to profile -- select among nsys, nvprof and ncu",
        "type": str
    },

    # Harness configuration arguments
    "log_dir": {
        "help": "Directory for all output logs.",
        "default": os.environ.get("LOG_DIR", "build/logs/default"),
    },
    "use_graphs": {
        "help": "Enable CUDA graphs",
        "action": "store_true",
    },

    # LWIS settings
    "devices": {
        "help": "Comma-separated list of numbered devices",
    },
    "map_path": {
        "help": "Path to map file for samples",
    },
    "tensor_path": {
        "help": "Path to preprocessed samples in .npy format",
    },
    "performance_sample_count": {
        "help": "Number of samples to load in performance set.  0=use default",
        "type": int,
    },
    "performance_sample_count_override": {
        "help": "If set, this overrides performance_sample_count.  0=don't override",
        "type": int,
    },
    "gpu_copy_streams": {
        "help": "Number of copy streams to use for GPU",
        "type": int,
    },
    "gpu_inference_streams": {
        "help": "Number of inference streams to use for GPU",
        "type": int,
    },
    "run_infer_on_copy_streams": {
        "help": "Run inference on copy streams.",
    },
    "warmup_duration": {
        "help": "Minimum duration to perform warmup for",
        "type": float,
    },
    "use_direct_host_access": {
        "help": "Use direct access to host memory for all devices",
    },
    "use_deque_limit": {
        "help": "Use a max number of elements dequed from work queue",
    },
    "deque_timeout_usec": {
        "help": "Timeout in us for deque from work queue.",
        "type": int,
    },
    "use_batcher_thread_per_device": {
        "help": "Enable a separate batcher thread per device",
    },
    "use_cuda_thread_per_device": {
        "help": "Enable a separate cuda thread per device",
    },
    "start_from_device": {
        "help": "Assuming that inputs start from device memory in QSL"
    },
    "end_on_device": {
        "help": "Copy outputs to host in untimed loadgen callback"
    },
    "coalesced_tensor": {
        "help": "Turn on if all the samples are coalesced into one single npy file"
    },
    "assume_contiguous": {
        "help": "Assume that the data in a query is already contiguous"
    },
    "complete_threads": {
        "help": "Number of threads per device for sending responses",
        "type": int,
    },
    "use_same_context": {
        "help": "Use the same TRT context for all copy streams (shape must be static and gpu_inference_streams must be 1).",
    },
    "use_spin_wait": {
        "help": "Use spin waiting for LWIS. Recommended for single stream",
        "action": "store_true",
    },


    # Shared settings
    "mlperf_conf_path": {
        "help": "Path to mlperf.conf",
    },
    "user_conf_path": {
        "help": "Path to user.conf",
    },

    # Loadgen settings
    "test_mode": {
        "help": "Testing mode for Loadgen",
        "choices": ["SubmissionRun", "AccuracyOnly", "PerformanceOnly", "FindPeakPerformance"],
    },
    "min_duration": {
        "help": "Minimum test duration",
        "type": int,
    },
    "max_duration": {
        "help": "Maximum test duration",
        "type": int,
    },
    "min_query_count": {
        "help": "Minimum number of queries in test",
        "type": int,
    },
    "max_query_count": {
        "help": "Maximum number of queries in test",
        "type": int,
    },
    "qsl_rng_seed": {
        "help": "Seed for RNG that specifies which QSL samples are chosen for performance set and the order in which samples are processed in AccuracyOnly mode",
        "type": int,
    },
    "sample_index_rng_seed": {
        "help": "Seed for RNG that specifies order in which samples from performance set are included in queries",
        "type": int,
    },
    "test_run": {
        "help": "If set, will set min_duration to 1 minute (60000ms). For Offline and Server, min_query_count is set to 1.",
        "action": "store_true",
    },

    # Loadgen logging settings
    "logfile_suffix": {
        "help": "Specify the filename suffix for the LoadGen log files",
    },
    "logfile_prefix_with_datetime": {
        "help": "Prefix filenames for LoadGen log files",
        "action": "store_true",
    },
    "log_copy_detail_to_stdout": {
        "help": "Copy LoadGen detailed logging to stdout",
        "action": "store_true",
    },
    "disable_log_copy_summary_to_stdout": {
        "help": "Disable copy LoadGen summary logging to stdout",
        "action": "store_true",
    },
    "log_mode": {
        "help": "Logging mode for Loadgen",
        "choices": ["AsyncPoll", "EndOfTestOnly", "Synchronous"],
    },
    "log_mode_async_poll_interval_ms": {
        "help": "Specify the poll interval for asynchrounous logging",
        "type": int,
    },
    "log_enable_trace": {
        "help": "Enable trace logging",
    },

    # Server harness arguments
    "accuracy_log_rng_seed": {
        "help": "Affects which samples have their query returns logged to the accuracy log in performance mode.",
        "type": int,
    },

    # Single stream harness arguments
    "single_stream_expected_latency_ns": {
        "help": "Inverse of desired target QPS",
        "type": int,
    },
    "single_stream_target_latency_percentile": {
        "help": "Desired latency percentile for single stream scenario",
        "type": float,
    },

    # Constant stream harness arguments
    "constant_stream_target_qps": {
        "help": "Target QPS for constant stream scenario",
        "type": int,
    },
    "constant_stream_target_latency_percentile": {
        "help": "Desired latency percentile to report as a performance metric, for constant stream scenario",
        "type": float,
    },

    # Args used by code.main
    "action": {
        "help": "generate_engines / run_harness / calibrate / generate_conf_files",
        "choices": ["generate_engines", "run_harness", "calibrate", "generate_conf_files", "run_audit_harness", "run_cpu_audit_harness", "run_audit_verification", "run_cpu_audit_verification"],
    },
    "benchmarks": {
        "help": "Specify the benchmark(s) with a comma-separated list. " +
        "Default: run all benchmarks.",
        "default": None,
    },
    "configs": {
        "help": "Specify the config files with a comma-separated list. " +
        "Wild card (*) is also allowed. If \"\", detect platform and attempt to load configs. " +
        "Default: \"\"",
        "default": "",
    },
    "config_ver": {
        "help": "Config version to run. Uses 'default' if not set.",
        "default": "default",
    },
    "scenarios": {
        "help": "Specify the scenarios with a comma-separated list. " +
        "Choices:[\"SingleStream\", \"FixedStream\"] " +
        "Default: \"*\"",
        "default": None,
    },
    "audit_test": {
        "help": "Defines audit test to run.",
        "choices": ["TEST01", "TEST04-A", "TEST04-B"],
    },
    "system_name": {
        "help": "Override the system name to run under",
        "type": str
    },

    # Args used for engine runners
    "engine_dir": {
        "help": "Set the engine directory to load from or serialize to",
        "type": str
    },
    "engine_file": {
        "help": "Set the engine file to load from",
        "type": str
    },
    "num_samples": {
        "help": "Number of samples to use for accuracy runner",
        "type": int,
    },
    "num_scenes": {
        "help": "Number of scenes to use for accuracy runner",
        "type": int,
    },
    "vboost_slider": {
        "help": "Control clock-propagation ratios between GPC-XBAR. Look at `nvidia-smi boost-slider --vboost`.",
        "type": int
    },
}

# ================== Argument groups ================== #

# Engine generation
PRECISION_ARGS = [
    "input_dtype",
    "input_format",
    "precision",
]
CALIBRATION_ARGS = [
    "cache_file",
    "calib_batch_size",
    "calib_data_map",
    "calib_max_batches",
    "force_calibration",
    "model_path",
    "verbose",
]
GENERATE_ENGINE_ARGS = [
    "active_sms",
    "gpu_batch_size",
    "gpu_copy_streams",
    "gpu_inference_streams",
    "power_limit",
    "verbose_nvtx",
    "workspace_size",
] + PRECISION_ARGS + CALIBRATION_ARGS

# Harness framework arguments
LOADGEN_ARGS = [
    "accuracy_log_rng_seed",
    "disable_log_copy_summary_to_stdout",
    "log_copy_detail_to_stdout",
    "log_enable_trace",
    "log_mode",
    "log_mode_async_poll_interval_ms",
    "logfile_prefix_with_datetime",
    "logfile_suffix",
    "max_duration",
    "max_query_count",
    "min_duration",
    "min_query_count",
    "qsl_rng_seed",
    "sample_index_rng_seed",
    "single_stream_target_latency_percentile",
    "constant_stream_target_latency_percentile",
    "constant_stream_target_qps", # FIXME: This should not be configurable; perhaps still useful for debugging
    "test_mode",
    "test_run",
]
LWIS_ARGS = [
    "assume_contiguous",
    "coalesced_tensor",
    "complete_threads",
    "deque_timeout_usec",
    "devices",
    "engine_file",
    "gpu_copy_streams",
    "gpu_inference_streams",
    "run_infer_on_copy_streams",
    "start_from_device",
    "end_on_device",
    "use_batcher_thread_per_device",
    "use_cuda_thread_per_device",
    "use_deque_limit",
    "use_direct_host_access",
    "use_spin_wait",
    "warmup_duration",
]
SHARED_ARGS = [
    "gpu_batch_size",
    "map_path",
    "mlperf_conf_path",
    "performance_sample_count",
    "performance_sample_count_override",
    "tensor_path",
    "use_graphs",
    "user_conf_path",
]
OTHER_HARNESS_ARGS = [
    "check_contiguity",
    "gpu_num_bundles",
    "log_dir",
    "max_pairs_per_staging_thread",
    "num_staging_batches",
    "num_staging_threads",
    "power_limit",
]
HARNESS_ARGS = ["verbose_glog", "verbose", "scenario"] + PRECISION_ARGS + LOADGEN_ARGS + LWIS_ARGS + SHARED_ARGS + OTHER_HARNESS_ARGS

# Scenario dependent arguments. These are prefixed with device: "gpu_", ...
SCENARIO_METRIC_PREFIXES = ["gpu_", ]
SINGLE_STREAM_PARAMS = ["single_stream_expected_latency_ns"]
CONSTANT_STREAM_PARAMS = ["constant_stream_target_qps"]
SERVER_PARAMS = []

#: Args for code.main
MAIN_ARGS = [
    "action",
    "audit_test",
    "benchmarks",
    "config_ver",
    "configs",
    "no_child_process",
    "power",
    "power_limit",
    "profile",
    "scenarios",
    "system_name",
]

# For accuracy runners
ACCURACY_ARGS = [
    "batch_size",
    "engine_file",
    "num_samples",
    "num_scenes",
    "verbose",
]


def parse_args(whitelist):
    """Parse whitelist args in user input and return parsed args."""

    parser = argparse.ArgumentParser(allow_abbrev=False)
    for flag in whitelist:

        # Check with global arg list
        if flag not in arguments_dict:
            raise IndexError("Unknown flag '{:}'".format(flag))

        parser.add_argument("--{:}".format(flag), **arguments_dict[flag])
    return vars(parser.parse_known_args()[0])


def check_args():
    """Create arg parser with global args and check if it works."""
    parser = argparse.ArgumentParser(allow_abbrev=False)
    for flag in arguments_dict:
        parser.add_argument("--{:}".format(flag), **arguments_dict[flag])
    parser.parse_args()


def apply_overrides(config, keys):
    """Apply overrides from user input on config file data."""
    # Make a copy so we don't modify original dict
    config = dict(config)
    override_args = parse_args(keys)
    for key in override_args:
        # Unset values (None) and unset store_true values (False) are both false-y
        if override_args[key]:
            config[key] = override_args[key]
    return config

##
# @brief Create an argument list based on scenario and benchmark name


def getScenarioMetricArgs(scenario, prefixes=("",)):
    """
    Returns a list of metric arguments specific to a scenario, prepended by all individual prefixes specified.

    i.e. for scenario key "foo" and prefixes ["a_", "b_"], this method would return ["a_foo", "b_foo"].

    By default, prefixes is SCENARIO_METRIC_PREFIXES
    """
    arglist = None
    if Scenario.SingleStream == scenario:
        arglist = SINGLE_STREAM_PARAMS
    elif Scenario.ConstantStream == scenario:
        arglist = CONSTANT_STREAM_PARAMS
    else:
        raise RuntimeError("Unknown Scenario \"{}\"".format(scenario))

    # Apply prefixes
    return [
        prefix + arg
        for prefix in prefixes
        for arg in arglist
    ]


def getScenarioBasedHarnessArgs(scenario, prefixes=("",)):
    """Return arguments for harness for a given scenario."""
    return HARNESS_ARGS + getScenarioMetricArgs(scenario, prefixes=prefixes)
