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

from collections import namedtuple
from importlib import import_module
from typing import Dict, Any

from code.common import logging
from code.common.constants import AliasedNameEnum, Benchmark

G_BENCHMARK_COMPONENT_ENUM_MAP: Dict[Benchmark, AliasedNameEnum] = {
    Benchmark.BEVFormer: None,
}
G_BENCHMARK_ACCELERATOR_COMPONENT_ALIAS_MAP = {
    Benchmark.BEVFormer: {"gpu": []},
}

# Instead of storing the objects themselves in maps, we store object locations, as we do not want to import redundant
# modules on every run. Some modules include CDLLs and TensorRT plugins, or have large imports that impact runtime.
# Dynamic imports are also preferred, as some modules (due to their legacy model / directory names) include dashes.
ModuleLocation = namedtuple("ModuleLocation", ("module_path", "cls_name"))
G_BENCHMARK_CLASS_MAP = {
    # Benchmark.ResNet50: ModuleLocation("code.resnet50.tensorrt.builder", "ResNet50"),
    Benchmark.BEVFormer: ModuleLocation("code.bevformer.tensorrt.builder", "BEVFormer"),
}
G_HARNESS_CLASS_MAP = {
    "lwis_harness": ModuleLocation("code.common.lwis_harness", "LWISHarness"),
    "bevformer_harness": ModuleLocation("code.bevformer.tensorrt.harness", "BEVFormerHarness"),
    "profiler_harness": ModuleLocation("code.internal.profiler", "ProfilerHarness"),
}


def get_cls(module_loc: ModuleLocation) -> type:
    """
    Returns the specified class denoted by a ModuleLocation.

    Args:
        module_loc (ModuleLocation):
            The ModuleLocation to specify the import path of the class

    Returns:
        type: The imported class located at module_loc
    """
    return getattr(import_module(module_loc.module_path), module_loc.cls_name)


def validate_batch_size(conf):
    benchmark = conf["benchmark"]

    # check if gpu_batch_size is of a valid component combination
    for accelerator in ("gpu",):
        batch_size_dict = conf.get(f"{accelerator}_batch_size")
        # If not using this accelerator then continue
        if not batch_size_dict:
            continue
        if G_BENCHMARK_COMPONENT_ENUM_MAP[benchmark] is not None and G_BENCHMARK_ACCELERATOR_COMPONENT_ALIAS_MAP[benchmark][accelerator] is not None:
            if not any(set(batch_size_dict.keys()) == valid_components for valid_components in G_BENCHMARK_ACCELERATOR_COMPONENT_ALIAS_MAP[benchmark][accelerator]):
                raise ValueError(f"[{benchmark.valstr()}] {accelerator}_batch_size : {batch_size_dict} does not have supported component combinations. Valid combinations are \
                                 {[valid_components for valid_components in G_BENCHMARK_ACCELERATOR_COMPONENT_ALIAS_MAP[benchmark][accelerator]]}")


def convert_to_component_aliased_enum(conf):
    benchmark = conf["benchmark"]
    for accelerator in ("gpu",):
        batch_size_dict = conf.get(f"{accelerator}_batch_size")
        # If not using this accelerator then continue
        if not batch_size_dict:
            continue
        for component in list(batch_size_dict.keys()):
            if G_BENCHMARK_COMPONENT_ENUM_MAP[benchmark] is not None:
                component_alias_enum = G_BENCHMARK_COMPONENT_ENUM_MAP[benchmark].get_match(component)
                if not component_alias_enum:
                    raise ValueError(f"[{benchmark.valstr()}] {accelerator}_batch_size : {batch_size_dict} has unsupported component {component}")
                batch_size_dict[component_alias_enum] = batch_size_dict.pop(component)


def get_benchmark(conf):
    """Return module of benchmark initialized with config."""

    benchmark = conf["benchmark"]
    if not isinstance(benchmark, Benchmark):
        logging.warning(f"'benchmark: {benchmark}' in config is not Benchmark Enum member. This behavior is deprecated.")
        benchmark = Benchmark.get_match(benchmark)
        if benchmark is None:
            ttype = type(conf["benchmark"])
            raise ValueError(f"'benchmark' in config is not of supported type '{ttype}'")

    if benchmark not in G_BENCHMARK_CLASS_MAP:
        raise ValueError(f"Unknown benchmark: {benchmark}")

    convert_to_component_aliased_enum(conf)
    # batch size validity check
    validate_batch_size(conf)

    builder_op = get_cls(G_BENCHMARK_CLASS_MAP[benchmark])
    return builder_op(conf)


def harness_matcher(benchmark: Benchmark):
    """Helper function to determine harness key and inference server type based on benchmark type."""
    # NOTE: Default to LWIS
    inference_server = "lwis"
    harness_key = "lwis_harness"

    if Benchmark.BEVFormer == benchmark:
        harness_key = "bevformer_harness"
        inference_server = "custom"

    return harness_key, inference_server


def get_harness(conf: Dict[str, Any], profile):
    """Refactors harness generation for use by functions other than handle_run_harness."""

    benchmark = conf["benchmark"]
    if not isinstance(benchmark, Benchmark):
        logging.warning("'benchmark' in config is not Benchmark Enum member. This behavior is deprecated.")
        benchmark = Benchmark.get_match(benchmark)
        if benchmark is None:
            ttype = type(conf["benchmark"])
            raise ValueError(f"'benchmark' in config is not of supported type '{ttype}'")

    convert_to_component_aliased_enum(conf)
    # batch size validity check
    validate_batch_size(conf)

    harness_key, inference_server = harness_matcher(benchmark)
    conf["inference_server"] = inference_server
    harness = get_cls(G_HARNESS_CLASS_MAP[harness_key])(conf, benchmark)

    # Attempt to run profiler. Note that this is only available internally.
    if profile is not None:
        try:
            harness = get_cls(G_HARNESS_CLASS_MAP["profiler_harness"])(harness, profile)
        except BaseException:
            logging.info("Could not load profiler: Are you an internal user?")

    return harness, conf
