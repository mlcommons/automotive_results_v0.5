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


from __future__ import annotations
from enum import Enum, unique

from ...aliased_name import AliasedName, AliasedNameEnum


__doc__ = """Stores constants and Enums related to MLPerf Automotive"""


VERSION: Final[str] = "v3.0"
"""str: Current version of MLPerf Automotive"""


@unique
class Benchmark(AliasedNameEnum):
    """Names of supported Benchmarks in MLPerf Automotive."""

    BEVFormer: AliasedName = AliasedName("bevformer")
    SSDResNet50: AliasedName = AliasedName("ssd_resnet50")
    DeepLabV3Plus: AliasedName = AliasedName("deeplabv3plus")

@unique
class Scenario(AliasedNameEnum):
    """Names of supported workload scenarios in MLPerf Automotive.

    IMPORTANT: This is **not** interchangeable or compatible with the TestScenario Enum from the mlperf_loadgen Python
    bindings. Make sure that any direct calls to mlperf_loadgen use the TestScenario Enum from loadgen, **not** this
    Enum.
    """

    SingleStream: AliasedName = AliasedName("SingleStream", ("single-stream", "single_stream"))
    ConstantStream: AliasedName = AliasedName("ConstantStream", ("constant-stream", "constant_stream"))


@unique
class AuditTest(AliasedNameEnum):
    """Audit test names"""

    TEST01: AliasedName = AliasedName("TEST01")
    TEST04: AliasedName = AliasedName("TEST04")
    TEST05: AliasedName = AliasedName("TEST05")


@unique
class AccuracyTarget(Enum):
    """Possible accuracy targets a benchmark must meet. Determined by MLPerf Automotive committee."""
    k_99: float = .99
    k_99_9: float = .999


G_HIGH_ACC_ENABLED_BENCHMARKS: Tuple[Benchmark, ...] = (
    Benchmark.BEVFormer,
)
"""Tuple[Benchmark, ...]: Benchmarks that have 99.9% accuracy targets"""


G_PERCEPTION_BENCHMARKS: Tuple[Benchmark, ...] = (
    Benchmark.BEVFormer,
    Benchmark.SSDResNet50,
    Benchmark.DeepLabV3Plus,
)
"""Tuple[Benchmark, ...]: Benchmarks for Perception"""
