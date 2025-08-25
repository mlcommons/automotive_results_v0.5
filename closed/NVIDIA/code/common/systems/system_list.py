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

from nvmitten.constants import ByteSuffix, CPUArchitecture
from nvmitten.interval import NumericRange
from nvmitten.json_utils import load
from nvmitten.nvidia.accelerator import GPU, DLA
from nvmitten.system.component import Description
from nvmitten.system.system import System
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Set

from code.common.systems.known_hardware import *


# Dynamically build Enum for known systems
_system_confs = dict()


def add_systems(name_format_string: str,
                id_format_string: str,
                cpu: KnownCPU,
                accelerator: KnownGPU,
                accelerator_counts: List[int],
                mem_requirement: Memory,
                target_dict: Dict[str, Description] = _system_confs,
                tags: List[str] = None,
                n_dlas: int = 0):
    """Adds a Description to a dictionary.

    Args:
        name_format_string (str): A Python format to generate the name for the Enum member. Can have a single format
                                  item to represent the count.
        id_format_string (str): A Python format to generate the system ID to use. The system ID is used for the systems/
                                json file. Can contain a single format item to represent the count.
        cpu (KnownCPU): The CPU that the system uses
        accelerator (KnownGPU): The Accelerator that the system uses
        accelerator_counts (List[int]): The list of various counts to use for accelerators.
        mem_requirement (Memory): The minimum memory requirement to have been tested for the hardware configuration.
        target_dict (Dict[str, Description]): The dictionary to add the Description to.
                                              (Default: _system_confs)
        tags (List[str]): A list of strings denoting certain tags used for classifying the system. (Default: None)
        n_dlas (int): The number of DLAs present on the system (Default: 0)
    """
    def _mem_cmp(m):
        thresh = NumericRange(mem_requirement._num_bytes * 0.95)
        return thresh.contains_numeric(m.capacity._num_bytes)

    for count in accelerator_counts:
        def _accelerator_cmp(count=0):
            def _f(d):
                # Check GPUs
                if len(d[GPU]) != count:
                    return False

                for i in range(count):
                    if not accelerator.matches(d[GPU][i]):
                        return False

                # Check DLAs
                if len(d[DLA]) != n_dlas:
                    return False
                return True
            return _f

        k = name_format_string.format(count)
        v = Description(System,
                        _match_ignore_fields=["extras"],
                        cpu=cpu,
                        host_memory=_mem_cmp,
                        accelerators=_accelerator_cmp(count=count),
                        extras={"id": id_format_string.format(count),
                                "tags": set(tags) if tags else set()})

        target_dict[k] = v


# Thor-X
add_systems("ThorX_Eval",
            "Thor-X",
            KnownCPU.ARMGeneric,
            KnownGPU.T264_ThorX,
            [1],
            Memory(32, ByteSuffix.GiB), # FIXME: how does ThorX advertise system memory and GPU memory to OS?
            # tags=["start_from_device_enabled", "end_on_device_enabled"] # FIXME: v0.5 doesn't allow them so commented out
            )

# Handle custom systems to better support partner drops.
custom_system_file = Path("code/common/systems/custom_list.json")
if custom_system_file.exists():
    with custom_system_file.open() as f:
        custom_systems = load(f)

    for k, v in custom_systems.items():
        if k in _system_confs:
            raise KeyError(f"SystemEnum member {k} already exists")

        # Set up 'extras'
        if "extras" not in v.mapping:
            v.mapping["extras"] = dict()
        v._match_ignore_fields.add("extras")

        if "tags" not in v.mapping["extras"]:
            v.mapping["extras"]["tags"] = set()

        v.mapping["extras"]["id"] = k
        v.mapping["extras"]["tags"].add("custom")

        _system_confs[k] = v

KnownSystem = SimpleNamespace(**_system_confs)


def classification_tags(system: System) -> Set[str]:
    tags = set()

    # This may break for non-homogeneous systems.
    gpus = system.accelerators[GPU]
    if len(gpus) > 0:
        tags.add("gpu_based")

        primary_sm = int(gpus[0].compute_sm)
        if primary_sm == (100, 101):
            tags.add("is_blackwell")
        if primary_sm == 90:
            tags.add("is_hopper")
        if primary_sm == 89:
            tags.add("is_ada")
        if primary_sm in (80, 86, 87, 89):
            tags.add("is_ampere")
        if primary_sm == 75:
            tags.add("is_turing")

        if gpus[0].name.startswith("Orin") and primary_sm == 87:
            tags.add("is_orin")
            tags.add("is_soc")

        if gpus[0].name.startswith("Thor") and primary_sm == 101:
            tags.add("is_thor")
            tags.add("is_soc")

    if len(gpus) > 1:
        tags.add("multi_gpu")

    if system.cpu.architecture == CPUArchitecture.aarch64:
        tags.add("is_aarch64")

    return tags


DETECTED_SYSTEM = System.detect()[0]
for name, sys_desc in _system_confs.items():
    if sys_desc.matches(DETECTED_SYSTEM):
        DETECTED_SYSTEM.extras["id"] = sys_desc.mapping["extras"]["id"]
        DETECTED_SYSTEM.extras["tags"] = sys_desc.mapping["extras"]["tags"].union(classification_tags(DETECTED_SYSTEM))
        DETECTED_SYSTEM.extras["name"] = name

        # Convenience field
        if len(DETECTED_SYSTEM.accelerators[GPU]) > 0:
            DETECTED_SYSTEM.extras["primary_compute_sm"] = DETECTED_SYSTEM.accelerators[GPU][0].compute_sm
        else:
            DETECTED_SYSTEM.extras["primary_compute_sm"] = None
        break
