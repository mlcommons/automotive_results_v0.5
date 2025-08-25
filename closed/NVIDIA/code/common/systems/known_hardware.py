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

from collections.abc import Iterable
from nvmitten.constants import CPUArchitecture
from nvmitten.memory import Memory
from nvmitten.interval import NumericRange
from nvmitten.nvidia.accelerator import GPU
from nvmitten.nvidia.constants import ComputeSM
from nvmitten.system.component import Description
from nvmitten.system.cpu import CPU


def PCIVendorID(s):
    return f"0x{s}10DE"


def GPUDescription(**kwargs):
    if "pci_id" in kwargs:
        if isinstance(kwargs["pci_id"], str):
            kwargs["pci_id"] = PCIVendorID(kwargs["pci_id"])
        elif isinstance(kwargs["pci_id"], Iterable):
            kwargs["pci_id"] = list(PCIVendorID(s) for s in kwargs["pci_id"]).__contains__
        else:
            raise TypeError("Unexpected type for PCI ID")
    if "vram" in kwargs:
        kwargs["vram"] = NumericRange(Memory.from_string(kwargs["vram"]), rel_tol=0.05)
    if "max_power_limit" in kwargs:
        kwargs["max_power_limit"] = NumericRange(kwargs["max_power_limit"], rel_tol=0.05)
    if "is_integrated" not in kwargs:
        kwargs["is_integrated"] = False

    return Description(GPU, **kwargs)


class KnownGPU:
    """By convention, we use the PCI ID but not the device name, as the name is prone and subject to change while PCI
    vendor IDs are not.
    """
    T264_ThorX = GPUDescription(
        name="Thor",
        compute_sm=ComputeSM(10, 1),
        vram="14.7 GiB",  # 512GiB max; in reality, likely 128GiB from 8x x32bit-wide 16GiB LPDDR5x modules?
        is_integrated=True)


class KnownCPU:
    ARMGeneric = Description(CPU,
                             architecture=CPUArchitecture.aarch64,
                             vendor="ARM")
    x86_64_AMD_Generic = Description(CPU,
                                     architecture=CPUArchitecture.x86_64,
                                     vendor="AuthenticAMD")
    x86_64_Intel_Generic = Description(CPU,
                                       architecture=CPUArchitecture.x86_64,
                                       vendor="GenuineIntel")
    x86_64_Generic = Description(CPU,
                                 architecture=CPUArchitecture.x86_64)
