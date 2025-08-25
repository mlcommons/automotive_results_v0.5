#! /usr/bin/env python3
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
import sys
import argparse
import json

import collections

submitter = "NVIDIA"

soc_sw_version_dict = \
    {
        "thor-x": {
            "trt": "TensorRT 10.13",
            "cuda": "CUDA 12.8",
            "cudnn": "cuDNN 9.8.0",
            "os": "DriveOS 7.0.3",
        },
    }


def get_soc_sw_version(accelerator_name, software_name):
    if software_name not in ["trt", "cuda", "cudnn", "os"]:
        raise KeyError(f"No version info for {software_name}. Options: {(list(soc_sw_version_dict[list(soc_sw_version_dict)[0]].keys()))}")
    if "thor-x" in accelerator_name.lower():
        return soc_sw_version_dict["thor-x"][software_name]
    else:
        # For v0.5 submission nothing else is expected
        raise KeyError("Only Thor-X is available for now.")


class Status:
    HARDENED_SYSTEMS = "hardened"
    DEVELOPMENT_SYSTEMS = "development"
    ENGINEERING_SAMPLES = "engineering_samples"
    PRESILICON = "presilicon"


class Division:
    CLOSED = "closed"
    OPEN = "open"


class SystemType:
    ADAS = "adas"


Machine = collections.namedtuple("Machine", [
    "status",
    "host_processor_model_name",
    "host_processors_per_node",
    "host_processor_core_count",
    "host_processor_frequency",
    "host_processor_caches",
    "host_memory_capacity",
    "host_storage_capacity",
    "host_storage_type",
    "disk_controllers",
    "disk_drives",
    "number_of_nodes",
    "accelerator_model_name",
    "accelerator_short_name",
    "mig_short_name",
    "accelerator_frequency",
    "accelerator_memory_capacity",
    "accelerator_memory_configuration",
    "hw_notes",
    "sw_notes",
    "system_id_prefix",
    "system_name_prefix",
    "host_memory_configuration",
    "host_networking",
    "host_network_card_count",
    "host_networking_topology",
    "accelerator_host_interconnect",
    "accelerator_interconnect",
    "cooling",
    "system_type_detail",
    "power_supply_details",
    "power_supply_quantity_and_rating_watts",
])

"""
Tutorial: how to get each field.
L40S = Machine(
    status=Status.AVAILABLE,
    host_processor_model_name="AMD EPYC 7742 64-Core Processor", # lscpu
    host_processors_per_node=2, # lscpu
    host_processor_core_count=64, # lscpu
    host_memory_capacity="1 TB", # free -h
    host_storage_capacity="3 TB SSD, 5 TB CIFS", # df -h /tmp
    host_storage_type="NVMe SSD, CIFS mounted disk storage",
    number_of_nodes=1,
    accelerator_model_name="NVIDIA L40S", # nvidia-smi
    accelerator_short_name="L40S",
    mig_short_name="",
    accelerator_memory_capacity="45 GB",
    accelerator_memory_configuration="GDDR6",
    hw_notes="",
    sw_notes="",
    system_id_prefix="",
    system_name_prefix="ASROCKRACK 4U8G-ROME2/4E", # csm find "partitions=mlperf-inference" -v
    host_memory_configuration="16x 64GB 36ASF8G72PZ-3G2E1", # sudo dmidecode --type 17
    host_networking="Gig Ethernet", # lspci | egrep -i --color 'network|ethernet'
    host_network_card_count="2x 10Gbe",
    host_networking_topology="Ethernet on switching network",
    accelerator_host_interconnect="PCIe Gen4 x16", # sudo lspci -vv | grep -E 'PCI bridge|LnkCap'
    accelerator_interconnect="N/A",
    cooling="Air-cooled",
    system_type_detail="N/A",
    power_supply_details="",
    power_supply_quantity_and_rating_watts="",
)
"""
# Orin-Jetson submission machine for MaxP
THOR_X_MAXP = Machine(
    status=Status.DEVELOPMENT_SYSTEMS,
    host_processor_model_name="ARM Neoverse V3AE CPU",
    host_processors_per_node=1,
    host_processor_core_count=14,
    host_processor_frequency="2.6GHz",
    host_processor_caches="64KB L1I 64KB L1D 1MB L2 per core, 16MB L3",
    host_memory_capacity="128 GB",
    host_storage_capacity="128 GB eMMC, 5TB CIFS",
    host_storage_type="eMMC 5.1, CIFS mounted disk storage",
    disk_controllers="eMMC 5.1",
    disk_drives="eMMC 5.1",
    number_of_nodes=1,
    accelerator_frequency="1.52GHz",
    accelerator_model_name="NVIDIA Thor-X",
    accelerator_short_name="Thor-X",
    mig_short_name="",
    accelerator_memory_capacity="Shared with host",
    accelerator_memory_configuration="LPDDR5x",
    hw_notes="",
    sw_notes="",
    system_id_prefix="",
    system_name_prefix="NVIDIA Thor-X with DriveOS",
    host_memory_configuration="64GB 256-bit bus LPDDR5x",
    host_networking="Gig Ethernet",
    host_network_card_count="1 Integrated",
    host_networking_topology="USB forwarded",
    accelerator_host_interconnect="NVLink C2C",
    accelerator_interconnect="N/A",
    cooling="Air-cooled",
    system_type_detail="N/A",
    power_supply_details="",
    power_supply_quantity_and_rating_watts="",
)


class System():
    def __init__(self, machine, division, system_type, gpu_count=1, mig_count=0, is_soc=True, is_maxq=False, additional_config=""):
        # Check if gpu_count is a multiple of machine.number_of_nodes
        if gpu_count % machine.number_of_nodes != 0:
            raise ValueError(f"gpu_count ({gpu_count}) must be a multiple of machine.number_of_nodes ({machine.number_of_nodes})")
        self.attr = {
            "system_id": self._get_system_id(machine, division, gpu_count, mig_count, is_soc, is_maxq, additional_config),
            "submitter": submitter,
            "division": division,
            "system_type": system_type,
            "system_type_detail": machine.system_type_detail,
            "status": machine.status if division == Division.CLOSED else Status.ENGINEERING_SAMPLES,
            "system_name": self._get_system_name(machine, division, gpu_count, mig_count, is_soc, is_maxq, additional_config),
            "number_of_nodes": machine.number_of_nodes,
            "host_processor_model_name": machine.host_processor_model_name,
            "host_processors_per_node": machine.host_processors_per_node,
            "host_processor_core_count": machine.host_processor_core_count,
            "host_processor_frequency": machine.host_processor_frequency,
            "host_processor_caches": machine.host_processor_caches,
            "host_processor_interconnect": "",
            "host_memory_configuration": machine.host_memory_configuration,
            "host_memory_capacity": machine.host_memory_capacity,
            "host_storage_capacity": machine.host_storage_capacity,
            "host_storage_type": machine.host_storage_type,
            "host_networking": machine.host_networking,
            "host_network_card_count": machine.host_network_card_count,
            "host_networking_topology": machine.host_networking_topology,
            "accelerators_per_node": gpu_count // machine.number_of_nodes,
            "accelerator_model_name": self._get_accelerator_model_name(machine, division, gpu_count, mig_count, is_soc, is_maxq, additional_config),
            "accelerator_frequency": machine.accelerator_frequency,
            "accelerator_host_interconnect": machine.accelerator_host_interconnect,
            "accelerator_interconnect": machine.accelerator_interconnect,
            "accelerator_interconnect_topology": "",
            "accelerator_memory_capacity": machine.accelerator_memory_capacity,
            "accelerator_memory_configuration": machine.accelerator_memory_configuration,
            "accelerator_on-chip_memories": "",
            "cooling": machine.cooling,
            "hw_notes": machine.hw_notes,
            "sw_notes": machine.sw_notes,
            "framework": self._get_framework(machine, division, gpu_count, mig_count, is_soc, is_maxq, additional_config),
            "operating_system": self._get_operating_system(machine, is_soc),
            "other_software_stack": self._get_software_stack(machine, division, gpu_count, mig_count, is_soc, is_maxq, additional_config),
            "power_management": "",
            "filesystem": "",
            "boot_firmware_version": "",
            "management_firmware_version": "",
            "other_hardware": "",
            "number_of_type_nics_installed": "",
            "nics_enabled_firmware": "",
            "nics_enabled_os": "",
            "nics_enabled_connected": "",
            "network_speed_mbit": "",
            "power_supply_quantity_and_rating_watts": machine.power_supply_quantity_and_rating_watts,
            "power_supply_details": machine.power_supply_details,
            "disk_drives": machine.disk_drives,
            "disk_controllers": machine.disk_controllers,
        }

    def _get_system_id(self, machine, division, gpu_count, mig_count, is_soc, is_maxq, additional_config):
        return "".join([
            (machine.system_id_prefix + "_") if machine.system_id_prefix != "" else "",
            machine.accelerator_short_name,
            ("x" + str(gpu_count)) if not is_soc and mig_count == 0 else "",
            "-MIG_{:}x{:}".format(mig_count * gpu_count, machine.mig_short_name) if mig_count > 0 else "",
            "_TRT" if division == Division.CLOSED else "",
            "_MaxQ" if is_maxq else "",
            "_{:}".format(additional_config) if additional_config != "" else "",
        ])

    def _get_system_name(self, machine, division, gpu_count, mig_count, is_soc, is_maxq, additional_config):
        system_details = []
        if not is_soc:
            system_details.append("{:d}x {:}{:}".format(
                gpu_count,
                machine.accelerator_short_name,
                "-MIG-{:}x{:}".format(mig_count, machine.mig_short_name) if mig_count > 0 else ""
            ))
        if is_maxq:
            system_details.append("MaxQ")
        if division == Division.CLOSED:
            system_details.append("TensorRT")
        if additional_config != "":
            system_details.append(additional_config)
        return "{:} ({:})".format(
            machine.system_name_prefix,
            ", ".join(system_details)
        )

    def _get_accelerator_model_name(self, machine, division, gpu_count, mig_count, is_soc, is_maxq, additional_config):
        return "{:}{:}".format(
            machine.accelerator_model_name,
            " ({:d}x{:} MIG)".format(mig_count, machine.mig_short_name) if mig_count > 0 else "",
        )

    def _get_operating_system(self, machine, is_soc):
        os = "Unknown"
        if is_soc:
            os = get_soc_sw_version(machine.accelerator_model_name, "os")
        else:
            os = os_version
        return os

    def _get_framework(self, machine, division, gpu_count, mig_count, is_soc, is_maxq, additional_config):
        frameworks = []
        if division == Division.CLOSED:
            # Distinguish different TRT version based on the arch/model
            if is_soc:
                version = get_soc_sw_version(machine.accelerator_model_name, "trt")
            else:
                version = trt_version
            frameworks.append(version)
        frameworks.append(cuda_version if not is_soc else get_soc_sw_version(machine.accelerator_model_name, "cuda"))
        return ", ".join(frameworks)

    def _get_software_stack(self, machine, division, gpu_count, mig_count, is_soc, is_maxq, additional_config):
        frameworks = []
        if division == Division.CLOSED:
            # Distinguish different TRT version based on the arch/model
            if is_soc:
                version = get_soc_sw_version(machine.accelerator_model_name, "trt")
            else:
                version = trt_version
            frameworks.append(version)
        frameworks.append(cuda_version if not is_soc else get_soc_sw_version(machine.accelerator_model_name, "cuda"))
        if division == Division.CLOSED:
            frameworks.append(cudnn_version if not is_soc else get_soc_sw_version(machine.accelerator_model_name, "cudnn"))
        if not frameworks:
            raise NotImplementedError(f"{machine.accelerator_short_name} not an available submission systems!")
        return ", ".join(frameworks)

    def __getitem__(self, key):
        return self.attr[key]


submission_systems = [
    #                                                        #gpu   SOC, MaxQ
    System(THOR_X_MAXP, Division.CLOSED, SystemType.ADAS, 1, 0, True, False),  # Thor-X
]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tsv", "-o",
        help="Specifies the output tab-separated file for system descriptions.",
        default="systems/system_descriptions.tsv"
    )
    parser.add_argument(
        "--dry_run",
        help="Don't actually copy files, just log the actions taken.",
        action="store_true"
    )
    parser.add_argument(
        "--manual_system_json",
        help="Path to the system json that is manually to the system description table.",
        nargs='+',
        default=[]
    )
    return parser.parse_args()


def main():
    args = get_args()

    tsv_file = args.tsv

    summary = []
    for system in submission_systems:
        json_file = os.path.join("..", "..", system["division"], system["submitter"], "systems", "{:}.json".format(system["system_id"]))
        print("Generating {:}".format(json_file))
        summary.append("\t".join([str(i) for i in [
            system["system_name"],
            system["system_id"],
            system["submitter"],
            system["division"],
            system["system_type"],
            system["system_type_detail"],
            system["status"],
            system["number_of_nodes"],
            system["host_processor_model_name"],
            system["host_processors_per_node"],
            system["host_processor_core_count"],
            system["host_processor_frequency"],
            system["host_processor_caches"],
            system["host_processor_interconnect"],
            system["host_memory_configuration"],
            system["host_memory_capacity"],
            system["host_storage_capacity"],
            system["host_storage_type"],
            system["host_networking"],
            system["host_networking_topology"],
            system["accelerators_per_node"],
            system["accelerator_model_name"],
            system["accelerator_frequency"],
            system["accelerator_host_interconnect"],
            system["accelerator_interconnect"],
            system["accelerator_interconnect_topology"],
            system["accelerator_memory_capacity"],
            system["accelerator_memory_configuration"],
            system["accelerator_on-chip_memories"],
            system["cooling"],
            system["hw_notes"],
            system["framework"],
            system["operating_system"],
            system["other_software_stack"],
            system["sw_notes"],
            system["power_management"],
            system["filesystem"],
            system["boot_firmware_version"],
            system["management_firmware_version"],
            system["other_hardware"],
            system["number_of_type_nics_installed"],
            system["nics_enabled_firmware"],
            system["nics_enabled_os"],
            system["nics_enabled_connected"],
            system["network_speed_mbit"],
            system["power_supply_quantity_and_rating_watts"],
            system["power_supply_details"],
            system["disk_drives"],
            system["disk_controllers"],
        ]]))
        del system.attr["system_id"]
        if not args.dry_run:
            with open(json_file, "w") as f:
                json.dump(system.attr, f, indent=4, sort_keys=True)
        else:
            print(json.dumps(system.attr, indent=4, sort_keys=True))

    # Add the systems to the summary, reading from the json file that's manually written.
    for fpath in args.manual_system_json:
        with open(fpath, "r") as fh:
            print(f"Adding {fpath} manually to the system description table.")
            system = json.load(fh)
            # Read the system_id directly from the file name.
            system_id = fpath.split("/")[-1].split(".")[0]
            summary.append("\t".join([str(i) for i in [
                system["system_name"],
                system_id,
                system["submitter"],
                system["division"],
                system["system_type"],
                system["system_type_detail"],
                system["status"],
                system["number_of_nodes"],
                system["host_processor_model_name"],
                system["host_processors_per_node"],
                system["host_processor_core_count"],
                system["host_processor_frequency"],
                system["host_processor_caches"],
                system["host_processor_interconnect"],
                system["host_memory_configuration"],
                system["host_memory_capacity"],
                system["host_storage_capacity"],
                system["host_storage_type"],
                system["host_networking"],
                system["host_networking_topology"],
                system["accelerators_per_node"],
                system["accelerator_model_name"],
                system["accelerator_frequency"],
                system["accelerator_host_interconnect"],
                system["accelerator_interconnect"],
                system["accelerator_interconnect_topology"],
                system["accelerator_memory_capacity"],
                system["accelerator_memory_configuration"],
                system["accelerator_on-chip_memories"],
                system["cooling"],
                system["hw_notes"],
                system["framework"],
                system["operating_system"],
                system["other_software_stack"],
                system["sw_notes"],
                system["power_management"],
                system["filesystem"],
                system["boot_firmware_version"],
                system["management_firmware_version"],
                system["other_hardware"],
                system["number_of_type_nics_installed"],
                system["nics_enabled_firmware"],
                system["nics_enabled_os"],
                system["nics_enabled_connected"],
                system["network_speed_mbit"],
                system["power_supply_quantity_and_rating_watts"],
                system["power_supply_details"],
                system["disk_drives"],
                system["disk_controllers"],
            ]]))

    print("Generating system description summary to {:}".format(tsv_file))
    if not args.dry_run:
        with open(tsv_file, "w") as f:
            for item in summary:
                print(item, file=f)
    else:
        print("\n".join(summary))

    print("Done!")


if __name__ == '__main__':
    main()
