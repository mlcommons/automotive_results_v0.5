import pytest
from pathlib import Path
from unittest.mock import patch

from nvmitten.constants import CPUArchitecture
from nvmitten.interval import Interval
from nvmitten.system.cpu import *


@patch("nvmitten.system.cpu.run_command")
def test_cpu_detect(mock_run):
    with Path("tests/assets/system_detect_spoofs/cpu/sample-system-1").open() as f:
        contents = f.read().split("\n")

    mock_run.return_value = contents

    cpu = CPU.detect()[0]
    mock_run.assert_called_once_with("lscpu -J", get_output=True, tee=False, verbose=False)
    assert cpu.name == "fake_cpu"
    assert cpu.architecture == CPUArchitecture.x86_64
    assert cpu.vendor == "DefinitelyRealVendor"
    assert cpu.cores_per_group == 64
    assert cpu.threads_per_core == 2
    assert cpu.n_groups == 2
    assert cpu.group_type == GroupType.Socket
    assert len(cpu.numa_nodes) == 8
    for i in range(8):
        assert cpu.numa_nodes[i] == [Interval(i * 16, (i + 1) * 16 - 1),
                                     Interval(i * 16 + 128, (i + 1) * 16 + 127)]
