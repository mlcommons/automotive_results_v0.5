import pytest
from pathlib import Path

from nvmitten.system.memory import *


def _load_asset(system_id):
    with Path(f"tests/assets/system_detect_spoofs/mem/sample-system-{system_id}").open() as f:
        contents = f.read()
    return contents


@pytest.fixture
def sample_systems():
    # Preload contents before pyfakefs takes over the filesystem.
    return {1: (_load_asset(1), 990640728000),
            2: (_load_asset(2), 32616612000),
            3: (_load_asset(3), 990640724000),
            4: (_load_asset(4), 528216216000)}


def test_hostmemory_detect_1(sample_systems, fs):
    contents, nbytes = sample_systems[1]

    fs.create_file("/proc/meminfo", contents=contents)
    HostMemory.detect.cache_clear()
    mem = HostMemory.detect()[0]
    assert mem.capacity._num_bytes == nbytes


def test_hostmemory_detect_2(sample_systems, fs):
    contents, nbytes = sample_systems[2]

    fs.create_file("/proc/meminfo", contents=contents)
    HostMemory.detect.cache_clear()
    mem = HostMemory.detect()[0]
    assert mem.capacity._num_bytes == nbytes


def test_hostmemory_detect_3(sample_systems, fs):
    contents, nbytes = sample_systems[3]

    fs.create_file("/proc/meminfo", contents=contents)
    HostMemory.detect.cache_clear()
    mem = HostMemory.detect()[0]
    assert mem.capacity._num_bytes == nbytes


def test_hostmemory_detect_4(sample_systems, fs):
    contents, nbytes = sample_systems[4]

    fs.create_file("/proc/meminfo", contents=contents)
    HostMemory.detect.cache_clear()
    mem = HostMemory.detect()[0]
    assert mem.capacity._num_bytes == nbytes
