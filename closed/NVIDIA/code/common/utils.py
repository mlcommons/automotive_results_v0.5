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

from functools import wraps
import heapq
import importlib
import importlib.util
from math import gcd
import os
import random
import re
from typing import Any, Callable, Dict, List, Optional, Union
import shutil
from contextlib import contextmanager
import venv
import platform
import sys

from code.common import logging, run_command
from code.common.constants import Benchmark

import nvtx

# Conditional import. Sometimes, we may use the constants files outside of an environment that has numpy installed, for
# example certain scripts in CI/CD. Provide an environment variable 'OUTSIDE_MLPINF_ENV' to allow using constants.py
# outside of the docker.

if importlib.util.find_spec("numpy") is not None or os.environ.get("OUTSIDE_MLPINF_ENV", "0") == "0":
    import numpy as np

    def get_dyn_ranges(cache_file: str) -> Dict[str, np.uint32]:
        """
        Get dynamic ranges from calibration file for network tensors.

        Args:
            cache_file (str):
                Path to INT8 calibration cache file.

        Returns:
            Dict[str, np.uint32]: Dictionary of tensor name -> dynamic range of tensor
        """
        dyn_ranges = {}
        if not os.path.exists(cache_file):
            raise FileNotFoundError("{} calibration file is not found.".format(cache_file))

        with open(cache_file, "rb") as f:
            lines = f.read().decode('ascii').splitlines()
        for line in lines:
            regex = r"(.+): (\w+)"
            results = re.findall(regex, line)
            # Omit unmatched lines
            if len(results) == 0 or len(results[0]) != 2:
                continue
            results = results[0]
            tensor_name = results[0]
            # Map dynamic range from [0.0 - 1.0] to [0.0 - 127.0]
            dynamic_range = np.uint32(int(results[1], base=16)).view(np.dtype('float32')).item() * 127.0
            dyn_ranges[tensor_name] = dynamic_range
        return dyn_ranges


def get_e2e_batch_size(batch_size_dict: Dict[str, int]):
    if not batch_size_dict:
        raise ValueError(f"{batch_size_dict} input batch_size_dict is empty.")

    batch_size_list = list(batch_size_dict.values())
    lcm_value = 1
    for val in batch_size_list:
        lcm_value = (lcm_value * val) // gcd(lcm_value, val)

    if lcm_value != max(batch_size_list):
        raise ValueError(f"End-to-end batch size {lcm_value} is not equal to any of the max engine batch size in {batch_size_dict}")

    return lcm_value


def check_eq(val1, val2, error_message="Values are not equal"):
    if val1 != val2:
        raise AssertionError(f"{error_message} ({val1} vs. {val2})")


class ScopeWrap:
    """
    Wrap scope with given enter & exit function calls.
    """

    def __init__(self,
                 enter_fn: Optional[Callable[[], None]],
                 exit_fn: Optional[Callable[[], None]]):
        self.enter_fn = enter_fn
        self.exit_fn = exit_fn

    def __enter__(self):
        if self.enter_fn:
            self.enter_fn()

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.exit_fn:
            self.exit_fn()


def add_nvtx_scope_wrap(prefix_attr: str = None, enable_attr: str = "verbose_nvtx"):
    """
    A decorator to add NVTX scope wrapping to a class.
    Wrapped class will be injected with self.nvtx_scope(scope_name: str) utility,
    which depending on value of `enable_attr` will record nvtx markers for scope.

    Args:
        prefix_attr (str): The attribute name to use for the prefix. If None, the class name will be used.
        enable_attr (str): The attribute name to determine if NVTX profiling is enabled.

    Returns:
        type: The decorated class.
    """
    class NVTXScopeWrap(ScopeWrap):
        """
        Wrap scope with nvtx markers.
        """

        def __init__(
            self,
            name: str,
            color: str = None,
        ):
            self.name = name
            self.color = color or random.choice(['green', 'blue', 'yellow', 'pink'])
            self.marker = None

            def enter_fn():
                self.marker = nvtx.start_range(message=self.name, color=self.color)

            def exit_fn():
                nvtx.end_range(self.marker)

            super().__init__(enter_fn=enter_fn, exit_fn=exit_fn)

    def decorator(cls: type):
        original_init = cls.__init__

        @wraps(original_init)
        def __init__(self, *args, **kwargs):
            prefix = self.__class__.__name__ if prefix_attr is None else getattr(self, prefix_attr, kwargs.get(prefix_attr, None))
            assert prefix is not None, f"{cls.__name__} must have non-None (str) attribute / __init__ param: {prefix_attr}"

            enabled = getattr(self, enable_attr, kwargs.get(enable_attr, None))
            assert enabled is not None, f"{cls.__name__} must have non-None (bool) attribute / __init__ param: {enable_attr}"

            self.nvtx_scope = lambda name: NVTXScopeWrap(f'{prefix}-{name}') if enabled else ScopeWrap(enter_fn=None, exit_fn=None)

            original_init(self, *args, **kwargs)

            if enabled:
                getattr(self, 'logger', logging).info("Enabled NVTX Scopes.")

        cls.__init__ = __init__
        return cls

    return decorator


class Tree:
    """
    Datastructure that allows storing and retrieving values in a generic tree structure, where a node can have a
    variable number of children.
    """

    def __init__(self, starting_val: Optional[Dict[str, Any]] = None):
        if starting_val is None:
            self.tree: Dict[str, Any] = dict()
        else:
            self.tree: Dict[str, Any] = starting_val

    def insert(self, keyspace: List[str], value: Any, append: bool = False):
        """
        Inserts a value into the tree. The keyspace represents the tree traversal or walk starting from the root of the
        tree to get to the leaf. `value` is the object stored at the leaf.

        If `append` is True, then the leaf is treated as a list, and `value` is appended to it instead of overwriting
        it.

        Args:
            keyspace (List[str]):
                The tree traversal to get to the node to insert at.
            value (Any):
                The value for the inserted leaf
            append (bool):
                Default: False. If True, the leaf is treated as a list, and `value` is appended to it instead of
                overwriting.
        """
        # pop(0) is O(k), but pop(-1) is O(1). Reverse keyspace
        keyspace = list(keyspace[::-1])

        curr = self.tree
        while len(keyspace) > 0:
            if len(keyspace) == 1:
                if append:
                    if keyspace[-1] not in curr:
                        curr[keyspace[-1]] = [value]
                    else:
                        if type(curr[keyspace[-1]]) is list:
                            curr[keyspace[-1]].append(value)
                        else:
                            curr[keyspace[-1]] = [curr[keyspace[-1]], value]
                else:
                    curr[keyspace[-1]] = value
            else:
                if keyspace[-1] not in curr:
                    curr[keyspace[-1]] = dict()

                curr = curr[keyspace[-1]]
            keyspace.pop(-1)

    def get(self, keyspace: List[str], default=None) -> Any:
        """
        Gets the value of a node in the tree. The keyspace represents the tree traversal or walk starting from the root
        to get to that node.

        Returns object stored at the leaf, if it exists, otherwise returns `default`.

        Args:
            keyspace (List[str]):
                The tree traversal to get to the node to insert at.

            default (Any):
                The value to return if the keyspace does not exist.

        Returns:
            Any: The value at the keyspace.
        """
        # pop(0) is O(k), but pop(-1) is O(1). Reverse keyspace
        keyspace = list(keyspace[::-1])

        curr = self.tree
        while len(keyspace) > 0:
            if keyspace[-1] not in curr:
                return default

            if len(keyspace) == 1:
                return curr[keyspace[-1]]
            else:
                curr = curr[keyspace[-1]]
                keyspace.pop(-1)

    def __getitem__(self, keyspace_str):
        return self.get(keyspace_str.split(","))

    def __setitem__(self, keyspace_str, value):
        self.insert(keyspace_str.split(","), value)

    def __iter__(self):
        return (k for k in self.tree)

    def __len__(self) -> int:
        """
        Number of leaves of the tree.

        Returns:
            int: The number of leaves of the tree. If a leaf is a list or tuple of length N, then that leaf is counted N
            times (representing N different leaves).
        """
        def _count_dict(d):
            _count = 0
            for k, v in d.items():
                if isinstance(v, dict):
                    _count += _count_dict(v)
                elif any([isinstance(v, t) for t in (list, tuple)]):
                    _count += len(v)
                else:
                    _count += 1
            return _count
        return _count_dict(self.tree)


class FastPercentileHeap:
    """
    A datastructure to maintain a running percentile of a stream of numbers using two heaps.

    Advantages:
        - fast calculation of current top self.k percentile
        - fast insertion of new values to track

    Attributes:
        percentile (float): The desired percentile (default is 99).
        small (list): Max heap to store the smaller half of the numbers (stored as negative values).
        large (list): Min heap to store the larger half of the numbers.
        n (int): The total number of elements added.
        k (float): The desired percentile as a fraction.
    """

    def __init__(self, percentile=99):
        """
        Initializes the FastPercentileHeap with the given percentile.

        Args:
            percentile (float): The desired percentile (default is 99).
        """
        self.small = []  # max heap (-ve values)
        self.large = []  # min heap

        self.n = 0
        self.k = 1.0 - (percentile / 100.0)

    def extend(self, nums: List[int]):
        """
        Extends the heap with a list of numbers.

        Args:
            nums (List[int]): The list of numbers to be added to the heap.
        """
        for num in nums:
            self.append(num)

    def append(self, num):
        """
        Appends a number to the heap in O(logN) time.

        Args:
            num (float): The number to be added to the heap.
        """
        self.n += 1
        target_large_size = max(1, int(self.n * self.k))

        if len(self.large) < target_large_size:
            heapq.heappush(self.large, num)
        else:
            heapq.heappush(self.small, -num)

        self._rebalance()

    def _rebalance(self):
        """
        Rebalances the heaps in O(1) time when the largest element in self.small is larger than the smallest element in self.large.
        """
        if not self.large or not self.small:
            return

        if -self.small[0] > self.large[0]:
            small_val = -heapq.heappop(self.small)
            large_val = heapq.heappop(self.large)
            heapq.heappush(self.small, -large_val)
            heapq.heappush(self.large, small_val)

    def p(self):
        """
        Calculates the top self.k percentile in O(1) time.

        Returns:
            float: The top self.k percentile value, or None if the heap is empty.
        """
        return self.large[0] if self.large else None


def parse_cli_flags(flags: Optional[Union[str, dict]] = None) -> Dict[str, Any]:
    """
    Parses a string or dictionary into a dictionary with string keys and values of appropriate types.

    Args:
        flags (Optional[Union[str, dict]]): A string in the format 'key1:value1,key2:value2,...' or a dictionary.

    Returns:
        Dict[str, Any]: A dictionary with string keys and values of appropriate types (bool, int, float, or str).
    """
    if flags is None or flags == "":
        return {}

    if type(flags) is dict:
        return flags

    flag_dict = {}
    for item in flags.split(','):
        key, value = item.split(':')
        if value.lower() in ['true', 'false']:
            value = value.lower() == 'true'
        else:
            try:
                if '.' in value:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                pass
        flag_dict[key] = value
    return flag_dict

