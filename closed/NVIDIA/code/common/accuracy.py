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
sys.path.append(os.getcwd())

import time
from code.common import logging
import numpy as np


class AccuracyRunner(object):
    def __init__(self, runner, val_map, image_dir, verbose=False):
        self.runner = runner
        self.val_map = val_map
        self.image_dir = image_dir
        self.verbose = verbose

        self.image_list = []
        self.class_list = []

    def reset(self):
        self.image_list = []
        self.class_list = []

    def load_val_images(self):
        self.reset()
        with open(self.val_map) as f:
            for line in f:
                self.image_list.append(line.split()[0])
                self.class_list.append(int(line.split()[1]))

    def run(self):
        raise NotImplementedError("AccuracyRunner.run() is not implemented")
