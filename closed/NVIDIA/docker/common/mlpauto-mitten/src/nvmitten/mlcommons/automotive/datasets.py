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

import sys

from .repo import MLCommonsAutomotiveRepoOp
from ...pipeline import Operation
from ...pipeline.resource import Resource, RemoteObjectType
from ...utils import run_command


def github_raw_url(repo_name: str, commit_hash: str, git_obj_path: str):
    """Returns the URL used for a GitHub raw content object given the repo data and the path of the object in the repo.

    Args:
        repo_name (str): The name of the repo, usually <user or org name>/<project name>.
        commit_hash (str): The full commit hash to check the file out from. Branch names can be used as well.
        git_obj_path (str): The relative path of the file within the git repo.
    """
    return f"https://raw.githubusercontent.com/{repo_name}/{commit_hash}/{git_obj_path}"

class NuScenesDatasetOp(Operation):
    """nuScenes dataset is a licensed dataset, and is restricted behind a Terms of Use Agreement. 
    As such, Mitten, and other libraries and MLPerf submission codebases, cannot distribute the URLs for
    download directly.

    Please check the following link for more details: http://nuscenes.mlcommons.org/ 

    IMPORTANT: Because of the above, make sure the config file storing the URLs is not distributed with the code, and is
    privated to avoid licensing and legal issues.
    """

    def __init__(self, nuscenes_url: str = "", force_generate: bool = False):
        """Creates a NuScenesDatasetOp.

        Args:
            nuscenes_url (str): The URL for the nuScenes dataset. After agreeing to the EULA, you can download the dataset
                                   from the following link: hhttps://drive.google.com/drive/folders/17CpM5eU8tjrxh_LpH_BTNTeT37PhzcnC.
                                   This is Google Drive link, so you need to use rclone with authentication to download the dataset. 
                                   (Default: "")
            force_generate (bool): If True, forces all resources to be regenerated. Existing resources will be deleted,
                                   even if they exist and are verified. (Default: False)
        """
        if not nuscenes_url:
            raise ValueError("nuscenes_url is not set. Did you read the instructions and accept the Terms "
                             "of Access?")

        self.nuscenes_url = nuscenes_url
        self.force_generate = force_generate

    @classmethod
    def immediate_dependencies(cls):
        return None

    def run(self, scratch_space, dependency_outputs):
        valset = Resource("build/data/nuScenes/nuscenes/samples.tar.gz",
                          scratch_space,
                          resource_remote_url=self.nuscenes_url,
                          resource_format="gztar",
                          resource_extract_dir="build/data/nuScenes/nuscenes")
        valset.generate(force=self.force_generate)
        return True

