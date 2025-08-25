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


from git import Repo
from git.exc import InvalidGitRepositoryError, NoSuchPathError, GitCommandError
from os import PathLike
from pathlib import Path
from tqdm import tqdm
from typing import Any, Dict

import importlib
import importlib.util
import logging
import shutil
import sys

from .ops import *
from ...pipeline import Operation
from ...utils import GitPyTqdm, run_command


def is_git_repo(path: PathLike) -> bool:
    """Checks if a valid Git repository is located at `path`.

    Args:
        path (PathLike): The path to check

    Returns:
        bool: True if the path is the root directory of a Git repository, False otherwise.
    """
    try:
        r = Repo(path)
        return True
    except (InvalidGitRepositoryError, NoSuchPathError):
        return False


class MLCommonsAutomotiveRepoOp(Operation):
    """Operation to clone the MLCommons Automotive repository into the scratch space.
    """

    def __init__(self,
                 repo_path: PathLike = "automotive",
                 repo_url: str = "https://github.com/mlcommons/mlperf_automotive.git",
                 repo_commit: str = "master"):
        """Creates a MLCommonsAutomotiveRepoOp.

        Args:
            repo_path (PathLike): Relative path in the scratch space to clone the MLCommons Inference repo to. (Default:
                                  inference)
            repo_url (str): The URL of the MLCommons Automotive repo. Useful to set if you have a fork you wish to use.
        """
        if not isinstance(repo_path, Path):
            self.repo_path = Path(repo_path)
        else:
            self.repo_path = repo_path
        self.repo_url = repo_url
        self.repo_commit = repo_commit

    @classmethod
    def immediate_dependencies(cls):
        return None

    def run(self, scratch_space, dependency_outputs):
        repo_abs_path = scratch_space.path / self.repo_path
        needs_clone = False
        if not is_git_repo(repo_abs_path):
            needs_clone = True
        else:
            repo = Repo(repo_abs_path)
            # Check that the remote we want to contains the specified remote, and set it to if it does.
            remote_exists = False
            for remote in repo.remotes:
                if remote.url == self.repo_url:
                    remote_exists = True
                    break

            # If the remote doesn't exist, or if it does but fails to update, wipe the repo and reclone.
            wipe_repo = True
            if remote_exists:
                repo.remotes.origin.set_url(self.repo_url)
                try:
                    for _pinfo in tqdm(repo.remotes.origin.pull(), desc="Pulling repository"):
                        pass
                    wipe_repo = False
                except GitCommandError:
                    logging.info(f"Failed to fetch origin from {self.repo_url} and/or apply changes to existing repo.")
            else:
                logging.info(f"Existing repo does not have requested remote {self.repo_url}")

            if wipe_repo:
                logging.info(f"Deleting existing repo")
                shutil.rmtree(repo_abs_path)
                needs_clone = True

        if needs_clone:
            repo = Repo.clone_from(self.repo_url,
                                   repo_abs_path,
                                   progress=GitPyTqdm(desc=f"git clone {self.repo_url}",
                                                      unit="B",
                                                      unit_scale=True))

        # Check out the hash
        logging.info(f"Checking out commit: {self.repo_commit}")
        repo.git.checkout(self.repo_commit)

        # Update all submodules
        for submodule in repo.submodules:
            submodule.update(recursive=True)

        return {"repo_root": repo_abs_path,
                "repo_hash": self.repo_commit}


class InstallLoadgenOp(Operation):
    """This Op will install Loadgen if it isn't already, and return the loadgen module as output.
    """

    @classmethod
    def immediate_dependencies(cls):
        return {MLCommonsAutomotiveRepoOp}

    def run(self, scratch_space, dependency_outputs):
        if not importlib.util.find_spec("mlperf_loadgen"):
            # Build and install loadgen
            mlcinf_reporoot = dependency_outputs[MLCommonsAutomotiveRepoOp]["repo_root"]
            loadgen_dir = mlcinf_reporoot / "loadgen"

            run_command(f"cd {loadgen_dir} && {sys.executable} setup.py install",
                        get_output=False,
                        tee=False,
                        verbose=False,
                        custom_env={"CFLAGS": "-std=c++14 -O3"})
        return {"loadgen": importlib.import_module("mlperf_loadgen")}


class LoadgenBenchmarkOp(LoadgenBenchmark):
    """Runs a Loadgen Test via the Python API
    """
    def __init__(self,
                 log_settings: Dict[str, Any] = None,
                 output_settings: Dict[str, Any] = None,
                 mlperf_conf_path: PathLike = None,
                 user_conf_path: PathLike = None,
                 audit_conf_path: PathLike = None,
                 test_setting_overrides: Dict[str, Any] = None,
                 performance_sample_count: int = -1):
        """Creates a LoadgenBenchmarkOp, which launches a benchmark using MLCommons Loadgen using the Python API.

        Args:
            log_settings (Dict[str, Any]): Settings for loadgen logging. Equivalent to
                                           loadgen::test_settings::LogSettings.
            output_settings (Dict[str, Any]): Settings for loadgen output. Equivalent to
                                              loadgen::test_settings::LogOutputSettings.
            mlperf_conf_path (PathLike): Path to mlperf.conf.
            user_conf_path (PathLike): Path to user.conf.
            audit_conf_path (PathLike): Path to audit.conf.
            test_setting_overrides (Dict[str, Any]): Overrides settings from mlperf.conf and user.conf.
        """
        if not log_settings:
            log_settings = dict()
        self.log_settings = log_settings

        if not output_settings:
            output_settings = dict()
        self.output_settings = output_settings

        self.mlperf_conf_path = Path(mlperf_conf_path)
        self.user_conf_path = Path(user_conf_path)
        self.audit_conf_path = Path(audit_conf_path)
        if not test_setting_overrides:
            test_setting_overrides = dict()
        self.test_setting_overrides = test_setting_overrides
        self.performance_sample_count = performance_sample_count

    @classmethod
    def immediate_dependencies(cls):
        return {InstallLoadgenOp, LoadgenWorkload}

    def run(self, scratch_space, dependency_outputs):
        lg = dependency_outputs[InstallLoadgenOp]["loadgen"]
        workload = dependency_outputs[LoadgenWorkload]

        # Set up test settings
        output_settings = lg.LogOutputSettings()
        for k, v in self.output_settings.items():
            setattr(output_settings, k, v)

        log_settings = lg.LogSettings()
        for k, v in self.log_settings.items():
            setattr(log_settings, k, v)
        log_settings.log_output = output_settings

        test_settings = lg.TestSettings()
        test_settings.FromConfig(str(self.mlperf_conf_path),
                                 workload["benchmark"],
                                 workload["scenario"])
        test_settings.FromConfig(str(self.user_conf_path),
                                 workload["benchmark"],
                                 workload["scenario"])
        test_settings.scenario = workload["scenario"]
        test_settings.mode = workload["mode"]
        for k, v in self.test_setting_overrides.items():
            setattr(test_settings, k, v)

        # Set up test
        sut_wrapper = workload["sut"]
        sut = lg.ConstructSUT(sut_wrapper.issue_queries, sut_wrapper.flush_queries)

        qsl_wrapper = workload["qsl"]
        psc = self.performance_sample_count if self.performance_sample_count > 0 else qsl_wrapper.count
        qsl = lg.ConstructQSL(qsl_wrapper.count,
                              psc,
                              qsl_wrapper.load_query_samples,
                              qsl_wrapper.unload_query_samples)

        # Do test
        logging.info("Starting test")
        sut_wrapper.start()
        lg.StartTestWithLogSettings(sut,
                                    qsl,
                                    test_settings,
                                    log_settings,
                                    str(self.audit_conf_path))

        # Cleanup
        results = sut_wrapper.stop()
        lg.DestroyQSL(qsl)
        lg.DestroySUT(sut)
        return True
