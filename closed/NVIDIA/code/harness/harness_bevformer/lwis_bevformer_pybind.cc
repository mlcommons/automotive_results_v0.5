/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* Include necessary header files */

// Pybind11
#include <pybind11/chrono.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

/* Include necessary header files */
// Loadgen
#include "loadgen.h"

// TensorRT
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "logger.h"
#include "logging.h"

// LWIS, for BEVFormer
#include "lwis_bevformer.hpp"

// Google Logging
#include <glog/logging.h>

// General C++
#include <chrono>
#include <dlfcn.h>
#include <iostream>
#include <memory>
#include <sys/stat.h>
#include <thread>

#include "callback.hpp"
#include "utils.hpp"

#include "cuda_profiler_api.h"

namespace py = pybind11;

// add error check
std::function<void(::mlperf::QuerySampleResponse*, std::vector<::mlperf::QuerySampleIndex>&, size_t)> getCallbackMap(
    std::string const& name)
{
    if (callbackMap.find(name) != callbackMap.end())
    {
        return callbackMap[name];
    }
    throw std::invalid_argument("Callback not found");
}

/* check if file exists; cannot use std::filesystem due to Xavier NX. Keeping as legacy behavior. */
inline bool doesFileExist(std::string const& name)
{
    struct stat buffer;
    return (stat(name.c_str(), &buffer) == 0);
}

std::shared_ptr<qsl::SampleLibraryBEVFormer> createQsl(std::shared_ptr<lwis::Server> sut,
                                                       std::string& mapPath,
                                                       std::vector<std::string>& tensorPaths, 
                                                       size_t perfSampleCount,
                                                       size_t padding)
{
    std::shared_ptr<qsl::SampleLibraryBEVFormer> qsl;

    // Release GIL while doing multi-thread computation
    {
        py::gil_scoped_release release;

        auto qsl = std::make_shared<qsl::SampleLibraryBEVFormer>(
            "LWIS_SampleLibrary", mapPath, tensorPaths, perfSampleCount, padding);
        sut->AddSampleLibrary(qsl);
    }
    
    return qsl;
}

void logDeviceStats(std::shared_ptr<lwis::Server> sut)
{
    // Log device stats
    auto devices = sut->GetDevices();
    for (auto& device : devices)
    {
        auto const& stats = device->GetStats();

        std::cout << "Device " << device->GetName() << " processed:" << std::endl;
        std::cout << "  Memcpy Calls: " << stats.m_MemcpyCalls << std::endl;
        std::cout << "  PerSampleCudaMemcpy Calls: " << stats.m_PerSampleCudaMemcpyCalls << std::endl;
    }
}

template <typename SampleLibraryType>
void startTest(std::shared_ptr<lwis::Server> server, 
               std::shared_ptr<SampleLibraryType> sampleLibrary,
               mlperf::TestSettings const& testSettings, 
               mlperf::LogSettings const& logSettings)
{
    py::gil_scoped_release release;
    mlperf::StartTest(server.get(), sampleLibrary.get(), testSettings, logSettings);
}

namespace lwis
{
PYBIND11_MODULE(lwis_bevformer_api, m)
{
    m.doc() = "MLPerf-Inference Python bindings for LWIS BEVFormer harness";

    m.def("get_callback_map", &getCallbackMap);
    m.def("init_glog", &initGlog);
    m.def("log_device_stats", &logDeviceStats);

    m.def("start_test", &startTest<qsl::SampleLibraryBEVFormer>);

    m.def("reset", [](std::shared_ptr<Server> sut) { sut.reset(); });
    m.def("reset", [](std::shared_ptr<qsl::SampleLibraryBEVFormer> qsl) { qsl.reset(); });

    py::class_<Server, std::shared_ptr<Server>>(m, "Server", py::module_local())
        .def(py::init<std::string>())
        .def("add_sample_library", &Server::AddSampleLibrary)
        .def("setup", &Server::Setup)
        .def("warmup", &Server::Warmup)
        .def("done", &Server::Done)
        .def("name", &Server::Name)
        .def("issue_query", &Server::IssueQuery)
        .def("flush_queries", &Server::FlushQueries)
        .def("set_response_callback", &Server::SetResponseCallback);

    py::class_<ServerParams>(m, "ServerParams", py::module_local())
        .def(py::init<>())
        .def_readwrite("device_names", &ServerParams::DeviceNames)
        .def_readwrite("engine_name", &ServerParams::EngineName);

    py::class_<ServerSettings_BEVFormer>(m, "ServerSettings_BEVFormer")
        .def(py::init<>())
        .def_readwrite("enable_sync_on_event", &ServerSettings_BEVFormer::EnableSyncOnEvent)
        .def_readwrite("enable_spin_wait", &ServerSettings_BEVFormer::EnableSpinWait)
        .def_readwrite("enable_device_schedule_spin", &ServerSettings_BEVFormer::EnableDeviceScheduleSpin)
        .def_readwrite("enable_response", &ServerSettings_BEVFormer::EnableResponse)
        .def_readwrite("enable_deque_limit", &ServerSettings_BEVFormer::EnableDequeLimit)
        .def_readwrite("enable_batcher_thread_per_device", &ServerSettings_BEVFormer::EnableBatcherThreadPerDevice)
        .def_readwrite("enable_cuda_thread_per_device", &ServerSettings_BEVFormer::EnableCudaThreadPerDevice)
        .def_readwrite("run_infer_on_copy_streams", &ServerSettings_BEVFormer::RunInferOnCopyStreams)
        .def_readwrite("use_same_context", &ServerSettings_BEVFormer::UseSameContext)
        .def_readwrite("verbose_nvtx", &ServerSettings_BEVFormer::VerboseNVTX)
        .def_readwrite("gpu_batch_size", &ServerSettings_BEVFormer::GPUBatchSize)
        .def_readwrite("gpu_copy_streams", &ServerSettings_BEVFormer::GPUCopyStreams)
        .def_readwrite("gpu_infer_streams", &ServerSettings_BEVFormer::GPUInferStreams)
        .def_readwrite("max_gpus", &ServerSettings_BEVFormer::MaxGPUs)
        .def_readwrite("complete_threads", &ServerSettings_BEVFormer::CompleteThreads)
        .def_readwrite("timeout", &ServerSettings_BEVFormer::Timeout);

    py::class_<qsl::SampleLibraryBEVFormer, std::shared_ptr<qsl::SampleLibraryBEVFormer>>(m, "SampleLibraryBEVFormer")
        .def(py::init<std::string, std::string, std::vector<std::string>, size_t, size_t>(),
            py::arg("name"), py::arg("map_path"), py::arg("tensor_paths"), py::arg("perf_sample_count"),
            py::arg("padding") = 0)
        .def("name", &qsl::SampleLibraryBEVFormer::Name)
        .def("total_sample_count", &qsl::SampleLibraryBEVFormer::TotalSampleCount)
        .def("performance_sample_count", &qsl::SampleLibraryBEVFormer::PerformanceSampleCount)
        .def("load_samples_to_ram", &qsl::SampleLibraryBEVFormer::LoadSamplesToRam)
        .def("unload_samples_from_ram", &qsl::SampleLibraryBEVFormer::UnloadSamplesFromRam);
} // pybind11
} // namespace lwis
