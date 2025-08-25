/*
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 * Copyright 2018 Google LLC
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

/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

#include <gflags/gflags.h>
#include <map>

// LWIS settings
DEFINE_string(engine_file, "", "Path to the TensorRT engine file");
DEFINE_string(plugins, "", "Comma-separated list of shared objects for plugins");
DEFINE_string(devices, "all", "Enable comma separated numbered devices");
DEFINE_bool(verbose, false, "Use verbose logging");
DEFINE_bool(verbose_nvtx, false, "Turn ProfilingVerbosity to kDETAILED so that layer detail is printed in NVTX.");
DEFINE_bool(use_spin_wait, true,
    "Actively wait for work completion.  This option may decrease multi-process "
    "synchronization time at the cost of additional CPU usage");
DEFINE_bool(use_device_schedule_spin, true,
    "Actively wait for results from the device.  May reduce latency at the the cost of "
    "less efficient CPU parallelization");
DEFINE_string(map_path, "",
    "Path to file containing map of sample ids to file names.");
DEFINE_string(tensor_path, "",
    "Path to dir containing preprocessed samples in npy format (*.npy). Comma-separated "
    "list if there are more than one input.");
DEFINE_bool(coalesced_tensor, false, "Turn on if all the samples are coalesced into one single npy file");
DEFINE_bool(use_graphs, false,
    "Enable cudaGraphs for TensorRT engines");
DEFINE_bool(use_deque_limit, false, "Enable a max number of elements dequed from work queue");
DEFINE_uint64(deque_timeout_usec, 10000, "Timeout for deque from work queue");
DEFINE_bool(use_batcher_thread_per_device, true, "Enable a separate batcher thread per device");
DEFINE_bool(use_cuda_thread_per_device, true, "Enable a separate cuda thread per device");
DEFINE_bool(use_same_context, true,
    "Use the same TRT context for all copy streams (shape must be static and "
    "gpu_inference_streams must be 1).");

DEFINE_uint32(gpu_copy_streams, 1, "Number of copy streams for inference");
DEFINE_uint32(gpu_inference_streams, 1, "Number of streams for inference");
DEFINE_uint32(gpu_batch_size, 1, "Max Batch size to use for all devices and engines");

DEFINE_bool(run_infer_on_copy_streams, false, "Runs inference on copy streams");
DEFINE_uint32(complete_threads, 1, "Number of threads per device for sending responses");
DEFINE_double(warmup_duration, 5.0, "Minimum duration to run warmup for");
DEFINE_string(response_postprocess, "", "Enable post-processing on query sample responses.");
DEFINE_uint64(performance_sample_count, 0, "Number of samples to load in performance set.  0=use default");

// Loadgen test settings
DEFINE_string(scenario, "SingleStream", "Scenario to run for Loadgen (SingleStream, ConstantStream)");
DEFINE_string(test_mode, "PerformanceOnly", "Testing mode for Loadgen");
DEFINE_string(model, "bevformer", "Model name");

// configuration files
DEFINE_string(mlperf_conf_path, "", "Path to mlperf.conf");
DEFINE_string(user_conf_path, "", "Path to user.conf");

// Loadgen logging settings
DEFINE_string(logfile_outdir, "", "Specify the existing output directory for the LoadGen logs");
DEFINE_string(logfile_prefix, "", "Specify the filename prefix for the LoadGen log files");
DEFINE_string(logfile_suffix, "", "Specify the filename suffix for the LoadGen log files");
DEFINE_bool(logfile_prefix_with_datetime, false, "Prefix filenames for LoadGen log files");
DEFINE_bool(log_copy_detail_to_stdout, false, "Copy LoadGen detailed logging to stdout");
DEFINE_bool(disable_log_copy_summary_to_stdout, false, "Disable copy LoadGen summary logging to stdout");
DEFINE_string(log_mode, "AsyncPoll", "Logging mode for Loadgen");
DEFINE_uint64(log_mode_async_poll_interval_ms, 1000, "Specify the poll interval for asynchrounous logging");
DEFINE_bool(log_enable_trace, false, "Enable trace logging");

/* Define a map to convert test mode input string into its corresponding enum value */
std::map<std::string, mlperf::TestMode> testModeMap = {{"SubmissionRun", mlperf::TestMode::SubmissionRun},
    {"AccuracyOnly", mlperf::TestMode::AccuracyOnly}, {"PerformanceOnly", mlperf::TestMode::PerformanceOnly},
    {"FindPeakPerformance", mlperf::TestMode::FindPeakPerformance}};

/* Define a map to convert logging mode input string into its corresponding enum value */
std::map<std::string, mlperf::LoggingMode> logModeMap = {{"AsyncPoll", mlperf::LoggingMode::AsyncPoll},
    {"EndOfTestOnly", mlperf::LoggingMode::EndOfTestOnly}, {"Synchronous", mlperf::LoggingMode::Synchronous}};

/* Define a map to convert test mode input string into its corresponding enum value */
std::map<std::string, mlperf::TestScenario> scenarioMap = {{"SingleStream", mlperf::TestScenario::SingleStream}, 
{"ConstantStream", mlperf::TestScenario::ConstantStream}};

/* Keep track of the GPU devices we are using */
std::vector<uint32_t> Devices;
std::vector<std::string> DeviceNames;

/* Helper function to actually perform inference using MLPerf Loadgen */
bool doInference()
{
    // Configure test settings
    mlperf::TestSettings test_settings;
    test_settings.scenario = scenarioMap[FLAGS_scenario];
    test_settings.mode = testModeMap[FLAGS_test_mode];

    gLogInfo << "mlperf.conf path: " << FLAGS_mlperf_conf_path << std::endl;
    gLogInfo << "user.conf path: " << FLAGS_user_conf_path << std::endl;
    test_settings.FromConfig(FLAGS_mlperf_conf_path, FLAGS_model, FLAGS_scenario, 2);
    test_settings.FromConfig(FLAGS_user_conf_path, FLAGS_model, FLAGS_scenario, 1);

    // Configure logging settings
    mlperf::LogSettings log_settings;
    log_settings.log_output.outdir = FLAGS_logfile_outdir;
    log_settings.log_output.prefix = FLAGS_logfile_prefix;
    log_settings.log_output.suffix = FLAGS_logfile_suffix;
    log_settings.log_output.prefix_with_datetime = FLAGS_logfile_prefix_with_datetime;
    log_settings.log_output.copy_detail_to_stdout = FLAGS_log_copy_detail_to_stdout;
    log_settings.log_output.copy_summary_to_stdout = !FLAGS_disable_log_copy_summary_to_stdout;
    log_settings.log_mode = logModeMap[FLAGS_log_mode];
    log_settings.log_mode_async_poll_interval_ms = FLAGS_log_mode_async_poll_interval_ms;
    log_settings.enable_trace = FLAGS_log_enable_trace;

    // Configure server settings
    lwis::ServerSettings_BEVFormer sut_settings;
    sut_settings.GPUBatchSize = FLAGS_gpu_batch_size;
    sut_settings.GPUCopyStreams = FLAGS_gpu_copy_streams;
    sut_settings.GPUInferStreams = FLAGS_gpu_inference_streams;

    sut_settings.EnableSpinWait = FLAGS_use_spin_wait;
    sut_settings.EnableDeviceScheduleSpin = FLAGS_use_device_schedule_spin;
    sut_settings.RunInferOnCopyStreams = FLAGS_run_infer_on_copy_streams;
    sut_settings.EnableDequeLimit = FLAGS_use_deque_limit;
    sut_settings.Timeout = std::chrono::microseconds(FLAGS_deque_timeout_usec);
    sut_settings.EnableBatcherThreadPerDevice = FLAGS_use_batcher_thread_per_device;
    sut_settings.EnableCudaThreadPerDevice = FLAGS_use_cuda_thread_per_device;
    sut_settings.CompleteThreads = FLAGS_complete_threads;
    sut_settings.UseSameContext = FLAGS_use_same_context;
    sut_settings.VerboseNVTX = FLAGS_verbose_nvtx;
    sut_settings.EnableCudaGraphs = FLAGS_use_graphs;

    lwis::ServerParams sut_params;
    sut_params.DeviceNames = FLAGS_devices;
    sut_params.EngineName = FLAGS_engine_file;

    // Autmotive v0.5 doesn't see any reason using multiple streams and batch size is fixed to 1
    CHECK_EQ(sut_settings.GPUCopyStreams, 1) << "Only 1 copy stream is supported for now";
    CHECK_EQ(sut_settings.GPUInferStreams, 1) << "Only 1 infer stream is supported for now";
    CHECK_EQ(sut_settings.GPUBatchSize, 1) << "Only batch size == 1 is supported for now";
    CHECK_EQ(sut_settings.EnableCudaGraphs, false) << "Cannot use CUDA graphs due to zero-copy scheme";

    // Instantiate Server
    lwis::ServerPtr_t sut = std::make_shared<lwis::Server>("Server_BevFormer");

    // Instantiate QSL
    gLogInfo << "Creating QSL." << std::endl;
    std::vector<std::string> tensor_paths = splitString(FLAGS_tensor_path, ",");
    const size_t padding = 0;
    auto oneQsl = std::make_shared<qsl::SampleLibraryBEVFormer>("LWIS_SampleLibrary", FLAGS_map_path,
        splitString(FLAGS_tensor_path, ","), FLAGS_performance_sample_count, /*padding*/ 0);
    sut->AddSampleLibrary(oneQsl);
    qsl::SampleLibraryBEVFormerPtr_t qsl = oneQsl;
    gLogInfo << "Finished Creating QSL." << std::endl;

    gLogInfo << "Setting up SUT." << std::endl;
    sut->Setup(sut_settings, sut_params); // Pass the requested sut settings and params to our SUT
    sut->SetResponseCallback(callbackMap[FLAGS_response_postprocess]); // Set QuerySampleResponse
                                                                       // post-processing callback
    gLogInfo << "Finished setting up SUT." << std::endl;

    // Perform a brief warmup
    gLogInfo << "Starting warmup. Running for a minimum of " << FLAGS_warmup_duration << " seconds." << std::endl;
    auto tStart = std::chrono::high_resolution_clock::now();
    sut->Warmup(FLAGS_warmup_duration);
    double elapsed = std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - tStart).count();
    gLogInfo << "Finished warmup. Ran for " << elapsed << "s." << std::endl;

    // Perform the inference testing
    gLogInfo << "Starting running actual test." << std::endl;
    cudaProfilerStart();
    mlperf::StartTest(sut.get(), qsl.get(), test_settings, log_settings);
    cudaProfilerStop();
    gLogInfo << "Finished running actual test." << std::endl;

    // Log device stats
    auto devices = sut->GetDevices();
    for (auto& device : devices)
    {
        const auto& stats = device->GetStats();

        gLogInfo << "Device " << device->GetName() << " processed:" << std::endl;
        gLogInfo << "  Memcpy Calls: " << stats.m_MemcpyCalls << std::endl;
        gLogInfo << "  PerSampleCudaMemcpy Calls: " << stats.m_PerSampleCudaMemcpyCalls << std::endl;
    }

    // Inform the SUT that we are done
    sut->Done();

    // Make sure CUDA RT is still in scope when we free the memory
    qsl.reset();
    sut.reset();

    return true;
}

int main(int argc, char* argv[])
{
    // Initialize logging
    FLAGS_alsologtostderr = 1; // Log to console
    ::google::InitGoogleLogging("TensorRT mlperf");
    ::google::ParseCommandLineFlags(&argc, &argv, true);
    const std::string gSampleName = "MLPerf_Automotive_BEVFormer_Harness";
    auto sampleTest = gLogger.defineTest(gSampleName, argc, const_cast<const char**>(argv));
    gLogger.reportTestStart(sampleTest);
    if (FLAGS_verbose)
    {
        setReportableSeverity(Severity::kVERBOSE);
    }
    else
    {
        setReportableSeverity(Severity::kINFO);
    }

    // Load all the needed shared objects for plugins.
    std::vector<std::string> plugin_files = splitString(FLAGS_plugins, ",");
    for (auto& s : plugin_files)
    {
        void* dlh = dlopen(s.c_str(), RTLD_LAZY);
        if (nullptr == dlh)
        {
            gLogError << "Error loading plugin library " << s << std::endl;
            return 1;
        }
        else
        {
            gLogInfo << "Loaded plugin library " << s << std::endl;
        }
    }

    // Perform inference
    bool pass = doInference();

    // Report test
    return gLogger.reportTest(sampleTest, pass);
}
