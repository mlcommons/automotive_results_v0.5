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

#include "lwis_bevformer.hpp"
#include "nvtx_common_wrapper.h" 
#include "loadgen.h"
#include "query_sample_library.h"

#include "NvInfer.h"
#include "NvInferPlugin.h"
#include <cstddef>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <fstream>
#include <stdexcept>

#include "logger.h"
#include <glog/logging.h>

namespace lwis
{
using namespace std::chrono_literals;

constexpr size_t const kFIRST_ENGINE = 0;
constexpr size_t const kFIRST_LOOP = 0;

inline int32_t getPersistentCacheSizeLimit()
{
    size_t persistentL2CacheSizeLimit = 0;
#if CUDART_VERSION >= 11030
    CHECK(cudaDeviceGetLimit(&persistentL2CacheSizeLimit, cudaLimitPersistingL2CacheSize) == 0);
#else
    persistentL2CacheSizeLimit = 0;
#endif
    return persistentL2CacheSizeLimit;
}

std::vector<std::string> split(const std::string& s, char delim)
{
    std::vector<std::string> res;
    std::stringstream ss;
    ss.str(s);
    std::string item;
    while (std::getline(ss, item, delim))
    {
        res.push_back(item);
    }
    return res;
}

void enqueueShim(nvinfer1::IExecutionContext* context, int batchSize, EngineTensors tensors,
    cudaStream_t inferStream)
{
    // v0.5 check
    CHECK_EQ(batchSize, 1) << "Need to set batch size to 1 for v0.5";

    auto& engine = context->getEngine();
        int32_t profileNum = context->getOptimizationProfile();
        CHECK_EQ(profileNum >= 0 && profileNum < engine.getNbOptimizationProfiles(), true);
        int32_t numIOTensors = engine.getNbIOTensors();
        for (int i = 0; i < numIOTensors; i++)
        {
            auto tensorName = engine.getIOTensorName(i);
            int tensorIdx = numIOTensors * profileNum + i;
            if (engine.getTensorIOMode(tensorName) == nvinfer1::TensorIOMode::kINPUT)
            {
                auto inputDims = context->getTensorShape(tensorName);
                CHECK_EQ(context->setInputShape(tensorName, inputDims), true);
            }
            context->setTensorAddress(tensorName, tensors[tensorIdx]);
        }
        CHECK_EQ(context->allInputShapesSpecified(), true);
        CHECK_EQ(context->enqueueV3(inferStream), true);
}

//----------------
// Device
//----------------
void Device::BuildGraphs()
{
    Issue();

    // build the graph by performing a single execution.
    size_t batchSize = 1;
        for (auto& streamState : m_StreamState)
        {
            auto& stream = streamState.first;
            auto& state = streamState.second;
            auto& bufferManager = std::get<0>(state);
            EngineTensors enqueueBuffers = bufferManager->getTensors();
            auto& context = std::get<4>(state);

            // need to issue enqueue to TRT to setup ressources properly _before_ starting graph construction
            enqueueShim(context, batchSize, enqueueBuffers, m_InferStreams[0]);

            cudaGraph_t graph;
#if (CUDA_VERSION >= 10010)
            CHECK_EQ(cudaStreamBeginCapture(m_InferStreams[0], cudaStreamCaptureModeThreadLocal), CUDA_SUCCESS);
#else
            CHECK_EQ(cudaStreamBeginCapture(m_InferStreams[0]), CUDA_SUCCESS);
#endif
            enqueueShim(context, batchSize, enqueueBuffers, m_InferStreams[0]);
            CHECK_EQ(cudaStreamEndCapture(m_InferStreams[0], &graph), CUDA_SUCCESS);

            cudaGraphExec_t graphExec;
            CHECK_EQ(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0), CUDA_SUCCESS);

            t_GraphKey key = std::make_pair(stream, batchSize);
            m_CudaGraphExecs[key] = graphExec;

            CHECK_EQ(cudaGraphDestroy(graph), CUDA_SUCCESS);
        }

    gLogInfo << "Capture " << m_CudaGraphExecs.size() << " CUDA graphs" << std::endl;
}

void Device::Setup()
{
    cudaSetDevice(m_Id);
    if (m_EnableDeviceScheduleSpin)
        cudaSetDeviceFlags(cudaDeviceScheduleSpin);

    unsigned int cudaEventFlags
        = (m_EnableSpinWait ? cudaEventDefault : cudaEventBlockingSync) | cudaEventDisableTiming;

    for (auto& inferStream : m_InferStreams)
    {
        CHECK_EQ(cudaStreamCreate(&inferStream), CUDA_SUCCESS);
    }

    // Setup execution context for the engine
    std::vector<std::shared_ptr<nvinfer1::ICudaEngine>> copyStream_Engines(m_CopyStreams.size());
    std::vector<nvinfer1::IExecutionContext*> copyStreamContexts(m_CopyStreams.size());
    std::vector<int32_t> copyStreamProfiles(m_CopyStreams.size());

    int32_t profileIdx{0};
    auto engine = m_Engine->GetCudaEngine();
    nvinfer1::IExecutionContext* context{nullptr};

    // Use the same TRT execution contexts for all copy streams
    // shape must be static and gpu_inference_streams must be 1
    if (m_UseSameContext)
    {
        context = engine->createExecutionContext();
        if (m_VerboseNVTX)
        {
            context->setNvtxVerbosity(nvinfer1::ProfilingVerbosity::kDETAILED);
        }
        else
        {
            context->setNvtxVerbosity(nvinfer1::ProfilingVerbosity::kNONE);
        }
        CHECK_EQ(context->getOptimizationProfile() == 0, true);
        CHECK_EQ(m_InferStreams.size() == 1, true);
        CHECK_EQ(context->allInputDimensionsSpecified(), true);
    }

    for (size_t i = 0; i < m_CopyStreams.size(); ++i)
    {
        // Create execution context for each profile
        // The number of engine profile should be the same as the number of copy streams
        if (!m_UseSameContext)
        {
            context = engine->createExecutionContext();
            if (m_VerboseNVTX)
            {
                context->setNvtxVerbosity(nvinfer1::ProfilingVerbosity::kDETAILED);
            }
            else
            {
                context->setNvtxVerbosity(nvinfer1::ProfilingVerbosity::kNONE);
            }
            CHECK_EQ(profileIdx < engine->getNbOptimizationProfiles(), true);
            CHECK_EQ(context->setOptimizationProfileAsync(profileIdx, m_InferStreams[0]), true);
        }
        CHECK_EQ(context->getOptimizationProfile() == profileIdx, true);
        copyStreamContexts[i] = context;

        // eviction last setup
        if (m_elRatio>0.0) {
            int32_t persistentCacheLimitCUDAValue = getPersistentCacheSizeLimit();
            context->setPersistentCacheLimit(m_elRatio*persistentCacheLimitCUDAValue);
        }

        std::shared_ptr<nvinfer1::ICudaEngine> emptyPtr{};
        std::shared_ptr<nvinfer1::ICudaEngine> aliasPtr(emptyPtr, engine);
        copyStream_Engines[i] = aliasPtr;
        copyStreamProfiles[i] = profileIdx;

        // Engines using static batch size can create multiple contexts with one profile.
        if (engine->getNbOptimizationProfiles() > 1 && !m_UseSameContext)
        {
            ++profileIdx;
        }
    }

    // Setup buffer manager and state for each copy stream
    for (size_t i = 0; i < m_CopyStreams.size(); ++i)
    {
        auto& copyStream = m_CopyStreams[i];
        CHECK_EQ(cudaStreamCreate(&copyStream), CUDA_SUCCESS);

        auto state = std::make_tuple(std::make_shared<BufferManager>(
                                         copyStream_Engines[i], m_BatchSize, copyStreamProfiles[i]),
            cudaEvent_t(), cudaEvent_t(), cudaEvent_t(), copyStreamContexts[i]);
        CHECK_EQ(cudaEventCreateWithFlags(&std::get<1>(state), cudaEventFlags), CUDA_SUCCESS);
        CHECK_EQ(cudaEventCreateWithFlags(&std::get<2>(state), cudaEventFlags), CUDA_SUCCESS);
        CHECK_EQ(cudaEventCreateWithFlags(&std::get<3>(state), cudaEventFlags), CUDA_SUCCESS);
        m_StreamState.insert(std::make_pair(copyStream, state));
        m_StreamQueue.emplace_back(copyStream);
    }
}

void Device::Issue()
{
    CHECK_EQ(cudaSetDevice(m_Id), cudaSuccess);
}

void Device::Done()
{
    // join before destroying all members
    for (auto& thread : m_Threads)
    {
        thread.join();
    }

    // destroy member objects
    cudaSetDevice(m_Id);

    for (auto& inferStream : m_InferStreams)
    {
        cudaStreamDestroy(inferStream);
    }
    for (auto& copyStream : m_CopyStreams)
    {
        auto& state = m_StreamState[copyStream];

        cudaStreamDestroy(copyStream);
        cudaEventDestroy(std::get<1>(state));
        cudaEventDestroy(std::get<2>(state));
        cudaEventDestroy(std::get<3>(state));
        if (!m_UseSameContext)
        {
            nvinfer1::IExecutionContext* engineContext = std::get<4>(state);
            delete engineContext;
            engineContext = nullptr;
        }
    }
    if (m_UseSameContext)
    {
        nvinfer1::IExecutionContext* engineContext = std::get<4>(m_StreamState[m_CopyStreams[0]]);
        delete engineContext;
        engineContext = nullptr;
    }
    for (auto& kv : m_CudaGraphExecs)
    {
        CHECK_EQ(cudaGraphExecDestroy(kv.second), CUDA_SUCCESS);
    }
}

void Device::Completion()
{
    // Testing for completion needs to be based on the main thread finishing submission and
    // providing events for the completion thread to wait on.  The resources exist as part of the
    // Device class.
    //

    // Flow:
    // Main thread
    // - Find Device (check may be based on data buffer availability)
    // - Enqueue work
    // - Enqueue CompletionQueue batch
    // ...
    // - Enqueue CompletionQueue null batch

    // Completion thread(s)
    // - Wait for entry
    // - Wait for queue head to have data ready (wait for event)
    // - Dequeue CompletionQueue

    while (true)
    {
        // TODO: with multiple CudaStream inference it may be beneficial to handle these out of order
        auto batch = m_CompletionQueue.front_then_pop();

        if (batch.Responses.empty())
            break;

        // wait on event completion
        CHECK_EQ(cudaEventSynchronize(batch.Event), cudaSuccess);

        // callback if it exists
        if (m_ResponseCallback)
        {
            CHECK(batch.SampleIds.size() == batch.Responses.size()) << "missing sample IDs";
            m_ResponseCallback(&batch.Responses[0], batch.SampleIds, batch.Responses.size());
        }

        // assume this function is reentrant for multiple devices
        TIMER_START(QuerySamplesComplete);
        mlperf::QuerySamplesComplete(
            &batch.Responses[0], batch.Responses.size(), batch.ResponseCb.value_or(mlperf::ResponseCallback{}));
        TIMER_END(QuerySamplesComplete);

        m_StreamQueue.emplace_back(batch.Stream);
    }
}


//----------------
// Server
//----------------

//! Setup
//!
//! Perform all necessary (untimed) setup in order to perform inference including: building
//! graphs and allocating device memory.
void Server::Setup(ServerSettings_BEVFormer& settings, ServerParams& params)
{
    initLibNvInferPlugins(&gLogger.getTRTLogger(), "");

    m_ServerSettings = settings;

    // enumerate devices
    std::vector<size_t> devices;
    if (params.DeviceNames == "all")
    {
        int numDevices = 0;
        cudaGetDeviceCount(&numDevices);
        for (int i = 0; i < numDevices; i++)
        {
            devices.emplace_back(i);
        }
    }
    else
    {
        auto deviceNames = split(params.DeviceNames, ',');
        for (auto& n : deviceNames)
            devices.emplace_back(std::stoi(n));
    }

    // check if an engine was specified
    if (params.EngineName.empty())
    {
        gLogError << "Engine file not specified" << std::endl;
    }

    auto runtime = nvinfer1::createInferRuntime(gLogger.getTRTLogger());
    size_t const batchSize = m_ServerSettings.GPUBatchSize;
    size_t const numCopyStreams = m_ServerSettings.GPUCopyStreams;
    size_t const numInferStreams = m_ServerSettings.GPUInferStreams;
    auto const& engineName = params.EngineName;

    for (auto const& deviceNum : devices)
    {
        cudaSetDevice(deviceNum);

        auto device = std::make_shared<lwis::Device>(deviceNum, numCopyStreams, numInferStreams,
            m_ServerSettings.CompleteThreads, m_ServerSettings.EnableSpinWait,
            m_ServerSettings.EnableDeviceScheduleSpin, m_ServerSettings.UseSameContext,
            m_ServerSettings.GPUBatchSize, m_ServerSettings.elRatio, m_ServerSettings.VerboseNVTX);

        std::vector<char> trtModelStream;
        auto size = GetModelStream(trtModelStream, engineName);
        auto engine = runtime->deserializeCudaEngine(trtModelStream.data(), size);
        device->AddEngine(std::make_shared<lwis::Engine>(engine), batchSize);
        std::ostringstream deviceName;
        deviceName << "Device:" << deviceNum << ".GPU";
        device->m_Name = deviceName.str();

        m_Devices.emplace_back(device);
        gLogInfo << device->m_Name << ": " << engineName
                    << " has been successfully loaded." << std::endl;
    }

    CHECK(m_Devices.size()) << "No devices or engines available";

    for (auto& device : m_Devices)
    {
        device->Setup();
    }

    if (m_ServerSettings.EnableCudaGraphs)
    {
        gLogInfo << "Start creating CUDA graphs" << std::endl;
        std::vector<std::thread> tmpGraphsThreads;
        for (auto& device : m_Devices)
        {
            tmpGraphsThreads.emplace_back(
                &Device::BuildGraphs, device.get());
        }
        for (auto& thread : tmpGraphsThreads)
        {
            thread.join();
        }
        gLogInfo << "Finish creating CUDA graphs" << std::endl;
    }

    delete runtime;
    runtime = nullptr;

    Reset();

    // create batchers
    for (size_t deviceNum = 0; deviceNum < (m_ServerSettings.EnableBatcherThreadPerDevice ? m_Devices.size() : 1);
         deviceNum++)
    {
        gLogInfo << "Creating batcher thread: " << deviceNum << " EnableBatcherThreadPerDevice: "
                 << (m_ServerSettings.EnableBatcherThreadPerDevice ? "true" : "false") << std::endl;
        m_Threads.emplace_back(std::thread(&Server::ProcessSamples, this));
    }

    // create issue threads
    if (m_ServerSettings.EnableCudaThreadPerDevice)
    {
        for (size_t deviceNum = 0; deviceNum < m_Devices.size(); deviceNum++)
        {
            gLogInfo << "Creating cuda thread: " << deviceNum << std::endl;
            m_IssueThreads.emplace_back(std::thread(&Server::ProcessBatches, this));
        }
    }
}

void Server::Done()
{
    // send dummy batch to signal completion
    for (auto& device : m_Devices)
    {
        for (size_t i = 0; i < m_ServerSettings.CompleteThreads; i++)
        {
            device->m_CompletionQueue.push_back(Batch{});
        }
    }
    for (auto& device : m_Devices)
        device->Done();

    // send end sample to signal completion
    while (!m_WorkQueue.empty())
    {
    }

    while (m_DeviceNum)
    {
        size_t currentDeviceId = m_DeviceNum;
        m_WorkQueue.emplace_back(mlperf::QuerySample{0, 0});
        while (currentDeviceId == m_DeviceNum)
        {
        }
    }

    if (m_ServerSettings.EnableCudaThreadPerDevice)
    {
        for (auto& device : m_Devices)
        {
            std::deque<mlperf::QuerySample> batch;
            auto pair = std::make_pair(std::move(batch), nullptr);
            device->m_IssueQueue.emplace_back(pair);
        }
        for (auto& thread : m_IssueThreads)
            thread.join();
    }

    // join after we insert the dummy sample
    for (auto& thread : m_Threads)
    {
        thread.join();
    }
}

void Server::IssueQuery(const std::vector<mlperf::QuerySample>& samples)
{
    TIMER_START(IssueQuery);
    m_WorkQueue.insert(samples);
    TIMER_END(IssueQuery);
}

DevicePtr_t Server::GetNextAvailableDevice(size_t deviceId)
{
    DevicePtr_t device;
    if (!m_ServerSettings.EnableBatcherThreadPerDevice)
    {
        do
        {
            device = m_Devices[m_DeviceIndex];
            m_DeviceIndex = (m_DeviceIndex + 1) % m_Devices.size();
        } while (device->m_StreamQueue.empty());
    }
    else
    {
        device = m_Devices[deviceId];
        while (device->m_StreamQueue.empty())
        {
        }
    }

    return device;
}

void Server::IssueBatch(DevicePtr_t device, size_t batchSize, std::deque<mlperf::QuerySample>::iterator begin,
    std::deque<mlperf::QuerySample>::iterator end, cudaStream_t copyStream)
{
    NVTX_RANGE_PUSH("IssueBatch");
    auto inferStream
        = (m_ServerSettings.RunInferOnCopyStreams) ? copyStream : device->m_InferStreams[device->m_InferStreamIdx];

    auto& state = device->m_StreamState[copyStream];
    auto& bufferManager = std::get<0>(state);
    auto& htod = std::get<1>(state);
    auto& inf = std::get<2>(state);
    auto& dtoh = std::get<3>(state);
    auto& context = std::get<4>(state);

    // setup Device
    device->Issue();

    // perform copy to device
#ifndef LWIS_DEBUG_DISABLE_INFERENCE
    auto enqueueBuffers = bufferManager->getTensors();
    TIMER_START(PrepareInputSamples);
    // enqueueBuffers = PrepareInputSamples(device, batchSize, begin, end, copyStream);
    enqueueBuffers = PrepareInputSamples(device, batchSize, begin, end, copyStream);
    TIMER_END(PrepareInputSamples);
    if (!m_ServerSettings.RunInferOnCopyStreams)
    {
        CHECK_EQ(cudaEventRecord(htod, copyStream), CUDA_SUCCESS);
    }
#ifndef LWIS_DEBUG_DISABLE_COMPUTE
    // perform inference
    if (!m_ServerSettings.RunInferOnCopyStreams)
    {
        CHECK_EQ(cudaStreamWaitEvent(inferStream, htod, 0), CUDA_SUCCESS);
    }
    Device::t_GraphKey key = std::make_pair(copyStream, device->m_BatchSize);
    auto g_it = device->m_CudaGraphExecs.lower_bound(key);
    if (g_it != device->m_CudaGraphExecs.end())
    {
        CHECK_EQ(cudaGraphLaunch(g_it->second, inferStream), CUDA_SUCCESS);
    }
    else
    {
        TIMER_START(enqueueShim);
        NVTX_RANGE_PUSH("enqueueShim");
        enqueueShim(context, batchSize, enqueueBuffers, inferStream);
        NVTX_RANGE_POP(); // "enqueueShim"
        TIMER_END(enqueueShim);
    }
#endif
    if (!m_ServerSettings.RunInferOnCopyStreams)
    {
        CHECK_EQ(cudaEventRecord(inf, inferStream), CUDA_SUCCESS);
    }

    // perform copy back to host
    if (!m_ServerSettings.RunInferOnCopyStreams)
    {
        CHECK_EQ(cudaStreamWaitEvent(copyStream, inf, 0), CUDA_SUCCESS);
    }
    auto engine = device->m_Engine->GetCudaEngine();

    for (auto i = 0; i < engine->getNbIOTensors(); i++)
    {
        auto tensorName = engine->getIOTensorName(i);
        if (engine->getTensorIOMode(tensorName) == nvinfer1::TensorIOMode::kOUTPUT)
        {
            bufferManager->prefetchOutputToHostAsync(copyStream, i);
        }
    }
    // make sure output buffers are available in host
    cudaStreamSynchronize(copyStream);
#endif
    CHECK_EQ(cudaEventRecord(dtoh, copyStream), CUDA_SUCCESS);

    // optional synchronization
    if (m_ServerSettings.EnableSyncOnEvent)
    {
        cudaEventSynchronize(dtoh);
    }

    // generate asynchronous response
    TIMER_START(asynchronous_response);
    if (m_ServerSettings.EnableResponse)
    {
        NVTX_RANGE_PUSH("ResponseCopy H2H");
        Batch batch;
  
        size_t totalOutputSize = bufferManager->getTotalOutputTensorSize();
        size_t singleOutputSize = totalOutputSize / batchSize;
        size_t batchIndex = 0;
        for (auto it = begin; it != end; ++it)
        {
            auto outputBuf = std::make_unique<std::vector<uint8_t>>(singleOutputSize);
            auto outputBufPtr = reinterpret_cast<uintptr_t>(outputBuf->data());

            // Need to interleave batched responses per category into per-batch response
            size_t categoryIndex = 0;
            for (auto& outputTensorInfo : bufferManager->getOutputTensorInfo())
            {
                auto tensorIndex = std::get<0>(outputTensorInfo);
                auto tensorName = std::get<1>(outputTensorInfo);
                auto tensorSize = std::get<2>(outputTensorInfo);
                auto singleTensorSize = std::get<3>(outputTensorInfo);
                auto buffer = std::get<4>(outputTensorInfo);

                auto targetPtr = reinterpret_cast<void*>(outputBufPtr + batchIndex * singleOutputSize + categoryIndex * singleTensorSize);
                memcpy(targetPtr, buffer, singleTensorSize);
                device->m_Stats.m_MemcpyCalls++;
                categoryIndex++;
            }

            batch.Responses.emplace_back(mlperf::QuerySampleResponse{it->id, outputBufPtr, singleOutputSize});
            if (device->m_ResponseCallback)
            {
                batch.SampleIds.emplace_back(it->index);
            }
            
            batchIndex++;
        }  

        batch.Event = dtoh;
        batch.Stream = copyStream;

        device->m_CompletionQueue.emplace_back(batch);
        NVTX_RANGE_POP(); // "ResponseCopy H2H"
    }
    TIMER_END(asynchronous_response);

    // Simple round-robin across inference streams.  These don't need to be managed like copy
    // streams since they are not tied with a resource that is re-used and not managed by hardware.
    device->m_InferStreamIdx = (device->m_InferStreamIdx + 1) % device->m_InferStreams.size();

    // m_IssueCount is used to track the number of batches issued
    m_IssueCount += batchSize;
    if (m_IssueCount % 1000 == 0)
    {
        gLogInfo << device->m_Name << " performed total " << m_IssueCount << " inferences" << std::endl;
    }
    NVTX_RANGE_POP(); // "IssueBatch"
}

EngineTensors Server::PrepareInputSamples(DevicePtr_t device, size_t batchSize, 
    std::deque<mlperf::QuerySample>::iterator begin, std::deque<mlperf::QuerySample>::iterator end, 
    cudaStream_t copyStream)
{
    NVTX_RANGE_PUSH("PrepareInputSamples");

    auto& bufferManager = std::get<0>(device->m_StreamState[copyStream]);
    auto deviceId = device->GetId();

    EngineTensors tensors = bufferManager->getTensors();

    if (m_SampleLibrary)
    {
        auto engine = device->m_Engine->GetCudaEngine();
        size_t numInputs{device->m_Engine->GetNumInputTensors()};

        auto internalTensorInfo = bufferManager->getInternalTensorInfo();
        auto internalTensorBuffer = std::get<4>(internalTensorInfo[0]);

        for (size_t i = 0; i < numInputs; i++)
        {
            auto tensorName = engine->getIOTensorName(i);
            if (strcmp(tensorName, "prev_bev") == 0)
            {
                tensors[i] = internalTensorBuffer;
                continue;
            }
            tensors[i] = m_SampleLibrary->GetSampleAddress(begin->index, i);
        }
    }
    NVTX_RANGE_POP(); // "PrepareInputSamples"

    return tensors;
}

void Server::Reset()
{
    m_DeviceIndex = 0;

    for (auto& device : m_Devices)
    {
        device->m_InferStreamIdx = 0;
        device->m_Stats.reset();
    }
}

void Server::ProcessSamples()
{
    // until Setup is called we may not have valid devices
    size_t deviceId = m_DeviceNum++;

    // initial device available
    auto device = GetNextAvailableDevice(deviceId);

    while (true)
    {
        std::deque<mlperf::QuerySample> samples;
        TIMER_START(m_WorkQueue_acquire_total);
        do
        {
            TIMER_START(m_WorkQueue_acquire);
            m_WorkQueue.acquire(samples, m_ServerSettings.Timeout,
                device->m_BatchSize,
                m_ServerSettings.EnableDequeLimit || m_ServerSettings.EnableBatcherThreadPerDevice);
            TIMER_END(m_WorkQueue_acquire);
        } while (samples.empty());
        TIMER_END(m_WorkQueue_acquire_total);

        auto begin = samples.begin();
        auto end = samples.end();

        // Use a null (0) id to represent the end of samples
        if (!begin->id)
        {
            m_DeviceNum--;
            break;
        }

        auto batchBegin = begin;

        // build batches up to maximum supported batchSize
        while (batchBegin != end)
        {
            // Input batch size depends on the first engine
            auto batchSizeTotal
                = std::min(device->m_BatchSize,
                    static_cast<size_t>(std::distance(batchBegin, end)));
            auto batchEnd = batchBegin + batchSizeTotal;

            // Acquire resources
            TIMER_START(m_StreamQueue_pop_front);
            auto copyStream = device->m_StreamQueue.front();
            device->m_StreamQueue.pop_front();
            TIMER_END(m_StreamQueue_pop_front);

            // Issue this batch
            if (!m_ServerSettings.EnableCudaThreadPerDevice)
            {
                // issue on this thread
                TIMER_START(IssueBatch);
                IssueBatch(device, batchSizeTotal, batchBegin, batchEnd, copyStream);
                TIMER_END(IssueBatch);
            }
            else
            {
                // issue on device specific thread
                std::deque<mlperf::QuerySample> batch(batchBegin, batchEnd);
                auto pair = std::make_pair(std::move(batch), copyStream);
                device->m_IssueQueue.emplace_back(pair);
            }

            // Advance to next batch
            batchBegin = batchEnd;

            // Get available device for next batch
            TIMER_START(GetNextAvailableDevice);
            device = GetNextAvailableDevice(deviceId);
            TIMER_END(GetNextAvailableDevice);
        }
    }
}

void Server::ProcessBatches()
{
    // until Setup is called we may not have valid devices
    size_t deviceId = m_IssueNum++;
    auto& device = m_Devices[deviceId];
    auto& issueQueue = device->m_IssueQueue;

    while (true)
    {
        auto pair = issueQueue.front();
        issueQueue.pop_front();

        auto& batch = pair.first;
        auto& stream = pair.second;

        if (batch.empty())
        {
            m_IssueNum--;
            break;
        }

        IssueBatch(device, batch.size(), batch.begin(), batch.end(), stream);
    }
}

void Server::Warmup(double duration)
{
    double elapsed = 0.0;
    auto tStart = std::chrono::high_resolution_clock::now();

    do
    {
        for (size_t deviceIndex = 0; deviceIndex < m_Devices.size(); ++deviceIndex)
        {
            // get next device to send batch to
            auto device = m_Devices[deviceIndex];
            for (auto copyStream : device->m_CopyStreams)
            {
                for (auto inferStream : device->m_InferStreams)
                {
                    auto& state = device->m_StreamState[copyStream];
                    auto& bufferManager = std::get<0>(state);
                    auto& htod = std::get<1>(state);
                    auto& inf = std::get<2>(state);
                    auto& dtoh = std::get<3>(state);
                    auto& context = std::get<4>(state);

                    device->Issue();
                    auto engine = device->m_Engine->GetCudaEngine();

                    for (auto i = 0; i < engine->getNbIOTensors(); i++)
                    {
                        auto tensorName = engine->getIOTensorName(i);
                        if (engine->getTensorIOMode(tensorName) == nvinfer1::TensorIOMode::kINPUT)
                        {
                            bufferManager->prefetchInputToDeviceAsync(copyStream, i);
                        }
                    }

                    if (!m_ServerSettings.RunInferOnCopyStreams)
                    {
                        CHECK_EQ(cudaEventRecord(htod, copyStream), CUDA_SUCCESS);
                        CHECK_EQ(cudaStreamWaitEvent(inferStream, htod, 0), CUDA_SUCCESS);
                    }

                    Device::t_GraphKey key = std::make_pair(copyStream, device->m_BatchSize);
                    auto g_it = device->m_CudaGraphExecs.lower_bound(key);
                    if (g_it != device->m_CudaGraphExecs.end())
                    {
                        CHECK_EQ(cudaGraphLaunch(
                                     g_it->second, (m_ServerSettings.RunInferOnCopyStreams) ? copyStream : inferStream),
                            CUDA_SUCCESS);
                    }
                    else
                    {
                        EngineTensors enqueueBuffers = bufferManager->getTensors();
                        enqueueShim(context,
                            device->m_BatchSize,
                            enqueueBuffers,
                            (m_ServerSettings.RunInferOnCopyStreams) ? copyStream : inferStream);
                    }

                    if (!m_ServerSettings.RunInferOnCopyStreams)
                    {
                        CHECK_EQ(cudaEventRecord(inf, inferStream), CUDA_SUCCESS);
                    }

                    if (!m_ServerSettings.RunInferOnCopyStreams)
                    {
                        CHECK_EQ(cudaStreamWaitEvent(copyStream, inf, 0), CUDA_SUCCESS);
                    }

                    for (auto i = 0; i < engine->getNbIOTensors(); i++)
                    {
                        auto tensorName = engine->getIOTensorName(i);
                        if (engine->getTensorIOMode(tensorName) == nvinfer1::TensorIOMode::kOUTPUT)
                        {
                            bufferManager->prefetchOutputToHostAsync(copyStream, i);
                        }
                    }
                    CHECK_EQ(cudaEventRecord(dtoh, copyStream), CUDA_SUCCESS);
                }
            }
        }
        elapsed = std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - tStart).count();
    } while (elapsed < duration);

    for (auto& device : m_Devices)
    {
        device->Issue();
        cudaDeviceSynchronize();
    }

    // reset server state
    Reset();
}

void Server::FlushQueries()
{
    // This function is called at the end of a series of IssueQuery calls (typically the end of a
    // region of queries that define a performance or accuracy test).  Its purpose is to allow a
    // SUT to force all remaining queued samples out to avoid implementing timeouts.

    // Currently, there is no use case for it in this IS.
}

void Server::SetResponseCallback(std::function<void(::mlperf::QuerySampleResponse* responses,
        std::vector<::mlperf::QuerySampleIndex>& sample_ids, size_t response_count)>
        callback)
{
    std::for_each(
        m_Devices.begin(), m_Devices.end(), [callback](DevicePtr_t device) { device->SetResponseCallback(callback); });
}

}; // namespace lwis
