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

#ifndef __LWIS_BEVFORMER_HPP__
#define __LWIS_BEVFORMER_HPP__

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <map>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>
#include <optional>

#include "NvInfer.h"
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#include "loadgen.h"
#include "system_under_test.h"
#include "test_settings.h"
#include "utils.hpp"

#include "lwis_bevformer_buffers.h"
#include "qsl_bevformer.hpp"


#define NVTX 0
#if NVTX
#define NVTX_MARK(message) nvtxMarkA(message);
#define NVTX_NAME_THIS_THREAD(name) nvtxNameOsThreadA(pthread_self(), name)
#define NVTX_RANGE_PUSH(message) nvtxRangePushA(message);
#define NVTX_RANGE_POP() nvtxRangePop();
#else
#define NVTX_MARK(message)
#define NVTX_NAME_THIS_THREAD(name)
#define NVTX_RANGE_PUSH(message)
#define NVTX_RANGE_POP()
#endif


// For debugging the timing of each part
class Timer
{
public:
    Timer(const std::string& tag_)
        : tag(tag_)
    {
        std::cout << "Timer " << tag << " created." << std::endl;
    }
    void add(const std::chrono::duration<double, std::milli>& in)
    {
        ++count;
        total += in;
    }
    ~Timer()
    {
        std::cout << "Timer " << tag << " reports " << total.count() / count << " ms per call for " << count
                  << " times." << std::endl;
    }

private:
    std::string tag;
    std::chrono::duration<double, std::milli> total{0};
    size_t count{0};
};

#define TIMER_ON 0

#if TIMER_ON
#define TIMER_START(s)                                                                                                 \
    static Timer timer##s(#s);                                                                                         \
    auto start##s = std::chrono::high_resolution_clock::now();
#define TIMER_END(s) timer##s.add(std::chrono::high_resolution_clock::now() - start##s);
#else
#define TIMER_START(s)
#define TIMER_END(s)
#endif

// This rewrites LWIS, toning down and adding some functionality such as:
// - defining parameters related to sliding window, i.e. ROI size, overlap factor, etc
// - loads and uses preconditioned (pre-normalized) Gaussian patches
// - handles sliding window inference by calling kernels for slicing, weighting, aggregating and
// ArgMax-ing

namespace lwis
{
using namespace std::chrono_literals;

class BufferManager;

class Device;
class Engine;
class Server;

typedef std::shared_ptr<Device> DevicePtr_t;
typedef std::shared_ptr<Engine> EnginePtr_t;
typedef std::shared_ptr<Server> ServerPtr_t;

typedef std::shared_ptr<::nvinfer1::ICudaEngine> ICudaEnginePtr_t;

struct Batch
{
    std::vector<mlperf::QuerySampleResponse> Responses;
    std::vector<mlperf::QuerySampleIndex> SampleIds;
    cudaEvent_t Event;
    cudaStream_t Stream;
    std::optional<mlperf::ResponseCallback> ResponseCb; // required if end on device enabled
};

struct ServerSettings_BEVFormer
{
    bool EnableSyncOnEvent{false};
    bool EnableSpinWait{false};
    bool EnableDeviceScheduleSpin{false};
    bool EnableResponse{true};
    bool EnableDequeLimit{false};
    bool EnableBatcherThreadPerDevice{false};
    bool EnableCudaThreadPerDevice{false};
    bool RunInferOnCopyStreams{true};
    bool UseSameContext{true};
    bool EnableCudaGraphs{false};
    bool VerboseNVTX{false};
    
    size_t GPUBatchSize{1};
    size_t GPUCopyStreams{1};
    size_t GPUInferStreams{1};
    int32_t MaxGPUs{-1};
    
    double elRatio{0.0};

    size_t CompleteThreads{1};

    std::chrono::microseconds Timeout{10000us};
};

struct ServerParams
{
    std::string DeviceNames;
    std::string EngineName;
};

template <typename T>
class SyncQueue
{
public:
    typedef typename std::deque<T>::iterator iterator;

    SyncQueue() {}

    bool empty()
    {
        std::unique_lock<std::mutex> l(m_Mutex);
        return m_Queue.empty();
    }

    size_t size()
    {
        std::unique_lock<std::mutex> l(m_Mutex);
        return m_Queue.size();
    }

    void insert(const std::vector<T>& values)
    {
        {
            std::unique_lock<std::mutex> l(m_Mutex);
            m_Queue.insert(m_Queue.end(), values.begin(), values.end());
        }
        m_Condition.notify_one();
    }
    void insert(const std::vector<T>& values, const size_t begin_idx, const size_t end_index)
    {
        {
            std::unique_lock<std::mutex> l(m_Mutex);
            m_Queue.insert(m_Queue.end(), values.begin() + begin_idx, values.begin() + end_index);
        }
        m_Condition.notify_one();
    }
    void acquire(
        std::deque<T>& values, std::chrono::microseconds duration = 10000us, size_t size = 1, bool limit = false)
    {
        size_t remaining = 0;

        TIMER_START(m_Mutex_create);
        {
            std::unique_lock<std::mutex> l(m_Mutex);
            TIMER_END(m_Mutex_create);
            TIMER_START(m_Condition_wait_for);
            m_Condition.wait_for(l, duration, [=, this] { return m_Queue.size() >= size; });
            TIMER_END(m_Condition_wait_for);

            if (!limit || m_Queue.size() <= size)
            {
                TIMER_START(swap);
                values.swap(m_Queue);
                TIMER_END(swap);
            }
            else
            {
                auto beg = m_Queue.begin();
                auto end = beg + size;
                TIMER_START(values_insert);
                values.insert(values.end(), beg, end);
                TIMER_END(values_insert);
                TIMER_START(m_Queue_erase);
                m_Queue.erase(beg, end);
                TIMER_END(m_Queue_erase);
                remaining = m_Queue.size();
            }
        }

        // wake up any waiting threads
        if (remaining)
            m_Condition.notify_one();
    }

    void push_back(T const& v)
    {
        {
            std::unique_lock<std::mutex> l(m_Mutex);
            m_Queue.push_back(v);
        }
        m_Condition.notify_one();
    }
    void emplace_back(T const& v)
    {
        {
            std::unique_lock<std::mutex> l(m_Mutex);
            m_Queue.emplace_back(v);
        }
        m_Condition.notify_one();
    }
    T front()
    {
        std::unique_lock<std::mutex> l(m_Mutex);
        m_Condition.wait(l, [this] { return !m_Queue.empty(); });
        T r(std::move(m_Queue.front()));
        return r;
    }
    T front_then_pop()
    {
        std::unique_lock<std::mutex> l(m_Mutex);
        m_Condition.wait(l, [this] { return !m_Queue.empty(); });
        T r(std::move(m_Queue.front()));
        m_Queue.pop_front();
        return r;
    }
    void pop_front()
    {
        std::unique_lock<std::mutex> l(m_Mutex);
        m_Queue.pop_front();
    }

private:
    mutable std::mutex m_Mutex;
    std::condition_variable m_Condition;

    std::deque<T> m_Queue;
};

enum SliceRelativePosition {
  StartingCorner = 0,
  Middle = 1,
  EndingCorner = 2
};

// captures execution engine for performing inference
class Device
{
    friend Server;

public:
    Device(size_t id, size_t numCopyStreams, size_t numInferStreams, size_t numCompleteThreads, bool enableSpinWait,
        bool enableDeviceScheduleSpin, bool useSameContext, size_t batchSize, double elRatio, bool verboseNVTX)
        : m_Id(id)
        , m_EnableSpinWait(enableSpinWait)
        , m_EnableDeviceScheduleSpin(enableDeviceScheduleSpin)
        , m_UseSameContext(useSameContext)
        , m_CopyStreams(numCopyStreams)
        , m_InferStreams(numInferStreams)
        , m_BatchSize(batchSize)
        , m_elRatio(elRatio)
        , m_VerboseNVTX(verboseNVTX)
    {
        for (size_t i = 0; i < numCompleteThreads; i++)
        {
            m_Threads.emplace_back(&Device::Completion, this);
        }
    }

    EnginePtr_t GetEngine()
    {
        return m_Engine;
    }

    size_t GetBatchSize()
    {
        return m_BatchSize;
    }

    std::string GetName()
    {
        return m_Name;
    }
    size_t GetId()
    {
        return m_Id;
    }

    void SetResponseCallback(std::function<void(::mlperf::QuerySampleResponse* responses,
            std::vector<::mlperf::QuerySampleIndex>& sample_ids, size_t response_count)>
            callback)
    {
        m_ResponseCallback = callback;
    }

    struct Stats
    {
        uint64_t m_MemcpyCalls{0};
        uint64_t m_PerSampleCudaMemcpyCalls{0};

        void reset()
        {
            m_MemcpyCalls = 0;
            m_PerSampleCudaMemcpyCalls = 0;
        }
    };

    const Stats& GetStats() const
    {
        return m_Stats;
    }

private:
    void AddEngine(EnginePtr_t engine, size_t batchSize)
    {
        m_Engine = engine;
    }

    void BuildGraphs();

    void Setup();
    void Issue();
    void Done();
    void Completion();

    EnginePtr_t m_Engine;
    const size_t m_Id{0};
    std::string m_Name{""};

    const bool m_EnableSpinWait{false};
    const bool m_EnableDeviceScheduleSpin{false};
    const bool m_UseSameContext{true};
    const bool m_VerboseNVTX{false};

    size_t m_BatchSize{1};
    double const m_elRatio{0.0};
    std::vector<cudaStream_t> m_CopyStreams;
    std::vector<cudaStream_t> m_InferStreams;
    size_t m_InferStreamIdx{0};
    std::map<cudaStream_t,
        std::tuple<std::shared_ptr<BufferManager>, cudaEvent_t, cudaEvent_t, cudaEvent_t,
            ::nvinfer1::IExecutionContext*>>
        m_StreamState;

    // collect sequence of sliding window for easy-batching
    std::deque<std::tuple<int,int,int,int>> m_Iterations;

    // Graphs
    typedef std::pair<cudaStream_t, size_t> t_GraphKey;
    std::map<t_GraphKey, cudaGraphExec_t> m_CudaGraphExecs;

    // Completion management
    SyncQueue<Batch> m_CompletionQueue;
    std::vector<std::thread> m_Threads;

    // Stream management
    SyncQueue<cudaStream_t> m_StreamQueue;

    // Query sample response callback.
    std::function<void(::mlperf::QuerySampleResponse* responses, std::vector<::mlperf::QuerySampleIndex>& sample_ids,
        size_t response_count)>
        m_ResponseCallback;

    // Batch management
    SyncQueue<std::pair<std::deque<mlperf::QuerySample>, cudaStream_t>> m_IssueQueue;

    Stats m_Stats;
};

// captures execution engine for performing inference
class Engine
{
public:
    Engine(::nvinfer1::ICudaEngine* cudaEngine)
        : m_CudaEngine(cudaEngine)
    {
        m_NumIOTensors = m_CudaEngine->getNbIOTensors();
        for (size_t i = 0; i < m_NumIOTensors; i++)
        {
            auto tensorName = m_CudaEngine->getIOTensorName(i);
            if (m_CudaEngine->getTensorIOMode(tensorName) == nvinfer1::TensorIOMode::kINPUT)
            {
                m_NumInputTensors++;
            }
            else if (m_CudaEngine->getTensorIOMode(tensorName) == nvinfer1::TensorIOMode::kOUTPUT)
            {
                m_NumOutputTensors++;
            }
        }
    }

    ::nvinfer1::ICudaEngine* GetCudaEngine() const
    {
        return m_CudaEngine;
    }

    size_t GetNumIOTensors() const
    {
        return m_NumIOTensors;
    }

    size_t GetNumInputTensors() const
    {
        return m_NumInputTensors;
    }

    size_t GetNumOutputTensors() const
    {
        return m_NumOutputTensors;
    }

private:
    ::nvinfer1::ICudaEngine* m_CudaEngine;
    size_t m_NumInputTensors{0};
    size_t m_NumOutputTensors{0};
    size_t m_NumIOTensors{0};
};

// Create buffers and other execution resources.
// Perform queuing, batching, and manage execution resources.
class Server : public mlperf::SystemUnderTest
{
public:
    Server(std::string name)
        : m_Name(name)
    {
    }
    ~Server() {}

    void Setup(ServerSettings_BEVFormer& settings, ServerParams& params);
    void AddSampleLibrary(qsl::SampleLibraryBEVFormerPtr_t sl)
    {
        m_SampleLibrary = sl;
    }
    void Warmup(double duration);
    void Done();

    std::vector<DevicePtr_t>& GetDevices()
    {
        return m_Devices;
    }

    // SUT virtual interface
    virtual const std::string& Name()
    {
        return m_Name;
    }
    virtual void IssueQuery(const std::vector<mlperf::QuerySample>& samples);
    virtual void FlushQueries();

    // Set query sample response callback to all the devices.
    void SetResponseCallback(std::function<void(::mlperf::QuerySampleResponse* responses,
            std::vector<::mlperf::QuerySampleIndex>& sample_ids, size_t response_count)>
            callback);

private:
    void ProcessSamples();
    void ProcessBatches();

    void IssueBatch(DevicePtr_t device, size_t batchSize, std::deque<mlperf::QuerySample>::iterator begin,
        std::deque<mlperf::QuerySample>::iterator end, cudaStream_t copyStream);

    DevicePtr_t GetNextAvailableDevice(size_t deviceId);
    
    EngineTensors PrepareInputSamples(DevicePtr_t device, size_t batchSize, 
        std::deque<mlperf::QuerySample>::iterator begin, std::deque<mlperf::QuerySample>::iterator end, 
        cudaStream_t copyStream);

    void Reset();

    const std::string m_Name;

    std::vector<DevicePtr_t> m_Devices;
    size_t m_DeviceIndex{0};
    qsl::SampleLibraryBEVFormerPtr_t m_SampleLibrary;

    ServerSettings_BEVFormer m_ServerSettings;

    // Query management
    SyncQueue<mlperf::QuerySample> m_WorkQueue;
    std::vector<std::thread> m_Threads;
    std::atomic<size_t> m_DeviceNum{0};

    // Batch management
    std::vector<std::thread> m_IssueThreads;
    std::atomic<size_t> m_IssueNum{0};
    uint64_t m_IssueCount{0};
};

} // namespace lwis

#endif // __LWIS_BEVFORMER_HPP__
