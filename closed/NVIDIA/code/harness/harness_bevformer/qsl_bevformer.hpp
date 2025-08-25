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

#ifndef __QSL_BEVFORMER_HPP__
#define __QSL_BEVFORMER_HPP__

#include <chrono>
#include <deque>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <sstream>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include "NvInfer.h"
#include "NvInferPlugin.h"

#include "numpy.hpp"

#include "logger.h"
#include <glog/logging.h>

#include "loadgen.h"
#include "query_sample_library.h"
#include "test_settings.h"

// For KiTS19, QSL also need to bookkeep the dimensions of each sample
// Rewriting QSL, adding some variables and functions for KiTS19 samples
namespace qsl
{

class SampleLibraryBEVFormer : public mlperf::QuerySampleLibrary
{
public:
    SampleLibraryBEVFormer(std::string name, std::string mapPath, std::vector<std::string> tensorPaths,
        size_t perfSampleCount, size_t padding = 0)
        : m_Name(name)
        , m_PerfSampleCount(perfSampleCount)
        , m_PerfSamplePadding(padding)
        , m_MapPath(mapPath)
        , m_TensorPaths(tensorPaths)
    {
        // Only provide dir path containing all the tensors
        CHECK_EQ(m_TensorPaths.size(), 1);

        // Get input size and allocate memory
        m_SampleSizes.resize(m_NumInputs);
        m_SampleMemory.resize(m_NumInputs);

        // Get number of samples
        // load and read in the sample map
        std::ifstream fs(m_MapPath);
        CHECK(fs) << "Unable to open sample map file: " << m_MapPath;

        char s[1024];
        while (fs.getline(s, 1024))
        {
            std::istringstream iss(s);
            std::vector<std::string> r(
                (std::istream_iterator<std::string>{iss}), std::istream_iterator<std::string>());

            m_FileLabelMap.insert(
                std::make_pair(m_SampleCount, std::make_tuple(r[0], (r.size() > 1 ? std::stoi(r[1]) : 0))));
            m_SampleCount++;
        }

        // as a safety, don't allow the perfSampleCount to be larger than sampleCount.
        m_PerfSampleCount = std::min(m_PerfSampleCount, m_SampleCount);

        AllocateMemory(m_PerfSampleCount, m_PerfSamplePadding);
    }

    ~SampleLibraryBEVFormer()
    {
        DeallocateMemory();
    }

    void AllocateMemory(size_t perfSampleCount, size_t padding)
    {
        for (size_t input_idx = 0; input_idx < m_NumInputs; input_idx++)
        {
            // input_idx == 1 is the prev_bev, fed from the previous frame
            if (input_idx == 1)
            {
                CHECK_EQ(getCategoryNameFromIndex(input_idx), "prev_bev");
                m_SampleSizes[input_idx] = 0;
                m_SampleMemory[input_idx].resize(1);
                continue;
            }
            std::string path = m_TensorPaths[0] + "/" + getCategoryNameFromIndex(input_idx) + "/" + std::get<0>(m_FileLabelMap[0]) + ".npy";
            npy::NpyFile npy(path);
            m_SampleSizes[input_idx] = npy.getTensorSize();
            m_SampleMemory[input_idx].resize(1);
            CHECK_EQ(cudaHostAlloc(&m_SampleMemory[input_idx][0],
                            (perfSampleCount + padding) * m_SampleSizes[input_idx], cudaHostAllocMapped),
                cudaSuccess);
            gLogVerbose << "SampleLibraryBEVFormer: Allocated " << (perfSampleCount + padding) * m_SampleSizes[input_idx] << " bytes for " 
                        << (perfSampleCount + padding) << " samples of " << getCategoryNameFromIndex(input_idx) << std::endl;
        }

        if (m_PerfSampleCountUsed < perfSampleCount)
        {
            m_PerfSampleCountUsed = perfSampleCount;
        }
    }

    void DeallocateMemory()
    {
        for (size_t input_idx = 0; input_idx < m_NumInputs; input_idx++)
        {
            // input_idx == 1 is the prev_bev, fed from the previous frame
            if (input_idx == 1)
            {
                continue;
            }
            CHECK_EQ(cudaFreeHost(m_SampleMemory[input_idx][0]), cudaSuccess);
        }
    }

    const std::string& Name() override
    {
        return m_Name;
    }
    size_t TotalSampleCount() override
    {
        return m_SampleCount;
    }
    size_t PerformanceSampleCount() override
    {
        return m_PerfSampleCount;
    }

    size_t PerformanceSampleCountUsed()
    {
        return m_PerfSampleCountUsed;
    }

    void LoadSamplesToRam(const std::vector<mlperf::QuerySampleIndex>& samples) override
    {
        gLogInfo << "Loading samples to RAM for " << samples.size() << " samples" << std::endl;
        // copy the samples into pinned memory

        if (m_PerfSampleCountUsed < samples.size())
        {
            gLogInfo << "Reallocating memory for " << samples.size() 
                     << " samples as its size is bigger than the current memory allocation" << std::endl;
            DeallocateMemory();
            AllocateMemory(samples.size(), m_PerfSamplePadding);
        }

        for (size_t sampleIndex = 0; sampleIndex < samples.size(); sampleIndex++)
        {
            auto& sampleId = samples[sampleIndex];
            for (size_t input_idx = 0; input_idx < m_NumInputs; input_idx++)
            {
                // input_idx == 1 is the prev_bev, fed from the previous frame
                if (input_idx == 1)
                {
                    continue;
                }
                std::string path = m_TensorPaths[0] + "/" + getCategoryNameFromIndex(input_idx) + "/" + std::get<0>(m_FileLabelMap[sampleId]) + ".npy";
                npy::NpyFile npy(path);
                std::vector<char> data;
                npy.loadAll(data);
                auto sampleAddress = static_cast<int8_t*>(m_SampleMemory[input_idx][0])
                    + sampleIndex * m_SampleSizes[input_idx];
                memcpy((char*) sampleAddress, data.data(), m_SampleSizes[input_idx]);
            }
        }

        // construct sample address map
        for (size_t sampleIndex = 0; sampleIndex < samples.size(); sampleIndex++)
        {
            auto& sampleId = samples[sampleIndex];
            m_SampleAddressMap[sampleId].push_back(std::vector<void*>(m_NumInputs, nullptr));
            for (size_t input_idx = 0; input_idx < m_NumInputs; input_idx++)
            {
                // input_idx == 1 is the prev_bev, fed from the previous frame
                if (input_idx == 1)
                {
                    continue;
                }
                auto hostAddress = static_cast<int8_t*>(m_SampleMemory[input_idx][0]) + sampleIndex * m_SampleSizes[input_idx];
                void * deviceAddress;
                CHECK_EQ(cudaHostGetDevicePointer(&deviceAddress, hostAddress, 0), cudaSuccess);
                m_SampleAddressMap[sampleId].back()[input_idx] = deviceAddress;
            }
        }
        gLogInfo<< "Loaded " << samples.size() << " samples to RAM" << std::endl;
    }

    void UnloadSamplesFromRam(const std::vector<mlperf::QuerySampleIndex>& samples) override
    {
        gLogInfo << "Unloading samples from RAM for " << samples.size() << " samples" << std::endl;
        // due to the removal of freelisting this code is currently a check and not required for functionality.
        for (auto& sampleId : samples)
        {
            {
                auto it = m_SampleAddressMap.find(sampleId);
                CHECK(it != m_SampleAddressMap.end()) << "Sample: " << sampleId << " not allocated properly";
                auto& sampleAddresses = it->second;
                CHECK(!sampleAddresses.empty()) << "Sample: " << sampleId << " not loaded";
                sampleAddresses.pop_back();
                if (sampleAddresses.empty())
                {
                    m_SampleAddressMap.erase(it);
                }
            }
        }

        CHECK(m_SampleAddressMap.empty()) << "Unload did not remove all samples";
        gLogInfo << "Unloaded " << samples.size() << " samples from RAM" << std::endl;
    }

    void* GetSampleAddress(
        mlperf::QuerySampleIndex sample_index, size_t input_idx, size_t device_idx = 0)
    {
        // input_idx == 1 is the prev_bev, fed from the previous frame
        if (input_idx == 1)
        {
            return nullptr;
        }
        auto it = m_SampleAddressMap.find(sample_index);
        CHECK(it != m_SampleAddressMap.end()) << "Sample: " << sample_index << " missing from RAM";
        CHECK(input_idx <= it->second.front().size()) << "invalid input_idx";
        return it->second.front()[input_idx];
    }

    size_t GetSampleSize(size_t input_idx) const
    {
        // input_idx == 1 is the prev_bev, fed from the previous frame
        if (input_idx == 1)
        {
            return 0;
        }
        return (m_SampleSizes.empty() ? 0 : m_SampleSizes[input_idx]);
    }

    int32_t GetNbInputs()
    {
        return m_NumInputs;
    }

protected:
    const size_t m_NumInputs{5};
    int m_NumDevices{1};

private:
    std::string getCategoryNameFromIndex(size_t index)
    {
        return m_SampleIdToFileName[index];
    }

    const std::string m_Name;
    size_t m_PerfSampleCount{0};
    size_t m_PerfSamplePadding{0};
    std::string m_MapPath;
    std::vector<std::string> m_TensorPaths;
    std::vector<size_t> m_SampleSizes;
    std::vector<std::vector<void*>> m_SampleMemory;
    std::vector<std::unique_ptr<npy::NpyFile>> m_NpyFiles;
    size_t m_SampleCount{0};
    size_t m_PerfSampleCountUsed{0};

    // maps sampleId to <fileName, label>
    std::map<mlperf::QuerySampleIndex, std::tuple<std::string, size_t>> m_FileLabelMap;
    // maps sampleId to num_inputs of <address>
    std::map<mlperf::QuerySampleIndex, std::vector<std::vector<void*>>> m_SampleAddressMap;

    nvinfer1::DataType m_Precision{nvinfer1::DataType::kINT8};

    std::map<size_t, std::string> m_SampleIdToFileName{
        {0, "image"},
        {1, "prev_bev"},
        {2, "use_prev_bev"},
        {3, "can_bus"},
        {4, "lidar2img"},
    };
};

typedef std::shared_ptr<qsl::SampleLibraryBEVFormer> SampleLibraryBEVFormerPtr_t;

} // namespace qsl

#endif // __QSL_BEVFORMER_HPP__