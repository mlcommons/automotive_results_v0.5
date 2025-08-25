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

#ifndef LWIS_BEVFORMER_BUFFERS_H
#define LWIS_BEVFORMER_BUFFERS_H

#include "NvInfer.h"
#include "half.h"
#include <NvInferRuntime.h>
#include <cassert>
#include <cuda_runtime_api.h>
#include <fstream>
#include <glog/logging.h>
#include <iostream>
#include <iterator>
#include <memory>
#include <new>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace lwis
{
#define DEBUG_LWIS_BUFFER 0

/* read in engine file into character array */
inline size_t GetModelStream(std::vector<char>& dst, std::string engineName)
{
    size_t size{0};
    std::ifstream file(engineName, std::ios::binary);
    if (file.good())
    {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        dst.resize(size);
        file.read(dst.data(), size);
        file.close();
    }

    return size;
}

inline unsigned int getElementSize(nvinfer1::DataType t)
{
    switch (t)
    {
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF: return 2;
        case nvinfer1::DataType::kINT8: return 1;
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kBOOL: return 1;
        case nvinfer1::DataType::kUINT8: return 1;
        case nvinfer1::DataType::kFP8: return 1;
        case nvinfer1::DataType::kBF16: return 2;
        case nvinfer1::DataType::kINT64: return 8;
        case nvinfer1::DataType::kINT4: assert(0 && "Int4 size calculation is not supported yet"); break;
        case nvinfer1::DataType::kFP4: assert(0 && "FP4 size calculation is not supported yet"); break;
        // case nvinfer1::DataType::kE8M0: return 1; // TODO: Add support for E8M0
        // Fall through to error
        default: break;
    }
    throw std::runtime_error("Invalid DataType.");
    return 0;
}

// Return m rounded up to nearest multiple of n
inline int roundUp(int m, int n)
{
    return ((m + n - 1) / n) * n;
}

inline size_t volume(const nvinfer1::Dims& d, const nvinfer1::TensorFormat& format)
{
    nvinfer1::Dims d_new = d;
    // Get number of scalars per vector.
    int spv{1};
    int channelDim{-1};
    switch (format)
    {
    case nvinfer1::TensorFormat::kCHW2:
        spv = 2;
        channelDim = d_new.nbDims - 3;
        break;
    case nvinfer1::TensorFormat::kCHW4:
        spv = 4;
        channelDim = d_new.nbDims - 3;
        break;
    case nvinfer1::TensorFormat::kHWC8:
        spv = 8;
        channelDim = d_new.nbDims - 3;
        break;
    case nvinfer1::TensorFormat::kDHWC8:
        spv = 8;
        channelDim = d_new.nbDims - 4;
        break;
    case nvinfer1::TensorFormat::kCHW16:
        spv = 16;
        channelDim = d_new.nbDims - 3;
        break;
    case nvinfer1::TensorFormat::kCHW32:
        spv = 32;
        channelDim = d_new.nbDims - 3;
        break;
    case nvinfer1::TensorFormat::kCDHW32:
        spv = 32;
        channelDim = d_new.nbDims - 4;
        break;
    case nvinfer1::TensorFormat::kLINEAR:
    default:
        spv = 1;
        channelDim = -1;
        break;
    }
    if (spv > 1)
    {
        assert(channelDim >= 0); // Make sure we have valid channel dimension.
        d_new.d[channelDim] = roundUp(d_new.d[channelDim], spv);
    }
    // Handle case where there is no dimension, just a scalar
    if (d_new.nbDims == 0)
    {
        return 1;
    }

    // BEVFormer specific: no need to skip the first dimension, which is batch dim as batch size is always 1
    return std::accumulate(d_new.d, d_new.d + d_new.nbDims, 1, std::multiplies<int64_t>());
}

inline size_t volume(const nvinfer1::Dims& d)
{
    return volume(d, nvinfer1::TensorFormat::kLINEAR);
}

//!
//! \brief  The GenericBuffer class is a templated class for buffers.
//!
//! \details This templated RAII (Resource Acquisition Is Initialization) class handles the
//! allocation,
//!          deallocation, querying of buffers on both the device and the host, using unified memory.
//!          It can handle data of arbitrary types because it stores byte buffers.
//!          The template parameters AllocFunc and FreeFunc are used for the
//!          allocation and deallocation of the buffer.
//!          AllocFunc must be a functor that takes in (void** ptr, size_t size)
//!          and returns bool. ptr is a pointer to where the allocated buffer address should be
//!          stored. size is the amount of memory in bytes to allocate. The boolean indicates
//!          whether or not the memory allocation was successful. FreeFunc must be a functor that
//!          takes in (void* ptr) and returns void. ptr is the allocated buffer address. It must
//!          work with nullptr input.
//!
template <typename AllocFunc, typename FreeFunc>
class GenericBuffer
{
public:
    //!
    //! \brief Construct an empty buffer.
    //!
    GenericBuffer()
        : mByteSize(0)
        , mBuffer(nullptr)
    {
    }

    //!
    //! \brief Construct a buffer with the specified allocation size in bytes.
    //!
    GenericBuffer(size_t size)
        : mByteSize(size)
    {
        if (!allocFn(&mBuffer, mByteSize))
            throw std::bad_alloc();
    }

    GenericBuffer(GenericBuffer&& buf)
        : mByteSize(buf.mByteSize)
        , mBuffer(buf.mBuffer)
    {
        buf.mByteSize = 0;
        buf.mBuffer = nullptr;
    }

    GenericBuffer& operator=(GenericBuffer&& buf)
    {
        if (this != &buf)
        {
            freeFn(mBuffer);
            mByteSize = buf.mByteSize;
            mBuffer = buf.mBuffer;
            buf.mByteSize = 0;
            buf.mBuffer = nullptr;
        }
        return *this;
    }

    //!
    //! \brief Returns pointer to underlying array.
    //!
    void* data()
    {
        return mBuffer;
    }

    //!
    //! \brief Returns pointer to underlying array.
    //!
    const void* data() const
    {
        return mBuffer;
    }

    //!
    //! \brief Returns the size (in bytes) of the buffer.
    //!
    size_t size() const
    {
        return mByteSize;
    }

    ~GenericBuffer()
    {
        freeFn(mBuffer);
    }

private:
    size_t mByteSize;
    void* mBuffer;
    AllocFunc allocFn;
    FreeFunc freeFn;
};

class BufferAllocator
{
public:
    bool operator()(void** ptr, size_t size) const
    {
        return cudaMallocManaged(ptr, size, cudaMemAttachGlobal) == cudaSuccess;
    }
};

class BufferFree
{
public:
    void operator()(void* ptr) const
    {
        cudaFree(ptr);
    }
};

using ManagedBuffer = GenericBuffer<BufferAllocator, BufferFree>;
using EngineTensors = std::vector<void*>;

//!
//! \brief  The BufferManager class handles host and device buffer allocation and deallocation.
//!
//! \details This RAII class handles host and device buffer allocation and deallocation,
//!          memcpy between host and device buffers to aid with inference,
//!          and debugging dumps to validate inference. The BufferManager class is meant to be
//!          used to simplify buffer management and any interactions between buffers and the engine.
//!
class BufferManager
{
public:
    static const size_t kINVALID_SIZE_VALUE = ~size_t(0);

    //!
    //! \brief Create a BufferManager for handling buffer interactions with engine.
    //!
    BufferManager(std::shared_ptr<nvinfer1::ICudaEngine> engine, const int& batchSize, const int profileIdx)
        : mEngine(engine)
        , mBatchSize(batchSize)
        , mProfileIdx(profileIdx)
    {
        // BEVFormer specific: batch size is always 1
        CHECK_EQ(mBatchSize, 1);

        // Each optimization profile owns numIOTensors tensors.
        int32_t numIOTensors = engine->getNbIOTensors();
        int32_t numTensorsTotal = numIOTensors * engine->getNbOptimizationProfiles();        
        int32_t tensorOffset = profileIdx * numIOTensors;

        // Reserve re spaces for each enqueue
        mTensors.resize(numTensorsTotal, nullptr);

        for (size_t IOTensorIndex = 0; IOTensorIndex < numIOTensors; IOTensorIndex++)
        {
            auto tensorName{engine->getIOTensorName(IOTensorIndex)};
            // Create host and device buffers
            bool isInput = engine->getTensorIOMode(tensorName) == nvinfer1::TensorIOMode::kINPUT;
            auto shape = engine->getTensorShape(tensorName);
            auto format = engine->getTensorFormat(tensorName, profileIdx);
            auto dataType = engine->getTensorDataType(tensorName);
            size_t vol = lwis::volume(shape, format);
            size_t elementSize = lwis::getElementSize(dataType);
            size_t tensorSize = batchSize * vol * elementSize;
            size_t singleTensorSize = vol * elementSize;

            // Construct both input and output buffers for the engine
            std::unique_ptr<ManagedBuffer> manBuf{new ManagedBuffer(tensorSize)};
            
            mTensors[tensorOffset + IOTensorIndex] = manBuf->data();
            if (isInput)
            {
                // mInputTensorInfo vector is ordered by tensorIndex
                mInputTensorInfo.emplace_back(std::tuple(IOTensorIndex, tensorName, tensorSize, singleTensorSize, manBuf->data()));
                mTotalInputTensorSize += tensorSize;
            }
            else
            {
                // first output tensor is for feedback (i.e. prev_bev features)
                if (mOutputTensorInfo.size() == 0 and mInternalTensorInfo.size() == 0)
                {
                    mTotalInternalTensorSize += tensorSize;
                    mInternalTensorInfo.emplace_back(std::tuple(IOTensorIndex, tensorName, tensorSize, singleTensorSize, manBuf->data()));
                }
                else
                {
                    // mOutputTensorInfo vector is ordered by tensorIndex
                    mOutputTensorInfo.emplace_back(std::tuple(IOTensorIndex, tensorName, tensorSize, singleTensorSize, manBuf->data()));
                    mTotalOutputTensorSize += tensorSize;
                }
            }
#if DEBUG_LWIS_BUFFER
            std::cout << "Allocated Address: address-" << manBuf->data() << " | ";
            std::cout << (isInput ? "Input" : "Output") << " tensorName: " << tensorName
                      << " | TensorSize: " << tensorSize << std::endl;
#endif
            // Move to the output buffer of the engine
            mManagedBuffers.emplace(tensorName, std::move(manBuf));
        }
#if DEBUG_LWIS_BUFFER
        // BEVFormer specific checks
        CHECK_EQ(mInputTensorInfo.size() + mOutputTensorInfo.size() + mInternalTensorInfo.size(), numIOTensors);
        CHECK_EQ(mInputTensorInfo.size(), 5);
        CHECK_EQ(mInternalTensorInfo.size(), 1);
        CHECK_EQ(mOutputTensorInfo.size(), 2);
        std::cout << "Engine- mManagedBuffers size: " << mManagedBuffers.size()
                    << std::endl;
#endif

    }

    ~BufferManager() = default;

    //! \brief Returns a vector of device tensors for all the engines that you can use directly as
    //!        tensors for the execute and enqueue methods of IExecutionContext.
    //!
    EngineTensors& getTensors()
    {
        return mTensors;
    }

    //!
    //! \brief Returns a vector of device tensors for all the engines.
    //!
    EngineTensors const& getTensors() const
    {
        return mTensors;
    }

    //!
    //! \brief Returns the size of the host and device buffers that correspond to tensorName.
    //!        Returns kINVALID_SIZE_VALUE if no such tensor can be found.
    //!
    size_t size(const std::string& tensorName) const
    {
        return mManagedBuffers.at(tensorName)->size();
    }

    //!
    //! \brief Returns the volume of the host and device buffers that correspond to tensorName.
    //!        Returns kINVALID_SIZE_VALUE if no such tensor can be found.
    //!
    size_t volume(const std::string& tensorName) const
    {
        auto tensorName_s = tensorName.c_str();
        return lwis::volume(
            mEngine->getTensorShape(tensorName_s),
            mEngine->getTensorFormat(tensorName_s, mProfileIdx)
        );
    }

    //!
    //! \brief Returns the elementSize of the host and device buffers that correspond to tensorName.
    //!        Returns kINVALID_SIZE_VALUE if no such tensor can be found.
    //!
    size_t elementSize(const std::string& tensorName) const
    {
        return lwis::getElementSize(mEngine->getTensorDataType(tensorName.c_str()));
    }

    //!
    //! \brief Dump host buffer with specified tensorName to ostream.
    //!        Prints error message to std::ostream if no such tensor can be found.
    //!
    void dumpBuffer(std::ostream& os, const std::string& tensorName)
    {
        void* buf = mManagedBuffers.at(tensorName)->data();
        size_t bufSize = mManagedBuffers.at(tensorName)->size();
        nvinfer1::Dims bufDims = mEngine->getTensorShape(tensorName.c_str());
        size_t rowCount
            = static_cast<size_t>(bufDims.nbDims >= 1 ? bufDims.d[bufDims.nbDims - 1] : mBatchSize);

        os << "[" << mBatchSize;
        for (int i = 0; i < bufDims.nbDims; i++)
        {
            os << ", " << bufDims.d[i];
        }
        os << "]" << std::endl;
        switch (mEngine->getTensorDataType(tensorName.c_str()))
        {
            case nvinfer1::DataType::kINT32: print<int32_t>(os, buf, bufSize, rowCount); break;
            case nvinfer1::DataType::kFLOAT: print<float>(os, buf, bufSize, rowCount); break;
            case nvinfer1::DataType::kHALF: print<half_float::half>(os, buf, bufSize, rowCount); break;
            // case nvinfer1::DataType::kBF16: print<__nv_bfloat16>(os, buf, bufSize, rowCount); break;
            case nvinfer1::DataType::kINT64: print<int64_t>(os, buf, bufSize, rowCount); break;
            case nvinfer1::DataType::kUINT8: print<uint8_t>(os, buf, bufSize, rowCount); break;
            // case nvinfer1::DataType::kFP8: print<__nv_fp8>(os, buf, bufSize, rowCount); break;
            // case nvinfer1::DataType::kE8M0: print<__nv_e8m0>(os, buf, bufSize, rowCount); break;
            case nvinfer1::DataType::kINT8: print<int8_t>(os, buf, bufSize, rowCount); break;
            case nvinfer1::DataType::kBOOL: print<bool>(os, buf, bufSize, rowCount); break;
        }
    }

    //!
    //! \brief Templated print function that dumps buffers of arbitrary type to std::ostream.
    //!        rowCount parameter controls how many elements are on each line.
    //!        A rowCount of 1 means that there is only 1 element on each line.
    //!
    template <typename T>
    void print(std::ostream& os, void* buf, size_t bufSize, size_t rowCount)
    {
        assert(rowCount != 0);
        assert(bufSize % sizeof(T) == 0);
        T* typedBuf = static_cast<T*>(buf);
        size_t numItems = bufSize / sizeof(T);
        for (int i = 0; i < static_cast<int>(numItems); i++)
        {
            // Handle rowCount == 1 case
            if (rowCount == 1 && i != static_cast<int>(numItems) - 1)
                os << typedBuf[i] << std::endl;
            else if (rowCount == 1)
                os << typedBuf[i];
            // Handle rowCount > 1 case
            else if (i % rowCount == 0)
                os << typedBuf[i];
            else if (i % rowCount == rowCount - 1)
                os << " " << typedBuf[i] << std::endl;
            else
                os << " " << typedBuf[i];
        }
    }

    size_t getTotalInputTensorSize() const
    {
        return mTotalInputTensorSize;
    }

    size_t getTotalOutputTensorSize() const
    {
        return mTotalOutputTensorSize;
    }

    size_t getTotalInternalTensorSize() const
    {
        return mTotalInternalTensorSize;
    }

    //!
    //! \brief prepare the contents of input buffers to be read in device asynchronously.
    //!
    void prefetchInputToDeviceAsync(const cudaStream_t& stream = 0, const size_t tensorIndex = 0, const size_t size = 0)
    {
        prefetchBuffers(tensorIndex, true, stream, size);
    }

    //!
    //! \brief prepare the contents of output buffers to be read in host asynchronously.
    //!
    void prefetchOutputToHostAsync(const cudaStream_t& stream = 0, const size_t tensorIndex = 0, const size_t size = 0)
    {
        prefetchBuffers(tensorIndex, false, stream, size);
    }

    //!
    //! \brief copy the contents of internal tensor to the input tensor
    //!
    void copyInternalToInput(const cudaStream_t& stream = 0)
    {
        auto internalTensorIndex = std::get<0>(mInternalTensorInfo[0]);
        auto internalTensorName = std::get<1>(mInternalTensorInfo[0]);
        auto internalTensorSize = std::get<2>(mInternalTensorInfo[0]);
        auto internalSingleTensorSize = std::get<3>(mInternalTensorInfo[0]);
        auto internalTensorBuffer = std::get<4>(mInternalTensorInfo[0]);
        std::string inputTensorName = "prev_bev";
        auto inputTensorIndex = std::get<0>(mInputTensorInfo[0]);
        copyBuffers(inputTensorIndex, true, stream, internalTensorBuffer, internalTensorSize);
    }

    std::vector<std::tuple<size_t, std::string, size_t, size_t, void*>> getInputTensorInfo() const
    {
        return mInputTensorInfo;
    }

    std::vector<std::tuple<size_t, std::string, size_t, size_t, void*>> getOutputTensorInfo() const
    {
        return mOutputTensorInfo;
    }

    std::vector<std::tuple<size_t, std::string, size_t, size_t, void*>> getInternalTensorInfo() const
    {
        return mInternalTensorInfo;
    }

    void* getBuffer(std::string const& tensorName) const
    {
        return mManagedBuffers.at(tensorName)->data();
    }

    void* getBuffer(size_t const tensorIdx) const
    {
        std::string tensorName{mEngine->getIOTensorName(tensorIdx)};
        return mManagedBuffers.at(tensorName)->data();
    }

    void copyBuffers(const size_t tensorIdx, const bool isInput, const cudaStream_t& stream = 0, 
        void* buf = nullptr, const size_t size = 0)
    {
        auto tensorName{mEngine->getIOTensorName(tensorIdx)};
        CHECK((isInput && mEngine->getTensorIOMode(tensorName) == nvinfer1::TensorIOMode::kINPUT)
            || (!isInput && mEngine->getTensorIOMode(tensorName) == nvinfer1::TensorIOMode::kOUTPUT))
            << "Expecting tensor " << tensorIdx << " to be " << (isInput ? "input" : "output")
            << "but get the opposite.";
        size_t numBufferTensors = mManagedBuffers.size();
        CHECK(mEngine->getIOTensorName(tensorIdx) != nullptr);
        CHECK(mManagedBuffers.find(tensorName) != mManagedBuffers.end())
            << "Invalid tensorIdx: " << tensorIdx << " numBufferTensors: " << numBufferTensors;
        
        const size_t byteSize = buf && size ? size : mManagedBuffers.at(tensorName)->size();
        auto srcPtr = isInput ? buf : mManagedBuffers.at(tensorName)->data();
        auto dstPtr = isInput ? mManagedBuffers.at(tensorName)->data() : buf;
        memcpy(dstPtr, srcPtr, byteSize);
    }

    void prefetchBuffers(const size_t tensorIdx, const bool isInput, const cudaStream_t& stream = 0, const size_t size = 0)
    {
        auto tensorName{mEngine->getIOTensorName(tensorIdx)};
        CHECK((isInput && mEngine->getTensorIOMode(tensorName) == nvinfer1::TensorIOMode::kINPUT)
            || (!isInput && mEngine->getTensorIOMode(tensorName) == nvinfer1::TensorIOMode::kOUTPUT))
            << "Expecting tensor " << tensorIdx << " to be " << (isInput ? "input" : "output")
            << "but get the opposite.";
        size_t numBufferTensors = mManagedBuffers.size();
        CHECK(mEngine->getIOTensorName(tensorIdx) != nullptr);
        CHECK(mManagedBuffers.find(tensorName) != mManagedBuffers.end())
            << "Invalid tensorIdx: " << tensorIdx << " numBufferTensors: " << numBufferTensors;

        const size_t byteSize = size ? size : mManagedBuffers.at(tensorName)->size();
        const unsigned int memAttachType = isInput ? cudaMemAttachGlobal : cudaMemAttachHost;
        CHECK_EQ(cudaStreamAttachMemAsync(stream, mManagedBuffers.at(tensorName)->data(), byteSize, memAttachType), cudaSuccess);
    }

private:
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine;              //!< The pointer to the engine
    int mBatchSize;                                              //!< The batch size
    int mProfileIdx;                                             //!< The optimization profile index
    std::unordered_map<std::string, std::unique_ptr<ManagedBuffer>> mManagedBuffers; //!< The map of pointers to managed buffers
    EngineTensors mTensors;                                 //!< The vector of buffers needed for engine execution
    std::vector<std::tuple<size_t, std::string, size_t, size_t, void*>> mInputTensorInfo;  //!< The vector of input tensor info: tensorIndex, tensorName, tensorSize, singleTensorSize, buffer
    std::vector<std::tuple<size_t, std::string, size_t, size_t, void*>> mOutputTensorInfo; //!< The vector of output tensor info: tensorIndex, tensorName, tensorSize, singleTensorSize, buffer
    std::vector<std::tuple<size_t, std::string, size_t, size_t, void*>> mInternalTensorInfo; //!< The vector of output tensor info that are fed back (i.e. prev_bev features) 
    size_t mTotalInputTensorSize{0};
    size_t mTotalOutputTensorSize{0};
    size_t mTotalInternalTensorSize{0};
};

} // namespace lwis

#endif // LWIS_BEVFORMER_BUFFERS_H
