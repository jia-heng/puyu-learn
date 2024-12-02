// -*- coding: utf-8 -*-
/* *******************************************************************************
Copyright Huawei Technologies Co., Ltd. 2024-2024. All rights reserved
Func: ME AI Model Infer API
Creator: s00827220
********************************************************************************** */

#ifndef INFERCORE_H
#define INFERCORE_H

#include <iostream>
#include <vector>
#include "common/argsParser.h"
#include "common/buffers.h"
#include "common/common.h"

#pragma once

using samplesCommon::SampleUniquePtr;
template <typename T>
SampleUniquePtr<T> makeUnique(T* t) { return SampleUniquePtr<T>{t}; }

struct DemoParams
{
    std::string trtFileName;
    std::vector<std::string> onnxModelDir;
    samplesCommon::OnnxSampleParams OnnxParams;
};

struct DenoiseBuffers {
    // denoiseU layers
    float* color{ nullptr };
    float* depth{ nullptr };
    float* normal{ nullptr };
    float* diffuse{ nullptr };
    float* motion{ nullptr };
    // float* refraction;  v1 not use
    // float* reflection;
    float* predict{ nullptr };
};

class ManagedModelBuffers
{
public:
    static const size_t kINVALID_SIZE_VALUE = ~size_t(0);
    ManagedModelBuffers(std::shared_ptr<nvinfer1::ICudaEngine> engine, int32_t const batchSize = 0);
    std::vector<void*>& getDeviceBindings();
    std::vector<void*> const& getDeviceBindings() const;
    samplesCommon::DeviceBuffer* getManagedDeviceBuffer(std::string const& tensorName);
    samplesCommon::HostBuffer* getManagedHostBuffer(std::string const& tensorName);
    // Overload of resize
    bool reBufferDevice(std::string const& tensorName, void* buffer);
    bool reBufferHost(std::string const& tensorName, void* buffer);
    bool resizeDevice(std::string const& tensorName, const nvinfer1::Dims& dims);

    ~ManagedModelBuffers() = default;

private:
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine;
    int mBatchSize;
    std::vector<std::unique_ptr<samplesCommon::ManagedBuffer>> mManagedBuffers;
    std::vector<void*> mDeviceBindings;
    std::unordered_map<std::string, int32_t> mNames;
};


/* infer class */
class InferImageME
{
public:
    InferImageME(const DemoParams& params);
    bool loadEngine();
    void temporalInit(uint16_t height = 736, uint16_t width = 1280); /* 第一次时必须调用 */
    bool inferDevice(DenoiseBuffers& modelBuffers, uint16_t height = 736, uint16_t width = 1280);
    bool inferHost(DenoiseBuffers& modelBuffers, uint16_t height = 736, uint16_t width = 1280);

private:
    bool setInputBufferDevice(int idx, void* buffer, uint16_t height = 736, uint16_t width = 1280);
    bool setOutBufferDevice(int idx, void* buffer, uint16_t height = 736, uint16_t width = 1280);
    bool buffersPrepareDevice(DenoiseBuffers& modelBuffers, uint16_t height = 736, uint16_t width = 1280);
    bool setInputBufferHostToDevice(int idx, float* buffer, uint16_t height = 736, uint16_t width = 1280);
    bool setOutBufferHost(int idx, float* buffer, uint16_t height = 736, uint16_t width = 1280);
    bool buffersPrepareHost(DenoiseBuffers& modelBuffers, uint16_t height = 736, uint16_t width = 1280);

    // bool validateOutput(int digit);
    DemoParams mParams;
    std::shared_ptr<nvinfer1::IRuntime> mRuntime{ nullptr };
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine{ nullptr };
    SampleUniquePtr<nvinfer1::IExecutionContext> mContext{ nullptr };
    SampleUniquePtr<ManagedModelBuffers> mBuffers{ nullptr };
    SampleUniquePtr<samplesCommon::DeviceBuffer> mTemporal{ nullptr };

    // bool dynamic{ false };
    // DynamicOptions dynamicOptions;
    uint16_t batchSize{ 1 };
    uint16_t mHeight{ 736 };
    uint16_t mWidth{ 1280 };
    nvinfer1::DataType mType{ nvinfer1::DataType::kFLOAT };
};

#endif
