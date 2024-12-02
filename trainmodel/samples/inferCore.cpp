// -*- coding: utf-8 -*-
/* *******************************************************************************
Copyright Huawei Technologies Co., Ltd. 2024-2024. All rights reserved
Func: ME AI Model Infer API
Creator: s00827220
********************************************************************************** */
#include "inferCore.h"
#include "dataloader.h"

using Eiotype = float;
/***************************************************
  class ManagedModelBuffers
****************************************************/
ManagedModelBuffers::ManagedModelBuffers(std::shared_ptr<nvinfer1::ICudaEngine> engine, int32_t const batchSize)
    : mEngine(engine), mBatchSize(batchSize)
{
    // Create host and device buffers
    for (int32_t i = 0, e = mEngine->getNbIOTensors(); i < e; i++)
    {
        auto const name = mEngine->getIOTensorName(i);
        mNames[name] = i;
        nvinfer1::DataType type = mEngine->getTensorDataType(name);
        std::unique_ptr<samplesCommon::ManagedBuffer> manBuf{ new samplesCommon::ManagedBuffer() };
        manBuf->deviceBuffer = samplesCommon::DeviceBuffer(type);
        manBuf->hostBuffer = samplesCommon::HostBuffer(type);
        void* deviceBuffer = manBuf->deviceBuffer.data();
        mDeviceBindings.emplace_back(deviceBuffer);
        mManagedBuffers.emplace_back(std::move(manBuf));
    }
}

std::vector<void*>& ManagedModelBuffers::getDeviceBindings()
{
    return mDeviceBindings;
}

std::vector<void*> const& ManagedModelBuffers::getDeviceBindings() const
{
    return mDeviceBindings;
}

samplesCommon::DeviceBuffer* ManagedModelBuffers::getManagedDeviceBuffer(std::string const& tensorName)
{
    auto record = mNames.find(tensorName);
    if (record == mNames.end())
        return nullptr;
    return &(mManagedBuffers[record->second]->deviceBuffer);
}

samplesCommon::HostBuffer* ManagedModelBuffers::getManagedHostBuffer(std::string const& tensorName)
{
    auto record = mNames.find(tensorName);
    if (record == mNames.end())
        return nullptr;
    return &(mManagedBuffers[record->second]->hostBuffer);
}

// Overload of resize
bool ManagedModelBuffers::reBufferDevice(std::string const& tensorName, void* buffer)
{
    auto record = mNames.find(tensorName);
    if (record == mNames.end())
        return false;
    mManagedBuffers[record->second]->deviceBuffer.reBuffer(buffer);
    mDeviceBindings[record->second] = buffer;

    return true;
}

bool ManagedModelBuffers::reBufferHost(std::string const& tensorName, void* buffer)
{
    auto record = mNames.find(tensorName);
    if (record == mNames.end())
        return false;
    mManagedBuffers[record->second]->hostBuffer.reBuffer(buffer);

    return true;
}

bool ManagedModelBuffers::resizeDevice(std::string const& tensorName, const nvinfer1::Dims& dims)
{
    auto record = mNames.find(tensorName);
    if (record == mNames.end())
        return false;
    mManagedBuffers[record->second]->deviceBuffer.resize(dims);
    mDeviceBindings[record->second] = mManagedBuffers[record->second]->deviceBuffer.data();
    return true;
}


/***************************************************
  class InferImageME
****************************************************/
InferImageME::InferImageME(const DemoParams& params)
    : mParams(params)
{
    // check trt file;
    mType = sizeof(Eiotype) == sizeof(float) ? nvinfer1::DataType::kFLOAT : nvinfer1::DataType::kHALF;
}


bool InferImageME::setInputBufferDevice(int idx, void* buffer, uint16_t height, uint16_t width)
{
    auto name = mEngine->getIOTensorName(idx);
    Dims4 inputDims{ batchSize, tensorChannel[idx], height, width };
    CHECK_RETURN_W_MSG(mContext->setInputShape(name, inputDims), false, "Invalid binding dimensions.");
    CHECK_RETURN_W_MSG(mBuffers->reBufferDevice(name, static_cast<void*>(buffer)), false, "Invalid BufferReset.");
    CHECK_RETURN_W_MSG(mContext->setTensorAddress(name, static_cast<void*>(buffer)), false, "Invalid bufferPtr.");

    return true;
}

bool InferImageME::setOutBufferDevice(int idx, void* buffer, uint16_t height, uint16_t width)
{
    auto name = mEngine->getIOTensorName(idx);
    Dims4 inputDims{ batchSize, tensorChannel[idx], height, width };
    CHECK_RETURN_W_MSG(mBuffers->reBufferDevice(name, static_cast<void*>(buffer)), false, "Invalid BufferReset.");
    CHECK_RETURN_W_MSG(mContext->setTensorAddress(name, static_cast<void*>(buffer)), false, "Invalid bufferPtr.");

    return true;
}

bool InferImageME::buffersPrepareDevice(DenoiseBuffers& modelBuffers, uint16_t height, uint16_t width)
{
    CHECK_RETURN_W_MSG(setInputBufferDevice(COLOR, modelBuffers.color, height, width), false, "set color buffer fail.");
    CHECK_RETURN_W_MSG(setInputBufferDevice(DEPTH, modelBuffers.depth, height, width), false, "set depth buffer fail.");
    CHECK_RETURN_W_MSG(setInputBufferDevice(NORMAL, modelBuffers.normal, height, width), false, "set noraml buffer fail.");
    CHECK_RETURN_W_MSG(setInputBufferDevice(DIFFUSE, modelBuffers.diffuse, height, width), false, "set diffuse buffer fail.");
    CHECK_RETURN_W_MSG(setInputBufferDevice(MOTION, modelBuffers.motion, height, width), false, "set motion buffer fail.");
    CHECK_RETURN_W_MSG(setInputBufferDevice(TEMPORAL, mTemporal->data(), height, width), false, "set temporal buffer fail.");
    CHECK_RETURN_W_MSG(setOutBufferDevice(PREDICT, modelBuffers.predict, height, width), false, "set predict buffer fail.");
    CHECK_RETURN_W_MSG(setOutBufferDevice(OUTPUT_TEMP, mTemporal->data(), height, width), false, "set output_temp buffer fail.");

    return true;
}

bool InferImageME::setInputBufferHostToDevice(int idx, float* buffer, uint16_t height, uint16_t width)
{
    auto name = mEngine->getIOTensorName(idx);
    Dims4 inputDims{ batchSize, tensorChannel[idx], height, width };
    auto deviceBufferPtr = mBuffers->getManagedDeviceBuffer(name);
    CHECK_RETURN_W_MSG(mBuffers->reBufferHost(name, static_cast<void*>(buffer)), false, "Invalid BufferReset.");
    CHECK_RETURN_W_MSG(mBuffers->resizeDevice(name, inputDims), false, "Invalid ResizeDevice.");
    CHECK_RETURN_W_MSG(mContext->setInputShape(name, inputDims), false, "Invalid binding dimensions.");
    CHECK_RETURN_W_MSG(mContext->setTensorAddress(name, deviceBufferPtr->data()), false, "Invalid deviceBufferPtr.");
    CHECK(cudaMemcpy(deviceBufferPtr->data(), static_cast<void*>(buffer),
        samplesCommon::volume(inputDims) * samplesCommon::getElementSize(mType), cudaMemcpyHostToDevice));

    return true;
}

bool InferImageME::setOutBufferHost(int idx, float* buffer, uint16_t height, uint16_t width)
{
    auto name = mEngine->getIOTensorName(idx);
    Dims4 inputDims{ batchSize, tensorChannel[idx], height, width };
    auto deviceBufferPtr = mBuffers->getManagedDeviceBuffer(name);
    CHECK_RETURN_W_MSG(mBuffers->reBufferHost(name, static_cast<void*>(buffer)), false, "Invalid BufferReset.");
    CHECK_RETURN_W_MSG(mBuffers->resizeDevice(name, inputDims), false, "Invalid ResizeDevice.");
    CHECK_RETURN_W_MSG(mContext->setTensorAddress(name, deviceBufferPtr->data()), false, "setTensorAddress fail.");

    return true;
}

bool InferImageME::buffersPrepareHost(DenoiseBuffers& modelBuffers, uint16_t height, uint16_t width)
{
    CHECK_RETURN_W_MSG(setInputBufferHostToDevice(COLOR, modelBuffers.color, height, width), false, "set color buffer fail.");
    CHECK_RETURN_W_MSG(setInputBufferHostToDevice(DEPTH, modelBuffers.depth, height, width), false, "set depth buffer fail.");
    CHECK_RETURN_W_MSG(setInputBufferHostToDevice(NORMAL, modelBuffers.normal, height, width), false, "set noraml buffer fail.");
    CHECK_RETURN_W_MSG(setInputBufferHostToDevice(DIFFUSE, modelBuffers.diffuse, height, width), false, "set diffuse buffer fail.");
    CHECK_RETURN_W_MSG(setInputBufferHostToDevice(MOTION, modelBuffers.motion, height, width), false, "set motion buffer fail.");
    CHECK_RETURN_W_MSG(setInputBufferDevice(TEMPORAL, mTemporal->data(), height, width), false, "set temporal buffer fail.");
    CHECK_RETURN_W_MSG(setOutBufferHost(PREDICT, modelBuffers.predict, height, width), false, "set predict buffer fail.");
    CHECK_RETURN_W_MSG(setOutBufferDevice(OUTPUT_TEMP, mTemporal->data(), height, width), false, "set output_temp buffer fail.");

    return true;
}

bool InferImageME::loadEngine()
{
    std::ifstream engineFile(mParams.trtFileName, std::ios::binary);
    if (!engineFile) {
        sample::gLogError << "Error open engine: " + mParams.trtFileName << std::endl;
        return false;
    }
    engineFile.seekg(0, engineFile.end);
    long int fsize = long int(engineFile.tellg());
    engineFile.seekg(0, engineFile.beg);
    char* modelData = new char[fsize];
    engineFile.read(modelData, fsize);
    engineFile.close();

    mRuntime = std::shared_ptr<nvinfer1::IRuntime>(createInferRuntime(sample::gLogger.getTRTLogger()));
    if (!mRuntime) { return false; }

    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        mRuntime->deserializeCudaEngine(modelData, fsize), samplesCommon::InferDeleter());
    if (!mEngine) {
        sample::gLogError << "Error deserializeCudaEngine: " + mParams.trtFileName << std::endl;
        return false;
    }

    mContext = makeUnique(mEngine->createExecutionContext());
    if (!mContext) {
        sample::gLogError << "context build failed." << std::endl;
        return false;
    }

    mBuffers = SampleUniquePtr<ManagedModelBuffers>(new ManagedModelBuffers(mEngine));
    mTemporal = SampleUniquePtr<samplesCommon::DeviceBuffer>(new samplesCommon::DeviceBuffer(mType));
    if (!mBuffers || !mTemporal) {
        sample::gLogError << "buffers build failed." << std::endl;
        return false;
    }

    return true;
}

/* ��һ��ʱ������� */
void InferImageME::temporalInit(uint16_t height, uint16_t width)
{
    Dims4 inputDims{ batchSize, tensorChannel[TEMPORAL], height, width };
    mTemporal->resize(inputDims);
    cudaMemset(mTemporal->data(), 0, mTemporal->nbBytes());
}

bool InferImageME::inferDevice(DenoiseBuffers& modelBuffers, uint16_t height, uint16_t width)
{
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    if (!buffersPrepareDevice(modelBuffers, height, width)) { return false; }
    bool status = mContext->executeV2(mBuffers->getDeviceBindings().data());
    if (!status) { return false; }

    CHECK(cudaStreamSynchronize(stream));

    return true;
}

bool InferImageME::inferHost(DenoiseBuffers& modelBuffers, uint16_t height, uint16_t width)
{
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    if (!buffersPrepareHost(modelBuffers, height, width)) { return false; }
    if (!mContext->allInputDimensionsSpecified()) { return false; }

    bool status = mContext->executeV2(mBuffers->getDeviceBindings().data());
    if (!status) { return false; }

    auto name = mEngine->getIOTensorName(PREDICT);
    auto hostBufferrPtr = mBuffers->getManagedHostBuffer(name);
    auto deviceBufferPtr = mBuffers->getManagedDeviceBuffer(name);
    CHECK(cudaMemcpy(hostBufferrPtr->data(), deviceBufferPtr->data(),
        deviceBufferPtr->nbBytes(), cudaMemcpyDeviceToHost));
    CHECK(cudaStreamSynchronize(stream));
    return true;

    // validateOutput(digit);
}

