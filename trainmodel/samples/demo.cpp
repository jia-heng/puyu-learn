// -*- coding: utf-8 -*-
/* *******************************************************************************
Copyright Huawei Technologies Co., Ltd. 2024-2024. All rights reserved
Func：ME AI denoise Demo
Creator：s00827220
********************************************************************************** */

// Define TRT entrypoints used in common code
#define DEFINE_TRT_ENTRYPOINTS 1
#define DEFINE_TRT_LEGACY_PARSER_ENTRYPOINT 0

#include "inferCore.h"
#include "dataloader.h"
#include "NvInfer.h"

using Eiotype = float;

DemoParams initializeDemoParams(const samplesCommon::Args& args)
{
    DemoParams params;
    params.trtFileName = "..\\model\\onnx\\temp1011.engine";
    if (args.dataDirs.empty()) // Use default directories if user hasn't provided directory paths
    {
        params.OnnxParams.dataDirs.push_back("E:/s00827220/works/engine/compare/train/720p");
        params.onnxModelDir.push_back("../model/onnx/");
    }
    else // Use the data directory provided by the user
    {
        params.OnnxParams.dataDirs = args.dataDirs;
    }
    params.OnnxParams.onnxFileName = "grender_model_v1_1008_dy.onnx";
    params.OnnxParams.inputTensorNames.push_back("color");
    params.OnnxParams.inputTensorNames.push_back("depth");
    params.OnnxParams.inputTensorNames.push_back("normal");
    params.OnnxParams.inputTensorNames.push_back("diffuse");
    params.OnnxParams.inputTensorNames.push_back("motion");
    params.OnnxParams.inputTensorNames.push_back("temporal");
    params.OnnxParams.outputTensorNames.push_back("ouput1");
    params.OnnxParams.outputTensorNames.push_back("ouput2");
    params.OnnxParams.int8 = args.runInInt8;
    params.OnnxParams.fp16 = args.runInFp16;
    params.OnnxParams.bf16 = args.runInBf16;
    params.OnnxParams.timingCacheFile = args.timingCacheFile;
    return params;
}

bool modelBuffersCudaMalloc(DenoiseBuffers& modelBuffers, uint16_t height = 736, uint16_t width = 1280)
{
    uint16_t batchSize = 1;
    size_t size = batchSize * tensorChannel[COLOR] * height * width * sizeof(Eiotype);
    cudaError_t err = cudaMalloc((void**)&modelBuffers.color, size);
    if (err != cudaSuccess) {
        sample::gLogError << "cudaMalloc failed." << std::endl;
        return false;
    }

    size = batchSize * tensorChannel[DEPTH] * height * width * sizeof(Eiotype);
    err = cudaMalloc((void**)&modelBuffers.depth, size);
    if (err != cudaSuccess) {
        sample::gLogError << "cudaMalloc failed." << std::endl;
        return false;
    }

    size = batchSize * tensorChannel[NORMAL] * height * width * sizeof(Eiotype);
    err = cudaMalloc((void**)&modelBuffers.normal, size);
    if (err != cudaSuccess) {
        sample::gLogError << "cudaMalloc failed." << std::endl;
        return false;
    }

    size = batchSize * tensorChannel[DIFFUSE] * height * width * sizeof(Eiotype);
    err = cudaMalloc((void**)&modelBuffers.diffuse, size);
    if (err != cudaSuccess) {
        sample::gLogError << "cudaMalloc failed." << std::endl;
        return false;
    }

    size = batchSize * tensorChannel[MOTION] * height * width * sizeof(Eiotype);
    err = cudaMalloc((void**)&modelBuffers.motion, size);
    if (err != cudaSuccess) {
        sample::gLogError << "cudaMalloc failed." << std::endl;
        return false;
    }

    size = batchSize * tensorChannel[PREDICT] * height * width * sizeof(Eiotype);
    err = cudaMalloc((void**)&modelBuffers.predict, size);
    if (err != cudaSuccess) {
        sample::gLogError << "cudaMalloc failed." << std::endl;
        return false;
    }

    return true;
}

void modelBuffersCudaFree(DenoiseBuffers& modelBuffers, uint16_t height = 736, uint16_t width = 1280)
{
    cudaFree(modelBuffers.color);
    cudaFree(modelBuffers.depth);
    cudaFree(modelBuffers.normal);
    cudaFree(modelBuffers.diffuse);
    cudaFree(modelBuffers.motion);
    cudaFree(modelBuffers.predict);
}

bool modelBuffersHostMalloc(DenoiseBuffers& modelBuffers, uint16_t height = 736, uint16_t width = 1280)
{
    uint16_t batchSize = 1;
    size_t size = batchSize * tensorChannel[COLOR] * height * width * sizeof(Eiotype);
    modelBuffers.color = (float*)malloc(size);
    if (modelBuffers.color == nullptr) {
        sample::gLogError << "malloc failed." << std::endl;
        return false;
    }

    size = batchSize * tensorChannel[DEPTH] * height * width * sizeof(Eiotype);
    modelBuffers.depth = (float*)malloc(size);
    if (modelBuffers.depth == nullptr) {
        sample::gLogError << "malloc failed." << std::endl;
        return false;
    }

    size = batchSize * tensorChannel[NORMAL] * height * width * sizeof(Eiotype);
    modelBuffers.normal = (float*)malloc(size);
    if (modelBuffers.normal == nullptr) {
        sample::gLogError << "malloc failed." << std::endl;
        return false;
    }

    size = batchSize * tensorChannel[DIFFUSE] * height * width * sizeof(Eiotype);
    modelBuffers.diffuse = (float*)malloc(size);
    if (modelBuffers.diffuse == nullptr) {
        sample::gLogError << "malloc failed." << std::endl;
        return false;
    }

    size = batchSize * tensorChannel[MOTION] * height * width * sizeof(Eiotype);
    modelBuffers.motion = (float*)malloc(size);
    if (modelBuffers.motion == nullptr) {
        sample::gLogError << "malloc failed." << std::endl;
        return false;
    }

    size = batchSize * tensorChannel[PREDICT] * height * width * sizeof(Eiotype);
    modelBuffers.predict = (float*)malloc(size);
    if (modelBuffers.predict == nullptr) {
        sample::gLogError << "malloc failed." << std::endl;
        return false;
    }

    return true;
}

void modelBuffersHostFree(DenoiseBuffers& modelBuffers, uint16_t height = 736, uint16_t width = 1280)
{
    free(modelBuffers.color);
    free(modelBuffers.depth);
    free(modelBuffers.normal);
    free(modelBuffers.diffuse);
    free(modelBuffers.motion);
    free(modelBuffers.predict);
}

void printHelpInfo()
{
    std::cout << "Usage: ./sample_dynamic_reshape [-h or --help] [-d or --datadir=<path to data directory>] "
        "[--timingCacheFile=<path to timing cache file>]"
        << std::endl;
    std::cout << "--help, -h         Display help information" << std::endl;
    std::cout << "--datadir          Specify path to a data directory, overriding the default. This option can be used "
        "multiple times to add multiple directories. If no data directories are given, the default is to use "
        "(data/samples/mnist/, data/mnist/)"
        << std::endl;
    std::cout << "--timingCacheFile  Specify path to a timing cache file. If it does not already exist, it will be "
        << "created." << std::endl;
    std::cout << "--int8             Run in Int8 mode." << std::endl;
    std::cout << "--fp16             Run in FP16 mode." << std::endl;
    std::cout << "--bf16             Run in BF16 mode." << std::endl;
}

void saveExrFileSingle(const std::string& filePath, const std::string& filename, float* data,
    int channel, int height, int width)
{
    Imf::Array2D<Imf::Rgba> pixels(height, width);
    int offset = height * width;
    if (channel == 3) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int index = y * width + x;
                pixels[y][x] = Imf::Rgba(data[index], data[index + offset], data[index + 2 * offset]);
            }
        }
    }
    if (channel == 2) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int index = y * width + x;
                pixels[y][x] = Imf::Rgba(data[index], data[index + offset], 0);
            }
        }
    }
    if (channel == 1) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int index = y * width + x;
                pixels[y][x] = Imf::Rgba(data[index], 0, 0);
            }
        }
    }

    try
    {
        Imf::RgbaOutputFile file(filename.c_str(), width, height, Imf::WRITE_RGBA);
        file.setFrameBuffer(&pixels[0][0], 1, width);
        file.writePixels(height);
    }
    catch (const std::exception& e)
    {
        std::cerr << "error writing image file hello.exr:" << e.what() << std::endl;
        return;
    }
}

int demoInferDevice(int argc, char** argv)
{
    samplesCommon::Args args;
    bool argsOK = samplesCommon::parseArgs(args, argc, argv);
    if (!argsOK)
    {
        sample::gLogError << "Invalid arguments" << std::endl;
        printHelpInfo();
        return EXIT_FAILURE;
    }
    if (args.help)
    {
        printHelpInfo();
        return EXIT_SUCCESS;
    }

    DemoParams params = initializeDemoParams(args);

    InferImageME demo{ params };

    if (!demo.loadEngine()) {
        sample::gLogError << "loadEngine failed." << std::endl;
        return 0;
    }

    DenoiseBuffers modelBuffers;
    if (!modelBuffersCudaMalloc(modelBuffers)) {
        return 0;
    }
    int width = 1280;
    int height = 736;

    float* hostPredict = new float[3 * height * width];
 
    demo.temporalInit();
    if (!demo.inferDevice(modelBuffers)) {
        sample::gLogError << "inferDevice failed." << std::endl;
        return 0;
    }

    cudaMemcpy(hostPredict, modelBuffers.predict, 3 * height * width * sizeof(float), cudaMemcpyDeviceToHost);
    saveExrFileSingle("temp", "output.exr", hostPredict, 3, height, width);

    delete[] hostPredict;
    modelBuffersCudaFree(modelBuffers);
    sample::gLogError << "Engine infer success." << std::endl;
    return 0;
}

int demoInferHost(int argc, char** argv)
{
    samplesCommon::Args args;
    std::mutex mtx;
    bool argsOK = samplesCommon::parseArgs(args, argc, argv);
    if (!argsOK)
    {
        sample::gLogError << "Invalid arguments" << std::endl;
        printHelpInfo();
        return EXIT_FAILURE;
    }
    if (args.help)
    {
        printHelpInfo();
        return EXIT_SUCCESS;
    }

    DemoParams params = initializeDemoParams(args);

    InferImageME demo{ params };

    if (!demo.loadEngine()) {
        sample::gLogError << "loadEngine failed." << std::endl;
        return 0;
    }

    DenoiseBuffers modelBuffers;
    if (!modelBuffersHostMalloc(modelBuffers)) {
        return 0;
    }

    auto data_path = "E:/s00827220/works/engine/compare/train/720p/scene0000/spp00";
    auto save_path = "./test";
    int width = 1280;
    int height = 720;
    auto dataloader = InferDataLoader(data_path, save_path, 64, width, height);
    demo.temporalInit();
    for (int i = 57; i < 61; i++) {
        dataloader.setTensorLayer(modelBuffers.color, COLOR, i);
        dataloader.setTensorLayer(modelBuffers.depth, DEPTH, i);
        dataloader.setTensorLayer(modelBuffers.normal, NORMAL, i);
        dataloader.setTensorLayer(modelBuffers.diffuse, DIFFUSE, i);
        dataloader.setTensorLayer(modelBuffers.motion, MOTION, i);
        //saveExrFileSingle("temp", "color0.exr", modelBuffers.color, 3, 736, width);
        //saveExrFileSingle("temp", "depth0.exr", modelBuffers.depth, 1, 736, width);
        //saveExrFileSingle("temp", "normal0.exr", modelBuffers.normal, 3, 736, width);
        //saveExrFileSingle("temp", "albedo0.exr", modelBuffers.diffuse, 3, 736, width);
        //saveExrFileSingle("temp", "motion0.exr", modelBuffers.motion, 2, 736, width);
        if (!demo.inferHost(modelBuffers)) {
            sample::gLogError << "inferDevice failed." << std::endl;
            return 0;
        }
        
        dataloader.saveExrFile(modelBuffers.predict, i);
    }
    // manageBuffer 析构自动释放；
    // modelBuffersHostFree(modelBuffers);
    sample::gLogError << "Engine infer success." << std::endl;

    return 0;
}

int main(int argc, char** argv)
{
    // demoInferDevice(argc, argv);
    demoInferHost(argc, argv);

    return 0;
}


