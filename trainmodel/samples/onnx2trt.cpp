//
//#include "NvInfer.h"
//#include <iostream>
//#include "NvOnnxParser.h"
//
//
//using namespace nvinfer1;
//using namespace nvonnxparser;
//
//
//
//
//class Logger : public ILogger
//{
//    void log(Severity severity, const char* msg) noexcept override
//    {
//        // suppress info-level messages
//        if (severity <= Severity::kWARNING)
//            std::cout << msg << std::endl;
//    }
//} logger;
//
//IBuilder* builder = createInferBuilder(logger);
//bool stronglyTyped = false;
//auto networkFlags = (stronglyTyped)
//? 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED)
//    : 0U;
//INetworkDefinition* network = builder->createNetworkV2(networkFlags);
//IParser* parser = createParser(*network, logger);
//auto modelFile = "E:\s00827220\works\projects\AI_Denoise\model\onnx\grender_model_v1_1008_dy.onnx";
//if (parser->parseFromFile(modelFile, static_cast<int32_t>(ILogger::Severity::kWARNING))) {
//    for (int32_t i = 0; i < parser->getNbErrors(); ++i)
//    {
//        std::cout << parser->getError(i)->desc() << std::endl;
//    }
//}
//
//IBuilderConfig* config = builder->createBuilderConfig();
//config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 2U << 30);
//// config->setMemoryPoolLimit(MemoryPoolType::kTACTIC_SHARED_MEMORY, 48 << 10);
//IHostMemory* serializedModel = builder->buildSerializedNetwork(*network, *config);
//if (!serializedModel) {
//    // ������
//    std::cerr << "ERROR: Failed to serialize the network." << std::endl;
//    return 1;
//}
//// ���ļ���д��
//std::ofstream engineFile("my_engine.engine", std::ios::binary);
//if (!engineFile.is_open()) {
//    // ������
//    std::cerr << "ERROR: Could not open the engine file." << std::endl;
//    return 1;
//}
//
//// д������
//engineFile.write(reinterpret_cast<const char*>(serializedModel->data()), serializedModel->size());
//if (!engineFile) {
//    // ������
//    std::cerr << "ERROR: Failed to write the engine data to file." << std::endl;
//    return 1;
//}
//// �ر��ļ�
//engineFile.close();
//
//delete parser;
//delete network;
//delete config;
//delete builder;
//delete serializedModel;
//save
//
//
//check onnx
//
//
////!
////! \brief Creates the network, configures the builder and creates the network engine
////!
////! \details This function creates the Onnx MNIST network by parsing the Onnx model and builds
////!          the engine that will be used to run MNIST (mEngine)
////!
////! \return true if the engine was created successfully and false otherwise
////!
//bool SampleOnnxMNIST::build()
//{
//    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
//    if (!builder)
//    {
//        return false;
//    }
//
//    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0));
//    if (!network)
//    {
//        return false;
//    }
//
//    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
//    if (!config)
//    {
//        return false;
//    }
//
//    auto parser
//        = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
//    if (!parser)
//    {
//        return false;
//    }
//
//    auto timingCache = SampleUniquePtr<nvinfer1::ITimingCache>();
//
//    auto constructed = constructNetwork(builder, network, config, parser, timingCache);
//    if (!constructed)
//    {
//        return false;
//    }
//
//    // CUDA stream used for profiling by the builder.
//    auto profileStream = samplesCommon::makeCudaStream();
//    if (!profileStream)
//    {
//        return false;
//    }
//    config->setProfileStream(*profileStream);
//
//    SampleUniquePtr<IHostMemory> plan{ builder->buildSerializedNetwork(*network, *config) };
//    if (!plan)
//    {
//        return false;
//    }
//
//    if (timingCache != nullptr && !mParams.timingCacheFile.empty())
//    {
//        samplesCommon::updateTimingCacheFile(
//            sample::gLogger.getTRTLogger(), mParams.timingCacheFile, timingCache.get(), *builder);
//    }
//
//    mRuntime = std::shared_ptr<nvinfer1::IRuntime>(createInferRuntime(sample::gLogger.getTRTLogger()));
//    if (!mRuntime)
//    {
//        return false;
//    }
//
//    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
//        mRuntime->deserializeCudaEngine(plan->data(), plan->size()), samplesCommon::InferDeleter());
//    if (!mEngine)
//    {
//        return false;
//    }
//
//    ASSERT(network->getNbInputs() == 1);
//    mInputDims = network->getInput(0)->getDimensions();
//    ASSERT(mInputDims.nbDims == 4);
//
//    ASSERT(network->getNbOutputs() == 1);
//    mOutputDims = network->getOutput(0)->getDimensions();
//    ASSERT(mOutputDims.nbDims == 2);
//
//    return true;
//}
//
//    class Params {
//    public:
//        Params() {};
//        ~Params() {};
//
//        EngineOptions trtOptions;
//        Path onnxPath, trtPath;
//        Path cacheAlgorithmPath;
//
//        int32_t curHeight{ 0 };
//        int32_t curWidth{ 0 };
//
//        bool readCacheAlgorithm{ true };
//        bool saving{ false };
//        std::string savingPath;
//        std::string cacheAlgorithmSaveingPath;
//    };
//
//    common::Params InitializeDenoiseParams(bool isDynamic, Status type, int32_t height, int32_t width)
//    {
//        common::Params params;
//        AssignCommonParamsTrtOptions(params, isDynamic, type);
//        params.trtOptions.dlaCore = -1;
//        if (isDynamic) {
//            if (type == Status::MOVE) {
//                params.trtOptions.scale = 1;
//                params.curHeight = height * params.trtOptions.scale;
//                params.curWidth = width * params.trtOptions.scale;
//            } else if (type == Status::STILL) {
//                params.trtOptions.scale = 2; // 2为超分的倍数
//                params.curHeight = height * params.trtOptions.scale;
//                params.curWidth = width * params.trtOptions.scale;
//            } else if (type == Status::STILLV2) {
//                params.trtOptions.scale = 1; // 辅助图层都是2x的,因此只做去噪,没有超分即倍数为1
//                params.curHeight = height * 2; // 和V1保持一致,传入的h和w是超分前渲染的h和w,2为超分的倍数
//                params.curWidth = width * 2; // 和V1保持一致,传入的h和w是超分前渲染的h和w,2为超分的倍数
//            }
//        }
//        return params;
//    }
//
//    class DynamicOptions {
//    public:
//        DynamicOptions() {};
//        ~DynamicOptions() {};
//        DynamicOptions(int32_t h, int32_t w)
//            : minHeight(h), optHeight(h), maxHeight(h), minWidth(w), optWidth(w), maxWidth(w) {};
//        DynamicOptions(int32_t minH, int32_t optH, int32_t maxH, int32_t minW, int32_t optW, int32_t maxW)
//            : minHeight(minH), optHeight(optH), maxHeight(maxH), minWidth(minW), optWidth(optW), maxWidth(maxW) {};
//
//        void SetDynamicOptions(const nvinfer1::Dims minDims4, const nvinfer1::Dims optDims4,
//                               const nvinfer1::Dims maxDims4)
//        {
//            minHeight = minDims4.d[2]; // 2
//            optHeight = optDims4.d[2]; // 2
//            maxHeight = maxDims4.d[2]; // 2
//            minWidth = minDims4.d[3]; // 3
//            optWidth = optDims4.d[3]; // 3
//            maxWidth = maxDims4.d[3]; // 3
//        }
//
//        void PrintDynamicOptions() const
//        {
//            logSpace::WriteLoggerContent("minH*W: (" + std::to_string(minHeight) + "* " +
//                std::to_string(minWidth) + "), maxH*W: (" + std::to_string(maxHeight) + "* " +
//                std::to_string(maxWidth) + ")", logSpace::InfoSev::kINFO);
//        }
//
//        bool CheckResolutionIn(int32_t height, int32_t width) const
//        {
//            if (height < minHeight || height > maxHeight || width < minWidth || width > maxWidth) {
//                logSpace::WriteLoggerContent("dynamic shape: (" + std::to_string(height) + ", " +
//                    std::to_string(width) + ") not allowed", logSpace::InfoSev::kERROR);
//                PrintDynamicOptions();
//                return false;
//            }
//            return true;
//        }
//
//        int32_t minHeight{ 176 };
//        int32_t optHeight{ 720 };
//        int32_t maxHeight{ 1440 };
//
//        int32_t minWidth{ 320 };
//        int32_t optWidth{ 1280 };
//        int32_t maxWidth{ 2560 };
//    };
//
////!
////! \enum DataType
////! \brief The type of weights and tensors.
////!
//enum class DataType : int32_t
//{
//    //! 32-bit floating point format.
//    kFLOAT = 0,
//
//    //! IEEE 16-bit floating-point format.
//    kHALF = 1,
//
//    //! 8-bit integer representing a quantized floating-point value.
//    kINT8 = 2,
//
//    //! Signed 32-bit integer format.
//    kINT32 = 3,
//
//    //! 8-bit boolean. 0 = false, 1 = true, other values undefined.
//    kBOOL = 4
//};