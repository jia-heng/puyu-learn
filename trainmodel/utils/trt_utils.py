import os
import stat
import platform
from typing import Tuple
import tensorrt as trt
from tensorrt import ICudaEngine
from tensorrt import Builder, IBuilderConfig, IElementWiseLayer, ILayer, INetworkDefinition, \
    IOptimizationProfile, IReduceLayer, Logger, OnnxParser, Runtime


class TRTLogger(trt.ILogger):
    levels = {
        'debug': [trt.ILogger.INTERNAL_ERROR,
                  trt.ILogger.ERROR,
                  trt.ILogger.WARNING,
                  trt.ILogger.INFO,
                  trt.ILogger.VERBOSE],

        'standard': [trt.ILogger.INTERNAL_ERROR,
                     trt.ILogger.ERROR,
                     trt.ILogger.WARNING,
                     trt.ILogger.INFO],

        'warning': [trt.ILogger.INTERNAL_ERROR,
                    trt.ILogger.ERROR,
                    trt.ILogger.WARNING],

        'error': [trt.ILogger.INTERNAL_ERROR,
                  trt.ILogger.ERROR],

        'silent': [],
    }

    def __init__(self, level='standard', logger=None):
        super(TRTLogger, self).__init__()
        self.level = self.levels[level]
        self.logger = logger

    def log(self, security, msg):
        if self.logger is None:
            if security in self.level:
                print(security, msg)
        else:
            if security == trt.ILogger.INFO:
                self.logger.info(msg)
            elif security == trt.ILogger.WARNING:
                self.logger.warning(msg)
            elif security == trt.ILogger.VERBOSE:
                self.logger.debug(msg)
            elif security == trt.ILogger.INTERNAL_ERROR or security == trt.ILogger.ERROR:
                self.logger.error(msg)
            else:
                self.logger.critical("Unknown TensorRT Logger Error")
                raise RuntimeError("Unknown TensorRT Logger Error")


trt_version = [int(n) for n in trt.__version__.split('.')]

# Array of TensorRT loggers. We need to keep global references to
# the TensorRT loggers that we create to prevent them from being
# garbage collected as those are referenced from C++ code without
# Python knowing about it.

tensorrt_loggers = []

DEFAULT_MAX_WORKSPACE_SIZE = 1 << 30


def get_read_permission():
    plat = platform.system().lower()
    if plat == 'windows':
        flag = os.O_BINARY | os.O_RDONLY  # Windows平台需要添加O_BINARY 才能以二进制模式打开文件
    elif plat == 'linux':
        flag = os.O_RDONLY
    mode = stat.S_IWUSR | stat.S_IRUSR
    return flag, mode


def get_write_permission():
    plat = platform.system().lower()
    if plat == 'windows':
        flag = os.O_BINARY | os.O_CREAT | os.O_WRONLY
    elif plat == 'linux':
        flag = os.O_CREAT | os.O_WRONLY
    mode = stat.S_IWUSR | stat.S_IRUSR
    return flag, mode


def load_tensorrt_engine(filename, logger):
    flag, mode = get_read_permission()
    print('TRT model path: ', filename)
    f = os.fdopen(os.open(filename, flag, mode), 'rb')
    with trt.Runtime(TRTLogger(logger=logger)) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine


def save_engine(engine: ICudaEngine, engine_file_path: str) -> None:
    """
    Serialize TensorRT engine to file.
    :param engine: TensorRT engine
    :param engine_file_path: output path
    """
    flag, mode = get_write_permission()
    with os.fdopen(os.open(engine_file_path, flag, mode), 'wb') as f:
        f.write(engine.serialize())


class TensorRTEngineBuilder():
    def __init__(
            self,
            onnx_file_path: str,
            logger: Logger,
            workspace_size: int,
            mode: str,
    ) -> ICudaEngine:
        """
        Convert ONNX file to TensorRT engine.
        It supports dynamic shape, however it's advised to keep sequence length fix as it hurts performance otherwise.
        Dynamic batch size don't hurt performance and is highly advised.
        :param runtime: global variable shared accross inference call / model building
        :param onnx_file_path: path to the ONNX file
        :param logger: specific logger to TensorRT
        :param workspace_size: GPU memory to use during the building, more is always better. If there is not enough memory,
        some optimization may fail, and the whole conversion process will crash.
        :param fp16: enable FP16 precision, it usually provide a 20-30% boost compared to ONNX Runtime.
        :param int8: enable INT-8 quantization, best performance but model should have been quantized.
        """
        with trt.Builder(logger) as builder:  # type: Builder
            with builder.create_network(
                    flags=1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            ) as network_definition:  # type: INetworkDefinition
                with trt.OnnxParser(network_definition, logger) as parser:  # type: OnnxParser
                    config: IBuilderConfig = builder.create_builder_config()
                    config.max_workspace_size = workspace_size
                    # to enable complete trt inspector debugging, only for TensorRT >= 8.2
                    # disable CUDNN optimizations
                    # config.set_tactic_sources(
                    #     tactic_sources=1 << int(trt.TacticSource.CUBLAS) | 1 << int(trt.TacticSource.CUBLAS_LT)
                    # )
                    if mode == "fp32":
                        print('fp32 mode enabled ......')
                        config.set_flag(trt.BuilderFlag.TF32)
                    if mode == "fp16":
                        print('fp16 mode enabled ......')
                        config.set_flag(trt.BuilderFlag.FP16)
                    config.set_flag(trt.BuilderFlag.DISABLE_TIMING_CACHE)
                    # https://github.com/NVIDIA/TensorRT/issues/1196 (sometimes big diff in output when using FP16)
                    config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
                    with open(onnx_file_path, "rb") as f:
                        parser.parse(f.read())
                    profile: IOptimizationProfile = builder.create_optimization_profile()
                    self.mode = mode
                    self.profile = profile
                    self.config = config
                    self.builder = builder
                    self.network_definition = network_definition

    def set_profile_shape(self,
                          min_shape: Tuple[int, int],
                          optimal_shape: Tuple[int, int],
                          max_shape: Tuple[int, int],
                          ):
        """
        :param min_shape: the minimal shape of input tensors. It's advised to set first dimension (batch size) to 1
        :param optimal_shape: input tensor shape used for optimizations
        :param max_shape: maximal input tensor shape
        """
        for num_input in range(self.network_definition.num_inputs):
            self.profile.set_shape(
                input=self.network_definition.get_input(num_input).name,
                min=min_shape,
                opt=optimal_shape,
                max=max_shape,
            )

    def create_engine(self,
                      runtime: Runtime,
                      timing_cache_file: str,
                      ):
        """
        :return: TensorRT engine to use during inference
        """
        # Create the network
        self.config.add_optimization_profile(self.profile)
        if self.mode == "fp16":
            self.network_definition = self.fix_fp16_network(self.network_definition)
        trt_engine = self.builder.build_serialized_network(self.network_definition, self.config)
        engine: ICudaEngine = runtime.deserialize_cuda_engine(trt_engine)
        if engine is None:
            print("error during engine generation, check error messages above :-(")

        # Speed up the engine build for trt major version >= 8
        # 1. load global timing cache
        if trt_version[0] >= 8:
            if timing_cache_file is not None:
                if os.path.exists(timing_cache_file):
                    flag, mode = get_read_permission()
                    with os.fdopen(os.open(timing_cache_file, flag, mode), 'rb') as f:
                        cache = self.config.create_timing_cache(f.read())
                        self.config.set_timing_cache(cache, ignore_mismatch=False)
                else:
                    cache = self.config.create_timing_cache(b"")
                    self.config.set_timing_cache(cache, ignore_mismatch=False)

        # save global timing cache
        if trt_version[0] >= 8 and timing_cache_file is not None:
            cache = self.config.get_timing_cache()
            with cache.serialize() as buffer:
                flag, mode = get_write_permission()
                with os.fdopen(os.open(timing_cache_file, flag, mode), 'wb') as f:
                    f.write(buffer)
                    f.flush()
                    os.fsync(f)

        return engine

    def fix_fp16_network(self, network_definition: INetworkDefinition) -> INetworkDefinition:
        """
        Mixed precision on TensorRT can generate scores very far from Pytorch because of some operator being saturated.
        Indeed, FP16 can't store very large and very small numbers like FP32.
        Here, we search for some patterns of operators to keep in FP32, in most cases, it is enough to fix the inference
        and don't hurt performances.
        :param network_definition: graph generated by TensorRT after parsing ONNX file (during the model building)
        :return: patched network definition
        """
        # search for patterns which may overflow in FP16 precision, we force FP32 precisions for those nodes
        for layer_index in range(network_definition.num_layers - 1):
            layer: ILayer = network_definition.get_layer(layer_index)
            next_layer: ILayer = network_definition.get_layer(layer_index + 1)
            # POW operation usually followed by mean reduce
            if layer.type == trt.LayerType.ELEMENTWISE and next_layer.type == trt.LayerType.REDUCE:
                # casting to get access to op attribute
                layer.__class__ = IElementWiseLayer
                next_layer.__class__ = IReduceLayer
                if layer.op == trt.ElementWiseOperation.POW:
                    layer.precision = trt.DataType.FLOAT
                    next_layer.precision = trt.DataType.FLOAT
                layer.set_output_type(index=0, dtype=trt.DataType.FLOAT)
                next_layer.set_output_type(index=0, dtype=trt.DataType.FLOAT)
        return network_definition
