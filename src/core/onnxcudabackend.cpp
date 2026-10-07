#include "onnxcudabackend.h"
#include "gpuruntime.h"
#include "../../vendor/onnxruntime/onnxruntime_c_api.h"

#include <QDir>
#include <QFileInfo>
#include <QLibrary>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace vision
{
namespace
{
[[noreturn]] void fail(const QString &message)
{
    throw std::runtime_error(message.toUtf8().constData());
}

template <typename T> using OrtHandle = std::unique_ptr<T, void(ORT_API_CALL *)(T *)>;

// Keep loaded NVIDIA/ORT libraries in the process. Provider worker threads and
// the driver's process-wide state must not outlive an unloaded shared library.
std::unique_ptr<QLibrary> openLibrary(const QString &path)
{
    auto library = std::make_unique<QLibrary>(path);
    library->setLoadHints(QLibrary::ResolveAllSymbolsHint | QLibrary::ExportExternalSymbolsHint |
                         QLibrary::PreventUnloadHint);
    if (!library->load())
        fail(QStringLiteral("无法加载 GPU 动态库 %1：%2").arg(path, library->errorString()));
    return library;
}

QString findLibrary(const QStringList &directories, const QString &name)
{
    for (const QString &directory : directories)
    {
        const QString path = QDir(directory).filePath(name);
        if (QFileInfo(path).isFile())
            return QFileInfo(path).absoluteFilePath();
    }
    return {};
}

} // namespace

struct OnnxCudaBackend::Impl
{
    const OrtApi *api = nullptr;
    OrtEnv *environment = nullptr;
    OrtSession *session = nullptr;
    OrtAllocator *allocator = nullptr; // Owned by ORT, never released here.
    std::vector<std::unique_ptr<QLibrary>> libraries;
    QByteArray inputName;
    std::vector<QByteArray> outputNames;
    QString name;
    bool profiling = false;

    ~Impl()
    {
        if (session)
        {
            if (profiling && allocator)
            {
                char *path = nullptr;
                if (OrtStatus *status = api->SessionEndProfiling(session, allocator, &path))
                    api->ReleaseStatus(status);
                else if (path)
                {
                    if (OrtStatus *status = api->AllocatorFree(allocator, path))
                        api->ReleaseStatus(status);
                }
            }
            api->ReleaseSession(session);
        }
        if (environment)
            api->ReleaseEnv(environment);
    }

    void check(OrtStatus *status) const
    {
        if (!status)
            return;
        const QString message = QString::fromUtf8(api->GetErrorMessage(status)).left(3000);
        api->ReleaseStatus(status);
        fail(QStringLiteral("ONNX Runtime CUDA：%1").arg(message));
    }

    QString tensorName(bool input, size_t index)
    {
        char *raw = nullptr;
        check(input ? api->SessionGetInputName(session, index, allocator, &raw)
                    : api->SessionGetOutputName(session, index, allocator, &raw));
        const QString value = QString::fromUtf8(raw);
        check(api->AllocatorFree(allocator, raw));
        if (value.isEmpty())
            fail(QStringLiteral("ONNX Runtime 模型张量名称为空。"));
        return value;
    }

    void preload(const GpuRuntimePaths &runtime, int deviceIndex)
    {
        auto driver = openLibrary(QStringLiteral("libcuda.so.1"));
        auto cuInit = reinterpret_cast<int (*)(unsigned int)>(driver->resolve("cuInit"));
        auto cuDeviceGetByPCIBusId =
            reinterpret_cast<int (*)(int *, const char *)>(driver->resolve("cuDeviceGetByPCIBusId"));
        auto cuDeviceGetName = reinterpret_cast<int (*)(char *, int, int)>(driver->resolve("cuDeviceGetName"));
        if (!cuInit || cuInit(0) != 0)
            fail(QStringLiteral("NVIDIA 驱动无法初始化 CUDA；请检查驱动和 GPU 是否可用。"));
        libraries.push_back(std::move(driver));

        // SONAME order follows the CUDA 12/cuDNN 9 wheel ELF dependencies:
        // cuSPARSE needs nvJitLink; cuBLAS needs cuBLASLt; cuSOLVER needs both.
        const QStringList cudaLibraries = {
            "libcudart.so.12", "libnvJitLink.so.12", "libnvrtc.so.12", "libcublasLt.so.12",
            "libcublas.so.12", "libcufft.so.11", "libcurand.so.10", "libcusparse.so.12", "libcusolver.so.11"};
        QLibrary *cudart = nullptr;
        for (const QString &soname : cudaLibraries)
        {
            const QString path = findLibrary(runtime.libraryDirectories, soname);
            if (path.isEmpty())
                fail(QStringLiteral("GPU 环境缺少 %1；请重新准备 GPU 环境。").arg(soname));
            libraries.push_back(openLibrary(path));
            if (soname == "libcudart.so.12")
                cudart = libraries.back().get();
        }
        auto cudaGetDeviceCount = reinterpret_cast<int (*)(int *)>(cudart->resolve("cudaGetDeviceCount"));
        auto cudaDeviceGetPCIBusId =
            reinterpret_cast<int (*)(char *, int, int)>(cudart->resolve("cudaDeviceGetPCIBusId"));
        int count = 0;
        if (!cudaGetDeviceCount || cudaGetDeviceCount(&count) != 0 || deviceIndex < 0 || deviceIndex >= count)
            fail(QStringLiteral("CUDA 设备 %1 不可用；请检查设备编号、NVIDIA 驱动和 GPU 环境。").arg(deviceIndex));
        name = QStringLiteral("NVIDIA CUDA #%1").arg(deviceIndex);
        char busId[64] = {}, deviceName[256] = {};
        int driverDevice = -1;
        // Resolve by PCI bus ID so CUDA_VISIBLE_DEVICES does not mislabel GPUs.
        if (cudaDeviceGetPCIBusId && cuDeviceGetByPCIBusId && cuDeviceGetName &&
            cudaDeviceGetPCIBusId(busId, sizeof(busId), deviceIndex) == 0 &&
            cuDeviceGetByPCIBusId(&driverDevice, busId) == 0 &&
            cuDeviceGetName(deviceName, sizeof(deviceName), driverDevice) == 0)
            name = QString::fromUtf8(deviceName);

        // cuDNN lazily opens its components by SONAME. Preload them from the
        // same managed environment, rather than depending on system CUDA.
        const QStringList cudnnLibraries = {
            "libcudnn_graph.so.9", "libcudnn_ops.so.9", "libcudnn_cnn.so.9",
            "libcudnn_engines_runtime_compiled.so.9", "libcudnn_engines_precompiled.so.9",
            "libcudnn_heuristic.so.9", "libcudnn_adv.so.9", "libcudnn.so.9"};
        for (const QString &soname : cudnnLibraries)
        {
            const QString path = findLibrary(runtime.libraryDirectories, soname);
            if (path.isEmpty())
                fail(QStringLiteral("GPU 环境缺少 %1；请重新准备 GPU 环境。").arg(soname));
            libraries.push_back(openLibrary(path));
        }
        libraries.push_back(openLibrary(runtime.onnxLibrary));
        const auto getBase = reinterpret_cast<const OrtApiBase *(ORT_API_CALL *)()>(
            libraries.back()->resolve("OrtGetApiBase"));
        if (!getBase)
            fail(QStringLiteral("所选 GPU 环境不是可用的 ONNX Runtime C API。"));
        const OrtApiBase *base = getBase();
        if (!base || QString::fromUtf8(base->GetVersionString()) != "1.23.2" ||
            !(api = base->GetApi(ORT_API_VERSION)))
            fail(QStringLiteral("GPU 环境需要 ONNX Runtime 1.23.2，与应用 C API 版本一致。"));
    }

    std::vector<cv::Mat> run(const cv::Mat &blob, const std::function<bool()> &cancel)
    {
        if (cancel && cancel())
            fail(QStringLiteral("ONNX CUDA 推理已取消。"));
        if (blob.dims != 4 || blob.type() != CV_32F || !blob.isContinuous())
            fail(QStringLiteral("ONNX CUDA 需要连续的 NCHW FP32 输入。"));
        OrtMemoryInfo *rawInfo = nullptr;
        check(api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &rawInfo));
        OrtHandle<OrtMemoryInfo> info(rawInfo, api->ReleaseMemoryInfo);
        std::vector<int64_t> shape;
        for (int i = 0; i < blob.dims; ++i)
            shape.push_back(blob.size[i]);
        OrtValue *rawInput = nullptr;
        check(api->CreateTensorWithDataAsOrtValue(info.get(), const_cast<uchar *>(blob.data),
                                               blob.total() * sizeof(float), shape.data(), shape.size(),
                                               ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &rawInput));
        OrtHandle<OrtValue> input(rawInput, api->ReleaseValue);
        OrtRunOptions *rawOptions = nullptr;
        check(api->CreateRunOptions(&rawOptions));
        OrtHandle<OrtRunOptions> options(rawOptions, api->ReleaseRunOptions);
        std::vector<const char *> names;
        for (const QByteArray &outputName : outputNames)
            names.push_back(outputName.constData());
        std::vector<OrtValue *> rawOutputs(names.size(), nullptr);
        const char *inputNames[] = {inputName.constData()};
        const OrtValue *inputs[] = {input.get()};
        bool done = false;
        std::mutex completionMutex;
        std::condition_variable completion;
        std::thread watcher;
        if (cancel)
            watcher = std::thread([&] {
                while (true)
                {
                    if (cancel())
                    {
                        if (OrtStatus *status = api->RunOptionsSetTerminate(options.get()))
                            api->ReleaseStatus(status);
                        break;
                    }
                    std::unique_lock<std::mutex> lock(completionMutex);
                    if (completion.wait_for(lock, std::chrono::milliseconds(10), [&] { return done; }))
                        break;
                }
            });
        OrtStatus *status = api->Run(session, options.get(), inputNames, inputs, 1,
                                     names.data(), names.size(), rawOutputs.data());
        {
            std::lock_guard<std::mutex> lock(completionMutex);
            done = true;
        }
        completion.notify_one();
        if (watcher.joinable())
            watcher.join();
        std::vector<OrtHandle<OrtValue>> values;
        for (OrtValue *value : rawOutputs)
            values.emplace_back(value, api->ReleaseValue);
        // Release every partially allocated output before propagating errors.
        check(status);
        if (cancel && cancel())
            fail(QStringLiteral("ONNX CUDA 推理已取消。"));
        std::vector<cv::Mat> outputs;
        for (const auto &value : values)
        {
            const OrtMemoryInfo *memory = nullptr;
            check(api->GetTensorMemoryInfo(value.get(), &memory));
            OrtMemoryInfoDeviceType location;
            api->MemoryInfoGetDeviceType(memory, &location);
            if (location != OrtMemoryInfoDeviceType_CPU)
                fail(QStringLiteral("ONNX CUDA 输出尚未复制到主机内存，已拒绝读取设备指针。"));
            OrtTensorTypeAndShapeInfo *rawShape = nullptr;
            check(api->GetTensorTypeAndShape(value.get(), &rawShape));
            OrtHandle<OrtTensorTypeAndShapeInfo> tensorShape(rawShape, api->ReleaseTensorTypeAndShapeInfo);
            ONNXTensorElementDataType type;
            check(api->GetTensorElementType(tensorShape.get(), &type));
            if (type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
                fail(QStringLiteral("当前 ONNX CUDA 支持 FP32 模型输入和原始 FP32 输出。"));
            size_t dimensions = 0;
            check(api->GetDimensionsCount(tensorShape.get(), &dimensions));
            if (!dimensions || dimensions > 32)
                fail(QStringLiteral("ONNX CUDA 输出张量维度不受支持。"));
            std::vector<int64_t> ortSizes(dimensions);
            std::vector<int> sizes;
            check(api->GetDimensions(tensorShape.get(), ortSizes.data(), dimensions));
            size_t elements = 1;
            for (int64_t dimension : ortSizes)
            {
                if (dimension <= 0 || dimension > std::numeric_limits<int>::max() ||
                    size_t(dimension) > 100000000 / elements)
                    fail(QStringLiteral("ONNX CUDA 输出形状无效或超过 1 亿元素。"));
                elements *= size_t(dimension);
                sizes.push_back(int(dimension));
            }
            void *data = nullptr;
            check(api->GetTensorMutableData(value.get(), &data));
            if (!data)
                fail(QStringLiteral("ONNX CUDA 输出张量为空。"));
            outputs.push_back(cv::Mat(int(dimensions), sizes.data(), CV_32F, data).clone());
        }
        return outputs;
    }
};

OnnxCudaBackend::OnnxCudaBackend() = default;
OnnxCudaBackend::~OnnxCudaBackend() = default;

void OnnxCudaBackend::load(const ModelConfig &config, const QByteArray &model)
{
    m_impl.reset();
    if (m_cancellationCheck && m_cancellationCheck())
        fail(QStringLiteral("ONNX CUDA 加载已取消。"));
    const auto runtime = gpuRuntimePaths();
    if (!runtime.prepared)
        fail(QStringLiteral("GPU 运行环境尚未准备完成：%1").arg(runtime.error));
    auto impl = std::make_unique<Impl>();
    impl->preload(runtime, config.deviceIndex);
    const OrtApi *api = impl->api;
    impl->check(api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "VisionStudio", &impl->environment));
    OrtSessionOptions *rawOptions = nullptr;
    impl->check(api->CreateSessionOptions(&rawOptions));
    OrtHandle<OrtSessionOptions> options(rawOptions, api->ReleaseSessionOptions);
    impl->check(api->SetIntraOpNumThreads(options.get(), 4));
    impl->check(api->SetInterOpNumThreads(options.get(), 1));
    impl->check(api->SetSessionGraphOptimizationLevel(options.get(), ORT_ENABLE_ALL));
    const QByteArray profilePrefix = qEnvironmentVariable("VISION_STUDIO_ORT_PROFILE_PREFIX").toUtf8();
    if (!profilePrefix.isEmpty())
    {
        impl->check(api->EnableProfiling(options.get(), profilePrefix.constData()));
        impl->profiling = true;
    }
    OrtCUDAProviderOptionsV2 *rawCuda = nullptr;
    impl->check(api->CreateCUDAProviderOptions(&rawCuda));
    OrtHandle<OrtCUDAProviderOptionsV2> cuda(rawCuda, api->ReleaseCUDAProviderOptions);
    const QByteArray index = QByteArray::number(config.deviceIndex);
    const char *keys[] = {"device_id", "use_tf32", "cudnn_conv_algo_search", "cudnn_conv_use_max_workspace"};
    const char *values[] = {index.constData(), "0", "HEURISTIC", "0"};
    impl->check(api->UpdateCUDAProviderOptions(cuda.get(), keys, values, 4));
    impl->check(api->SessionOptionsAppendExecutionProvider_CUDA_V2(options.get(), cuda.get()));
    impl->check(api->CreateSessionFromArray(impl->environment, model.constData(), size_t(model.size()),
                                          options.get(), &impl->session));
    impl->check(api->GetAllocatorWithDefaultOptions(&impl->allocator));
    size_t inputs = 0, outputs = 0;
    impl->check(api->SessionGetInputCount(impl->session, &inputs));
    impl->check(api->SessionGetOutputCount(impl->session, &outputs));
    if (inputs != 1 || !outputs || outputs > 16)
        fail(QStringLiteral("ONNX CUDA 当前支持单图像输入，且模型输出数量不超过 16。"));
    impl->inputName = impl->tensorName(true, 0).toUtf8();
    for (size_t i = 0; i < outputs; ++i)
        impl->outputNames.push_back(impl->tensorName(false, i).toUtf8());
    OrtTypeInfo *rawType = nullptr;
    impl->check(api->SessionGetInputTypeInfo(impl->session, 0, &rawType));
    OrtHandle<OrtTypeInfo> type(rawType, api->ReleaseTypeInfo);
    const OrtTensorTypeAndShapeInfo *shape = nullptr;
    impl->check(api->CastTypeInfoToTensorInfo(type.get(), &shape));
    if (!shape)
        fail(QStringLiteral("ONNX CUDA 模型输入不是图像张量。"));
    ONNXTensorElementDataType elementType;
    impl->check(api->GetTensorElementType(shape, &elementType));
    size_t rank = 0;
    impl->check(api->GetDimensionsCount(shape, &rank));
    if (rank != 4 || elementType != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
        fail(QStringLiteral("ONNX CUDA 当前需要 NCHW FP32 模型输入。"));
    int64_t dimensions[4];
    impl->check(api->GetDimensions(shape, dimensions, 4));
    const int sizes[] = {1, config.inputChannels, config.inputSize, config.inputSize};
    for (int i = 0; i < 4; ++i)
        if (dimensions[i] > 0 && dimensions[i] != sizes[i])
            fail(QStringLiteral("ONNX CUDA 输入形状与设置不匹配；请使用模型对应的通道数和输入尺寸。"));
    // Real warmup verifies driver/kernel compatibility before announcing CUDA.
    // It is excluded from per-frame timings and retains the exact input shape.
    impl->run(cv::Mat(4, sizes, CV_32F, cv::Scalar(0)), m_cancellationCheck);
    m_impl = std::move(impl);
}

bool OnnxCudaBackend::loaded() const noexcept { return m_impl && m_impl->session; }
QString OnnxCudaBackend::deviceName() const { return m_impl ? m_impl->name : QString(); }
void OnnxCudaBackend::setCancellationCheck(std::function<bool()> check) { m_cancellationCheck = std::move(check); }
std::vector<cv::Mat> OnnxCudaBackend::infer(const cv::Mat &blob)
{
    if (!loaded())
        fail(QStringLiteral("ONNX CUDA 模型尚未加载。"));
    return m_impl->run(blob, m_cancellationCheck);
}
} // namespace vision
