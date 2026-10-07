#include "ptbackend.h"
#include "gpuruntime.h"

#include <QBuffer>
#include <QCoreApplication>
#include <QDir>
#include <QElapsedTimer>
#include <QFileInfo>
#include <QImageWriter>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonParseError>
#include <QProcess>
#include <QProcessEnvironment>
#include <QStandardPaths>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace vision
{
namespace
{
constexpr qsizetype responseLimit = 8 * 1024 * 1024;
constexpr qsizetype requestLimit = 100 * 1024 * 1024;

[[noreturn]] void fail(const QString &message)
{
    throw std::runtime_error(message.toUtf8().constData());
}

QString projectDirectory()
{
#ifdef VISION_PROJECT_DIR
    return QString::fromUtf8(VISION_PROJECT_DIR);
#else
    return QDir::currentPath();
#endif
}

QString helperPath()
{
    const QString overridePath = qEnvironmentVariable("VISION_STUDIO_PT_WORKER");
    if (!overridePath.isEmpty())
    {
        if (!QFileInfo(overridePath).isFile())
            fail(QStringLiteral("找不到 PyTorch 推理脚本：%1").arg(overridePath));
        return QFileInfo(overridePath).absoluteFilePath();
    }
    const QString applicationPath = QCoreApplication::applicationDirPath();
    const QStringList candidates = {applicationPath + "/../../scripts/pt_worker.py",
                                    applicationPath + "/../scripts/pt_worker.py",
                                    projectDirectory() + "/scripts/pt_worker.py"};
    for (const QString &path : candidates)
        if (QFileInfo(path).isFile())
            return QDir::cleanPath(QFileInfo(path).absoluteFilePath());
    fail(QStringLiteral("找不到 scripts/pt_worker.py。请保留完整的 VisionStudio 项目目录。"));
}

QString interpreterPath(const QString &script)
{
    const QString overridePath = qEnvironmentVariable("VISION_STUDIO_PYTHON");
    if (!overridePath.isEmpty())
    {
        const QString executable = QFileInfo(overridePath).isAbsolute()
                                       ? overridePath
                                       : QStandardPaths::findExecutable(overridePath);
        if (executable.isEmpty() || !QFileInfo(executable).isFile() || !QFileInfo(executable).isExecutable())
            fail(QStringLiteral("VISION_STUDIO_PYTHON 指定的 Python 无法执行：%1").arg(overridePath));
        return executable;
    }
    const QStringList candidates = {QFileInfo(script).absolutePath() + "/../runtime/bin/python",
                                    projectDirectory() + "/runtime/bin/python",
                                    QDir::homePath() + "/VisionStudio/runtime/bin/python"};
    for (const QString &path : candidates)
        if (QFileInfo(path).isFile() && QFileInfo(path).isExecutable())
            return QDir::cleanPath(QFileInfo(path).absoluteFilePath());
    fail(QStringLiteral("找不到 PyTorch 运行环境。请安装 VisionStudio/runtime，或用 "
                        "VISION_STUDIO_PYTHON 指定已安装 torch、torchvision、ultralytics 的 Python。"));
}

bool finiteNumber(const QJsonValue &value)
{
    return value.isDouble() && std::isfinite(value.toDouble());
}

int operationTimeout(const char *environmentName, int defaultValue)
{
    bool ok = false;
    const int value = qEnvironmentVariableIntValue(environmentName, &ok);
    return ok && value > 0 ? std::min(value, defaultValue) : defaultValue;
}

QString errorMessage(const QJsonObject &message)
{
    QString error = message.value("error").toString();
    if (error.isEmpty())
        error = QStringLiteral("PyTorch 推理进程返回未知错误。");
    const QString detail = message.value("detail").toString();
    if (!detail.isEmpty())
        error += "\n" + detail.left(1500);
    return error;
}
} // namespace

PtBackend::PtBackend() = default;

PtBackend::~PtBackend()
{
    reset();
}

bool PtBackend::loaded() const noexcept
{
    return m_loaded && m_process && m_process->state() == QProcess::Running;
}

void PtBackend::reset() noexcept
{
    m_loaded = false;
    if (m_process && m_process->state() != QProcess::NotRunning)
    {
        m_process->write("{\"command\":\"quit\"}\n");
        m_process->waitForBytesWritten(100);
        m_process->closeWriteChannel();
        if (!m_process->waitForFinished(250))
        {
            m_process->terminate();
            if (!m_process->waitForFinished(250))
            {
                m_process->kill();
                m_process->waitForFinished(250);
            }
        }
    }
    m_process.reset();
    m_stdout.clear();
    m_stderr.clear();
    m_backendName.clear();
}

void PtBackend::collectOutput()
{
    if (!m_process)
        return;
    m_stdout += m_process->readAllStandardOutput();
    m_stderr += m_process->readAllStandardError();
    if (m_stderr.size() > 65536)
        m_stderr = m_stderr.right(65536);
}

[[noreturn]] void PtBackend::processFailure(const QString &message)
{
    collectOutput();
    const QString detail = QString::fromUtf8(m_stderr).trimmed().right(1500);
    const QString fullMessage = detail.isEmpty() ? message : message + "\n" + detail;
    reset();
    fail(fullMessage);
}

QJsonObject PtBackend::receive(int timeoutMs)
{
    QElapsedTimer timer;
    timer.start();
    while (true)
    {
        if (m_cancellationCheck && m_cancellationCheck())
            processFailure(QStringLiteral("PyTorch 推理已取消。"));
        collectOutput();
        if (m_stdout.size() > responseLimit)
            processFailure(QStringLiteral("PyTorch 推理响应过大，进程已停止。"));
        const qsizetype newline = m_stdout.indexOf('\n');
        if (newline >= 0)
        {
            const QByteArray line = m_stdout.left(newline);
            m_stdout.remove(0, newline + 1);
            QJsonParseError error;
            const QJsonDocument response = QJsonDocument::fromJson(line, &error);
            if (error.error != QJsonParseError::NoError || !response.isObject())
                processFailure(QStringLiteral("PyTorch 推理协议响应无效：%1").arg(error.errorString()));
            return response.object();
        }
        if (!m_process || m_process->state() == QProcess::NotRunning)
            processFailure(QStringLiteral("PyTorch 推理进程意外退出。请检查运行环境和模型文件。"));
        const qint64 remaining = timeoutMs - timer.elapsed();
        if (remaining <= 0)
            processFailure(QStringLiteral("PyTorch 操作超时（%1 秒），推理进程已停止。")
                               .arg(timeoutMs / 1000.0, 0, 'g', 4));
        m_process->waitForReadyRead(int(std::min<qint64>(remaining, 250)));
    }
}

void PtBackend::send(const QJsonObject &message)
{
    const QByteArray bytes = QJsonDocument(message).toJson(QJsonDocument::Compact) + '\n';
    if (bytes.size() > requestLimit)
        fail(QStringLiteral("图像过大，无法发送给 PyTorch 推理进程。请缩小原图后重新运行。"));
    if (!m_process || m_process->state() != QProcess::Running || m_process->write(bytes) != bytes.size())
        processFailure(QStringLiteral("无法向 PyTorch 推理进程发送图像。"));
}

void PtBackend::load(const ModelConfig &config)
{
    reset();
    if (config.deviceIndex < 0 || config.deviceIndex > 255)
        fail(QStringLiteral("CUDA 设备编号应在 0 到 255 之间。"));
    if (config.device != ComputeDevice::CPU)
    {
        try
        {
            loadOnDevice(config, true);
            return;
        }
        catch (const std::exception &error)
        {
            reset();
            if (config.device == ComputeDevice::CUDA ||
                (m_cancellationCheck && m_cancellationCheck()))
                throw;
            const QString notice = QStringLiteral("CUDA 初始化失败，已自动使用 CPU：%1")
                                       .arg(QString::fromUtf8(error.what()).left(1500));
            loadOnDevice(config, false);
            m_config.deviceNotice = notice;
            return;
        }
    }
    loadOnDevice(config, false);
}

void PtBackend::loadOnDevice(const ModelConfig &config, bool cuda)
{
    // The UI's spin box rounds the default 1/255 value to eight decimal places.
    if ((config.colorMode == InputColorMode::Color && !config.swapRB) ||
        std::abs(config.scale - 1.0 / 255.0) > 1e-8 || config.meanR != 0 ||
        config.meanG != 0 || config.meanB != 0)
        fail(QStringLiteral(".pt 支持 RGB 彩色或灰度输入。彩色模式请使用 RGB；灰度模式忽略通道交换。"
                            "缩放系数应为 1/255、三通道均值应为 0。"));
    const QString script = helperPath();
    QString python;
    if (cuda)
    {
        const auto runtime = gpuRuntimePaths();
        if (!runtime.prepared)
            fail(QStringLiteral("GPU 运行环境尚未准备完成：%1").arg(runtime.error));
        python = runtime.python;
    }
    else
        python = interpreterPath(script);
    auto process = std::make_unique<QProcess>();
    process->setProcessChannelMode(QProcess::SeparateChannels);
    QProcessEnvironment environment = QProcessEnvironment::systemEnvironment();
    environment.insert("PYTHONUNBUFFERED", "1");
    environment.insert("PYTHONDONTWRITEBYTECODE", "1");
    environment.insert("YOLO_AUTOINSTALL", "false");
    environment.insert("YOLOv5_AUTOINSTALL", "false");
    environment.insert("YOLO_CONFIG_DIR",
                       qEnvironmentVariable("YOLO_CONFIG_DIR", QStandardPaths::writableLocation(
                                                                   QStandardPaths::GenericConfigLocation) +
                                                                   "/vision-studio/ultralytics"));
    process->setProcessEnvironment(environment);
    process->setProgram(python);
    const QString task = config.task == ModelTask::Classification ? "classify"
                         : config.task == ModelTask::YoloV5       ? "v5"
                                                                  : "detect";
    process->setArguments(
        {"-u", script, "--model", QFileInfo(config.modelPath).absoluteFilePath(), "--task", task, "--size",
         QString::number(config.inputSize), "--labels",
         QString::fromUtf8(
             QJsonDocument(QJsonArray::fromStringList(config.labels)).toJson(QJsonDocument::Compact)),
         "--color-mode", config.colorMode == InputColorMode::Grayscale ? "grayscale" : "color",
         "--device", cuda ? "cuda" : "cpu", "--device-index", QString::number(config.deviceIndex)});
    m_process = std::move(process);
    if (m_cancellationCheck && m_cancellationCheck())
        processFailure(QStringLiteral("PyTorch 推理已取消。"));
    m_process->start();
    QElapsedTimer startupTimer;
    startupTimer.start();
    while (!m_process->waitForStarted(100))
    {
        if (m_cancellationCheck && m_cancellationCheck())
            processFailure(QStringLiteral("PyTorch 推理已取消。"));
        if (m_process->state() == QProcess::NotRunning || startupTimer.elapsed() >= 10000)
            processFailure(QStringLiteral("无法启动 PyTorch 运行环境：%1").arg(m_process->errorString()));
    }
    try
    {
        const QJsonObject ready = receive(operationTimeout("VISION_STUDIO_PT_LOAD_TIMEOUT_MS", 90000));
        if (!ready.value("ok").isBool() || !ready.value("ok").toBool())
            fail(errorMessage(ready));
        if (ready.value("event").toString() != "ready" || !finiteNumber(ready.value("protocol")) ||
            ready.value("protocol").toDouble() != 1)
            fail(QStringLiteral("PyTorch 推理脚本与应用的协议版本不匹配。"));
        const QString loadedModelPath = ready.value("model_path").toString();
        const QString expectedPath = QFileInfo(config.modelPath).canonicalFilePath();
        if (loadedModelPath.isEmpty() || expectedPath.isEmpty() ||
            QFileInfo(loadedModelPath).canonicalFilePath() != expectedPath)
            fail(QStringLiteral("PyTorch 推理进程加载的文件与所选模型不一致，已拒绝继续推理。"));
        const QJsonValue channels = ready.value("input_channels");
        if (!finiteNumber(channels) || (channels.toDouble() != 1 && channels.toDouble() != 3))
            fail(QStringLiteral("PyTorch 模型输入通道元数据无效；当前支持 1 或 3 通道模型。"));
        if (config.colorMode == InputColorMode::Color && channels.toInt() != 3)
            fail(QStringLiteral("此模型需要单通道输入；请选择灰度模式后重新运行。"));
        const QString detectedTask = ready.value("task").toString();
        if (detectedTask != "detect" && detectedTask != "classify")
            fail(QStringLiteral("此 .pt 模型任务尚不受支持；请选择目标检测或分类模型。"));
        const QJsonValue labelsValue = ready.value("labels");
        if (!labelsValue.isArray() || labelsValue.toArray().isEmpty() ||
            labelsValue.toArray().size() > 100000)
            fail(QStringLiteral(".pt 模型没有有效的类别元数据。"));
        QStringList labels;
        for (const QJsonValue &value : labelsValue.toArray())
        {
            if (!value.isString() || value.toString().isEmpty() || value.toString().size() > 1000)
                fail(QStringLiteral(".pt 模型类别名称无效。"));
            labels.append(value.toString());
        }
        if (!config.labels.isEmpty() && config.labels.size() != labels.size())
            fail(QStringLiteral(
                     "标签文件有 %1 个类别，.pt 模型有 %2 个类别；请使用匹配的标签文件或模型自带类别。")
                     .arg(config.labels.size())
                     .arg(labels.size()));
        m_config = config;
        const QString actualDevice = ready.value("device").toString(cuda ? QString() : "cpu");
        if (actualDevice != (cuda ? "cuda" : "cpu"))
            fail(QStringLiteral("PyTorch 返回的实际设备与请求不一致，已拒绝继续推理。"));
        const QJsonValue deviceIndex = ready.value("device_index");
        const QJsonObject versions = ready.value("versions").toObject();
        if (!cuda && !deviceIndex.isUndefined() &&
            (!finiteNumber(deviceIndex) || deviceIndex.toDouble() != -1))
            fail(QStringLiteral("PyTorch CPU 设备元数据无效，已拒绝继续推理。"));
        if (cuda && (!finiteNumber(deviceIndex) || deviceIndex.toDouble() != config.deviceIndex ||
                     ready.value("device_name").toString().trimmed().isEmpty() ||
                     versions.value("cuda").toString() != "12.8" ||
                     versions.value("torch").toString() != "2.9.1+cu128"))
            fail(QStringLiteral("PyTorch CUDA 设备元数据无效，已拒绝继续推理。"));
        m_config.resolvedDevice = cuda ? ComputeDevice::CUDA : ComputeDevice::CPU;
        m_config.deviceName = cuda ? ready.value("device_name").toString().left(256) : QStringLiteral("CPU");
        m_config.deviceNotice.clear();
        m_config.inputChannels = channels.toInt();
        const QString layout = ready.value("layout").toString("v8");
        if (detectedTask == "detect" && layout != "v5" && layout != "v8")
            fail(QStringLiteral("PyTorch 检测模型格式元数据无效。"));
        m_config.task = detectedTask == "classify" ? ModelTask::Classification
                        : layout == "v5"           ? ModelTask::YoloV5
                                                   : ModelTask::YoloV8;
        m_config.labels = config.labels.isEmpty() ? labels : config.labels;
        m_backendName = ready.value("backend").toString("PyTorch / CPU");
        m_loaded = true;
    }
    catch (...)
    {
        reset();
        throw;
    }
}

InferenceResult PtBackend::infer(const QImage &image, const QString &source)
{
    if (!loaded())
        fail(QStringLiteral("PyTorch 模型未加载，或推理进程已经退出。请重新运行。"));
    if (image.isNull())
        fail(QStringLiteral("输入图像为空或无法解码。"));
    if (qint64(image.width()) * image.height() > 100000000)
        fail(QStringLiteral("原图超过 1 亿像素，请缩小后重新运行。"));
    if (m_cancellationCheck && m_cancellationCheck())
        processFailure(QStringLiteral("PyTorch 推理已取消。"));
    QElapsedTimer timer;
    timer.start();
    QByteArray encoded;
    QBuffer buffer(&encoded);
    buffer.open(QIODevice::WriteOnly);
    QImageWriter writer(&buffer, "PNG");
    // Qt 6.8 maps [0,100] to zlib levels [0,9]; 11 selects fast, lossless level 1.
    writer.setCompression(11);
    // The Qt preview and ONNX path both reduce source pixels through RGB888.
    // A 16-bit PNG decoded by OpenCV IMREAD_COLOR instead truncates its high
    // byte, which can disagree by one intensity with Qt's 16-to-8 rounding.
    // Send the same 8-bit RGB pixels while retaining the original result image.
    const QImage modelImage = image.convertToFormat(QImage::Format_RGB888);
    if (modelImage.isNull() || !writer.write(modelImage))
        fail(QStringLiteral("无法编码图像并发送给 PyTorch。"));
    if (m_cancellationCheck && m_cancellationCheck())
        processFailure(QStringLiteral("PyTorch 推理已取消。"));
    const qint64 id = ++m_sequence;
    send({{"command", "infer"},
          {"id", double(id)},
          {"image", QString::fromLatin1(encoded.toBase64())},
          {"input_size", m_config.inputSize},
          {"confidence", m_config.confidence},
          {"iou", m_config.iou}});
    const QJsonObject response = receive(operationTimeout("VISION_STUDIO_PT_INFER_TIMEOUT_MS", 120000));
    if (!finiteNumber(response.value("id")) || response.value("id").toDouble() != double(id))
        processFailure(QStringLiteral("PyTorch 推理响应序号不匹配。"));
    if (!response.value("ok").isBool() || !response.value("ok").toBool())
        fail(errorMessage(response));
    if (response.value("task").toString() !=
        (m_config.task == ModelTask::Classification ? "classify" : "detect"))
        processFailure(QStringLiteral("PyTorch 响应任务与已加载模型不匹配。"));
    if (!finiteNumber(response.value("inference_ms")) || response.value("inference_ms").toDouble() < 0 ||
        !response.value("predictions").isArray())
        processFailure(QStringLiteral("PyTorch 推理响应缺少有效的耗时或预测列表。"));
    const QJsonArray predictions = response.value("predictions").toArray();
    if (predictions.size() > (m_config.task == ModelTask::Classification ? 5 : 300))
        processFailure(QStringLiteral("PyTorch 返回了过多预测结果。"));
    InferenceResult result;
    result.image = inputPreviewImage(image, m_config.colorMode);
    result.originalImage = image;
    result.source = source;
    result.modelName = QFileInfo(m_config.modelPath).fileName();
    result.task = m_config.task;
    result.backend = m_backendName;
    result.requestedDevice = m_config.device;
    result.device = m_config.resolvedDevice;
    result.deviceIndex = m_config.resolvedDevice == ComputeDevice::CUDA ? m_config.deviceIndex : -1;
    result.deviceName = m_config.deviceName;
    result.deviceNotice = m_config.deviceNotice;
    result.inferenceMs = response.value("inference_ms").toDouble();
    const QRectF imageBounds(0, 0, image.width(), image.height());
    for (const QJsonValue &value : predictions)
    {
        if (!value.isObject())
            processFailure(QStringLiteral("PyTorch 预测结果格式无效。"));
        const QJsonObject prediction = value.toObject();
        const QJsonValue classValue = prediction.value("class_id");
        const QJsonValue confidenceValue = prediction.value("confidence");
        if (!finiteNumber(classValue) || std::floor(classValue.toDouble()) != classValue.toDouble() ||
            classValue.toDouble() < 0 || classValue.toDouble() >= m_config.labels.size() ||
            !finiteNumber(confidenceValue) || confidenceValue.toDouble() < 0 ||
            confidenceValue.toDouble() > 1)
            processFailure(QStringLiteral("PyTorch 返回了无效类别或置信度。"));
        const int classId = classValue.toInt();
        QRectF box;
        if (m_config.task != ModelTask::Classification)
        {
            const QJsonValue boxValue = prediction.value("box");
            if (!boxValue.isArray() || boxValue.toArray().size() != 4)
                processFailure(QStringLiteral("PyTorch 检测框格式无效。"));
            const QJsonArray coordinates = boxValue.toArray();
            for (const QJsonValue &coordinate : coordinates)
                if (!finiteNumber(coordinate))
                    processFailure(QStringLiteral("PyTorch 检测框含无效数值。"));
            if (coordinates[2].toDouble() <= 0 || coordinates[3].toDouble() <= 0)
                processFailure(QStringLiteral("PyTorch 检测框尺寸无效。"));
            box = QRectF(coordinates[0].toDouble(), coordinates[1].toDouble(), coordinates[2].toDouble(),
                         coordinates[3].toDouble())
                      .intersected(imageBounds);
            if (box.width() <= 0 || box.height() <= 0)
                continue;
        }
        result.predictions.append(
            {classId, m_config.labels.at(classId), float(confidenceValue.toDouble()), box});
    }
    result.totalMs = timer.nsecsElapsed() / 1e6;
    return result;
}

} // namespace vision
