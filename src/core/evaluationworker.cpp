#include "evaluationworker.h"
#include "visionengine.h"

#include <QCoreApplication>
#include <QCryptographicHash>
#include <QDateTime>
#include <QDir>
#include <QElapsedTimer>
#include <QFile>
#include <QFileInfo>
#include <QImageReader>
#include <QJsonArray>
#include <QJsonDocument>
#include <QProcess>
#include <QProcessEnvironment>
#include <QSaveFile>
#include <QSet>
#include <QStandardPaths>
#include <QTemporaryDir>
#include <QUuid>
#include <algorithm>
#include <cmath>
#include <functional>
#include <stdexcept>

namespace vision
{
namespace
{
[[noreturn]] void fail(const QString &message)
{
    throw std::runtime_error(message.toUtf8().constData());
}

QString absolute(const QString &path)
{
    return QDir::cleanPath(QFileInfo(path).absoluteFilePath());
}

void saveJson(const QString &path, const QJsonObject &value)
{
    QSaveFile file(path);
    const auto bytes = QJsonDocument(value).toJson(QJsonDocument::Indented);
    if (!file.open(QIODevice::WriteOnly) || file.write(bytes) != bytes.size() || !file.commit())
        fail(QStringLiteral("无法保存评估记录 %1：%2").arg(path, file.errorString()));
}

QJsonObject readJson(const QString &path)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly))
        fail(QStringLiteral("无法读取评估记录 %1：%2").arg(path, file.errorString()));
    QJsonParseError error;
    const auto document = QJsonDocument::fromJson(file.readAll(), &error);
    if (error.error != QJsonParseError::NoError || !document.isObject())
        fail(QStringLiteral("评估记录不是有效 JSON 对象：%1").arg(path));
    return document.object();
}

QString hashFile(const QString &path, const std::function<bool()> &cancelled = {})
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly))
        fail(QStringLiteral("无法读取待校验文件：%1").arg(path));
    QCryptographicHash hash(QCryptographicHash::Sha256);
    while (!file.atEnd())
    {
        if (cancelled && cancelled())
            fail(QStringLiteral("评估已停止。"));
        const auto bytes = file.read(1024 * 1024);
        if (bytes.isEmpty() && file.error() != QFile::NoError)
            fail(QStringLiteral("文件校验失败：%1").arg(path));
        hash.addData(bytes);
    }
    return QString::fromLatin1(hash.result().toHex());
}

void checkSample(const QJsonObject &sample, const std::function<bool()> &cancelled)
{
    for (const auto &pair : {qMakePair(QStringLiteral("image"), QStringLiteral("image_sha256")),
                             qMakePair(QStringLiteral("label"), QStringLiteral("label_sha256"))})
    {
        const auto path = sample.value(pair.first).toString();
        const auto expected = sample.value(pair.second).toString();
        if (path.isEmpty() || expected.isEmpty() || hashFile(path, cancelled) != expected)
            fail(QStringLiteral("数据集文件在预检查后发生变化，请重新导入：%1").arg(path));
    }
}

QString helperPath(const QString &projectRoot)
{
    const auto home = qEnvironmentVariable("VISION_STUDIO_HOME");
    const auto app = QCoreApplication::applicationDirPath();
    const QStringList candidates = {projectRoot + "/scripts/evaluation_worker.py",
        home + "/scripts/evaluation_worker.py", app + "/../../scripts/evaluation_worker.py",
        app + "/../scripts/evaluation_worker.py",
#ifdef VISION_PROJECT_DIR
        QString::fromUtf8(VISION_PROJECT_DIR) + "/scripts/evaluation_worker.py"
#else
        QDir::currentPath() + "/scripts/evaluation_worker.py"
#endif
    };
    for (const auto &path : candidates)
        if (QFileInfo(path).isFile())
            return absolute(path);
    fail(QStringLiteral("找不到 scripts/evaluation_worker.py，请保留完整的安装目录。"));
}

QString interpreterPath(const QString &helper)
{
    const auto override = qEnvironmentVariable("VISION_STUDIO_PYTHON");
    if (!override.isEmpty())
    {
        const auto result = QFileInfo(override).isAbsolute() ? override : QStandardPaths::findExecutable(override);
        if (QFileInfo(result).isFile() && QFileInfo(result).isExecutable())
            return absolute(result);
        fail(QStringLiteral("VISION_STUDIO_PYTHON 指定的 Python 无法执行：%1").arg(override));
    }
    const QString root = QFileInfo(helper).absolutePath() + "/..";
#ifdef Q_OS_WIN
    const QString suffix = "/runtime/python.exe";
#else
    const QString suffix = "/runtime/bin/python";
#endif
    const QStringList candidates = {root + suffix, qEnvironmentVariable("VISION_STUDIO_HOME") + suffix,
#ifdef VISION_PROJECT_DIR
        QString::fromUtf8(VISION_PROJECT_DIR) + suffix,
#endif
        QDir::homePath() + "/VisionStudio" + suffix};
    for (const auto &path : candidates)
        if (QFileInfo(path).isFile() && QFileInfo(path).isExecutable())
            return absolute(path);
    fail(QStringLiteral("找不到评估 Python 环境，请使用完整安装包或指定 VISION_STUDIO_PYTHON。"));
}

// All waits and disk writes occur exclusively on EvaluationWorker::run().
// Stop never waits for this process from the GUI thread.
bool runHelper(const QString &helper, const QStringList &arguments, const QString &directory,
               const std::function<bool()> &cancelled, bool allowCancellation)
{
    QProcess process;
    auto environment = QProcessEnvironment::systemEnvironment();
    environment.insert("PYTHONDONTWRITEBYTECODE", "1");
    environment.insert("PYTHONUNBUFFERED", "1");
    environment.insert("MPLBACKEND", "Agg");
    environment.insert("MPLCONFIGDIR", directory + "/plot-config");
    environment.insert("YOLO_AUTOINSTALL", "false");
    environment.insert("YOLOv5_AUTOINSTALL", "false");
    process.setProcessEnvironment(environment);
    process.setWorkingDirectory(directory);
    process.setProcessChannelMode(QProcess::MergedChannels);
    QFile log(directory + "/helper.log");
    if (!log.open(QIODevice::WriteOnly | QIODevice::Append))
        fail(QStringLiteral("无法创建评估日志：%1").arg(log.errorString()));
    QStringList args{helper};
    args.append(arguments);
    process.start(interpreterPath(helper), args);
    QElapsedTimer timer;
    timer.start();
    QByteArray detail;
    auto collect = [&]
    {
        const auto output = process.readAll();
        if (!output.isEmpty())
        {
            if (log.write(output) != output.size())
                fail(QStringLiteral("评估日志写入失败：%1").arg(log.errorString()));
            detail = (detail + output).right(8192);
        }
    };
    while (process.state() != QProcess::NotRunning)
    {
        process.waitForReadyRead(40);
        collect();
        if (allowCancellation && cancelled())
        {
            process.terminate();
            if (!process.waitForFinished(300))
            {
                process.kill();
                process.waitForFinished(1000);
            }
            collect();
            return false;
        }
        if (timer.elapsed() > 15 * 60 * 1000)
        {
            process.kill();
            process.waitForFinished(1000);
            collect();
            fail(QStringLiteral("评估辅助进程超时，日志已保留。"));
        }
    }
    collect();
    if (process.exitStatus() != QProcess::NormalExit || process.exitCode() != 0 ||
        process.error() == QProcess::FailedToStart)
    {
        const QString reason = process.error() == QProcess::FailedToStart ? process.errorString() :
            process.exitStatus() != QProcess::NormalExit ? QStringLiteral("进程异常退出") :
                QStringLiteral("退出代码 %1").arg(process.exitCode());
        fail(QStringLiteral("评估辅助进程失败：%1\n%2")
                 .arg(reason, QString::fromUtf8(detail).trimmed()));
    }
    return true;
}

QJsonArray pathsJson(const QStringList &paths)
{
    QJsonArray result;
    for (const auto &path : paths)
        result.append(path);
    return result;
}

QJsonObject predictionJson(const InferenceResult &result, const QJsonObject &sample, int index)
{
    QJsonArray predictions;
    for (const auto &p : result.predictions)
        predictions.append(QJsonObject{{"class_id", p.classId}, {"label", p.label},
            {"confidence", p.confidence},
            {"box", QJsonArray{p.box.left(), p.box.top(), p.box.right(), p.box.bottom()}}});
    QJsonObject value = sample;
    value.insert("index", index);
    value.insert("width", result.image.width());
    value.insert("height", result.image.height());
    value.insert("predictions", predictions);
    value.insert("inference_ms", result.inferenceMs);
    value.insert("total_ms", result.totalMs);
    value.insert("backend", result.backend);
    value.insert("device", result.device == ComputeDevice::CUDA ? "cuda" : "cpu");
    value.insert("device_index", result.deviceIndex);
    value.insert("device_name", result.deviceName);
    return value;
}
} // namespace

EvaluationWorker::EvaluationWorker(QObject *parent) : QThread(parent)
{
}

void EvaluationWorker::configureDataset(const QString &yaml, const QString &split, const QString &metadata,
                                       const QString &projectRoot)
{
    if (isRunning())
        return;
    operation_ = Operation::Dataset;
    yaml_ = yaml;
    split_ = split;
    metadata_ = metadata;
    projectRoot_ = projectRoot;
    stop_.store(false, std::memory_order_release);
}

void EvaluationWorker::configureEvaluation(const EvaluationRequest &request)
{
    if (isRunning())
        return;
    operation_ = Operation::Evaluation;
    request_ = request;
    projectRoot_ = request.projectRoot;
    stop_.store(false, std::memory_order_release);
}

void EvaluationWorker::requestStop() noexcept
{
    stop_.store(true, std::memory_order_release);
}

void EvaluationWorker::configureReanalysis(const QString &source, const QJsonObject &options,
                                          const QString &outputRoot, const QString &projectRoot)
{
    if (isRunning()) return;
    operation_ = Operation::Reanalysis;
    analysisSource_ = source;
    analysisOptions_ = options;
    analysisOutputRoot_ = outputRoot;
    projectRoot_ = projectRoot;
    stop_.store(false, std::memory_order_release);
}

void EvaluationWorker::configureExternalImport(const QString &source, const QJsonObject &options,
                                              const QString &outputRoot, const QString &projectRoot)
{
    if (isRunning()) return;
    configureReanalysis(source, options, outputRoot, projectRoot);
    operation_ = Operation::ExternalImport;
}

QJsonObject EvaluationWorker::modelConfigJson(const ModelConfig &config)
{
    QJsonArray labels, classes;
    for (const auto &label : config.labels)
        labels.append(label);
    for (int id : config.classFilter)
        classes.append(id);
    return {{"model_path", absolute(config.modelPath)}, {"labels", labels},
        {"class_filter", classes}, {"model_class_count", config.modelClassCount},
        {"task", config.task == ModelTask::YoloV5 ? "yolov5" :
                  config.task == ModelTask::YoloV8 ? "yolov8" : "classification"},
        {"input_size", config.inputSize}, {"confidence", config.confidence},
        {"nms_iou", config.iou}, {"swap_rb", config.swapRB},
        {"color_mode", config.colorMode == InputColorMode::Grayscale ? "grayscale" : "color"},
        {"input_channels", config.inputChannels}, {"scale", config.scale},
        {"mean_r", config.meanR}, {"mean_g", config.meanG}, {"mean_b", config.meanB},
        {"device", config.device == ComputeDevice::CUDA ? "cuda" :
                    config.device == ComputeDevice::Auto ? "auto" : "cpu"},
        {"device_index", config.deviceIndex},
        {"resolved_device", config.resolvedDevice == ComputeDevice::CUDA ? "cuda" : "cpu"},
        {"device_name", config.deviceName}, {"device_notice", config.deviceNotice},
        {"stereo_view", "full"}, {"exif_auto_transform", false},
        {"max_detections", config.maxDetections}, {"max_candidates", 30000},
        {"cpu_threads", config.cpuThreads}, {"resolved_cpu_threads", config.resolvedCpuThreads},
        {"nms_class_aware", true}, {"nms_before_threshold_scan", true}};
}

QJsonObject EvaluationWorker::readSample(const QString &directory, int index, QString *error)
{
    if (error)
        error->clear();
    try
    {
        if (index < 0)
            fail(QStringLiteral("评估样本索引无效。"));
        const QString base = directory + "/samples/" + QString::number(index);
        QFile checksumFile(base + ".sha256");
        if (!checksumFile.open(QIODevice::ReadOnly))
            fail(QStringLiteral("评估样本缺少校验文件：%1").arg(base));
        const auto expected = checksumFile.read(66);
        if (checksumFile.error() != QFile::NoError || expected.size() != 65 || expected.back() != '\n' ||
            !std::all_of(expected.cbegin(), expected.cbegin() + 64, [](char c) {
                return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
            }))
            fail(QStringLiteral("评估样本校验文件格式损坏：%1").arg(base));
        QFile sampleFile(base + ".json");
        if (!sampleFile.open(QIODevice::ReadOnly))
            fail(QStringLiteral("无法读取评估样本缓存：%1").arg(base));
        const auto bytes = sampleFile.readAll();
        if (sampleFile.error() != QFile::NoError ||
            QCryptographicHash::hash(bytes, QCryptographicHash::Sha256).toHex() != expected.left(64))
            fail(QStringLiteral("评估样本缓存校验失败，文件可能被修改或错拷贝：%1").arg(base));
        QJsonParseError parseError;
        const auto document = QJsonDocument::fromJson(bytes, &parseError);
        if (parseError.error != QJsonParseError::NoError || !document.isObject())
            fail(QStringLiteral("评估样本缓存不是有效 JSON 对象：%1").arg(base));
        return document.object(); // Parse exactly the bytes that were hashed, without reopening the path.
    }
    catch (const std::exception &exception)
    {
        if (error)
            *error = QString::fromUtf8(exception.what());
        return {};
    }
}

void EvaluationWorker::run()
{
    const auto stopping = [this] { return stop_.load(std::memory_order_acquire); };
    QString directory;
    QJsonObject status;
    try
    {
        const QString helper = helperPath(projectRoot_);
        if (operation_ == Operation::Reanalysis || operation_ == Operation::ExternalImport)
        {
            directory = analysisOutputRoot_ + "/evaluations/" +
                QDateTime::currentDateTimeUtc().toString("yyyyMMdd-HHmmss-zzz") + "-analysis-" +
                QUuid::createUuid().toString(QUuid::WithoutBraces).left(8);
            if (!QDir().mkpath(directory)) fail(QStringLiteral("无法创建新分析报告目录。"));
            status = {{"schema_version", 1}, {"status", "analyzing"}, {"completed", 0}};
            saveJson(directory + "/status.json", status);
            saveJson(directory + "/analysis-options.json", analysisOptions_);
            const QString snapshotHelper = QFileInfo(helper).absolutePath() +
                (operation_ == Operation::ExternalImport ? "/evaluation_import.py" : "/evaluation_snapshot.py");
            if (!QFileInfo(snapshotHelper).isFile()) fail(QStringLiteral("找不到历史快照分析程序。"));
            emit progress(0, 0, QStringLiteral("正在校验历史快照并重新分析，不执行模型推理…"));
            if (!runHelper(snapshotHelper, {"--source", analysisSource_, "--options",
                directory + "/analysis-options.json", "--output", directory}, directory, stopping, true))
            {
                status.insert("status", "cancelled_analysis");
                saveJson(directory + "/status.json", status);
                return;
            }
            const auto report = readJson(directory + "/report.json");
            emit reportReady(report, directory);
            return;
        }
        if (operation_ == Operation::Dataset)
        {
            QTemporaryDir temp(QDir::tempPath() + "/vision-studio-dataset-XXXXXX");
            if (!temp.isValid())
                fail(QStringLiteral("无法创建数据集预检查目录。"));
            saveJson(temp.path() + "/request.json", {{"dataset_yaml", absolute(yaml_)},
                {"split", split_}, {"metadata_path", metadata_}});
            emit progress(0, 0, QStringLiteral("正在检查 YOLO 数据集与标注…"));
            if (!runHelper(helper, {"prepare", "--request", temp.path() + "/request.json",
                "--output", temp.path() + "/manifest.json"}, temp.path(), stopping, true))
                return;
            auto manifest = readJson(temp.path() + "/manifest.json");
            if (!stopping())
                emit datasetReady(manifest);
            return;
        }

        const auto &request = request_;
        if (request.config.task == ModelTask::Classification)
            fail(QStringLiteral("数据集评估首版仅支持目标检测，不支持图像分类。"));
        if (request.manifest.value("requires_attention").toBool())
            fail(QStringLiteral("数据集有缺失或无效标注，请修复预检查问题并重新导入后再评估。"));
        if (request.files.isEmpty())
            fail(QStringLiteral("评估范围中没有图片。"));
        if (request.modelClassId < 0 || request.datasetClassId < 0 ||
            request.datasetClassId >= request.manifest.value("names").toArray().size())
            fail(QStringLiteral("评估类别映射超出数据集类别范围。"));
        if (!std::isfinite(request.scoreFloor) || request.scoreFloor <= 0 || request.scoreFloor > 1 ||
            !std::isfinite(request.confidence) || request.confidence < request.scoreFloor || request.confidence > 1 ||
            !std::isfinite(request.matchIou) || request.matchIou <= 0 || request.matchIou > 1 ||
            !std::isfinite(request.scanStart) || !std::isfinite(request.scanEnd) ||
            !std::isfinite(request.scanStep) || request.scanStart < request.scoreFloor ||
            request.scanEnd > 1 || request.scanStart > request.scanEnd || request.scanStep <= 0 ||
            (request.scanEnd - request.scanStart) / request.scanStep > 10000)
            fail(QStringLiteral("评估阈值无效：分析阈值及扫描范围不能低于采集最低置信度。"));
        if (request.batchSize < 1 || request.batchSize > 32 || request.figureDpi < 72 || request.figureDpi > 1200 ||
            request.figureFormats.isEmpty())
            fail(QStringLiteral("评估 batch 须为 1～32；图表分辨率须为 72～1200 DPI，且至少选择一种格式。"));
        for (const auto &format : request.figureFormats)
            if (format != "png" && format != "svg" && format != "pdf")
                fail(QStringLiteral("评估图表仅支持 PNG／SVG／PDF。"));

        QHash<QString, QJsonObject> byImage;
        QStringList manifestImages;
        for (const auto &value : request.manifest.value("samples").toArray())
        {
            const auto sample = value.toObject();
            const auto path = absolute(sample.value("image").toString());
            if (byImage.contains(path))
                fail(QStringLiteral("数据集包含重复图片：%1").arg(path));
            byImage.insert(path, sample);
            manifestImages.append(path);
        }
        QSet<QString> seen;
        QStringList files;
        for (const auto &file : request.files)
        {
            const auto path = absolute(file);
            if (!byImage.contains(path) || seen.contains(path))
                fail(QStringLiteral("评估范围含有重复或不属于当前数据集的图片：%1").arg(path));
            seen.insert(path);
            files.append(path);
        }

        const QString parent = request.outputRoot + "/evaluations";
        directory = parent + "/" + QDateTime::currentDateTimeUtc().toString("yyyyMMdd-HHmmss-zzz") +
                    "-" + QUuid::createUuid().toString(QUuid::WithoutBraces).left(8);
        if (!QDir().mkpath(directory + "/samples"))
            fail(QStringLiteral("无法创建评估输出目录：%1").arg(directory));
        status = {{"schema_version", 1}, {"status", "running"}, {"completed", 0},
                  {"total", files.size()}, {"directory", directory}};
        saveJson(directory + "/status.json", status);
        saveJson(directory + "/manifest.json", request.manifest);

        const auto modelSha = hashFile(request.config.modelPath);
        const QFileInfo modelInfo(request.config.modelPath);
        const QString modelIdentity = modelInfo.absoluteFilePath() + "|" + QString::number(modelInfo.size()) +
            "|" + QString::number(modelInfo.lastModified().toMSecsSinceEpoch());
        ModelConfig config = request.config;
        config.confidence = float(request.scoreFloor);
        config.classFilter = {request.modelClassId};
        QJsonObject parameters{{"schema_version", 1}, {"model_path", absolute(config.modelPath)},
            {"model_sha256", modelSha}, {"model_identity", modelIdentity}, {"model_class_id", request.modelClassId},
            {"dataset_class_id", request.datasetClassId}, {"confidence", request.confidence},
            {"score_floor", request.scoreFloor}, {"match_iou", request.matchIou},
            {"scan_start", request.scanStart}, {"scan_end", request.scanEnd},
            {"scan_step", request.scanStep}, {"sample_images", pathsJson(files)},
            {"requested_batch_size", request.batchSize}, {"figure_formats", pathsJson(request.figureFormats)},
            {"figure_dpi", request.figureDpi},
            {"full_split", files == manifestImages}, {"config", modelConfigJson(config)}};
        saveJson(directory + "/request.json", parameters);
        QFile predictions(directory + "/predictions.jsonl");
        if (!predictions.open(QIODevice::WriteOnly | QIODevice::NewOnly))
            fail(QStringLiteral("无法创建评估预测缓存：%1").arg(predictions.errorString()));
        emit progress(0, files.size(), QStringLiteral("正在加载评估模型…"));
        int completed = 0;
        {
            VisionEngine engine;
            engine.setCancellationCheck(stopping);
            if (!stopping())
            {
                try
                {
                    engine.load(config);
                    config = engine.config();
                    if (config.task == ModelTask::Classification)
                        fail(QStringLiteral("当前模型实际为图像分类模型，不能用于目标检测评估。"));
                    parameters.insert("config", modelConfigJson(config));
                    parameters.insert("model_class_count", config.modelClassCount);
                    const int batchSize = std::min(request.batchSize, engine.maximumBatchSize());
                    QJsonArray executedBatches;
                    parameters.insert("effective_batch_size", std::min(batchSize, int(files.size())));
                    parameters.insert("batch_notice", request.batchSize > engine.maximumBatchSize() ?
                        engine.batchNotice() : QString());
                    saveJson(directory + "/request.json", parameters);
                    if (hashFile(config.modelPath) != modelSha)
                        fail(QStringLiteral("模型在加载期间发生变化，请重新开始评估。"));
                    for (int begin = 0; begin < files.size() && !stopping(); begin += batchSize)
                    {
                        QVector<QImage> images;
                        QStringList sources;
                        for (int index = begin; index < std::min(begin + batchSize, int(files.size())) && !stopping(); ++index)
                        {
                            const auto &path = files.at(index);
                            const auto sample = byImage.value(path);
                            checkSample(sample, stopping);
                            QImageReader reader(path);
                            reader.setAutoTransform(false); // GT uses unrotated encoded pixel coordinates.
                            const auto image = reader.read();
                            if (image.isNull())
                                fail(QStringLiteral("无法读取评估图片 %1：%2").arg(path, reader.errorString()));
                            if (image.width() != sample.value("width").toInt() ||
                                image.height() != sample.value("height").toInt())
                                fail(QStringLiteral("评估图片尺寸与数据集快照不一致：%1").arg(path));
                            images.append(image);
                            sources.append(path);
                        }
                        if (stopping())
                            break;
                        const auto results = engine.inferBatch(images, sources);
                        config = engine.config(); // Includes actual PT thread count after the first forward.
                        executedBatches.append(images.size());
                        parameters.insert("executed_batch_sizes", executedBatches);
                        // A stop during forward discards the uncommitted batch. No next
                        // batch is scheduled; already persisted samples remain a prefix.
                        if (stopping())
                            break;
                        if (results.size() != images.size())
                            fail(QStringLiteral("评估批量推理返回数量与输入不一致。"));
                        for (int offset = 0; offset < results.size() && !stopping(); ++offset)
                        {
                            const int index = begin + offset;
                            const auto sample = byImage.value(files.at(index));
                            const auto &result = results.at(offset);
                            checkSample(sample, stopping);
                            auto row = predictionJson(result, sample, index);
                            row.insert("batch_size", images.size());
                            row.insert("batch_index", executedBatches.size() - 1);
                            const auto line = QJsonDocument(row).toJson(QJsonDocument::Compact) + '\n';
                            if (predictions.write(line) != line.size() || !predictions.flush())
                                fail(QStringLiteral("预测缓存写入失败：%1").arg(predictions.errorString()));
                            const QString sampleBase = directory + "/samples/" + QString::number(index);
                            const auto sampleBytes = QJsonDocument(row).toJson(QJsonDocument::Indented);
                            QSaveFile sampleFile(sampleBase + ".json");
                            if (!sampleFile.open(QIODevice::WriteOnly) || sampleFile.write(sampleBytes) != sampleBytes.size() ||
                                !sampleFile.commit())
                                fail(QStringLiteral("评估样本缓存写入失败：%1").arg(sampleFile.errorString()));
                            const auto checksum = QCryptographicHash::hash(sampleBytes, QCryptographicHash::Sha256).toHex() + '\n';
                            QSaveFile checksumFile(sampleBase + ".sha256");
                            if (!checksumFile.open(QIODevice::WriteOnly) || checksumFile.write(checksum) != checksum.size() ||
                                !checksumFile.commit())
                                fail(QStringLiteral("评估样本校验文件写入失败：%1").arg(checksumFile.errorString()));
                            ++completed;
                            status.insert("completed", completed);
                            saveJson(directory + "/status.json", status);
                            emit progress(completed, files.size(), QStringLiteral("正在评估数据集…"));
                        }
                    }
                }
                catch (const std::exception &)
                {
                    if (!stopping())
                        throw;
                }
            }
        }
        predictions.close();
        if (hashFile(config.modelPath) != modelSha)
            fail(QStringLiteral("模型在评估期间发生变化，缓存不能作为有效评估结果。"));
        const bool cancelled = stopping() || completed != files.size();
        const auto state = cancelled ? QStringLiteral("cancelled") : QStringLiteral("complete");
        parameters.insert("status", state);
        parameters.insert("model_class_count", config.modelClassCount);
        parameters.insert("config", modelConfigJson(config));
        saveJson(directory + "/request.json", parameters);
        status.insert("status", cancelled ? "finalizing_cancelled" : "analyzing");
        saveJson(directory + "/status.json", status);
        emit progress(completed, files.size(), cancelled ? QStringLiteral("已停止，正在保存已完成结果…")
                                                         : QStringLiteral("正在计算指标与阈值曲线…"));
        // Computing/persisting a partial report is worker-only finalization. A GUI stop
        // never waits for it and no further image is dispatched during this phase.
        runHelper(helper, {"analyze", "--request", directory + "/request.json", "--manifest",
            directory + "/manifest.json", "--predictions", directory + "/predictions.jsonl",
            "--output", directory + "/report.json"}, directory, stopping, false);
        auto report = readJson(directory + "/report.json");
        report.insert("directory", directory);
        report.insert("manifest", request.manifest);
        report.insert("request", parameters);
        report.insert("config", modelConfigJson(config));
        report.insert("model_class_id", request.modelClassId);
        report.insert("dataset_class_id", request.datasetClassId);
        report.insert("full_split", files == manifestImages && !cancelled);
        saveJson(directory + "/report.json", report);
        status.insert("status", state);
        status.insert("completed", completed);
        saveJson(directory + "/status.json", status);
        emit reportReady(report, directory);
    }
    catch (const std::exception &error)
    {
        const QString message = QString::fromUtf8(error.what());
        if (!directory.isEmpty())
        {
            status.insert("status", "failed");
            status.insert("error", message);
            try { saveJson(directory + "/status.json", status); } catch (...) { }
        }
        emit failed(message);
    }
}
} // namespace vision
