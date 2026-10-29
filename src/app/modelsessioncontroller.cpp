#include "modelsessioncontroller.h"
#include "core/cpuruntime.h"
#include "core/visionengine.h"

#include <QDateTime>
#include <QFileInfo>
#include <QJsonArray>
#include <QPointer>
#include <QSignalBlocker>
#include <QThread>
#include <QTimer>
#include <exception>
#include <utility>

namespace vision::app
{
ModelSessionController::ModelSessionController(QObject *parent)
    : ModelSessionController([](const ModelConfig &config, std::function<bool()> cancelled)
      { return VisionEngine::inspectModel(config, std::move(cancelled)); }, parent) {}

ModelSessionController::ModelSessionController(Inspector inspector, QObject *parent)
    : QObject(parent), inspector_(std::move(inspector))
{
    qRegisterMetaType<ModelSessionSnapshot>();
}

ModelSessionController::~ModelSessionController()
{
    for (const auto &cancelled : inspections_) cancelled->store(true);
    // Threads own only copied configuration, cancellation and output values.
    // Normal window closure waits asynchronously for shutdownReady.
}

bool ModelSessionController::editable(QString *error) const
{
    if (error) error->clear();
    if (!shuttingDown_) return true;
    if (error) *error = QStringLiteral("窗口正在关闭，不能再修改模型会话。");
    return false;
}

bool ModelSessionController::contains(const QString &path, QString *error) const
{
    if (!editable(error)) return false;
    if (state_.models.contains(path)) return true;
    if (error) *error = QStringLiteral("模型已不在当前模型库中。");
    return false;
}

void ModelSessionController::initialize(const ModelSessionSnapshot &state)
{
    if (shuttingDown_) return;
    cancelInspection();
    const auto generation = state_.inspectionGeneration;
    state_ = state;
    state_.inspectionGeneration = generation;
    state_.inspectionPending = false;
    emit changed();
}

QString ModelSessionController::fileIdentity(const QString &path)
{
    const QFileInfo file(path);
    return file.isFile() ? file.absoluteFilePath() + "|" + QString::number(file.size()) + "|" +
                              QString::number(file.lastModified().toMSecsSinceEpoch()) : QString();
}

bool ModelSessionController::isPtModel(const QString &path)
{
    const QString suffix = QFileInfo(path).suffix().toLower();
    return suffix == "pt" || suffix == "torchscript";
}

QStringList ModelSessionController::convertedLabels(const QJsonObject &metadata)
{
    QStringList result;
    const auto names = metadata.value("names").toObject();
    if (names.isEmpty() || names.size() > 100000)
        return {};
    for (int index = 0; index < names.size(); ++index)
    {
        const QJsonValue value = names.value(QString::number(index));
        if (!value.isString() || value.toString().trimmed().isEmpty() || value.toString().size() > 1000)
            return {};
        result.append(value.toString());
    }
    return result;
}
bool ModelSessionController::validConvertedMetadata(const QString &path, const QJsonObject &metadata)
{
    const QString format = metadata.value("format").toString();
    const QString suffix = QFileInfo(path).suffix().toLower();
    if ((format != "onnx" || suffix != "onnx") &&
        (format != "torchscript" || (suffix != "torchscript" && suffix != "pt")))
        return false;
    if (QFileInfo(metadata.value("output").toString()).absoluteFilePath() != QFileInfo(path).absoluteFilePath())
        return false;
    const QString task = metadata.value("task").toString(), layout = metadata.value("layout").toString();
    if ((task != "detect" || (layout != "v5" && layout != "v8")) &&
        (task != "classify" || layout != "classify"))
        return false;
    const double channels = metadata.value("input_channels").toDouble(-1);
    if ((channels != 1 && channels != 3) || metadata.value("ch").toDouble(-1) != channels ||
        convertedLabels(metadata).isEmpty() || metadata.value("precision").toString() != "fp32" ||
        !metadata.value("nms").isBool() || metadata.value("nms").toBool() ||
        metadata.value("dynamic").toBool())
        return false;
    const auto shape = metadata.value("shape").toArray();
    if (shape.size() != 4 || shape[0].toDouble(-1) != 1 || shape[1].toDouble(-1) != channels)
        return false;
    const double size = shape[2].toDouble(-1);
    return size >= 32 && size <= 4096 && size == int(size) && int(size) % 32 == 0 &&
           shape[3].toDouble(-1) == size;
}
bool ModelSessionController::validateImport(const QStringList &input, QStringList *paths, QString *error) const
{
    QString ignoredError;
    QStringList ignoredPaths;
    if (!error) error = &ignoredError;
    if (!paths) paths = &ignoredPaths;
    error->clear();
    paths->clear();
    if (input.isEmpty()) { *error = "请拖入 ONNX / PT / TorchScript 本地模型文件。"; return false; }
    // Keep an existing library key (including its display name, notes and
    // profiles) when another spelling or symlink names the same real file.
    QHash<QString, QString> existing;
    for (const auto &path : state_.models) {
        const QString canonical = QFileInfo(path).canonicalFilePath();
        if (!canonical.isEmpty() && !existing.contains(canonical)) existing.insert(canonical, path);
    }
    QSet<QString> seen;
    for (const auto &path : input) {
        const QFileInfo file(path);
        const QString suffix = file.suffix().toLower();
        if (!file.isFile() || !file.isReadable() || file.size() <= 0 ||
            (suffix != "onnx" && suffix != "pt" && suffix != "torchscript")) {
            *error = "仅支持可读取且非空的 .onnx / .pt / .torchscript 文件，不能混入文件夹或其他文件；本次未导入任何模型。";
            return false;
        }
        const QString canonical = file.canonicalFilePath();
        if (canonical.isEmpty()) { *error = "无法确定模型文件的本地路径；本次未导入任何模型。"; return false; }
        const QString actualSuffix = QFileInfo(canonical).suffix().toLower();
        if (actualSuffix != "onnx" && actualSuffix != "pt" && actualSuffix != "torchscript") {
            *error = "模型文件实际路径的格式不受支持；本次未导入任何模型。"; return false;
        }
        if (!seen.contains(canonical)) {
            seen.insert(canonical);
            paths->append(existing.value(canonical, canonical));
        }
    }
    return true;
}

bool ModelSessionController::importOrdered(const QStringList &input, int *added, QString *error)
{
    if (added) *added = 0;
    if (!editable(error)) return false;
    QStringList paths;
    if (!validateImport(input, &paths, error)) return false;
    QStringList next = state_.models;
    int count = 0;
    for (const auto &path : paths)
        if (!next.contains(path)) { next.append(path); ++count; }
    // Runtime availability and operation permissions are checked by the caller
    // before this atomic state change. Only the final path is globally selected.
    const QString previousSelection = state_.activeModel;
    const bool invalidProfile = state_.convertedProfiles.contains(paths.last()) && convertedProfile(paths.last()).isEmpty();
    QSignalBlocker blocker(this);
    if (!select(paths.last(), false, error)) return false;
    state_.models = next;
    if (added) *added = count;
    blocker.unblock();
    emit changed();
    if (previousSelection != state_.activeModel || invalidProfile)
    {
        emit selectionChanged(state_.activeModel);
        emit metadataChanged();
    }
    return true;
}

bool ModelSessionController::reorder(const QStringList &paths, QString *error)
{
    if (!editable(error)) return false;
    QHash<QString, int> before, after;
    for (const auto &path : state_.models) ++before[path];
    for (const auto &path : paths) ++after[path];
    if (before != after)
    {
        if (error) *error = QStringLiteral("模型顺序必须包含当前模型库中的全部条目。");
        return false;
    }
    if (state_.models == paths) return true;
    state_.models = paths;
    emit changed();
    return true;
}

bool ModelSessionController::rename(const QString &path, const QString &name, QString *error)
{
    if (!contains(path, error)) return false;
    const QString value = name.simplified();
    if (value.isEmpty()) state_.displayNames.remove(path);
    else state_.displayNames.insert(path, value);
    emit changed();
    return true;
}

bool ModelSessionController::setNote(const QString &path, const QString &note, QString *error)
{
    if (!contains(path, error)) return false;
    const QString value = note.trimmed();
    if (value.isEmpty()) state_.notes.remove(path);
    else state_.notes.insert(path, value);
    emit changed();
    return true;
}

bool ModelSessionController::remove(const QString &path, QString *error)
{
    if (!contains(path, error)) return false;
    state_.models.removeAll(path);
    if (state_.activeModel == path)
    {
        QString next;
        for (const auto &candidate : state_.models)
            if (QFileInfo(candidate).isFile()) { next = candidate; break; }
        QSignalBlocker blocker(this);
        select(next);
        blocker.unblock();
        emit selectionChanged(state_.activeModel);
        emit metadataChanged();
    }
    // Preserve annotations and profiles when removing only a library entry,
    // exactly as before; adding the same path later restores them.
    emit changed();
    return true;
}

void ModelSessionController::resetMetadata(bool advanceGeneration)
{
    if (advanceGeneration) cancelInspection();
    else
    {
        if (currentCancellation_) currentCancellation_->store(true);
        state_.inspectionPending = false;
    }
    state_.nativeLabels.clear();
    state_.inputChannels = 0;
    state_.classCount = 0;
    state_.inspectedIdentity.clear();
    state_.inspectionError.clear();
    state_.hasSuggestedTask = false;
}

bool ModelSessionController::select(const QString &path, bool force, QString *error)
{
    if (!editable(error)) return false;
    if (path.isEmpty())
    {
        state_.activeModel.clear();
        resetMetadata(true);
        emit changed(); emit selectionChanged({}); emit metadataChanged();
        return true;
    }
    const QFileInfo file(path);
    if (!file.isFile())
    {
        if (error) *error = QStringLiteral("模型文件不存在：") + path;
        return false;
    }
    const QString absolute = file.absoluteFilePath();
    if (state_.activeModel == absolute && !force &&
        (!state_.convertedProfiles.contains(absolute) || !convertedProfile(absolute).isEmpty()))
        return true;
    resetMetadata();
    state_.activeModel = absolute;
    if (!state_.models.contains(absolute)) state_.models.append(absolute);
    const QString name = file.fileName().toLower();
    if (name.contains("yolov5"))
    {
        state_.suggestedTask = ModelTask::YoloV5; state_.hasSuggestedTask = true;
    }
    else if (name.contains("yolov8") || name.contains("yolo11"))
    {
        state_.suggestedTask = ModelTask::YoloV8; state_.hasSuggestedTask = true;
    }
    const auto metadata = convertedProfile(absolute);
    if (!metadata.isEmpty())
    {
        state_.nativeLabels = convertedLabels(metadata);
        state_.inputChannels = metadata.value("input_channels").toInt();
        state_.suggestedTask = metadata.value("task").toString() == "classify" ? ModelTask::Classification :
            metadata.value("layout").toString() == "v5" ? ModelTask::YoloV5 : ModelTask::YoloV8;
        state_.hasSuggestedTask = true;
    }
    emit changed(); emit selectionChanged(absolute); emit metadataChanged();
    return true;
}

QString ModelSessionController::displayName(const QString &path) const
{
    const QString name = state_.displayNames.value(path).toString().trimmed();
    return name.isEmpty() ? QFileInfo(path).fileName() : name;
}

bool ModelSessionController::rememberConverted(const QString &path, const QJsonObject &metadata, QString *error)
{
    if (!editable(error)) return false;
    const QFileInfo file(path);
    if (!file.isFile() || file.isSymLink() || !validConvertedMetadata(file.absoluteFilePath(), metadata) ||
        metadata.value("bytes").toDouble(-1) != double(file.size()))
    {
        if (error) *error = QStringLiteral("转换结果的模型元数据或文件大小不匹配，请重新转换后再加入模型库。");
        return false;
    }
    state_.convertedProfiles.insert(file.absoluteFilePath(),
        QVariantMap{{"fileSize", file.size()}, {"fileModifiedMs", file.lastModified().toMSecsSinceEpoch()},
                    {"metadata", metadata.toVariantMap()}});
    emit changed();
    return true;
}

QJsonObject ModelSessionController::convertedProfile(const QString &path) const
{
    const QFileInfo file(path);
    const auto profile = state_.convertedProfiles.value(file.absoluteFilePath()).toMap();
    if (!file.isFile() || file.isSymLink() || profile.value("fileSize").toLongLong() != file.size() ||
        profile.value("fileModifiedMs").toLongLong() != file.lastModified().toMSecsSinceEpoch()) return {};
    const auto metadata = QJsonObject::fromVariantMap(profile.value("metadata").toMap());
    return validConvertedMetadata(file.absoluteFilePath(), metadata) ? metadata : QJsonObject{};
}

void ModelSessionController::setEvaluationProfile(const QString &path, const QVariantMap &profile)
{
    if (shuttingDown_) return;
    state_.evaluationProfiles.insert(path, profile);
    emit changed();
}

void ModelSessionController::setConvertedProfileRecord(const QString &path, const QVariantMap &profile)
{
    if (shuttingDown_) return;
    state_.convertedProfiles.insert(path, profile);
    emit changed();
}

void ModelSessionController::addKnownModel(const QString &path)
{
    if (shuttingDown_ || state_.models.contains(path)) return;
    state_.models.append(path);
    emit changed();
}

void ModelSessionController::acceptInferenceClassCount(int count)
{
    if (count <= 0 || shuttingDown_) return;
    state_.classCount = count;
    emit metadataChanged();
}

void ModelSessionController::cancelInspection()
{
    if (currentCancellation_) currentCancellation_->store(true);
    ++state_.inspectionGeneration;
    state_.inspectionPending = false;
}

void ModelSessionController::inspect(const ModelConfig &requested)
{
    if (shuttingDown_) return;
    if (isPtModel(state_.activeModel) && cpuRuntimeDownloadNeeded()) return;
    cancelInspection();
    currentCancellation_ = std::make_shared<std::atomic_bool>(false);
    const auto cancelled = currentCancellation_;
    const auto generation = state_.inspectionGeneration;
    const auto identity = fileIdentity(state_.activeModel);
    state_.inspectedIdentity.clear(); state_.inspectionError.clear(); state_.classCount = 0;
    if (identity.isEmpty()) { emit metadataChanged(); return; }
    state_.inspectionPending = true;
    ModelConfig config = requested;
    config.modelPath = state_.activeModel;
    struct Inspection { ModelConfig config; QString error; };
    const auto output = std::make_shared<Inspection>();
    const auto inspector = inspector_;
    auto *worker = QThread::create([config, cancelled, output, inspector]
    {
        try { output->config = inspector(config, [cancelled] { return cancelled->load(); }); }
        catch (const std::exception &error) { output->error = QString::fromUtf8(error.what()); }
        catch (...) { output->error = QStringLiteral("模型信息读取失败。"); }
    });
    worker->setObjectName("modelClassInspection");
    inspections_.insert(worker, cancelled);
    connect(worker, &QThread::finished, this, [this, worker, generation, identity, cancelled, output]
    {
        inspections_.remove(worker);
        if (shuttingDown_) { notifyShutdownIfReady(); return; }
        if (generation != state_.inspectionGeneration || cancelled->load()) return;
        state_.inspectionPending = false;
        if (identity != fileIdentity(state_.activeModel))
        {
            emit metadataChanged();
            emit inspectionRetryRequested();
            return;
        }
        state_.inspectionError = output->error;
        if (output->error.isEmpty())
        {
            state_.classCount = output->config.modelClassCount;
            state_.inputChannels = output->config.inputChannels;
            state_.inspectedIdentity = identity;
            if (isPtModel(state_.activeModel) || output->config.task == ModelTask::Classification)
            {
                if (isPtModel(state_.activeModel)) state_.nativeLabels = output->config.labels;
                state_.suggestedTask = output->config.task;
                state_.hasSuggestedTask = true;
            }
        }
        emit metadataChanged();
    });
    connect(worker, &QThread::finished, worker, &QObject::deleteLater);
    worker->start();
    emit metadataChanged();
}

bool ModelSessionController::acceptLoadedMetadata(const ModelConfig &config, const QString &capturedIdentity)
{
    if (shuttingDown_ || QFileInfo(config.modelPath).absoluteFilePath() != state_.activeModel) return false;
    if (state_.inspectionPending) cancelInspection();
    else if (currentCancellation_) currentCancellation_->store(true);
    state_.classCount = config.modelClassCount;
    state_.inputChannels = config.inputChannels;
    state_.inspectedIdentity = capturedIdentity.isEmpty() ? fileIdentity(state_.activeModel) : capturedIdentity;
    state_.inspectionError.clear();
    if (isPtModel(state_.activeModel))
    {
        state_.nativeLabels = config.labels;
        state_.suggestedTask = config.task; state_.hasSuggestedTask = true;
    }
    emit metadataChanged();
    return true;
}

void ModelSessionController::notifyShutdownIfReady()
{
    if (shuttingDown_ && inspections_.isEmpty() && !shutdownReadyEmitted_)
    {
        shutdownReadyEmitted_ = true;
        emit shutdownReady();
    }
}

void ModelSessionController::requestShutdown()
{
    if (shuttingDown_) return;
    shuttingDown_ = true;
    cancelInspection();
    for (const auto &cancelled : inspections_) cancelled->store(true);
    if (inspections_.isEmpty()) QTimer::singleShot(0, this, &ModelSessionController::notifyShutdownIfReady);
}
} // namespace vision::app
