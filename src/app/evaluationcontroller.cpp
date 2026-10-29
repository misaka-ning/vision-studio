#include "evaluationcontroller.h"
#include "asyncthreadretirement.h"
#include "core/evaluationmatch.h"

#include <QCryptographicHash>
#include <QDateTime>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QThread>
#include <QTimer>
#include <algorithm>
#include <cmath>
#include <limits>

namespace vision::app
{
namespace
{
QString identity(const QString &path)
{
    const QFileInfo f(path);
    return f.isFile() ? f.absoluteFilePath() + "|" + QString::number(f.size()) + "|" +
                           QString::number(f.lastModified().toMSecsSinceEpoch()) : QString();
}
QString sha256(const QString &path)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) return {};
    QCryptographicHash hash(QCryptographicHash::Sha256);
    while (!file.atEnd()) {
        auto chunk = file.read(1024 * 1024);
        if (chunk.isEmpty() && file.error() != QFile::NoError) return {};
        hash.addData(chunk);
    }
    return QString::fromLatin1(hash.result().toHex());
}
QJsonObject jsonFile(const QString &path)
{
    QFile f(path);
    if (!f.open(QIODevice::ReadOnly)) return {};
    return QJsonDocument::fromJson(f.readAll()).object();
}
QRectF boxOf(const QJsonObject &value)
{
    const auto b = value.value("box").toArray();
    return b.size() == 4 ? QRectF(QPointF(b[0].toDouble(), b[1].toDouble()),
                                  QPointF(b[2].toDouble(), b[3].toDouble())) : QRectF();
}
QString datasetName(const QJsonObject &manifest, int id)
{
    const auto names = manifest.value("names").toArray();
    return id >= 0 && id < names.size() ? names[id].toString() : QString::number(id);
}
double initialThreshold(const QJsonObject &report, double previous)
{
    auto request = report.value("request").toObject();
    if (request.isEmpty()) request = report.value("config").toObject();
    const double value = request.value("analysis_threshold").toDouble(request.value("confidence").toDouble(previous));
    double best = value, distance = std::numeric_limits<double>::max();
    for (const auto &entry : report.value("rows").toArray())
    {
        const auto candidate = entry.toObject().value("confidence");
        if (!candidate.isDouble() || !std::isfinite(candidate.toDouble())) continue;
        const double delta = std::abs(candidate.toDouble() - value);
        if (delta < distance) { distance = delta; best = candidate.toDouble(); }
    }
    return best;
}
} // namespace

ModelConfig EvaluationController::configFromJson(const QJsonObject &o)
{
    vision::ModelConfig c;
    c.modelPath = o.value("model_path").toString();
    c.task = o.value("task").toString() == "yolov5" ? vision::ModelTask::YoloV5 :
             o.value("task").toString() == "classification" ? vision::ModelTask::Classification : vision::ModelTask::YoloV8;
    c.inputSize = o.value("input_size").toInt(640);
    c.inputChannels = o.value("input_channels").toInt();
    c.modelClassCount = o.value("model_class_count").toInt();
    c.confidence = float(o.value("confidence").toDouble(.25));
    c.iou = float(o.value("nms_iou").toDouble(.45));
    c.swapRB = o.value("swap_rb").toBool(true);
    c.colorMode = o.value("color_mode").toString() == "grayscale" ? vision::InputColorMode::Grayscale : vision::InputColorMode::Color;
    c.scale = o.value("scale").toDouble(1. / 255);
    c.meanR = o.value("mean_r").toDouble(); c.meanG = o.value("mean_g").toDouble(); c.meanB = o.value("mean_b").toDouble();
    c.device = o.value("device").toString() == "cuda" ? vision::ComputeDevice::CUDA :
               o.value("device").toString() == "auto" ? vision::ComputeDevice::Auto : vision::ComputeDevice::CPU;
    c.resolvedDevice = o.value("resolved_device").toString() == "cuda" ? vision::ComputeDevice::CUDA : vision::ComputeDevice::CPU;
    c.deviceIndex = o.value("device_index").toInt();
    c.deviceName = o.value("device_name").toString();
    c.deviceNotice = o.value("device_notice").toString();
    c.maxDetections = o.value("max_detections").toInt(300);
    c.cpuThreads = o.value("cpu_threads").toInt();
    c.resolvedCpuThreads = o.value("resolved_cpu_threads").toInt();
    for (const auto &v : o.value("labels").toArray()) c.labels.append(v.toString());
    for (const auto &v : o.value("class_filter").toArray()) c.classFilter.append(v.toInt());
    return c;
}

EvaluationController::EvaluationController(QObject *parent) : QObject(parent)
{
    qRegisterMetaType<EvaluationFilterOutcome>();
    qRegisterMetaType<EvaluationPreviewOutcome>();
}

EvaluationController::~EvaluationController()
{
    if (worker_) worker_->requestStop();
    for (const auto &cancelled : reviewThreads_) cancelled->store(true);
    // Retirement tickets outlive an early controller deletion and reap their
    // threads only after native exit. Normal closure waits for this owner's gate.
}

bool EvaluationController::mayStart(QString *error) const
{
    if (error) error->clear();
    if (shuttingDown_ || worker_)
    {
        if (error) *error = shuttingDown_ ? QStringLiteral("窗口正在关闭，不能开始评估操作。") : QStringLiteral("请先结束当前评估任务。");
        return false;
    }
    return true;
}

bool EvaluationController::start(const EvaluationRequest &request, const QString &fingerprint, QString *error)
{
    if (!mayStart(error)) return false;
    state_.runConfig = request.config;
    state_.runFingerprint = fingerprint;
    state_.taskFiles = request.files;
    state_.rangeFirst = request.files.isEmpty() ? 0 : contextFiles_.indexOf(request.files.first()) + 1;
    state_.rangeLast = request.files.isEmpty() ? 0 : contextFiles_.indexOf(request.files.last()) + 1;
    resetReview(); state_.review.active = false;
    // reviewChanged is synchronous and may close the owner before registration.
    if (!mayStart(error)) return false;
    auto *worker = new EvaluationWorker;
    worker->configureEvaluation(request);
    startWorker(worker, EvaluationOperation::Evaluation);
    return true;
}

bool EvaluationController::reanalyze(const QJsonObject &options, const QString &outputRoot,
                                    const QString &projectRoot, QString *error)
{
    if (!mayStart(error)) return false;
    if (state_.report.isEmpty() || state_.directory.isEmpty())
    {
        if (error) *error = QStringLiteral("请先打开有效评估报告再重新分析。");
        return false;
    }
    suspendReview();
    if (!mayStart(error)) return false;
    auto *worker = new EvaluationWorker;
    worker->configureReanalysis(state_.directory, options, outputRoot, projectRoot);
    startWorker(worker, EvaluationOperation::Reanalysis);
    return true;
}

bool EvaluationController::importExternal(const QString &source, const QJsonObject &options,
    const QString &outputRoot, const QString &projectRoot, QString *error)
{
    if (!mayStart(error)) return false;
    if (source.isEmpty()) { if (error) *error = QStringLiteral("请选择待导入的评估报告。"); return false; }
    suspendReview();
    if (!mayStart(error)) return false;
    auto *worker = new EvaluationWorker;
    worker->configureExternalImport(source, options, outputRoot, projectRoot);
    startWorker(worker, EvaluationOperation::ExternalImport);
    return true;
}

void EvaluationController::startWorker(EvaluationWorker *worker, EvaluationOperation operation)
{
    worker_ = worker;
    state_.operation = operation;
    state_.completed = state_.total = 0; state_.stage.clear();
    const auto imported = std::make_shared<QString>();
    connect(worker, &EvaluationWorker::progress, this, [this](int completed, int total, const QString &stage)
    {
        state_.completed = completed; state_.total = total; state_.stage = stage;
        emit progress(completed, total, stage);
    });
    connect(worker, &EvaluationWorker::failed, this, &EvaluationController::failed);
    connect(worker, &EvaluationWorker::reportReady, this,
        [this, imported, operation](const QJsonObject &report, const QString &directory)
    {
        if (operation == EvaluationOperation::ExternalImport) { *imported = directory; return; }
        acceptReport(report, directory);
        emit reportReady(report, directory);
    });
    AsyncThreadRetirement::watch(worker, this, [this, worker, imported, operation]
    {
        if (worker_ == worker) worker_ = nullptr;
        state_.operation = EvaluationOperation::Idle;
        emit busyChanged(false);
        if (!shuttingDown_ && operation == EvaluationOperation::ExternalImport && !imported->isEmpty())
            emit externalImportReady(*imported);
        emit finished();
        notifyShutdownIfReady();
    });
    worker->start();
    emit busyChanged(true);
}

void EvaluationController::requestStop()
{
    if (worker_) worker_->requestStop();
}

LoadedEvaluationReport EvaluationController::readReport(const QString &directory)
{
    LoadedEvaluationReport loaded;
    loaded.directory = directory;
    loaded.report = jsonFile(directory + "/report.json");
    const auto status = jsonFile(directory + "/status.json");
    if (loaded.report.value("schema_version").toInt() != 1 ||
        (status.value("status").toString() != "complete" && status.value("status").toString() != "cancelled"))
    {
        loaded.error = QStringLiteral("该评估记录缺失、损坏或尚未完成保存。");
        return loaded;
    }
    loaded.manifest = loaded.report.value("manifest").toObject();
    if (loaded.manifest.isEmpty()) loaded.manifest = jsonFile(directory + "/manifest.json");
    if (loaded.manifest.value("samples").toArray().isEmpty())
    {
        loaded.error = QStringLiteral("评估报告没有有效数据集快照。");
        return loaded;
    }
    loaded.valid = true;
    return loaded;
}

void EvaluationController::acceptReport(const QJsonObject &report, const QString &directory, bool historical)
{
    resetReview();
    state_.report = report; state_.directory = directory;
    state_.runConfig = configFromJson(report.value("config").toObject());
    if (historical) state_.runFingerprint.clear();
    state_.sampleIndices.clear();
    const auto selected = report.value("request").toObject().value("sample_images").toArray();
    const int completed = report.value("per_image").toArray().size();
    for (int i = 0; i < std::min(int(selected.size()), completed); ++i)
        state_.sampleIndices.insert(selected[i].toString(), i);
    state_.review.active = true;
    state_.review.threshold = initialThreshold(report, state_.review.threshold);
    emit reportChanged(); emit reviewChanged();
}

void EvaluationController::cancelReviewWork()
{
    if (filterCancellation_) filterCancellation_->store(true);
    if (previewCancellation_) previewCancellation_->store(true);
    ++state_.review.filterGeneration; ++state_.review.previewGeneration;
    state_.review.pending = state_.review.previewPending = false;
}

void EvaluationController::resetReview()
{
    cancelReviewWork();
    const bool active = state_.review.active;
    const double threshold = state_.review.threshold;
    const auto filterGeneration = state_.review.filterGeneration, previewGeneration = state_.review.previewGeneration;
    state_.review = {};
    state_.review.active = active; state_.review.threshold = threshold;
    state_.review.filterGeneration = filterGeneration; state_.review.previewGeneration = previewGeneration;
    emit reviewChanged();
}

void EvaluationController::clear()
{
    resetReview();
    const auto review = state_.review;
    const auto operation = state_.operation;
    state_ = {}; state_.review.filterGeneration = review.filterGeneration;
    state_.review.previewGeneration = review.previewGeneration; state_.operation = operation;
    contextFiles_.clear(); contextSamples_.clear(); contextManifest_ = {};
    emit reportChanged(); emit reviewChanged();
}

void EvaluationController::suspendReview()
{
    cancelReviewWork();
    state_.review.available = false;
    emit reviewChanged();
}

void EvaluationController::setReviewContext(const QStringList &files, const QHash<QString, QJsonObject> &samples,
                                           const QJsonObject &manifest)
{
    if (contextFiles_ == files && contextSamples_ == samples && contextManifest_ == manifest) return;
    resetReview(); contextFiles_ = files; contextSamples_ = samples; contextManifest_ = manifest;
}

void EvaluationController::setReviewActive(bool active)
{
    cancelReviewWork();
    state_.review.active = active && !state_.report.isEmpty();
    if (!state_.review.active) resetReview();
    emit reviewChanged();
}

bool EvaluationController::invalidateForFingerprint(const QString &fingerprint)
{
    if (!state_.review.active || state_.runFingerprint == fingerprint) return false;
    setReviewActive(false);
    return true;
}

bool EvaluationController::applyFilter(const QString &filter, double threshold, int currentRow,
                                       bool selectFirst, QString *error)
{
    if (!mayStart(error)) return false;
    if (state_.report.isEmpty()) { if (error) *error = QStringLiteral("没有可用评估报告。"); return false; }
    QStringList sceneImages;
    if (filter.startsWith("scene:"))
    {
        const auto scene = state_.report.value("groups").toObject().value(filter.mid(6)).toObject();
        if (!scene.value("available").toBool() || !scene.value("sample_images").isArray())
        { if (error) *error = QStringLiteral("该报告没有可用的已审核场景名单，请生成新版评估或重新分析。"); return false; }
        for (const auto &value : scene.value("sample_images").toArray()) sceneImages.append(value.toString());
    }
    else if (!QStringList{"all", "errors", "fp", "fn", "negative_fp"}.contains(filter))
    { if (error) *error = QStringLiteral("不支持的评估图片筛选条件。"); return false; }
    cancelReviewWork();
    state_.review.filter = filter; state_.review.threshold = threshold; state_.review.active = true;
    state_.review.pending = true; state_.review.available = false; state_.review.error.clear();
    state_.review.rows.clear(); state_.review.errorRows.clear();
    const auto cancelled = filterCancellation_ = std::make_shared<std::atomic_bool>(false);
    const quint64 generation = state_.review.filterGeneration;
    const auto files = contextFiles_;
    const QString directory = state_.directory;
    const auto indices = state_.sampleIndices;
    const auto parameters = state_.report.value("request").toObject();
    const auto output = std::make_shared<EvaluationReviewSummary>();
    auto *thread = QThread::create([output, files, indices, directory, parameters, threshold, filter, cancelled, sceneImages]
    {
        *output = summarizeEvaluationSamples(files, indices, directory,
            parameters.value("model_class_id").toInt(), parameters.value("dataset_class_id").toInt(), threshold,
            parameters.value("match_iou").toDouble(.5), filter, [cancelled] { return cancelled->load(); }, sceneImages);
    });
    reviewThreads_.insert(thread, cancelled);
    AsyncThreadRetirement::watch(thread, this,
        [this, thread, cancelled, generation, files, directory, output, selectFirst, currentRow]
    {
        reviewThreads_.remove(thread);
        if (shuttingDown_) { notifyShutdownIfReady(); return; }
        if (cancelled->load() || generation != state_.review.filterGeneration || isBusy() ||
            !state_.review.active || directory != state_.directory || files != contextFiles_) return;
        state_.review.pending = false;
        EvaluationFilterOutcome outcome;
        outcome.files = files; outcome.directory = directory; outcome.generation = generation;
        outcome.selectFirst = selectFirst;
        outcome.selectedRow = currentRow;
        if (!output->valid || output->visible.size() != files.size())
        {
            state_.review.available = false;
            state_.review.error = output->error.isEmpty() ? QStringLiteral("评估缓存校验失败，无法整理错例。") : output->error;
            outcome.error = state_.review.error;
            emit reviewChanged(); emit filterReady(outcome); return;
        }
        state_.review.available = true;
        state_.review.errorRows = output->errorRows;
        state_.review.evaluated = output->evaluatedRows.size();
        state_.review.falsePositive = output->falsePositiveRows.size();
        state_.review.falseNegative = output->falseNegativeRows.size();
        state_.review.negativeFalsePositive = output->negativeFalsePositiveRows.size();
        state_.review.rows.clear();
        outcome.visible.resize(files.size());
        for (int row = 0; row < files.size(); ++row)
        {
            const bool datasetImage = contextSamples_.contains(files[row]);
            outcome.visible[row] = output->visible[row] && (datasetImage || state_.review.filter == "all");
            if (outcome.visible[row] && datasetImage) state_.review.rows.append(row);
        }
        const bool temporarySelection = state_.review.filter == "all" && currentRow >= 0 && currentRow < files.size() &&
            !contextSamples_.contains(files[currentRow]);
        if (selectFirst || (!temporarySelection && !state_.review.rows.contains(currentRow)))
            outcome.selectedRow = state_.review.rows.isEmpty() ? -1 : state_.review.rows.first();
        outcome.valid = true;
        emit reviewChanged(); emit filterReady(outcome);
    });
    thread->start(); emit reviewChanged();
    return true;
}

bool EvaluationController::requestPreview(int row)
{
    if (shuttingDown_ || isBusy() || state_.review.pending || !state_.review.active ||
        row < 0 || row >= contextFiles_.size()) return false;
    if (previewCancellation_) previewCancellation_->store(true);
    const auto cancelled = previewCancellation_ = std::make_shared<std::atomic_bool>(false);
    const quint64 generation = ++state_.review.previewGeneration;
    state_.review.previewPending = true;
    const auto path = contextFiles_[row];
    EvaluationPreviewRequest request;
    request.directory = state_.directory; request.index = state_.sampleIndices.value(path, -1);
    request.sample = contextSamples_.value(path); request.config = state_.runConfig;
    const auto parameters = state_.report.value("request").toObject();
    request.modelClassId = parameters.value("model_class_id").toInt();
    request.datasetClassId = parameters.value("dataset_class_id").toInt();
    const auto manifest = contextManifest_.isEmpty() ? state_.report.value("manifest").toObject() : contextManifest_;
    request.datasetClassName = datasetName(manifest, request.datasetClassId);
    request.confidence = state_.review.threshold; request.matchIou = parameters.value("match_iou").toDouble(.5);
    const auto output = std::make_shared<EvaluationPreview>();
    auto *thread = QThread::create([request, cancelled, output]
    { *output = loadEvaluationPreview(request, [cancelled] { return cancelled->load(); }); });
    reviewThreads_.insert(thread, cancelled);
    AsyncThreadRetirement::watch(thread, this, [this, thread, cancelled, generation, path, row, request, output]
    {
        reviewThreads_.remove(thread);
        if (shuttingDown_) { notifyShutdownIfReady(); return; }
        if (cancelled->load() || generation != state_.review.previewGeneration || isBusy() || !state_.review.active ||
            request.directory != state_.directory || row >= contextFiles_.size() || contextFiles_[row] != path) return;
        state_.review.previewPending = false;
        EvaluationPreviewOutcome outcome;
        outcome.row = row; outcome.source = path; outcome.directory = request.directory;
        outcome.generation = generation; outcome.preview = *output; outcome.config = request.config;
        outcome.config.confidence = float(request.confidence); outcome.config.classFilter = {request.modelClassId};
        emit reviewChanged(); emit previewReady(outcome);
    });
    thread->start(); emit reviewChanged();
    return true;
}

int EvaluationController::navigationTarget(int currentRow, int step) const
{
    if (isBusy() || state_.review.pending || !state_.review.available || !state_.review.active) return -1;
    const int position = state_.review.rows.indexOf(currentRow), next = position + step;
    return position >= 0 && next >= 0 && next < state_.review.rows.size() ? state_.review.rows[next] : -1;
}

EvaluationConfigurationPatch EvaluationController::configurationPatch(const QString &activeModel, double threshold,
                                                                      const QVariantMap &previousProfile)
{
    EvaluationConfigurationPatch patch;
    if (isBusy() || shuttingDown_ || state_.report.isEmpty()) { patch.error = QStringLiteral("请先结束当前任务并打开有效评估报告。"); return patch; }
    if (!state_.report.value("allow_apply_profile").toBool(true))
    { patch.error = QStringLiteral("导入报告缺少完整的原始后端配置，仅供分析，不能自动采用检测配置。"); return patch; }
    const auto request = state_.report.value("request").toObject();
    const QString path = request.value("model_path").toString();
    const QString expected = request.value("model_sha256").toString();
    if (path.isEmpty() || expected.size() != 64 ||
        QFileInfo(path).absoluteFilePath() != QFileInfo(activeModel).absoluteFilePath() || sha256(path) != expected)
    { patch.error = QStringLiteral("请先在模型库选用报告对应的同一模型文件，再采用该检测配置。"); return patch; }
    if (!std::isfinite(threshold) || threshold < 0 || threshold > 1)
    { patch.error = QStringLiteral("检测阈值必须在 0～1 之间。"); return patch; }
    patch.config = configFromJson(state_.report.value("config").toObject());
    patch.config.modelPath = path; patch.config.confidence = float(threshold);
    patch.classId = request.value("model_class_id").toInt(); patch.config.classFilter = {patch.classId};
    patch.profile = previousProfile;
    patch.profile.insert("config", state_.report.value("config").toObject().toVariantMap());
    patch.profile.insert("model_class_id", patch.classId); patch.profile.insert("adopted_threshold", threshold);
    patch.profile.insert("adopted", true); patch.profile.insert("report_directory", state_.directory);
    patch.profile.insert("adopted_report_directory", state_.directory);
    patch.profile.insert("model_identity", identity(activeModel)); patch.profile.insert("model_sha256", expected);
    patch.valid = true; state_.review.active = true;
    return patch;
}

QJsonObject EvaluationController::historyEntry(const QString &modelDisplayName) const
{
    const auto request = state_.report.value("request").toObject();
    const auto manifest = contextManifest_.isEmpty() ? state_.report.value("manifest").toObject() : contextManifest_;
    return {{"time", QDateTime::currentDateTime().toString(Qt::ISODateWithMs)}, {"kind", "evaluation"},
        {"source", manifest.value("data_yaml")}, {"model", modelDisplayName},
        {"model_path", request.value("model_path")}, {"report_directory", state_.directory},
        {"backend", state_.report.value("config").toObject().value("resolved_device")},
        {"objects", state_.report.value("metrics").toObject().value("gt_boxes")},
        {"task", state_.report.value("status").toString() == "cancelled" ? "评估 · 已停止" :
            state_.report.value("full_split").toBool() ? "评估 · 完整集合" : "评估 · 子集"},
        {"split", manifest.value("split")}, {"metrics", state_.report.value("metrics")}};
}

QVariantMap EvaluationController::reportAssociation(const QVariantMap &previousProfile) const
{
    const auto request = state_.report.value("request").toObject();
    auto association = previousProfile;
    if (association.value("model_sha256").toString() != request.value("model_sha256").toString()) association.clear();
    association.insert("report_directory", state_.directory);
    if (state_.report.value("allow_apply_profile").toBool(true) && !request.value("model_identity").toString().isEmpty())
        association.insert("model_identity", request.value("model_identity").toString());
    association.insert("model_sha256", request.value("model_sha256").toString());
    const auto manifest = contextManifest_.isEmpty() ? state_.report.value("manifest").toObject() : contextManifest_;
    association.insert("split", manifest.value("split").toString());
    association.insert("full_split", state_.report.value("full_split").toBool());
    association.insert("metrics", state_.report.value("metrics").toObject().toVariantMap());
    return association;
}

QString EvaluationController::configurationFingerprint(const EvaluationRequest &request)
{
    auto config = EvaluationWorker::modelConfigJson(request.config);
    for (const auto &key : {"confidence", "class_filter", "labels", "resolved_device", "device_name", "device_notice"}) config.remove(key);
    config.insert("model_identity", identity(request.config.modelPath));
    config.insert("model_class_id", request.modelClassId); config.insert("dataset_class_id", request.datasetClassId);
    config.insert("dataset_fingerprint", request.manifest.value("fingerprint"));
    config.insert("max_detections", request.config.maxDetections); config.insert("score_floor", request.scoreFloor);
    config.insert("cpu_threads", request.config.cpuThreads);
    return QString::fromLatin1(QCryptographicHash::hash(QJsonDocument(config).toJson(QJsonDocument::Compact), QCryptographicHash::Sha256).toHex());
}

void EvaluationController::attachGroundTruth(InferenceResult &result, const QJsonObject &manifest,
    const QHash<QString, QJsonObject> &samples, int modelClassId, int datasetClassId, int modelClassCount, double matchIou)
{
    result.groundTruth.clear(); result.matchedPredictions.clear(); result.matchedGroundTruth.clear(); result.ignoredPredictions.clear();
    result.hasGroundTruth = false; result.hasEvaluationMatches = false;
    if (manifest.isEmpty() || result.task == ModelTask::Classification || !samples.contains(result.source) ||
        datasetClassId < 0 || modelClassId < 0) return;
    result.hasGroundTruth = true;
    const auto sample = samples.value(result.source);
    for (const auto &value : sample.value("gt").toArray())
    {
        const auto gt = value.toObject();
        if (gt.value("class_id").toInt(-1) != datasetClassId) continue;
        result.groundTruth.append({datasetClassId, datasetName(manifest, datasetClassId), 1, boxOf(gt)});
    }
    for (int i = 0; i < result.predictions.size(); ++i)
        if (result.predictions[i].classId != modelClassId) result.ignoredPredictions.append(i);
    if (!result.modelName.isEmpty() && modelClassId < modelClassCount && datasetClassId < manifest.value("names").toArray().size())
        matchEvaluation(result, matchIou);
}

void EvaluationController::notifyShutdownIfReady()
{
    if (isShutdownReady() && !shutdownReadyEmitted_)
    { shutdownReadyEmitted_ = true; emit shutdownReady(); }
}

void EvaluationController::requestShutdown()
{
    if (shuttingDown_) return;
    shuttingDown_ = true; requestStop(); cancelReviewWork();
    for (const auto &cancelled : reviewThreads_) cancelled->store(true);
    if (!worker_ && reviewThreads_.isEmpty()) QTimer::singleShot(0, this, &EvaluationController::notifyShutdownIfReady);
}
} // namespace vision::app
