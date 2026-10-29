#include "resultwriter.h"
#include "annotationrenderer.h"
#include "core/gpuruntime.h"

#include <QCoreApplication>
#include <QDateTime>
#include <QDir>
#include <QFileInfo>
#include <QImageWriter>
#include <QJsonArray>
#include <QJsonDocument>
#include <QMutex>
#include <QMutexLocker>
#include <QPointer>
#include <QQueue>
#include <QRunnable>
#include <QSaveFile>
#include <QTemporaryDir>
#include <QThread>
#include <QThreadPool>

namespace vision::app
{
namespace
{
QString stereoCode(vision::StereoView view)
{
    return view == vision::StereoView::Left ? "left" : (view == vision::StereoView::Right ? "right" : "full");
}
QString taskName(vision::ModelTask t)
{
    return t == vision::ModelTask::YoloV5   ? "YOLOv5"
           : t == vision::ModelTask::YoloV8 ? "YOLOv8 / 11"
                                            : "图像分类";
}
QString cleanName(const QString &s)
{
    QString v = QFileInfo(s).completeBaseName();
    if (v.isEmpty())
        v = "frame";
    v.replace(QRegularExpression("[^\\p{L}\\p{N}_-]"), "_");
    return v.left(70);
}
bool isPtModel(const QString &path)
{
    const QString suffix = QFileInfo(path).suffix().toLower();
    return suffix == "pt" || suffix == "torchscript";
}
} // namespace

PreparedResultExport ResultWriter::prepare(const vision::InferenceResult &r,
                                                          const QString &dir,
                                                          const vision::ModelConfig &configValue,
                                                          const QJsonObject &historicalConfig)
{
    if (r.image.isNull() || r.demonstration)
        return {};
    const QString stamp = QDateTime::currentDateTime().toString("yyyyMMdd-HHmmss-zzz");
    const QString base = dir + "/" + cleanName(r.source) + "-" + stamp +
                         (r.frameNumber > 0 ? QString("-f%1").arg(r.frameNumber) : "");
    // Capture metadata only; annotation rendering and encoding run in the writer.
    QJsonArray predictions;
    QByteArray csv = r.hasGroundTruth ? "class_id,label,confidence,x,y,width,height,match_status\n" : "class_id,label,confidence,x,y,width,height\n";
    int predictionIndex = 0;
    for (const auto &p : r.predictions)
    {
        QJsonObject pred{{"class_id", p.classId},   {"label", p.label}, {"confidence", double(p.confidence)},
                         {"x", p.box.x()},          {"y", p.box.y()},   {"width", p.box.width()},
                         {"height", p.box.height()}};
        const QString matchStatus = !r.hasEvaluationMatches ? "unavailable" : r.ignoredPredictions.contains(predictionIndex) ? "ignored" :
            r.matchedPredictions.contains(predictionIndex) ? "TP" : "FP";
        if (r.hasGroundTruth) pred.insert("match_status", matchStatus);
        predictions.append(pred);
        QString label = p.label;
        label.replace('"', "\"\"");
        if (!label.isEmpty() && QString("=+-@\t\r").contains(label.front()))
            label.prepend('\'');
        csv += QString("%1,\"%2\",%3,%4,%5,%6,%7\n")
                   .arg(p.classId)
                   .arg(label)
                   .arg(p.confidence, 0, 'f', 6)
                   .arg(p.box.x(), 0, 'f', 2)
                   .arg(p.box.y(), 0, 'f', 2)
                   .arg(p.box.width(), 0, 'f', 2)
                   .arg(p.box.height(), 0, 'f', 2)
                   .toUtf8();
        if (r.hasGroundTruth) { csv.chop(1); csv += ',' + matchStatus.toUtf8() + '\n'; }
        ++predictionIndex;
    }
    const QJsonObject preprocess{
        {"mode", isPtModel(configValue.modelPath) ? "model_native" : "configured"},
        {"color_mode", configValue.colorMode == vision::InputColorMode::Grayscale
                           ? "grayscale"
                           : (configValue.swapRB ? "rgb" : "bgr")},
        {"swap_rb", configValue.swapRB},
        {"scale", configValue.scale},
        {"mean_rgb", QJsonArray{configValue.meanR, configValue.meanG, configValue.meanB}}};
    QJsonArray filterIds;
    for (int id : configValue.classFilter) filterIds.append(id);
    const QJsonObject config{{"class_filter", filterIds},
                             {"model_class_count", configValue.modelClassCount},
                             {"input_size", configValue.inputSize},
                             {"input_channels", configValue.inputChannels},
                             {"color_mode", configValue.colorMode == vision::InputColorMode::Grayscale
                                                ? "grayscale"
                                                : (configValue.swapRB ? "rgb" : "bgr")},
                             {"confidence_threshold", double(configValue.confidence)},
                             {"nms_iou", double(configValue.iou)},
                             {"preprocess", preprocess}};
    QJsonObject root{
        {"application", "Vision Studio"},
        {"version", QCoreApplication::applicationVersion()},
        {"timestamp", QDateTime::currentDateTime().toString(Qt::ISODateWithMs)},
        {"source", r.source},
        {"model", r.modelName},
        {"backend", r.backend},
        {"requested_device", vision::computeDeviceKey(r.requestedDevice)},
        {"actual_device", vision::computeDeviceKey(r.device)},
        {"device_index", r.deviceIndex},
        {"device_name", r.deviceName},
        {"device_notice", r.deviceNotice},
        {"model_file", configValue.modelPath},
        {"config", config},
        {"task", taskName(r.task)},
        {"width", r.image.width()},
        {"height", r.image.height()},
        {"stereo_view", stereoCode(r.stereoView)},
        {"source_frame_size",
         QJsonObject{
             {"width", r.sourceFrameSize.isEmpty() ? r.image.width() : r.sourceFrameSize.width()},
             {"height", r.sourceFrameSize.isEmpty() ? r.image.height() : r.sourceFrameSize.height()}}},
        {"frame", r.frameNumber},
        {"inference_ms", r.inferenceMs},
        {"total_ms", r.totalMs},
        {"predictions", predictions}};
    if (!r.hasVerifiedDevice)
    {
        for (const auto &key : {"requested_device", "actual_device", "device_index", "device_name"})
            root.insert(key, QJsonValue::Null);
        root.insert("source_runtime_device", r.sourceRuntimeDevice);
        root.insert("device_verified", false);
    }
    if (!r.hasInferenceTiming)
    {
        root.insert("inference_ms", QJsonValue::Null);
        root.insert("total_ms", QJsonValue::Null);
    }
    if (!r.hasCompleteInferenceConfig)
    {
        root.insert("task", "目标检测 · 历史预测");
        auto recorded = historicalConfig;
        recorded.insert("metadata_complete", false);
        recorded.insert("confidence_threshold", double(configValue.confidence));
        for (const auto &key : {"preprocess", "color_mode", "input_channels"}) recorded.insert(key, QJsonValue::Null);
        root.insert("config", recorded);
    }
    if (r.hasGroundTruth) {
        QJsonArray truth, matched, matchedGt, ignored;
        for (const auto &gt : r.groundTruth) truth.append(QJsonObject{{"class_id", gt.classId}, {"label", gt.label},
            {"x", gt.box.x()}, {"y", gt.box.y()}, {"width", gt.box.width()}, {"height", gt.box.height()}});
        for (int id : r.matchedPredictions) matched.append(id);
        for (int id : r.matchedGroundTruth) matchedGt.append(id);
        for (int id : r.ignoredPredictions) ignored.append(id);
        root.insert("ground_truth", truth);
        root.insert("evaluation", QJsonObject{{"has_ground_truth", true}, {"matching_available", r.hasEvaluationMatches},
            {"matched_predictions", matched}, {"matched_ground_truth", matchedGt}, {"ignored_predictions", ignored},
            {"report_directory", r.evaluationReport}, {"matching_iou", r.matchingIou},
            {"analysis_confidence", double(configValue.confidence)}});
    }
    return PreparedResultExport{base, r, QJsonDocument(root).toJson(), csv};
}

bool ResultWriter::writePrepared(const PreparedResultExport &result, QString *error)
{
    if (result.result.image.isNull() || result.base.isEmpty())
    {
        if (error)
            *error = "没有可导出的真实结果。";
        return false;
    }
    if (!QDir().mkpath(QFileInfo(result.base).absolutePath()))
    {
        if (error)
            *error = "无法创建目录。";
        return false;
    }
    QSaveFile png(result.base + ".png");
    if (!png.open(QIODevice::WriteOnly))
    {
        if (error)
            *error = png.errorString();
        return false;
    }
    QImageWriter writer(&png, "PNG");
    writer.setCompression(11);
    const QImage image = AnnotationRenderer::render(result.result);
    if (!writer.write(image) || !png.commit())
    {
        if (error)
            *error = writer.errorString().isEmpty() ? png.errorString() : writer.errorString();
        return false;
    }
    return writeAtomically(result.base + ".json", result.json, error) &&
           writeAtomically(result.base + ".csv", result.csv, error);
}


bool ResultWriter::writeAtomically(const QString &path, const QByteArray &data, QString *error)
{
    QSaveFile file(path);
    if (!file.open(QIODevice::WriteOnly) || file.write(data) != data.size() || !file.commit())
    {
        if (error) *error = file.errorString();
        return false;
    }
    return true;
}

struct ResultWriter::QueueState
{
    QMutex mutex;
    QQueue<ResultWriteRequest> jobs;
    bool running = false;
};

ResultWriter::ResultWriter(QObject *parent) : QObject(parent), queue_(std::make_shared<QueueState>())
{
    qRegisterMetaType<ResultWriteOutcome>();
}

ResultWriter::~ResultWriter() = default;

ResultWriteOutcome ResultWriter::write(const ResultWriteRequest &request)
{
    ResultWriteOutcome outcome;
    outcome.source = request.result.source;
    outcome.ticket = request.ticket;
    outcome.generation = request.generation;
    outcome.directory = request.directory;
    outcome.snapshotPath = request.snapshotPath;
    outcome.automaticExport = request.automaticExport;
    outcome.streamFinalization = request.streamFinalization;
    if (!request.snapshotPath.isEmpty())
    {
        QSaveFile file(request.snapshotPath);
        if (file.open(QIODevice::WriteOnly))
        {
            QImageWriter writer(&file, "PNG");
            writer.setCompression(11);
            const QImage &original = request.result.originalImage.isNull()
                                         ? request.result.image : request.result.originalImage;
            outcome.snapshotSaved = writer.write(original) && file.commit();
            if (!outcome.snapshotSaved)
                outcome.snapshotError = writer.errorString().isEmpty() ? file.errorString() : writer.errorString();
        }
        else outcome.snapshotError = file.errorString();
    }
    outcome.exportSaved = !request.automaticExport || writePrepared(request.exported, &outcome.exportError);
    outcome.historySaved = request.historyPath.isEmpty() ||
        writeAtomically(request.historyPath, request.history, &outcome.historyError);
    return outcome;
}

bool ResultWriter::enqueue(ResultWriteRequest request)
{
    Q_ASSERT(QThread::currentThread() == thread());
    if (shuttingDown_) return false;
    ++pendingCount_;
    const auto state = queue_;
    {
        QMutexLocker lock(&state->mutex);
        state->jobs.enqueue(std::move(request));
        if (state->running)
        {
            lock.unlock();
            emit pendingChanged(pendingCount_);
            return true;
        }
        state->running = true;
    }
    const QPointer<ResultWriter> receiver(this);
    QObject *application = QCoreApplication::instance();
    QThreadPool::globalInstance()->start(QRunnable::create([state, receiver, application]
    {
        for (;;)
        {
            ResultWriteRequest request;
            {
                QMutexLocker lock(&state->mutex);
                if (state->jobs.isEmpty()) { state->running = false; break; }
                request = state->jobs.dequeue();
            }
            const auto outcome = write(request);
            // The guard is checked on the GUI/owner thread, never dereferenced
            // by the writer. Accepted disk writes survive accidental early deletion.
            QMetaObject::invokeMethod(application, [receiver, outcome]
            {
                if (receiver) receiver->deliver(outcome);
            }, Qt::QueuedConnection);
        }
    }));
    emit pendingChanged(pendingCount_);
    return true;
}

void ResultWriter::deliver(const ResultWriteOutcome &outcome)
{
    Q_ASSERT(QThread::currentThread() == thread());
    --pendingCount_;
    const QPointer<ResultWriter> guard(this);
    emit pendingChanged(pendingCount_);
    if (!guard) return;
    emit completed(outcome);
    if (!guard) return;
    if (shuttingDown_ && pendingCount_ == 0 && !shutdownReadyEmitted_)
    {
        shutdownReadyEmitted_ = true;
        emit shutdownReady();
    }
}

void ResultWriter::requestShutdown()
{
    Q_ASSERT(QThread::currentThread() == thread());
    shuttingDown_ = true;
    if (pendingCount_ == 0 && !shutdownReadyEmitted_)
    {
        shutdownReadyEmitted_ = true;
        QMetaObject::invokeMethod(this, [this] { emit shutdownReady(); }, Qt::QueuedConnection);
    }
}
} // namespace vision::app
