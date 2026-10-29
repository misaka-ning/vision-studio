#include "historyrepository.h"
#include "resultwriter.h"
#include "core/gpuruntime.h"

#include <QDateTime>
#include <QFile>
#include <QJsonDocument>
#include <QJsonParseError>
#include <utility>

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
} // namespace

HistoryRepository::HistoryRepository(QString filePath) : filePath_(std::move(filePath))
{
    reload();
}

void HistoryRepository::reload(QString *error)
{
    entries_ = load(error);
}

void HistoryRepository::appendResult(const InferenceResult &result)
{
    prepend(entries_, result);
}

void HistoryRepository::appendEvaluation(const QJsonObject &entry)
{
    entries_.prepend(entry);
    while (entries_.size() > 200) entries_.removeLast();
}

QByteArray HistoryRepository::serialize() const { return serialize(entries_); }

bool HistoryRepository::save(QString *error) const { return save(entries_, error); }

bool HistoryRepository::exportTo(const QString &destination, QString *error) const
{
    return ResultWriter::writeAtomically(destination, serialize(), error);
}

QJsonArray HistoryRepository::load(QString *error) const
{
    if (error) error->clear();
    QFile file(filePath_);
    if (!file.open(QIODevice::ReadOnly))
    {
        if (error && file.exists()) *error = file.errorString();
        return {};
    }
    QJsonParseError parseError;
    const auto document = QJsonDocument::fromJson(file.readAll(), &parseError);
    if (error && parseError.error != QJsonParseError::NoError) *error = parseError.errorString();
    return document.array();
}

bool HistoryRepository::save(const QJsonArray &entries, QString *error) const
{
    return ResultWriter::writeAtomically(filePath_, serialize(entries), error);
}

QByteArray HistoryRepository::serialize(const QJsonArray &entries)
{
    return QJsonDocument(entries).toJson();
}

QJsonObject HistoryRepository::entry(const InferenceResult &r)
{
    QJsonObject o{
        {"time", QDateTime::currentDateTime().toString(Qt::ISODateWithMs)},
        {"source", r.source},
        {"model", r.modelName},
        {"backend", r.backend},
        {"requested_device", vision::computeDeviceKey(r.requestedDevice)},
        {"actual_device", vision::computeDeviceKey(r.device)},
        {"device_index", r.deviceIndex},
        {"device_name", r.deviceName},
        {"device_notice", r.deviceNotice},
        {"objects", r.predictions.size()},
        {"inference_ms", r.inferenceMs},
        {"task", taskName(r.task)},
        {"width", r.image.width()},
        {"height", r.image.height()},
        {"stereo_view", stereoCode(r.stereoView)},
        {"source_frame_size",
         QJsonObject{
             {"width", r.sourceFrameSize.isEmpty() ? r.image.width() : r.sourceFrameSize.width()},
             {"height", r.sourceFrameSize.isEmpty() ? r.image.height() : r.sourceFrameSize.height()}}},
        {"frame", r.frameNumber}};
    if (!r.hasVerifiedDevice) {
        for (const auto &key : {"requested_device", "actual_device", "device_index", "device_name"}) o.insert(key, QJsonValue::Null);
        o.insert("source_runtime_device", r.sourceRuntimeDevice); o.insert("device_verified", false);
    }
    if (!r.hasInferenceTiming) o.insert("inference_ms", QJsonValue::Null);
    if (!r.hasCompleteInferenceConfig) o.insert("task", "目标检测 · 历史预测");
    return o;
}

void HistoryRepository::prepend(QJsonArray &entries, const InferenceResult &result)
{
    entries.prepend(entry(result));
    while (entries.size() > 200) entries.removeLast();
}
} // namespace vision::app
