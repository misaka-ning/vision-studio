// SPDX-License-Identifier: AGPL-3.0-only
#include "evaluationreview.h"
#include "evaluationmatch.h"
#include "evaluationworker.h"

#include <QBuffer>
#include <QCryptographicHash>
#include <QFile>
#include <QFileInfo>
#include <QImageReader>
#include <QJsonArray>
#include <QSet>
#include <cmath>
#include <stdexcept>

namespace vision
{
namespace
{
[[noreturn]] void fail(const QString &message)
{
    throw std::runtime_error(message.toUtf8().constData());
}

void checkCancelled(const std::function<bool()> &cancelled)
{
    if (cancelled && cancelled())
        fail(QStringLiteral("评估回看已取消。"));
}

void validateParameters(int modelId, int datasetId, double confidence, double iou)
{
    if (modelId < 0 || datasetId < 0 || !std::isfinite(confidence) || confidence < 0 || confidence > 1 ||
        !std::isfinite(iou) || iou <= 0 || iou > 1)
        fail(QStringLiteral("评估回看类别或阈值无效。"));
}

bool finite(const QJsonValue &value)
{
    return value.isDouble() && std::isfinite(value.toDouble());
}

QRectF checkedBox(const QJsonObject &value, int width, int height)
{
    const auto box = value.value("box").toArray();
    if (box.size() != 4)
        fail(QStringLiteral("评估缓存中的框格式损坏。"));
    for (const auto &number : box)
        if (!finite(number))
            fail(QStringLiteral("评估缓存中的框含有无效数值。"));
    const double x1 = box[0].toDouble(), y1 = box[1].toDouble(), x2 = box[2].toDouble(), y2 = box[3].toDouble();
    if (x2 <= x1 || y2 <= y1 || x1 < -.001 || y1 < -.001 || x2 > width + .001 || y2 > height + .001)
        fail(QStringLiteral("评估缓存中的框与原图坐标范围不一致。"));
    return QRectF(QPointF(x1, y1), QPointF(x2, y2));
}

int checkedId(const QJsonObject &value)
{
    const auto id = value.value("class_id");
    if (!finite(id) || id.toDouble() < 0 || id.toDouble() != id.toInt(-1))
        fail(QStringLiteral("评估缓存中的类别 ID 无效。"));
    return id.toInt();
}

InferenceResult compactResult(const QJsonObject &sample, int modelId, int datasetId, double confidence,
                              const QString &datasetName, bool includePredictions,
                              const std::function<bool()> &cancelled)
{
    const int width = sample.value("width").toInt(), height = sample.value("height").toInt();
    if (sample.value("image").toString().isEmpty() || width <= 0 || height <= 0 ||
        !sample.value("gt").isArray() || (includePredictions && !sample.value("predictions").isArray()))
        fail(QStringLiteral("评估缓存缺少有效的图片尺寸、标注或预测。"));
    InferenceResult result;
    result.source = sample.value("image").toString();
    result.hasGroundTruth = true;
    for (const auto &value : sample.value("gt").toArray())
    {
        checkCancelled(cancelled);
        if (!value.isObject())
            fail(QStringLiteral("评估缓存中的标注格式损坏。"));
        const auto object = value.toObject();
        const int id = checkedId(object);
        const auto box = checkedBox(object, width, height);
        if (id == datasetId)
            result.groundTruth.append({datasetId, datasetName.isEmpty() ? QString::number(datasetId) : datasetName, 1, box});
    }
    if (includePredictions)
        for (const auto &value : sample.value("predictions").toArray())
        {
            checkCancelled(cancelled);
            if (!value.isObject())
                fail(QStringLiteral("评估缓存中的预测格式损坏。"));
            const auto object = value.toObject();
            const int id = checkedId(object);
            const auto box = checkedBox(object, width, height);
            const auto score = object.value("confidence");
            if (!finite(score) || score.toDouble() < 0 || score.toDouble() > 1)
                fail(QStringLiteral("评估缓存中的置信度无效。"));
            if (id == modelId && score.toDouble() >= confidence)
                result.predictions.append({modelId, object.value("label").toString(QString::number(modelId)),
                    float(score.toDouble()), box});
        }
    return result;
}

QImage verifiedImage(const QJsonObject &sample, const std::function<bool()> &cancelled)
{
    const auto image = sample.value("image").toString();
    const auto expected = sample.value("image_sha256").toString();
    if (expected.size() != 64)
        fail(QStringLiteral("评估缓存缺少原图校验值。"));
    QFile file(image);
    if (!file.open(QIODevice::ReadOnly))
        fail(QStringLiteral("原图已移动或无法读取：%1").arg(image));
    QCryptographicHash hash(QCryptographicHash::Sha256);
    QByteArray encoded;
    while (!file.atEnd())
    {
        checkCancelled(cancelled);
        const auto chunk = file.read(1024 * 1024);
        if (chunk.isEmpty() && file.error() != QFile::NoError)
            fail(QStringLiteral("原图读取失败：%1").arg(image));
        hash.addData(chunk);
        encoded.append(chunk);
    }
    if (QString::fromLatin1(hash.result().toHex()) != expected)
        fail(QStringLiteral("原图已改变，不能与历史标注或预测叠加：%1").arg(image));
    checkCancelled(cancelled);
    // Decode exactly the bytes just hashed, rather than reopening a path that
    // another process could replace between verification and image decoding.
    QBuffer buffer(&encoded);
    buffer.open(QIODevice::ReadOnly);
    QImageReader reader(&buffer);
    reader.setAutoTransform(false);
    const auto result = reader.read();
    if (result.isNull() || result.width() != sample.value("width").toInt() ||
        result.height() != sample.value("height").toInt())
        fail(QStringLiteral("原图无法解码或尺寸与评估快照不一致：%1").arg(image));
    checkCancelled(cancelled);
    return result;
}
} // namespace

EvaluationPreview loadEvaluationPreview(const EvaluationPreviewRequest &request,
                                        std::function<bool()> cancelled)
{
    EvaluationPreview preview;
    try
    {
        checkCancelled(cancelled);
        validateParameters(request.modelClassId, request.datasetClassId, request.confidence, request.matchIou);
        const bool completed = request.index >= 0;
        QString error;
        const auto sample = completed ? EvaluationWorker::readSample(request.directory, request.index, &error)
                                      : request.sample;
        if (sample.isEmpty())
            fail(error.isEmpty() ? QStringLiteral("评估样本缓存不存在或已损坏。") : error);
        if (completed && !request.sample.isEmpty())
            for (const auto &key : {"image", "width", "height", "image_sha256", "label_sha256", "gt"})
                if (sample.value(key) != request.sample.value(key))
                    fail(QStringLiteral("缓存样本与报告中的数据集快照不一致。"));
        auto result = compactResult(sample, request.modelClassId, request.datasetClassId, request.confidence,
                                    request.datasetClassName, completed, cancelled);
        result.originalImage = verifiedImage(sample, cancelled);
        result.image = inputPreviewImage(result.originalImage, request.config.colorMode);
        result.sourceFrameSize = result.originalImage.size();
        result.modelName = QFileInfo(request.config.modelPath).fileName();
        result.task = request.config.task;
        result.classFilter = {request.modelClassId};
        result.modelClassCount = request.config.modelClassCount;
        result.requestedDevice = request.config.device;
        result.backend = sample.value("backend").toString();
        result.device = sample.value("device").toString() == "cuda" ? ComputeDevice::CUDA : ComputeDevice::CPU;
        result.deviceIndex = sample.value("device_index").toInt(-1);
        result.deviceName = sample.value("device_name").toString();
        result.deviceNotice = request.config.deviceNotice;
        result.inferenceMs = sample.value("inference_ms").toDouble();
        result.totalMs = sample.value("total_ms").toDouble();
        result.evaluationReport = request.directory;
        if (completed)
            matchEvaluation(result, request.matchIou);
        else
        {
            result.hasEvaluationMatches = false;
            result.matchingIou = request.matchIou;
        }
        checkCancelled(cancelled);
        preview.result = std::move(result);
        preview.completed = completed;
    }
    catch (const std::exception &error)
    {
        preview.error = QString::fromUtf8(error.what());
    }
    return preview;
}

EvaluationReviewSummary summarizeEvaluationSamples(const QStringList &files, const QHash<QString, int> &indices,
    const QString &directory, int modelClassId, int datasetClassId, double confidence,
    double matchIou, const QString &filter, std::function<bool()> cancelled)
{
    EvaluationReviewSummary summary;
    try
    {
        checkCancelled(cancelled);
        validateParameters(modelClassId, datasetClassId, confidence, matchIou);
        if (filter != "all" && filter != "errors" && filter != "fp" && filter != "fn" && filter != "negative_fp")
            fail(QStringLiteral("不支持的评估图片筛选条件。"));
        summary.visible = QVector<bool>(files.size(), filter == "all");
        QSet<QString> seen;
        for (int row = 0; row < files.size(); ++row)
        {
            checkCancelled(cancelled);
            if (!indices.contains(files[row]) || seen.contains(files[row]))
                continue;
            seen.insert(files[row]);
            QString readError;
            const auto sample = EvaluationWorker::readSample(directory, indices.value(files[row]), &readError);
            if (sample.isEmpty() || sample.value("image").toString() != files[row])
                fail(readError.isEmpty() ? QStringLiteral("评估缓存中的图片顺序与任务不一致。") : readError);
            auto result = compactResult(sample, modelClassId, datasetClassId, confidence, {}, true, cancelled);
            matchEvaluation(result, matchIou);
            const bool fp = result.predictions.size() > result.matchedPredictions.size();
            const bool fn = result.groundTruth.size() > result.matchedGroundTruth.size();
            const bool negativeFp = result.groundTruth.isEmpty() && !result.predictions.isEmpty();
            summary.evaluatedRows.append(row);
            if (fp || fn) summary.errorRows.append(row);
            if (fp) summary.falsePositiveRows.append(row);
            if (fn) summary.falseNegativeRows.append(row);
            if (negativeFp) summary.negativeFalsePositiveRows.append(row);
            if (filter != "all")
                summary.visible[row] = filter == "errors" ? fp || fn : filter == "fp" ? fp : filter == "fn" ? fn : negativeFp;
        }
        checkCancelled(cancelled);
        summary.valid = true;
    }
    catch (const std::exception &exception)
    {
        summary = {};
        summary.error = QString::fromUtf8(exception.what());
    }
    return summary;
}

QVector<bool> filterEvaluationSamples(const QStringList &files, const QHash<QString, int> &indices,
    const QString &directory, int modelClassId, int datasetClassId, double confidence,
    double matchIou, const QString &filter, std::function<bool()> cancelled, QString *error)
{
    if (error) error->clear();
    if (filter == "all") {
        try {
            checkCancelled(cancelled);
            validateParameters(modelClassId, datasetClassId, confidence, matchIou);
            return QVector<bool>(files.size(), true);
        } catch (const std::exception &exception) {
            if (error) *error = QString::fromUtf8(exception.what());
            return {};
        }
    }
    const auto summary = summarizeEvaluationSamples(files, indices, directory, modelClassId, datasetClassId,
                                                    confidence, matchIou, filter, cancelled);
    if (!summary.valid && error) *error = summary.error;
    return summary.visible;
}
} // namespace vision
