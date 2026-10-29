// SPDX-License-Identifier: AGPL-3.0-only
#include "workbenchconfiguration.h"
#include <QFileInfo>
#include <QDateTime>
#include <QCryptographicHash>
#include <QDataStream>
#include <QIODevice>

namespace vision::app
{
QString WorkbenchConfiguration::configurationFingerprint(ModelConfig config, const QString &modelIdentity)
{
    std::sort(config.classFilter.begin(), config.classFilter.end());
    QByteArray bytes;
    QDataStream out(&bytes, QIODevice::WriteOnly);
    out << modelIdentity << int(config.task) << config.inputSize << config.confidence << config.iou
        << int(config.colorMode) << config.swapRB << config.scale << config.meanR << config.meanG << config.meanB
        << int(config.device) << config.deviceIndex << config.labels << config.classFilter
        << config.maxDetections << config.cpuThreads;
    return QString::fromLatin1(QCryptographicHash::hash(bytes, QCryptographicHash::Sha256).toHex());
}

ClassFilterStatus WorkbenchConfiguration::classFilterStatus(const ModelSessionSnapshot &model, const QString &identity) const
{
    ClassFilterStatus status;
    const auto task = draft_.parameters.task;
    if (task == ModelTask::Classification) {
        status.hint = "图像分类不使用目标检测类别筛选";
        return status;
    }
    if (!parseClassFilter(draft_.classFilterText, &status.ids, &status.error)) {
        status.hint = status.error;
        return status;
    }
    if (status.ids.isEmpty()) {
        status.hint = "留空检测全部类别";
        return status;
    }
    const bool current = !identity.isEmpty() && model.inspectedIdentity == identity;
    if (!current || model.inspectionPending) {
        status.error = model.inspectionPending ? "正在后台读取模型类别，完成后可开始"
            : (model.inspectionError.isEmpty() ? "尚未取得模型类别信息，请重新选择模型"
                : "无法校验类别：" + model.inspectionError.left(180));
    } else validateClassFilter(status.ids, model.classCount, task, &status.error);
    if (!status.error.isEmpty()) status.hint = status.error;
    else {
        QStringList values;
        for (int id : status.ids) values.append(QString::number(id));
        status.hint = QString("保留原始 ID %1 · 模型 %2 类").arg(values.join(",")).arg(model.classCount);
        if (status.ids.contains(32)) status.hint += "\nCOCO ID 32 是 sports ball，包含多种球类，并非只识别足球。";
    }
    return status;
}

ModelConfig WorkbenchConfiguration::compose(const ModelSessionSnapshot &model, bool convertedProfileAvailable) const
{
    auto config = draft_.parameters;
    config.confidence = float(draft_.confidence);
    config.iou = float(draft_.iou);
    config.modelPath = model.activeModel; config.modelClassCount = model.classCount;
    config.inputChannels = model.inputChannels;
    config.classFilter.clear();
    if (config.task != ModelTask::Classification) parseClassFilter(draft_.classFilterText, &config.classFilter);
    if (config.labels.isEmpty() && !ModelSessionController::isPtModel(model.activeModel) && convertedProfileAvailable)
        config.labels = model.nativeLabels;
    if (config.colorMode == InputColorMode::Grayscale) config.meanG = config.meanB = config.meanR;
    if (ModelSessionController::isPtModel(model.activeModel)) {
        config.swapRB = true; config.scale = 1.0 / 255.0; config.meanR = config.meanG = config.meanB = 0;
    }
    const auto profile = model.evaluationProfiles.value(model.activeModel).toMap();
    const auto identity = ModelSessionController::fileIdentity(model.activeModel);
    if (profile.value("adopted").toBool() && !identity.isEmpty() && profile.value("model_identity").toString() == identity) {
        const auto adopted = profile.value("config").toMap();
        config.maxDetections = qBound(1, adopted.value("max_detections", 300).toInt(), 10000);
        config.cpuThreads = qBound(0, adopted.value("cpu_threads", 0).toInt(), 256);
    }
    return config;
}
}
