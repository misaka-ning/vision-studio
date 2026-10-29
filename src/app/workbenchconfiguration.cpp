// SPDX-License-Identifier: AGPL-3.0-only
#include "workbenchconfiguration.h"
#include <QFileInfo>
#include <QDateTime>

namespace vision::app
{
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
