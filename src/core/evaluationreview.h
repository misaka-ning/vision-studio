// SPDX-License-Identifier: AGPL-3.0-only
#pragma once

#include "visiontypes.h"
#include <QHash>
#include <QJsonObject>
#include <functional>

namespace vision
{
struct EvaluationPreviewRequest
{
    QString directory;
    int index = -1;
    // The manifest snapshot also identifies the expected image for a cached row.
    QJsonObject sample;
    ModelConfig config;
    int modelClassId = 0;
    int datasetClassId = 0;
    QString datasetClassName;
    double confidence = .25;
    double matchIou = .5;
};

struct EvaluationPreview
{
    InferenceResult result;
    QString error;
    bool completed = false;
};

// Pure, UI-free operations: call from a worker and discard stale generations.
// Errors/cancellation never return partially trusted image overlays.
EvaluationPreview loadEvaluationPreview(const EvaluationPreviewRequest &request,
    std::function<bool()> cancelled = {});

// An empty vector means cancellation or invalid cache, never "no matches".
// The all filter returns one true per file without reading cache/image files.
QVector<bool> filterEvaluationSamples(const QStringList &files, const QHash<QString, int> &indices,
    const QString &directory, int modelClassId, int datasetClassId, double confidence,
    double matchIou, const QString &filter, std::function<bool()> cancelled = {},
    QString *error = nullptr);
} // namespace vision
