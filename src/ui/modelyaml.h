#pragma once

#include <QJsonValue>
#include <QString>

namespace vision
{
struct ModelYamlDocument
{
    bool available = false;
    QString text;
    QString notice;
    int layers = 0;
};

// Only JSON primitives and documented architecture fields are accepted.
// No model construction, expression evaluation or filesystem lookup occurs.
ModelYamlDocument modelYamlDocument(const QJsonValue &architecture);

// Reads only metadata already parsed by the pinned Netron JavaScript reader.
QString modelYamlExtractionScript();

// Explicit new destination only. Same-directory staging plus an atomic
// no-replace hard link prevents overwriting even a concurrently created file.
bool saveModelYaml(const QString &destination, const QString &sourceModel,
                   const ModelYamlDocument &document, QString *error = nullptr);
}
