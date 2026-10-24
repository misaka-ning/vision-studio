// SPDX-License-Identifier: AGPL-3.0-only
#pragma once
#include <QJsonObject>
#include <QString>

namespace vision
{
QString packageRoot();
QString cpuRuntimeCacheRoot();
QJsonObject cpuRuntimeDescriptor();
QString cpuRuntimeDirectory(const QJsonObject &descriptor);
QString cpuPython(const QString &helper = {});
bool cpuRuntimeDownloadNeeded();
} // namespace vision
