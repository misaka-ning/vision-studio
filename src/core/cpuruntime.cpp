// SPDX-License-Identifier: AGPL-3.0-only
#include "cpuruntime.h"
#include <QCoreApplication>
#include <QCryptographicHash>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QRegularExpression>
#include <QStandardPaths>

namespace vision
{
namespace {
QJsonObject readObject(const QString &path)
{
    QFile f(path);
    return f.open(QIODevice::ReadOnly) ? QJsonDocument::fromJson(f.readAll()).object() : QJsonObject{};
}
QString executable(const QString &path)
{
    const QString resolved = QFileInfo(path).isAbsolute() ? path : QStandardPaths::findExecutable(path);
    return QFileInfo(resolved).isFile() && QFileInfo(resolved).isExecutable()
        ? QFileInfo(resolved).absoluteFilePath() : QString();
}
}
QString packageRoot()
{
    const QString home = qEnvironmentVariable("VISION_STUDIO_HOME");
    return QDir(home.isEmpty() ? QCoreApplication::applicationDirPath() + "/.." : home).absolutePath();
}
QString cpuRuntimeCacheRoot()
{
    QString data = qEnvironmentVariable("VISION_STUDIO_DATA_DIR");
    if (data.isEmpty()) data = QStandardPaths::writableLocation(QStandardPaths::GenericDataLocation) + "/vision-studio";
    return QDir(data).absoluteFilePath("runtimes/cpu");
}
QJsonObject cpuRuntimeDescriptor()
{
    return readObject(packageRoot() + "/cpu-runtime.json");
}
QString cpuRuntimeDirectory(const QJsonObject &descriptor)
{
    const QString id = descriptor.value("sha256").toString();
    if (!QRegularExpression("^[a-f0-9]{64}$").match(id).hasMatch()) return {};
    return QDir(cpuRuntimeCacheRoot()).filePath(id);
}
QString cpuPython(const QString &helper)
{
    const QString override = qEnvironmentVariable("VISION_STUDIO_PYTHON");
    // An invalid explicit override must remain an error, never silently download.
    if (!override.isEmpty()) return executable(override);
#ifdef Q_OS_WIN
    const QString suffix = "/runtime/python.exe";
#else
    const QString suffix = "/runtime/bin/python";
#endif
    QStringList candidates{packageRoot() + suffix};
    if (!helper.isEmpty()) candidates.append(QFileInfo(helper).absolutePath() + "/.." + suffix);
#ifdef VISION_PROJECT_DIR
    // Installed builds must never fall back to the build machine's source tree.
    if (!qEnvironmentVariableIsSet("VISION_STUDIO_SYSTEM_INSTALL"))
        candidates.append(QString::fromUtf8(VISION_PROJECT_DIR) + suffix);
#endif
    if (!qEnvironmentVariableIsSet("VISION_STUDIO_SYSTEM_INSTALL"))
        candidates.append(QDir::homePath() + "/VisionStudio" + suffix);
    for (const auto &candidate : candidates) {
        const auto found = executable(candidate);
        if (found.isEmpty()) continue;
#ifdef Q_OS_WIN
        const auto descriptor = cpuRuntimeDescriptor();
        if (qEnvironmentVariableIsSet("VISION_STUDIO_SYSTEM_INSTALL") && descriptor.contains("inventory_sha256")) {
            QFile inventory(QFileInfo(found).absolutePath() + "/CPU-FILES.json");
            if (!inventory.open(QIODevice::ReadOnly) ||
                QString::fromLatin1(QCryptographicHash::hash(inventory.readAll(), QCryptographicHash::Sha256).toHex())
                    != descriptor.value("inventory_sha256").toString()) continue;
        }
#endif
        return found;
    }
#ifdef Q_OS_WIN
    const auto descriptor = cpuRuntimeDescriptor();
    const auto directory = cpuRuntimeDirectory(descriptor);
    if (!directory.isEmpty() && readObject(directory + "/ready.json").value("sha256") == descriptor.value("sha256"))
        return executable(directory + "/python.exe");
#endif
    return {};
}
bool cpuRuntimeDownloadNeeded()
{
#ifdef Q_OS_WIN
    return qEnvironmentVariable("VISION_STUDIO_PYTHON").isEmpty() && cpuPython().isEmpty()
        && QFileInfo(packageRoot() + "/cpu-runtime.json").isFile();
#else
    return false;
#endif
}
} // namespace vision
