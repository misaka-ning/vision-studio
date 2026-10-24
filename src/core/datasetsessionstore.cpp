// SPDX-License-Identifier: AGPL-3.0-only
#include "datasetsessionstore.h"
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonParseError>
#include <QSaveFile>
#include <QTemporaryFile>
#include <QThread>
#ifdef Q_OS_WIN
#include <qt_windows.h>
#include <fcntl.h>
#include <io.h>
#endif

namespace vision
{
namespace
{
QString validate(const DatasetSession &s)
{
    if (s.sourcePath.isEmpty() || s.sourcePath.contains(QChar::Null) || !QDir::isAbsolutePath(s.sourcePath))
        return QStringLiteral("数据集路径记录必须包含有效的绝对路径。");
    if (s.split != QLatin1String("train") && s.split != QLatin1String("val") && s.split != QLatin1String("test"))
        return QStringLiteral("数据集路径记录的集合必须是 train、val 或 test。");
    if (s.metadataPath.contains(QChar::Null) || (!s.metadataPath.isEmpty() && !QDir::isAbsolutePath(s.metadataPath)))
        return QStringLiteral("审核文件路径记录必须为空或有效的绝对路径。");
    return {};
}
}

DatasetSessionLoadResult DatasetSessionStore::load(const QString &filePath)
{
    DatasetSessionLoadResult r;
    const QFileInfo info(filePath);
    if (!info.exists()) return r;
    r.status = DatasetSessionStatus::IoError;
    if (!info.isFile()) {
        r.error = QStringLiteral("数据集路径记录不是普通文件。"); return r;
    }
    QFile file(filePath);
#ifdef Q_OS_WIN
    // Readers must permit the writer's atomic replacement. The ordinary CRT
    // file-sharing mode denies deletion while a concurrent reader is open.
    const HANDLE handle = CreateFileW(reinterpret_cast<LPCWSTR>(filePath.utf16()), GENERIC_READ,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    const int descriptor = handle == INVALID_HANDLE_VALUE ? -1 :
        _open_osfhandle(reinterpret_cast<intptr_t>(handle), _O_RDONLY | _O_BINARY);
    if (descriptor < 0 && handle != INVALID_HANDLE_VALUE) CloseHandle(handle);
    const bool opened = descriptor >= 0 && file.open(descriptor, QIODevice::ReadOnly, QFileDevice::AutoCloseHandle);
    if (!opened && descriptor >= 0) _close(descriptor);
#else
    const bool opened = file.open(QIODevice::ReadOnly);
#endif
    if (!opened) {
        r.error = QStringLiteral("无法读取数据集路径记录：%1").arg(file.errorString()); return r;
    }
    // Bound the read too, in case an external non-atomic editor grows the file.
    if (file.size() > MaximumFileBytes) {
        r.status = DatasetSessionStatus::Invalid;
        r.error = QStringLiteral("数据集路径记录超过 64 KiB，已跳过自动恢复。"); return r;
    }
    const auto data = file.read(MaximumFileBytes + 1);
    if (file.error() != QFileDevice::NoError) {
        r.error = QStringLiteral("读取数据集路径记录失败：%1").arg(file.errorString()); return r;
    }
    r.status = DatasetSessionStatus::Invalid;
    if (data.size() > MaximumFileBytes) {
        r.error = QStringLiteral("数据集路径记录超过 64 KiB，已跳过自动恢复。"); return r;
    }
    QJsonParseError parse;
    const auto doc = QJsonDocument::fromJson(data, &parse);
    if (parse.error != QJsonParseError::NoError || !doc.isObject()) {
        r.error = QStringLiteral("数据集路径记录不是有效的 JSON 对象，已跳过自动恢复。"); return r;
    }
    const auto o = doc.object();
    if (!o.value("schema_version").isDouble() || o.value("schema_version").toDouble() != 1.0) {
        r.error = QStringLiteral("数据集路径记录版本不受支持，已跳过自动恢复。"); return r;
    }
    if (!o.value("source_path").isString() || !o.value("split").isString() ||
        (o.contains("metadata_path") && !o.value("metadata_path").isString())) {
        r.error = QStringLiteral("数据集路径记录字段无效，已跳过自动恢复。"); return r;
    }
    DatasetSession session{o.value("source_path").toString(), o.value("split").toString(), o.value("metadata_path").toString()};
    r.error = validate(session);
    if (!r.error.isEmpty()) return r;
    r.status = DatasetSessionStatus::Loaded; r.session = session;
    return r;
}

bool DatasetSessionStore::save(const QString &filePath, const DatasetSession &session, QString *error)
{
    auto fail = [error](const QString &message) { if (error) *error = message; return false; };
    if (error) error->clear();
    const auto invalid = validate(session);
    if (!invalid.isEmpty()) return fail(invalid);
    const auto data = QJsonDocument(QJsonObject{{"schema_version", 1}, {"source_path", session.sourcePath},
        {"split", session.split}, {"metadata_path", session.metadataPath}}).toJson(QJsonDocument::Indented);
    if (data.size() > MaximumFileBytes) return fail(QStringLiteral("数据集路径记录超过 64 KiB，未保存。"));
    if (filePath.isEmpty() || !QDir().mkpath(QFileInfo(filePath).absolutePath()))
        return fail(QStringLiteral("无法创建数据集路径记录目录。"));
#ifdef Q_OS_WIN
    QString stagedPath;
    {
        QTemporaryFile staged(QFileInfo(filePath).absolutePath() + "/.vision-dataset-session-XXXXXX");
        if (!staged.open() || staged.write(data) != data.size() || !staged.flush())
            return fail(QStringLiteral("无法写入数据集路径记录临时文件。"));
        stagedPath = staged.fileName(); staged.setAutoRemove(false);
    }
    bool replaced = false;
    for (int attempt = 0; attempt < 20; ++attempt) {
        replaced = MoveFileExW(reinterpret_cast<LPCWSTR>(stagedPath.utf16()),
            reinterpret_cast<LPCWSTR>(filePath.utf16()), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH);
        if (replaced) break;
        const auto error = GetLastError();
        if (error != ERROR_SHARING_VIOLATION && error != ERROR_ACCESS_DENIED) break;
        QThread::msleep(2);
    }
    QFile::remove(stagedPath);
    if (!replaced) return fail(QStringLiteral("数据集路径记录原子提交失败，原记录保留；请检查文件权限或占用。"));
    return true;
#else
    QSaveFile file(filePath);
    // Never truncate the old record when a same-directory replacement fails.
    file.setDirectWriteFallback(false);
    if (!file.open(QIODevice::WriteOnly))
        return fail(QStringLiteral("无法保存数据集路径记录：%1").arg(file.errorString()));
    if (!file.setPermissions(QFileDevice::ReadOwner | QFileDevice::WriteOwner)) {
        file.cancelWriting();
        return fail(QStringLiteral("无法设置数据集路径记录权限：%1").arg(file.errorString()));
    }
    if (file.write(data) != data.size()) {
        file.cancelWriting();
        return fail(QStringLiteral("数据集路径记录写入失败：%1").arg(file.errorString()));
    }
    if (!file.commit()) return fail(QStringLiteral("数据集路径记录原子提交失败：%1").arg(file.errorString()));
    return true;
#endif
}
}
