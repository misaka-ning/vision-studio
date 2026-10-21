// SPDX-License-Identifier: AGPL-3.0-only
#pragma once
#include <QString>

namespace vision
{
struct DatasetSession
{
    QString sourcePath;
    QString split = QStringLiteral("val");
    QString metadataPath;
};
enum class DatasetSessionStatus { Missing, Loaded, Invalid, IoError };
struct DatasetSessionLoadResult
{
    DatasetSessionStatus status = DatasetSessionStatus::Missing;
    DatasetSession session;
    QString error;
};
// Readers open one complete snapshot; only QSaveFile::commit replaces it.
class DatasetSessionStore
{
  public:
    static constexpr qint64 MaximumFileBytes = 64 * 1024;
    static DatasetSessionLoadResult load(const QString &filePath);
    static bool save(const QString &filePath, const DatasetSession &session,
                     QString *error = nullptr);
};
}
