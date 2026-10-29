#pragma once

#include "core/visiontypes.h"
#include <QJsonArray>
#include <QJsonObject>

namespace vision::app
{
// The caller owns the in-memory history. This repository owns only its file path,
// so there cannot be a second independently mutable copy of the same history.
class HistoryRepository final
{
  public:
    explicit HistoryRepository(QString filePath);
    QJsonArray load(QString *error = nullptr) const;
    bool save(const QJsonArray &entries, QString *error = nullptr) const;
    QString filePath() const { return filePath_; }
    static QJsonObject entry(const InferenceResult &result);
    static void prepend(QJsonArray &entries, const InferenceResult &result);
    static QByteArray serialize(const QJsonArray &entries);

  private:
    QString filePath_;
};
} // namespace vision::app
