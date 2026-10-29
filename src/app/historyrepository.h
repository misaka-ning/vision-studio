#pragma once

#include "core/visiontypes.h"
#include <QJsonArray>
#include <QJsonObject>

namespace vision::app
{
// Owns the in-memory history and its existing JSON file. UI consumers receive
// value snapshots; all production mutations go through append commands.
class HistoryRepository final
{
  public:
    explicit HistoryRepository(QString filePath);
    QJsonArray snapshot() const { return entries_; }
    void reload(QString *error = nullptr);
    void appendResult(const InferenceResult &result);
    void appendEvaluation(const QJsonObject &entry);
    QByteArray serialize() const;
    bool save(QString *error = nullptr) const;
    bool exportTo(const QString &destination, QString *error = nullptr) const;
    // Compatibility helpers for frozen snapshot tests and independent readers.
    QJsonArray load(QString *error = nullptr) const;
    bool save(const QJsonArray &entries, QString *error = nullptr) const;
    QString filePath() const { return filePath_; }
    static QJsonObject entry(const InferenceResult &result);
    static void prepend(QJsonArray &entries, const InferenceResult &result);
    static QByteArray serialize(const QJsonArray &entries);

  private:
    QString filePath_;
    QJsonArray entries_;
};
} // namespace vision::app
