// SPDX-License-Identifier: AGPL-3.0-only
#pragma once
#include "imageresultstore.h"
#include <QObject>

namespace vision::app
{
class HistoryRepository;

struct ResultDeliveryOptions
{
    bool automaticExport = false;
    QString exportDirectory;
};

// Owns accepted image results and their bounded, serial persistence pipeline.
// DetectionController owns this service; presentation only reads cache snapshots.
class ResultDeliveryService final : public QObject
{
    Q_OBJECT
  public:
    explicit ResultDeliveryService(QObject *parent = nullptr);
    void setHistoryRepository(HistoryRepository *repository) { history_ = repository; }
    bool beginTask(const ModelConfig &config, const QString &fingerprint,
                   const QHash<QString, QString> &sourceIdentities,
                   const ResultDeliveryOptions &options);
    void updateLoadedConfiguration(const ModelConfig &config) { config_ = config; }
    void setConfigurationFingerprint(const QString &fingerprint) { fingerprint_ = fingerprint; }
    // Every accepted or rejected ticket is eventually acknowledged exactly once.
    void acceptImage(const InferenceResult &result, quint64 ticket);
    void finalizeStream(const InferenceResult &result);
    const ImageResultStore &cache() const { return cache_; }
    bool hasResultsForConfiguration(const QString &fingerprint) const
    { return !cache_.entries().isEmpty() && cache_.fingerprint() == fingerprint; }
    bool invalidateConfiguration(const QString &fingerprint) { return cache_.invalidateConfiguration(fingerprint); }
    const CachedImageResult *findCached(const QString &source, const QString &identity) { return cache_.findValid(source, identity); }
    void clearCache() { cache_.clear(); }
    void removeCached(const QString &source) { cache_.remove(source); }
    int pendingCount() const { return writer_->pendingCount(); }
    void requestShutdown();
    bool isShutdownReady() const { return shutdownReady_; }
  signals:
    void acknowledged(quint64 ticket);
    void notice(QString message, bool error);
    void fatalError(QString message);
    void historyChanged();
    void streamFinalized();
    void pendingChanged(int count);
    void shutdownReady();
  private:
    void complete(const ResultWriteOutcome &outcome);
    ImageResultStore cache_;
    ResultWriter *writer_;
    HistoryRepository *history_ = nullptr;
    ModelConfig config_;
    QString fingerprint_;
    QHash<QString, QString> sourceIdentities_;
    ResultDeliveryOptions options_;
    bool shuttingDown_ = false, shutdownReady_ = false;
};
}
