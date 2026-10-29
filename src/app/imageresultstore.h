// SPDX-License-Identifier: AGPL-3.0-only
#pragma once
#include "resultwriter.h"
#include <QHash>

namespace vision::app
{
struct CachedImageResult
{
    InferenceResult result;
    ModelConfig config;
    QString snapshotPath, sourceIdentity;
};
struct CacheCommitOutcome { bool current = false, warnMemoryFallback = false; };

class ImageResultStore final
{
  public:
    const QHash<QString, CachedImageResult> &entries() const { return entries_; }
    const QString &fingerprint() const { return fingerprint_; }
    quint64 generation() const { return generation_; }
    void setFingerprint(const QString &fingerprint) { fingerprint_ = fingerprint; }
    // Changing the inference identity invalidates the complete result generation.
    // The caller owns preview decoration; the store owns cache validity.
    bool invalidateConfiguration(const QString &fingerprint);
    // An unavailable source identity retains the existing entry, as before.
    // A known replacement invalidates only this source.
    const CachedImageResult *findValid(const QString &source, const QString &sourceIdentity);
    void clear();
    void remove(const QString &source) { entries_.remove(source); }
    void accept(const InferenceResult &result, const ModelConfig &config,
                const QString &fingerprint, const QString &sourceIdentity);
    std::shared_ptr<QTemporaryDir> snapshotDirectory();
    CacheCommitOutcome commit(const ResultWriteOutcome &outcome);
  private:
    QHash<QString, CachedImageResult> entries_;
    std::shared_ptr<QTemporaryDir> directory_;
    QString fingerprint_;
    quint64 generation_ = 0;
    bool warned_ = false;
};
}
