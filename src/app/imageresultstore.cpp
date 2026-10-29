// SPDX-License-Identifier: AGPL-3.0-only
#include "imageresultstore.h"
#include <QDir>
#include <QTemporaryDir>

namespace vision::app
{
bool ImageResultStore::invalidateConfiguration(const QString &fingerprint)
{
    if (entries_.isEmpty() || fingerprint_ == fingerprint) return false;
    clear();
    return true;
}
const CachedImageResult *ImageResultStore::findValid(const QString &source, const QString &sourceIdentity)
{
    auto found = entries_.find(source);
    if (found == entries_.end()) return nullptr;
    if (!sourceIdentity.isEmpty() && sourceIdentity != found->sourceIdentity) {
        entries_.erase(found);
        return nullptr;
    }
    return &found.value();
}
void ImageResultStore::clear()
{
    ++generation_; entries_.clear(); directory_.reset(); fingerprint_.clear(); warned_ = false;
}
void ImageResultStore::accept(const InferenceResult &result, const ModelConfig &config,
                              const QString &fingerprint, const QString &sourceIdentity)
{
    if (result.image.isNull() || result.demonstration) return;
    fingerprint_ = fingerprint;
    entries_.insert(result.source, CachedImageResult{result, config, {}, sourceIdentity});
}
std::shared_ptr<QTemporaryDir> ImageResultStore::snapshotDirectory()
{
    if (!directory_) directory_ = std::make_shared<QTemporaryDir>(QDir::tempPath() + "/vision-studio-image-results-XXXXXX");
    return directory_;
}
CacheCommitOutcome ImageResultStore::commit(const ResultWriteOutcome &outcome)
{
    CacheCommitOutcome accepted;
    if (outcome.generation != generation_) return accepted;
    accepted.current = true;
    auto found = entries_.find(outcome.source);
    if (found == entries_.end()) return accepted;
    if (outcome.snapshotSaved) {
        found->snapshotPath = outcome.snapshotPath;
        found->result.image = {}; found->result.originalImage = {};
    } else if (!warned_) {
        warned_ = true; accepted.warnMemoryFallback = true;
    }
    return accepted;
}
}
