// SPDX-License-Identifier: AGPL-3.0-only
#include "resultdeliveryservice.h"
#include "historyrepository.h"
#include <QMetaObject>
#include <QTemporaryDir>
#include <QUuid>

namespace vision::app
{
ResultDeliveryService::ResultDeliveryService(QObject *parent)
    : QObject(parent), writer_(new ResultWriter(this))
{
    connect(writer_, &ResultWriter::pendingChanged, this, &ResultDeliveryService::pendingChanged);
    connect(writer_, &ResultWriter::completed, this, &ResultDeliveryService::complete);
    connect(writer_, &ResultWriter::shutdownReady, this, [this] {
        shutdownReady_ = true;
        emit shutdownReady();
    });
}

bool ResultDeliveryService::beginTask(const ModelConfig &config, const QString &fingerprint,
    const QHash<QString, QString> &sourceIdentities, const ResultDeliveryOptions &options)
{
    if (shuttingDown_) return false;
    const bool invalidated = cache_.fingerprint() != fingerprint;
    if (invalidated) cache_.clear();
    cache_.setFingerprint(fingerprint);
    config_ = config;
    fingerprint_ = fingerprint;
    sourceIdentities_ = sourceIdentities;
    options_ = options;
    return invalidated;
}

void ResultDeliveryService::acceptImage(const InferenceResult &result, quint64 ticket)
{
    if (shuttingDown_) { emit acknowledged(ticket); return; }
    if (result.modelClassCount > 0) config_.modelClassCount = result.modelClassCount;
    if (sourceIdentities_.contains(result.source))
        cache_.accept(result, config_, fingerprint_, sourceIdentities_.value(result.source));
    if (history_) history_->appendResult(result);
    ResultWriteRequest request;
    request.result = result;
    request.ticket = ticket;
    request.generation = cache_.generation();
    request.directory = cache_.snapshotDirectory();
    if (request.directory->isValid())
        request.snapshotPath = request.directory->filePath(QUuid::createUuid().toString(QUuid::Id128) + ".png");
    request.automaticExport = options_.automaticExport;
    if (request.automaticExport)
        request.exported = ResultWriter::prepare(result, options_.exportDirectory, config_);
    if (history_) {
        request.historyPath = history_->filePath();
        request.history = history_->serialize();
    }
    if (!writer_->enqueue(std::move(request))) emit acknowledged(ticket);
    emit historyChanged();
}

void ResultDeliveryService::finalizeStream(const InferenceResult &result)
{
    if (result.image.isNull()) {
        QMetaObject::invokeMethod(this, [this] { emit streamFinalized(); }, Qt::QueuedConnection);
        return;
    }
    if (history_) history_->appendResult(result);
    ResultWriteRequest request;
    request.result = result;
    request.generation = cache_.generation();
    request.streamFinalization = true;
    request.automaticExport = options_.automaticExport;
    if (request.automaticExport)
        request.exported = ResultWriter::prepare(result, options_.exportDirectory, config_);
    if (history_) {
        request.historyPath = history_->filePath();
        request.history = history_->serialize();
    }
    if (!writer_->enqueue(std::move(request))) {
        const QString message = "自动导出失败：结果保存服务已关闭。";
        emit fatalError(message);
        emit notice(message, true);
        QMetaObject::invokeMethod(this, [this] { emit streamFinalized(); }, Qt::QueuedConnection);
    }
    emit historyChanged();
}

void ResultDeliveryService::complete(const ResultWriteOutcome &outcome)
{
    // Stream finalization has no image-cache snapshot and no backpressure ticket.
    // It still observes the same serial history/export writes and failure ordering.
    const auto accepted = outcome.streamFinalization
        ? CacheCommitOutcome{true, false} : cache_.commit(outcome);
    if (accepted.current) {
        if (accepted.warnMemoryFallback)
            emit notice("无法写入临时结果缓存，本轮结果保存在内存中。请检查临时目录空间或权限。", true);
        if (!outcome.historySaved) emit notice("运行记录保存失败：" + outcome.historyError, true);
        if (outcome.automaticExport && !outcome.exportSaved) {
            const QString message = "自动导出失败：" + outcome.exportError;
            emit fatalError(message);
            emit notice(message, true);
        }
    }
    // Stale writes and all failures release the same worker ticket as successes.
    if (outcome.streamFinalization) emit streamFinalized();
    else emit acknowledged(outcome.ticket);
}

void ResultDeliveryService::requestShutdown()
{
    if (shuttingDown_) return;
    shuttingDown_ = true;
    writer_->requestShutdown();
}
}
