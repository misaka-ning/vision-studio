// SPDX-License-Identifier: AGPL-3.0-only
#include "detectioncontroller.h"
#include "asyncthreadretirement.h"
#include "core/inferenceworker.h"
#include <QThread>

namespace vision::app
{
DetectionController::DetectionController(QObject *parent)
    : QObject(parent), thread_(new QThread), worker_(new InferenceWorker), results_(new ResultDeliveryService(this))
{
    connect(results_, &ResultDeliveryService::acknowledged, this, &DetectionController::acknowledgeImageResult);
    connect(results_, &ResultDeliveryService::fatalError, this, [this](const QString &error) {
        reportFailure(error);
        requestStop();
    });
    connect(results_, &ResultDeliveryService::shutdownReady, this, &DetectionController::checkShutdownReady);
    connect(results_, &ResultDeliveryService::streamFinalized, this, &DetectionController::completeFinish);
    worker_->setImageResultBackpressureEnabled(true);
    worker_->moveToThread(thread_);
    connect(thread_, &QThread::finished, worker_, &QObject::deleteLater);
    AsyncThreadRetirement::watch(thread_, this, [this] {
        thread_ = nullptr;
        threadFinished_ = true;
        if (state_.running) beginFinish(true);
        else if (shuttingDown_) results_->requestShutdown();
        checkShutdownReady();
    });
    connect(this, &DetectionController::dispatch, worker_, &InferenceWorker::run, Qt::QueuedConnection);
    connect(worker_, &InferenceWorker::modelReady, this, [this](const QString &backend, const ModelConfig &config) {
        state_.loadedConfig = config;
        results_->updateLoadedConfiguration(config);
        emit modelReady(backend, config);
    });
    connect(worker_, &InferenceWorker::resultReady, this, [this](const InferenceResult &result) {
        if (state_.request.sourceKind != SourceKind::Images && state_.running) lastStreamResult_ = result;
        emit resultReady(result);
    });
    connect(worker_, &InferenceWorker::imageResultReady, this, [this](const InferenceResult &result, quint64 ticket) {
        tickets_.insert(ticket);
        if (shuttingDown_ || state_.stopping || !state_.running) {
            acknowledgeImageResult(ticket); return;
        }
        ++state_.completed;
        emit imageResultReady(result, ticket);
    });
    connect(worker_, &InferenceWorker::progress, this, [this](int completed, int total) {
        if (state_.request.sourceKind != SourceKind::Images) state_.completed = completed;
        state_.total = total; emit progress(completed, total);
    });
    connect(worker_, &InferenceWorker::status, this, &DetectionController::status);
    connect(worker_, &InferenceWorker::failed, this, [this](const QString &message) {
        state_.failed = true; state_.error = message; emit failed(message);
    });
    connect(worker_, &InferenceWorker::finished, this, &DetectionController::beginFinish);
    connect(worker_, &InferenceWorker::recordingStarted, this, &DetectionController::recordingStarted);
    connect(worker_, &InferenceWorker::recordingFinished, this, &DetectionController::recordingFinished);
    connect(worker_, &InferenceWorker::recordingFailed, this, &DetectionController::recordingFailed);
    thread_->start();
}

DetectionController::~DetectionController()
{
    if (!shuttingDown_) requestShutdown();
    // The retirement ticket owns the native join and QThread object even when
    // this owner is destroyed early; no GUI-thread implicit destructor wait.
}

void DetectionController::prepare()
{
    if (shuttingDown_) return;
    state_ = {}; state_.running = true; worker_->prepare();
    lastStreamResult_ = {}; finishing_ = false; finishCancelled_ = false;
}
void DetectionController::run(JobRequest request)
{
    if (shuttingDown_) return;
    state_.request = request; state_.total = request.files.size();
    emit dispatch(std::move(request));
}
void DetectionController::prepareTask(const JobRequest &request, int first, int last,
    const QString &modelIdentity, const QString &fingerprint,
    const QHash<QString, QString> &sourceIdentities)
{
    prepare();
    if (shuttingDown_) return;
    state_.request = request; state_.total = request.files.size();
    state_.rangeFirst = first; state_.rangeLast = last;
    state_.modelIdentity = modelIdentity; state_.configurationFingerprint = fingerprint;
    state_.sourceIdentities = sourceIdentities;
}
void DetectionController::reportFailure(const QString &error)
{
    state_.failed = true; state_.error = error;
}
void DetectionController::setConfigurationFingerprint(const QString &fingerprint)
{
    state_.configurationFingerprint = fingerprint;
    results_->setConfigurationFingerprint(fingerprint);
}
bool DetectionController::beginResultDelivery(const ResultDeliveryOptions &options)
{
    return results_->beginTask(state_.request.config, state_.configurationFingerprint,
                               state_.sourceIdentities, options);
}
bool DetectionController::acceptImageResult(const InferenceResult &result, quint64 ticket)
{
    if (ticket && (!state_.running || state_.stopping || shuttingDown_)) {
        acknowledgeImageResult(ticket);
        return false;
    }
    results_->acceptImage(result, ticket);
    return true;
}
void DetectionController::prepareRecording(const QString &directory) { if (!shuttingDown_) worker_->prepareRecording(directory); }
void DetectionController::requestStop() { state_.stopping = true; if (!shuttingDown_ && !threadFinished_ && worker_) worker_->requestStop(); }
void DetectionController::acknowledgeImageResult(quint64 ticket)
{
    if (tickets_.remove(ticket) && !shuttingDown_ && !threadFinished_ && worker_) worker_->acknowledgeImageResult(ticket);
}
void DetectionController::requestStartRecording() { if (!shuttingDown_ && worker_) worker_->requestStartRecording(); }
void DetectionController::requestStopRecording() { if (!shuttingDown_ && !threadFinished_ && worker_) worker_->requestStopRecording(); }
void DetectionController::requestShutdown()
{
    if (shuttingDown_) return;
    requestStop(); shuttingDown_ = true;
    if (!threadFinished_) thread_->quit();
    if (threadFinished_ && state_.running) beginFinish(true);
    if (!state_.running && !finishing_) results_->requestShutdown();
    checkShutdownReady();
}
void DetectionController::beginFinish(bool cancelled)
{
    if (finishing_) return;
    if (!state_.running) {
        if (shuttingDown_) results_->requestShutdown();
        checkShutdownReady();
        return;
    }
    finishing_ = true;
    finishCancelled_ = cancelled;
    if (state_.request.sourceKind != SourceKind::Images && !lastStreamResult_.image.isNull())
        results_->finalizeStream(lastStreamResult_);
    else completeFinish();
}
void DetectionController::completeFinish()
{
    if (!finishing_) return;
    finishing_ = false;
    state_.running = false;
    if (shuttingDown_) results_->requestShutdown();
    emit finished(finishCancelled_);
    checkShutdownReady();
}
bool DetectionController::isShutdownReady() const
{
    return shuttingDown_ && threadFinished_ && results_->isShutdownReady();
}
void DetectionController::checkShutdownReady()
{
    if (isShutdownReady() && !shutdownReadyEmitted_) {
        shutdownReadyEmitted_ = true;
        emit shutdownReady();
    }
}
}
