// SPDX-License-Identifier: AGPL-3.0-only
#include "detectioncontroller.h"
#include "core/inferenceworker.h"
#include <QThread>

namespace vision::app
{
DetectionController::DetectionController(QObject *parent)
    : QObject(parent), thread_(new QThread), worker_(new InferenceWorker)
{
    worker_->setImageResultBackpressureEnabled(true);
    worker_->moveToThread(thread_);
    connect(thread_, &QThread::finished, worker_, &QObject::deleteLater);
    connect(thread_, &QThread::finished, this, [this] {
        threadFinished_ = true;
        if (shuttingDown_) emit shutdownReady();
    });
    connect(this, &DetectionController::dispatch, worker_, &InferenceWorker::run, Qt::QueuedConnection);
    connect(worker_, &InferenceWorker::modelReady, this, [this](const QString &backend, const ModelConfig &config) {
        state_.loadedConfig = config; emit modelReady(backend, config);
    });
    connect(worker_, &InferenceWorker::resultReady, this, &DetectionController::resultReady);
    connect(worker_, &InferenceWorker::imageResultReady, this, [this](const InferenceResult &result, quint64 ticket) {
        tickets_.insert(ticket);
        if (shuttingDown_ || state_.stopping || !state_.running) {
            acknowledgeImageResult(ticket); return;
        }
        emit imageResultReady(result, ticket);
    });
    connect(worker_, &InferenceWorker::progress, this, [this](int completed, int total) {
        state_.completed = completed; state_.total = total; emit progress(completed, total);
    });
    connect(worker_, &InferenceWorker::status, this, &DetectionController::status);
    connect(worker_, &InferenceWorker::failed, this, [this](const QString &message) {
        state_.failed = true; state_.error = message; emit failed(message);
    });
    connect(worker_, &InferenceWorker::finished, this, [this](bool cancelled) {
        state_.running = false; emit finished(cancelled);
    });
    connect(worker_, &InferenceWorker::recordingStarted, this, &DetectionController::recordingStarted);
    connect(worker_, &InferenceWorker::recordingFinished, this, &DetectionController::recordingFinished);
    connect(worker_, &InferenceWorker::recordingFailed, this, &DetectionController::recordingFailed);
    thread_->start();
}

DetectionController::~DetectionController()
{
    if (!shuttingDown_) requestShutdown();
    if (!thread_->isFinished()) {
        // Safety for early owner destruction: the running QThread outlives its
        // owner and reaps itself; normal UI destruction follows shutdownReady.
        connect(thread_, &QThread::finished, thread_, &QObject::deleteLater);
    } else delete thread_;
}

void DetectionController::prepare()
{
    if (shuttingDown_) return;
    state_ = {}; state_.running = true; worker_->prepare();
}
void DetectionController::run(JobRequest request)
{
    if (shuttingDown_) return;
    state_.request = request; state_.total = request.files.size();
    emit dispatch(std::move(request));
}
void DetectionController::prepareRecording(const QString &directory) { if (!shuttingDown_) worker_->prepareRecording(directory); }
void DetectionController::requestStop() { state_.stopping = true; if (!shuttingDown_ && !threadFinished_) worker_->requestStop(); }
void DetectionController::acknowledgeImageResult(quint64 ticket)
{
    if (tickets_.remove(ticket) && !shuttingDown_ && !threadFinished_) worker_->acknowledgeImageResult(ticket);
}
void DetectionController::requestStartRecording() { if (!shuttingDown_) worker_->requestStartRecording(); }
void DetectionController::requestStopRecording() { if (!shuttingDown_ && !threadFinished_) worker_->requestStopRecording(); }
void DetectionController::requestShutdown()
{
    if (shuttingDown_) return;
    requestStop(); shuttingDown_ = true;
    if (threadFinished_) emit shutdownReady();
    else thread_->quit();
}
bool DetectionController::isShutdownReady() const { return shuttingDown_ && threadFinished_; }
}
