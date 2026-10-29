// SPDX-License-Identifier: AGPL-3.0-only
#include "mainwindow.h"
#include "app/asyncthreadretirement.h"
#include "pages/workbenchpage.h"
#include "app/gpuenvironmentservice.h"
#include "app/recordingcontroller.h"
#include "asyncimagepreview.h"
#include "lazyimagequeue.h"
#include "modelconversionpage.h"
#include "modelviewer.h"
#include "runtimesetup.h"
#include <QApplication>
#include <QThreadPool>
#include <QTimer>

bool MainWindow::isWorkbenchBusy() const
{
    return operationCoordinator_ && operationCoordinator_->activity().workbenchBusy();
}

int MainWindow::completedCount() const
{
    if (!operationCoordinator_) return 0;
    if (operationCoordinator_->visibleTask() == vision::app::VisibleTask::Evaluation) return evaluationState().completed;
    if (operationCoordinator_->visibleTask() == vision::app::VisibleTask::Detection) return detection_->snapshot().completed;
    return 0;
}
QStringList MainWindow::taskFiles() const
{
    if (!operationCoordinator_) return {};
    if (operationCoordinator_->visibleTask() == vision::app::VisibleTask::Evaluation) return evaluationState().taskFiles;
    if (operationCoordinator_->visibleTask() == vision::app::VisibleTask::Detection) return detection_->snapshot().request.files;
    return {};
}
int MainWindow::taskFirst() const
{
    return operationCoordinator_->visibleTask() == vision::app::VisibleTask::Evaluation
        ? evaluationState().rangeFirst : operationCoordinator_->visibleTask() == vision::app::VisibleTask::Detection
            ? detection_->snapshot().rangeFirst : 0;
}
int MainWindow::taskLast() const
{
    return operationCoordinator_->visibleTask() == vision::app::VisibleTask::Evaluation
        ? evaluationState().rangeLast : operationCoordinator_->visibleTask() == vision::app::VisibleTask::Detection
            ? detection_->snapshot().rangeLast : 0;
}

void MainWindow::initializeOperationCoordinator()
{
    connect(operationCoordinator_, &vision::app::OperationCoordinator::shutdownStarted,
            this, [this] { setBusy(true); setEnabled(false); });
    operationCoordinator_->setActivityProvider([this] {
        vision::app::ActivitySnapshot state;
        state.detection = detection_->snapshot().running;
        state.input = inputState().importing;
        state.projection = workbenchPage_->isQueueProjectionActive();
        state.evaluation = evaluationController_->isBusy();
        state.conversion = modelConversion_->isBusy();
        state.gpuPreparation = gpuService_->isPreparing();
        state.saving = pendingResultWrites() > 0;
        state.modal = QApplication::activeModalWidget();
        return state;
    });
    auto add = [this](const QString &name, auto *service) {
        operationCoordinator_->addShutdownParticipant(name, service,
            [service] { service->requestShutdown(); }, [service] { return service->isShutdownReady(); });
    };
    add("input", inputSession_); add("model", modelSession_); add("evaluation", evaluationController_);
    add("gpu", gpuService_); add("recordings", recordingsController_); add("detection", detection_);
    add("model-viewer", modelViewer_); add("conversion", modelConversion_);
    operationCoordinator_->addShutdownParticipant("preview", workbenchPage_,
        [this] { workbenchPage_->requestPreviewShutdown(); },
        [this] { return workbenchPage_->isPreviewShutdownReady(); });
    operationCoordinator_->addShutdownParticipant("thumbnail-pool", qApp,
        [this] { static_cast<LazyImageQueue *>(queue_)->cancelPendingThumbnails(); }, [] {
        const auto *pool = qApp->findChild<QThreadPool *>("visionThumbnailPool");
        return !pool || pool->activeThreadCount() == 0;
    });
    operationCoordinator_->addShutdownParticipant("preview-pool", qApp, [] {}, [] {
        return AsyncImagePreview::applicationPool()->activeRequestCount() == 0;
    });
    // Window shutdown cancels preparations but does not permanently disable a
    // QApplication-wide service needed by a subsequently created window.
    auto *runtime = ui::runtimeSetup();
    operationCoordinator_->addShutdownParticipant("runtime-setup", runtime,
        [runtime] { runtime->cancelPendingPreparations(); }, [runtime] { return runtime->isShutdownReady(); });
    operationCoordinator_->addShutdownParticipant("shared-background-jobs", qApp, [] {}, [] {
        return QThreadPool::globalInstance()->activeThreadCount() == 0;
    });
    operationCoordinator_->addShutdownParticipant("thread-retirement", qApp, [] {}, [] {
        return vision::app::AsyncThreadRetirement::activeCount() == 0;
    });
    connect(operationCoordinator_, &vision::app::OperationCoordinator::shutdownReady, this,
        [this] {
            QTimer::singleShot(0, this, [this] {
                close();
                if (exitRequested_) QApplication::exit(requestedExitCode_);
            });
        });
}

void MainWindow::requestExit(int code)
{
    requestedExitCode_ = code;
    exitRequested_ = true;
    QApplication::setQuitOnLastWindowClosed(false);
    close();
    if (isShutdownReady()) QApplication::exit(requestedExitCode_);
}
