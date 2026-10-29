// SPDX-License-Identifier: AGPL-3.0-only
#include "mainwindow.h"
#include "pages/workbenchpage.h"
#include "app/gpuenvironmentservice.h"
#include "core/cpuruntime.h"
#include "asyncimagepreview.h"
#include "evaluationpanel.h"
#include "icons.h"
#include "imagecanvas.h"
#include "lazyimagequeue.h"
#include "modelconversionpage.h"
#include "runtimesetup.h"

#include <QApplication>
#include <QCheckBox>
#include <QComboBox>
#include <QElapsedTimer>
#include <QFileInfo>
#include <QLabel>
#include <QListWidget>
#include <QProgressBar>
#include <QPushButton>
#include <QSignalBlocker>
#include <QTableWidget>
#include <QTabWidget>
#include <QTimer>

vision::app::InputSessionSnapshot MainWindow::inputState() const
{
    return inputSession_ ? inputSession_->snapshot() : vision::app::InputSessionSnapshot{};
}

void MainWindow::connectInputSession()
{
    connect(workbenchPage_, &WorkbenchPage::projectionChanged, this, &MainWindow::updateStartEligibility);
    connect(workbenchPage_, &WorkbenchPage::queueProjectionFinished, this,
        [this](quint64 generation, int projectedCount) {
        if (generation != inputState().projectionGeneration || isClosing()) return;
        const bool wasImporting = inputState().importing;
        inputSession_->acknowledgeQueueProjection(generation, projectedCount);
        if (!wasImporting && !workbenchPage_->inputPresentation().taskActive && !inputState().importing) {
            const auto state = inputState();
            { const QSignalBlocker blocker(queue_); queue_->setCurrentRow(state.selectedRow);
              if (!state.hasSelection) queue_->clearSelection(); }
            setBusy(false); updateSourceUi(); selectQueueImage(state.selectedRow);
        }
    });
        connect(workbenchPage_, &WorkbenchPage::previewReady, this,
            [this](quint64 generation, const QString &, const QImage &original, const QImage &image, bool primaryUsed, const QString &) {
            if (generation != workbenchPage_->previewGeneration() || isWorkbenchBusy() || isClosing() ||
                (evaluationState().review.active && inputState().datasetSamples.contains(inputState().files.value(queue_->currentRow()))) ||
                inputState().sourceKind != vision::SourceKind::Images || queue_->currentRow() < 0 || queue_->currentRow() >= inputState().files.size()) return;
            const QString source = inputState().files.at(queue_->currentRow());
            const bool hadCache = imageResults().entries().contains(source);
            const auto *cached = detection_->results()->findCached(source,
                vision::app::ModelSessionController::fileIdentity(source));
            if (hadCache && !cached) {
                queue_->item(queue_->currentRow())->setForeground(QBrush());
                selectQueueImage(queue_->currentRow()); return;
            }
            if (hadCache && primaryUsed && !image.isNull() &&
                (cached->result.sourceFrameSize.isEmpty() || original.size() == cached->result.sourceFrameSize)) {
                auto result = cached->result; result.originalImage = original; result.image = image;
                workbenchPage_->setDisplayedConfig(cached->config); displayResult(result); return;
            }
            if (hadCache) {
                detection_->results()->removeCached(source); queue_->item(queue_->currentRow())->setForeground(QBrush());
                showNotice("此图片的临时结果缓存无法读取，请重新检测。", true);
                if (primaryUsed) { selectQueueImage(queue_->currentRow()); return; }
            }
            vision::InferenceResult preview;
            preview.source = source; preview.task = currentConfig().task;
            preview.originalImage = original; preview.image = image;
            attachGroundTruth(preview); canvas_->setResult(preview);
            sizeMetric_->setText(image.isNull() ? "—" : QString("%1 × %2").arg(image.width()).arg(image.height()));
            resultInfo_->setText(image.isNull() ? "无法读取此图片，请检查文件是否存在。" :
                !inputState().datasetSamples.contains(source) ? "临时图片已就绪 · 无 GT，仅参与检测，不计入数据集评估。" : "图片已就绪，点击开始检测。");
        });

    inputSession_->setRestoreEnabled(settings_->value("restoreDatasetOnStartup", true).toBool());
    inputSession_->setRestoreGuard([this] {
        return !isClosing() && !isWorkbenchBusy() && !workbenchPage_->isQueueProjectionActive() && !gpuService_->isPreparing() &&
            (!evaluationController_ || !evaluationController_->isBusy()) &&
            (!modelConversion_ || !modelConversion_->isBusy()) && !QApplication::activeModalWidget();
    });
    connect(inputSession_, &vision::app::InputSessionController::stateChanged, this, [this] {
        if (inputState().importing && !workbenchPage_->inputPresentation().taskActive) {
            workbenchPage_->beginInputPresentation();
            operationCoordinator_->setVisibleTask(vision::app::VisibleTask::Input);
            setBusy(true);
        }
        updateSourceUi();
    });
    connect(inputSession_, &vision::app::InputSessionController::datasetReady, this,
        [this](const QJsonObject &manifest) {
        evaluation_->setDataset(manifest); showGroundTruth_->show();
    });
    connect(inputSession_, &vision::app::InputSessionController::queueReset, this,
        [this](const QStringList &files, quint64 generation) {
        workbenchPage_->cancelQueueProjection();
        { const QSignalBlocker blocker(queue_); queue_->clear(); }
        workbenchPage_->cancelImagePreview();
        canvas_->clear(); workbenchPage_->clearDisplayedResult(); predictionTable_->setRowCount(0);
        emptyResults_->show(); exportButton_->setEnabled(false);
        if (inputState().importing && files.isEmpty()) workbenchPage_->markInputReplacing();
        if (!inputState().importing && files.isEmpty()) {
            clearDatasetContext(); clearImageResults();
        }
        if (!files.isEmpty()) projectInputQueue(files, 0, generation);
    });
    connect(inputSession_, &vision::app::InputSessionController::queueAppend,
        this, &MainWindow::projectInputQueue);
    connect(inputSession_, &vision::app::InputSessionController::selectionChanged, this,
        [this](int row, bool selected) {
        if (workbenchPage_->isQueueProjectionActive() || row >= queue_->count()) return;
        { const QSignalBlocker blocker(queue_); queue_->setCurrentRow(row);
          if (!selected) queue_->clearSelection(); }
        if (!isWorkbenchBusy()) selectQueueImage(row);
        updateRangeUi();
    });
    connect(inputSession_, &vision::app::InputSessionController::progress, this,
        [this](int completed, int total, const QString &stage) {
        progress_->setRange(0, total); progress_->setValue(completed); showNotice(stage);
    });
    connect(inputSession_, &vision::app::InputSessionController::notice, this,
        [this](const QString &message) { showNotice(message); });
    connect(inputSession_, &vision::app::InputSessionController::failed, this,
        [this](const QString &message) {
        workbenchPage_->markInputFailed();
        if (inputState().datasetManifest.isEmpty()) evaluation_->showDatasetImportError(message);
        else evaluation_->setDataset(inputState().datasetManifest);
        showNotice(message, true);
    });
    connect(inputSession_, &vision::app::InputSessionController::finished, this, [this](bool cancelled) {
        workbenchPage_->endInputPresentation();
        const auto state = inputState();
        if (!cancelled && !workbenchPage_->inputPresentation().failed && workbenchPage_->inputPresentation().replacing) {
            clearImageResults(); clearDatasetContext();
            if (!state.datasetManifest.isEmpty()) {
                evaluation_->setDataset(state.datasetManifest); showGroundTruth_->show();
                detectionRange_->setCurrentIndex(0);
            }
        } else if (!state.datasetManifest.isEmpty()) evaluation_->setDataset(state.datasetManifest);
        // Appended temporary pictures must also be reachable after error review.
        if (!cancelled && !workbenchPage_->inputPresentation().failed && !workbenchPage_->inputPresentation().replacing && evaluationState().review.active) {
            evaluationController_->resetReview();
            for (int row = 0; row < queue_->count(); ++row) queue_->item(row)->setHidden(false);
        }
        if (!workbenchPage_->isQueueProjectionActive()) {
            setBusy(false); updateEvaluationModel(); updateSourceUi();
            selectQueueImage(state.selectedRow);
            static_cast<LazyImageQueue *>(queue_)->scheduleVisibleThumbnails();
        }
        if (cancelled) showNotice("已取消输入载入，启动恢复记录保留。");
        else if (!workbenchPage_->inputPresentation().failed) showNotice(QString("输入已就绪 · %1 张图片").arg(state.files.size()));
        if (isClosing()) QTimer::singleShot(0, this, &QWidget::close);
    });
    connect(inputSession_, &vision::app::InputSessionController::shutdownReady, this, [this] {
        if (isClosing()) QTimer::singleShot(0, this, &QWidget::close);
    });
}

void MainWindow::projectInputQueue(const QStringList &files, int firstRow, quint64 generation)
{
    const auto state = inputState();
    if (generation != state.projectionGeneration || isClosing()) return;
    workbenchPage_->projectQueue(files, firstRow, generation, state.datasetSamples,
        currentConfig().colorMode, !state.datasetManifest.isEmpty());
}

void MainWindow::clearDatasetContext()
{
    if (!evaluation_) return;
    workbenchPage_->cancelImagePreview();
    if (evaluationController_) evaluationController_->clear();
    evaluation_->clearDataset(); evaluation_->clearReport();
    workbenchTabs_->setCurrentIndex(0); showGroundTruth_->hide();
    for (int row = 0; queue_ && row < queue_->count(); ++row) queue_->item(row)->setHidden(false);
}

void MainWindow::loadDataset(const QString &source, const QString &split, const QString &metadata)
{
    cancelPendingDatasetSessionRestore();
    beginDatasetImport(source, split, metadata);
}

QString MainWindow::datasetSessionFilePath() const { return dataRoot_ + "/dataset-session.json"; }

void MainWindow::cancelPendingDatasetSessionRestore() { inputSession_->noteUserInteraction(); }

void MainWindow::restoreDatasetSession()
{
    if (vision::cpuRuntimeDownloadNeeded()) return;
    inputSession_->setRestoreEnabled(settings_->value("restoreDatasetOnStartup", true).toBool());
    inputSession_->restore();
}

void MainWindow::beginDatasetImport(const QString &source, const QString &split, const QString &metadata)
{
    if (previewDropLocked()) return;
    if (!ui::ensureCpuRuntime(this, [this, source, split, metadata] { beginDatasetImport(source, split, metadata); })) return;
    evaluation_->setDatasetSource(source, split, metadata);
    inputSession_->loadDataset(source, split, metadata);
}

void MainWindow::populateDatasetQueue(const QJsonObject &manifest, const std::function<void()> &completed)
{
    const auto connection = std::make_shared<QMetaObject::Connection>();
    *connection = connect(inputSession_, &vision::app::InputSessionController::finished, this,
        [this, completed, connection](bool cancelled) {
        disconnect(*connection);
        if (!cancelled && !workbenchPage_->inputPresentation().failed && completed) completed();
    });
    if (!inputSession_->openDatasetSnapshot(manifest)) disconnect(*connection);
}
