#include "runtimesetup.h"
#include "core/cpuruntime.h"
// SPDX-License-Identifier: AGPL-3.0-only
#include "mainwindow.h"
#include "smoke/smokedrivers.h"
#include "pages/workbenchpage.h"
#include "app/historyrepository.h"
#include "app/gpuenvironmentservice.h"
#include "evaluationpanel.h"
#include "imagecanvas.h"
#include "lazyimagequeue.h"
#include "asyncimagepreview.h"
#include "modelconversionpage.h"
#include "icons.h"
#include "core/evaluationworker.h"
#include "core/evaluationmatch.h"
#include "core/evaluationreview.h"
#include "core/datasetsessionstore.h"
#include "core/gpuruntime.h"
#include <QApplication>
#include <QCheckBox>
#include <QComboBox>
#include <QCryptographicHash>
#include <QDateTime>
#include <QDir>
#include <QDoubleSpinBox>
#include <QFile>
#include <QFileInfo>
#include <QHeaderView>
#include <QImageReader>
#include <QJsonDocument>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QProgressBar>
#include <QPushButton>
#include <QPixmap>
#include <QSaveFile>
#include <QScrollArea>
#include <QScrollBar>
#include <QSettings>
#include <QSignalBlocker>
#include <QSpinBox>
#include <QSplitter>
#include <QTabWidget>
#include <QTableWidget>
#include <QTimer>
#include <cmath>


void MainWindow::connectEvaluation()
{
    evaluationController_ = new vision::app::EvaluationController(this);
    connect(evaluation_, &EvaluationPanel::datasetSelectionStarted, this, &MainWindow::cancelPendingDatasetSessionRestore);
    connect(evaluation_, &EvaluationPanel::datasetRequested, this, &MainWindow::loadDataset);
    connect(evaluation_, &EvaluationPanel::configChanged, this, [this] {
        if (isWorkbenchBusy()) return;
        if (evaluationController_->invalidateForFingerprint(evaluationFingerprint())) {
            resetEvaluationReviewNavigation();
            for (int row = 0; row < queue_->count(); ++row) queue_->item(row)->setHidden(false);
        }
        updateStartEligibility(); selectQueueImage(queue_->currentRow());
    });
    connect(evaluation_, &EvaluationPanel::modeChanged, this, [this](bool mode) {
        if (isWorkbenchBusy()) return;
        evaluationController_->setReviewActive(mode);
        if (!mode) for (int row = 0; row < queue_->count(); ++row) queue_->item(row)->setHidden(false);
        if (!evaluationState().review.active) resetEvaluationReviewNavigation();
        workbenchTabs_->setCurrentIndex(mode ? 1 : 0);
        updateStartEligibility(); selectQueueImage(queue_->currentRow());
        if (evaluationState().review.active) applyEvaluationFilter(evaluationState().review.filter);
    });
    connect(workbenchTabs_, &QTabWidget::currentChanged, this, [this](int index) {
        evaluation_->hideDatasetSettings(); updateStartEligibility();
        if (!isWorkbenchBusy() && index == 0) selectQueueImage(queue_->currentRow());
    });
    connect(evaluation_, &EvaluationPanel::thresholdChanged, this, [this](double) {
        if (!isWorkbenchBusy() && !evaluationState().report.isEmpty()) applyEvaluationFilter(evaluationState().review.filter);
    });
    connect(evaluation_, &EvaluationPanel::imageFilterChanged, this, &MainWindow::applyEvaluationFilter);
    connect(evaluation_, &EvaluationPanel::errorReviewRequested, this, &MainWindow::showEvaluationErrors);
    connect(evaluation_, &EvaluationPanel::applyRequested, this, &MainWindow::applyEvaluationProfile);
    connect(evaluation_, &EvaluationPanel::reanalysisRequested, this, &MainWindow::reanalyzeEvaluation);
    connect(evaluation_, &EvaluationPanel::sceneReviewRequested, this, &MainWindow::showEvaluationSceneErrors);
    connect(evaluation_, &EvaluationPanel::externalReportImportRequested, this, &MainWindow::importEvaluationReport);
    connect(historyTable_, &QTableWidget::cellDoubleClicked, this, [this](int row, int) {
        const auto history = historySnapshot();
        if (isWorkbenchBusy() || row < 0 || row >= history.size()) return;
        const auto entry = history[row].toObject();
        if (entry.value("kind").toString() == "evaluation") loadEvaluationReport(entry.value("report_directory").toString());
    });
    connect(evaluationController_, &vision::app::EvaluationController::busyChanged, this, &MainWindow::setBusy);
    connect(evaluationController_, &vision::app::EvaluationController::progress, this,
        [this](int completed, int total, const QString &stage) {
        progress_->setRange(0, total > 0 ? total : 0); progress_->setValue(completed);
        updateRangeUi(); showNotice(total > 0 ? stage + QString(" · %1/%2").arg(completed).arg(total) : stage);
    });
    connect(evaluationController_, &vision::app::EvaluationController::failed, this,
        [this](const QString &error) { showNotice(error, true); });
    connect(evaluationController_, &vision::app::EvaluationController::reportReady, this,
        [this](const QJsonObject &report, const QString &directory) { acceptEvaluationReport(report, directory, true); });
    connect(evaluationController_, &vision::app::EvaluationController::externalImportReady, this,
        [this](const QString &directory) { if (!isClosing()) loadEvaluationReport(directory, true); });
    connect(evaluationController_, &vision::app::EvaluationController::finished, this, [this] {
        if (!isWorkbenchBusy() && evaluationState().review.active && !evaluationState().review.pending && !evaluationState().review.available)
            applyEvaluationFilter(evaluationState().review.filter);
        refreshHistory(); if (isClosing()) QTimer::singleShot(0, this, &QWidget::close);
    });
    connect(evaluationController_, &vision::app::EvaluationController::reviewChanged, this, &MainWindow::updateEvaluationReviewUi);
    connect(evaluationController_, &vision::app::EvaluationController::filterReady, this,
        [this](const vision::app::EvaluationFilterOutcome &outcome) {
        if (isWorkbenchBusy() || isClosing() || outcome.files != inputState().files || outcome.directory != evaluationState().directory) return;
        if (!outcome.valid) {
            resultInfo_->setText("错例统计不可用，请检查报告缓存；不能将校验失败视作零错例。");
            showNotice(outcome.error, true); updateEvaluationReviewUi(); return;
        }
        for (int row = 0; row < outcome.visible.size(); ++row) queue_->item(row)->setHidden(!outcome.visible[row]);
        // Selection can change while the summary is running.
        const int currentRow = queue_->currentRow();
        const bool keepSelection = !outcome.selectFirst && currentRow >= 0 && currentRow < outcome.visible.size() && outcome.visible[currentRow];
        const auto reviewRows = evaluationState().review.rows;
        const int selectedRow = keepSelection ? currentRow : (reviewRows.isEmpty() ? -1 : reviewRows.first());
        const bool changed = selectedRow != currentRow;
        queue_->setCurrentRow(selectedRow);
        if (selectedRow < 0) {
            queue_->clearSelection(); workbenchPage_->clearDisplayedResult(); canvas_->clear(); predictionTable_->setRowCount(0);
            exportButton_->setEnabled(false);
            resultInfo_->setText("当前阈值下没有符合此筛选的已完成图片。评估范围保持不变。");
        } else {
            queue_->scrollToItem(queue_->item(selectedRow), QAbstractItemView::EnsureVisible);
            if (!changed) selectQueueImage(selectedRow);
        }
        updateRangeUi(); updateEvaluationReviewUi();
    });
    connect(evaluationController_, &vision::app::EvaluationController::previewReady, this,
        [this](const vision::app::EvaluationPreviewOutcome &outcome) {
        if (isWorkbenchBusy() || isClosing() || !evaluationState().review.active || outcome.row != queue_->currentRow() ||
            outcome.directory != evaluationState().directory || inputState().files.value(outcome.row) != outcome.source) return;
        const auto &preview = outcome.preview;
        if (!preview.error.isEmpty() || preview.result.image.isNull()) {
            workbenchPage_->clearDisplayedResult(); canvas_->clear(); predictionTable_->setRowCount(0); exportButton_->setEnabled(false);
            resultInfo_->setText("原图已移动、改变或缓存损坏。评估数值快照仍保留；为避免错配，暂不叠加预测。");
            showNotice(preview.error, true); return;
        }
        workbenchPage_->setDisplayedConfig(outcome.config); displayResult(preview.result);
        if (preview.completed) queue_->item(outcome.row)->setForeground(QColor("#60CDFF"));
        else {
            workbenchPage_->clearDisplayedResult(); exportButton_->setEnabled(false); countMetric_->setText("—"); latencyMetric_->setText("—"); classMetric_->setText("—");
            resultInfo_->setText("该图片未在当前评估任务中完成推理，GT 仅用于标注预览。");
        }
    });
}

void MainWindow::updateEvaluationModel()
{
    if (!evaluation_ || !taskBox_) return;
    auto c = currentConfig();
    if (c.labels.isEmpty()) c.labels = modelState().nativeLabels;
    if (c.labels.isEmpty() && c.modelClassCount == 80) c.labels = vision::cocoLabels();
    evaluation_->setModel(c);
}

QString MainWindow::evaluationFingerprint() const
{
    auto config = vision::EvaluationWorker::modelConfigJson(currentConfig());
    for (const auto &key : {"confidence", "class_filter", "labels", "resolved_device", "device_name", "device_notice"}) config.remove(key);
    config.insert("model_identity", vision::app::ModelSessionController::fileIdentity(modelState().activeModel));
    config.insert("model_class_id", evaluation_ ? evaluation_->modelClassId() : 0);
    config.insert("dataset_class_id", evaluation_ ? evaluation_->datasetClassId() : 0);
    config.insert("dataset_fingerprint", inputState().datasetManifest.value("fingerprint"));
    if (evaluation_) {
        config.insert("max_detections", evaluation_->maxDetections());
        config.insert("score_floor", evaluation_->scoreFloor());
        config.insert("cpu_threads", evaluation_->cpuThreads());
    }
    return QString::fromLatin1(QCryptographicHash::hash(QJsonDocument(config).toJson(QJsonDocument::Compact), QCryptographicHash::Sha256).toHex());
}

void MainWindow::startEvaluation()
{
    if (!vision::app::OperationCoordinator::availability(operationCoordinator_->activity()).start) return;
    if (!ui::ensureCpuRuntime(this, [this] { startEvaluation(); })) return;
    if (isWorkbenchBusy() || evaluationController_->isBusy() || pendingResultWrites() || !evaluation_->canEvaluate()) return;
    if (modelState().inspectionPending || modelState().inspectedIdentity != modelFileIdentity()) {
        inspectSelectedModel(); showNotice("请等待当前模型类别信息加载完成后开始评估"); return;
    }
    QString error;
    const auto range = selectedEvaluationRange(&error);
    if (!error.isEmpty()) { showNotice(error, true); return; }
    vision::EvaluationRequest request;
    request.config = currentConfig(); request.manifest = inputState().datasetManifest; request.files = range;
    request.modelClassId = evaluation_->modelClassId(); request.datasetClassId = evaluation_->datasetClassId();
    request.confidence = configurationState().confidence; request.matchIou = evaluation_->matchIou();
    request.scanStart = evaluation_->scanStart(); request.scanEnd = evaluation_->scanEnd(); request.scanStep = evaluation_->scanStep();
    request.scoreFloor = evaluation_->scoreFloor(); request.batchSize = evaluation_->batchSize();
    request.config.maxDetections = evaluation_->maxDetections(); request.config.cpuThreads = evaluation_->cpuThreads();
    const auto outputOptions = evaluation_->analysisOptions();
    request.figureFormats.clear();
    for (const auto &format : outputOptions.value("figure_formats").toArray()) request.figureFormats.append(format.toString());
    request.figureDpi = outputOptions.value("figure_dpi").toInt(300);
    request.outputRoot = dataRoot_; request.projectRoot = projectRoot_;
    for (int row = 0; row < queue_->count(); ++row) queue_->item(row)->setHidden(false);
    operationCoordinator_->setVisibleTask(vision::app::VisibleTask::Evaluation);
    evaluationController_->setReviewContext(inputState().files, inputState().datasetSamples, inputState().datasetManifest);
    if (!evaluationController_->start(request, evaluationFingerprint(), &error)) { showNotice(error, true); return; }
    workbenchTabs_->setCurrentIndex(1);
}

void MainWindow::reanalyzeEvaluation(const QJsonObject &options)
{
    if (isClosing()) return;
    if (!ui::ensureCpuRuntime(this, [this, options] { reanalyzeEvaluation(options); })) return;
    if (isWorkbenchBusy() || evaluationController_->isBusy()) return;
    QString error;
    if (!evaluationController_->reanalyze(options, dataRoot_, projectRoot_, &error) && !error.isEmpty()) showNotice(error, true);
}

void MainWindow::showEvaluationSceneErrors(const QString &group)
{
    if (isWorkbenchBusy() || evaluationState().report.isEmpty()) return;
    const auto details = evaluationState().report.value("groups").toObject().value(group).toObject();
    if (!details.value("available").toBool() || !details.value("sample_images").isArray()) {
        showNotice("该报告没有可用的已审核场景名单，请生成新版评估或重新分析。", true); return;
    }
    const QString filter = "scene:" + group;
    if (evaluationPreviewFilter_->findData(filter) < 0)
        evaluationPreviewFilter_->addItem("场景错例 · " + group, filter);
    workbenchPage_->selectFirstReviewOnNextFilter();
    applyEvaluationFilter(filter);
    workbenchTabs_->setCurrentIndex(0);
}

void MainWindow::importEvaluationReport(const QString &source, const QJsonObject &options)
{
    if (isClosing()) return;
    if (!ui::ensureCpuRuntime(this, [this, source, options] { importEvaluationReport(source, options); })) return;
    if (isWorkbenchBusy() || evaluationController_->isBusy() || source.isEmpty()) return;
    QString error;
    if (!evaluationController_->importExternal(source, options, dataRoot_, projectRoot_, &error)) showNotice(error, true);
}

void MainWindow::acceptEvaluationReport(const QJsonObject &report, const QString &directory, bool record)
{
    if (evaluationState().directory != directory || evaluationState().report != report)
        evaluationController_->acceptReport(report, directory);
    evaluationController_->setReviewContext(inputState().files, inputState().datasetSamples, inputState().datasetManifest);
    evaluation_->setReport(report, directory);
    const int completed = report.value("per_image").toArray().size();
    workbenchTabs_->setTabEnabled(1, true);
    workbenchTabs_->setCurrentIndex(1);
    if (record) {
        const auto request = report.value("request").toObject();
        historyRepository_->appendEvaluation(evaluationController_->historyEntry(modelDisplayName(request.value("model_path").toString())));
        if (!historyRepository_->save())
            showNotice("评估已完成，但运行记录保存失败，请从评估目录打开报告。", true);
        auto association = modelState().evaluationProfiles.value(request.value("model_path").toString()).toMap();
        if (association.value("model_sha256").toString() != request.value("model_sha256").toString()) association.clear();
        association.insert("report_directory", directory);
        if (report.value("allow_apply_profile").toBool(true) && !request.value("model_identity").toString().isEmpty())
            association.insert("model_identity", request.value("model_identity").toString());
        association.insert("model_sha256", request.value("model_sha256").toString());
        association.insert("split", inputState().datasetManifest.value("split").toString());
        association.insert("full_split", report.value("full_split").toBool());
        association.insert("metrics", report.value("metrics").toObject().toVariantMap());
        modelSession_->setEvaluationProfile(request.value("model_path").toString(), association);
        persist(); refreshModelLibrary(); refreshHistory();
    }
    showNotice(QString("%1 · %2 张已完成样本 · 可分析阈值、回看错例与导出报告")
        .arg(report.value("status").toString() == "cancelled" ? "评估已停止" :
             report.value("full_split").toBool() ? "完整集合评估完成" : "子集评估完成").arg(completed));
    updateEvaluationReviewUi();
    if (!isWorkbenchBusy()) applyEvaluationFilter("all");
}

void MainWindow::loadEvaluationReport(const QString &directory, bool record)
{
    if (isWorkbenchBusy() || directory.isEmpty()) return;
    cancelPendingDatasetSessionRestore();
    const auto loaded = vision::app::EvaluationController::readReport(directory);
    if (!loaded.valid) { showNotice(loaded.error, true); return; }
    clearImageResults(); evaluation_->setDataset(loaded.manifest); showGroundTruth_->show();
    selectRoute(ui::Route::Workbench);
    populateDatasetQueue(loaded.manifest, [this, loaded, record] {
        evaluationController_->acceptReport(loaded.report, loaded.directory, true);
        setBusy(false); acceptEvaluationReport(loaded.report, loaded.directory, record);
        queue_->setCurrentRow(inputState().files.isEmpty() ? -1 : 0); updateSourceUi();
    });
}

void MainWindow::attachGroundTruth(vision::InferenceResult &result) const
{
    if (!evaluation_) return;
    vision::app::EvaluationController::attachGroundTruth(result, inputState().datasetManifest, inputState().datasetSamples,
        evaluation_->modelClassId(), evaluation_->datasetClassId(), modelState().classCount, evaluation_->matchIou());
}

void MainWindow::reviewEvaluationImage(int row)
{
    if (isWorkbenchBusy() || evaluationState().review.pending || row < 0 || row >= inputState().files.size()) return;
    evaluationController_->setReviewContext(inputState().files, inputState().datasetSamples, inputState().datasetManifest);
    if (!evaluationController_->requestPreview(row)) return;
    workbenchPage_->clearDisplayedResult(); canvas_->clear(); predictionTable_->setRowCount(0); exportButton_->setEnabled(false);
    resultInfo_->setText("正在读取并校验评估图像…"); updateEvaluationReviewUi();
}

void MainWindow::resetEvaluationReviewNavigation()
{
    evaluationController_->resetReview(); workbenchPage_->selectFirstReviewOnNextFilter(false);
    if (evaluation_) evaluation_->setImageFilter("all", false);
    if (evaluationPreviewFilter_) {
        QSignalBlocker blocker(evaluationPreviewFilter_);
        evaluationPreviewFilter_->setCurrentIndex(evaluationPreviewFilter_->findData("all"));
    }
    updateEvaluationReviewUi();
}

void MainWindow::updateEvaluationReviewUi()
{
    if (!evaluationReviewBar_) return;
    const bool report = !evaluationState().report.isEmpty() && inputState().sourceKind == vision::SourceKind::Images;
    evaluationReviewBar_->setVisible(report && evaluationState().review.active && !isWorkbenchBusy());
    evaluationReviewButton_->setEnabled(report && !isWorkbenchBusy() && !inputState().files.isEmpty());
    evaluationPreviewFilter_->setEnabled(!isWorkbenchBusy() && report);
    const int position = evaluationState().review.rows.indexOf(queue_->currentRow());
    const bool ready = !isWorkbenchBusy() && evaluationState().review.active && evaluationState().review.available && !evaluationState().review.pending;
    evaluationReviewPreviousButton_->setEnabled(ready && position > 0);
    evaluationReviewNextButton_->setEnabled(ready && position >= 0 && position + 1 < evaluationState().review.rows.size());
    QString summary = QString("阈值 %1 · ").arg(evaluation_->analysisThreshold(), 0, 'f', 3);
    if (evaluationState().review.pending) summary += "正在整理错例…";
    else if (!evaluationState().review.available) summary += evaluationState().review.error.isEmpty() ? "错例统计尚未就绪" : "错例统计不可用";
    else summary += QString("错图 %1 / 已评估 %2 / 集合 %3 · 当前 %4/%5")
        .arg(evaluationState().review.errorRows.size()).arg(evaluationState().review.evaluated).arg(inputState().datasetManifest.value("samples").toArray().size())
        .arg(position < 0 ? 0 : position + 1).arg(evaluationState().review.rows.size());
    evaluationReviewSummaryLabel_->setText(summary);
    evaluationReviewSummaryLabel_->setToolTip(evaluationState().review.error.isEmpty() ? summary +
        "\n错图按图片去重，包含当前阈值下的误检或漏检；导航按当前筛选的队列顺序。" : evaluationState().review.error);
    evaluation_->setErrorSummary(evaluationState().review.evaluated, evaluationState().review.errorRows.size(), evaluationState().review.falsePositive,
        evaluationState().review.falseNegative, evaluationState().review.negativeFalsePositive, inputState().datasetManifest.value("samples").toArray().size(), evaluationState().review.pending,
        evaluationState().review.available, evaluationState().review.error);
}

void MainWindow::showEvaluationErrors()
{
    if (isWorkbenchBusy() || evaluationState().report.isEmpty() || inputState().files.isEmpty() || inputState().sourceKind != vision::SourceKind::Images) return;
    evaluationController_->setReviewActive(true);
    workbenchPage_->selectFirstReviewOnNextFilter();
    applyEvaluationFilter("errors");
    workbenchTabs_->setCurrentIndex(0);
}

void MainWindow::navigateEvaluationReview(int step)
{
    if (isWorkbenchBusy()) return;
    const int row = evaluationController_->navigationTarget(queue_->currentRow(), step);
    if (row < 0) return;
    queue_->setCurrentRow(row); queue_->scrollToItem(queue_->item(row), QAbstractItemView::EnsureVisible);
    updateEvaluationReviewUi();
}

void MainWindow::applyEvaluationFilter(const QString &filter)
{
    if (isWorkbenchBusy() || evaluationState().report.isEmpty()) return;
    evaluationController_->setReviewContext(inputState().files, inputState().datasetSamples, inputState().datasetManifest);
    QString error;
    const bool selectFirst = workbenchPage_->takeSelectFirstReview();
    if (!evaluationController_->applyFilter(filter, evaluation_->analysisThreshold(), queue_->currentRow(), selectFirst, &error)) {
        if (!error.isEmpty()) showNotice(error, true);
        return;
    }
    evaluation_->setImageFilter(filter.startsWith("scene:") ? "errors" : filter, false);
    { QSignalBlocker blocker(evaluationPreviewFilter_); evaluationPreviewFilter_->setCurrentIndex(evaluationPreviewFilter_->findData(filter)); }
    workbenchPage_->clearDisplayedResult(); canvas_->clear(); predictionTable_->setRowCount(0); exportButton_->setEnabled(false);
    resultInfo_->setText("正在整理当前阈值的评估图片与错例…"); updateEvaluationReviewUi();
}

void MainWindow::applyEvaluationProfile(double threshold)
{
    if (isWorkbenchBusy() || evaluationState().report.isEmpty()) return;
    const auto patch = evaluationController_->configurationPatch(modelState().activeModel, threshold,
        modelState().evaluationProfiles.value(modelState().activeModel).toMap());
    if (!patch.valid) { showNotice(patch.error, true); return; }
    const auto c = patch.config;
    const auto request = evaluationState().report.value("request").toObject();
    const auto normalized = [](auto *editor, auto requested) {
        const QSignalBlocker blocker(editor);
        editor->setValue(requested);
        return editor->value();
    };
    auto draft = configurationState();
    draft.parameters.task = c.task; draft.parameters.inputSize = normalized(inputSize_, c.inputSize);
    draft.iou = normalized(iou_, c.iou);
    draft.parameters.colorMode = c.colorMode; draft.parameters.swapRB = c.swapRB;
    draft.parameters.scale = normalized(scale_, c.scale);
    draft.parameters.meanR = normalized(meanR_, c.meanR);
    draft.parameters.meanG = normalized(meanG_, c.meanG);
    draft.parameters.meanB = normalized(meanB_, c.meanB);
    draft.confidence = normalized(confidence_, threshold);
    draft.classFilterText = QString::number(request.value("model_class_id").toInt());
    configuration_->replaceDraft(draft);
    updateTaskUi();
    setComputeDevice(c.device, c.deviceIndex);
    modelSession_->setEvaluationProfile(modelState().activeModel, patch.profile);
    evaluationController_->setReviewActive(true);
    selectQueueImage(queue_->currentRow()); persist(); refreshModelLibrary();
    showNotice(QString("已采用检测配置 · 置信度 %1 · 原始类别 ID %2 · 每图最多 %3 框 · 尺寸/颜色/NMS/设备与报告一致")
        .arg(threshold, 0, 'f', 3).arg(patch.classId).arg(c.maxDetections));
}

void MainWindow::runEvaluationSmoke(const QString &directory, const QString &requestPath)
{
    vision::ui::smoke::runEvaluationSmoke(*this, directory, requestPath);
}

void MainWindow::runDatasetSessionSmoke(const QString &directory)
{
    vision::ui::smoke::runDatasetSessionSmoke(*this, directory);
}
