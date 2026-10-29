#include "runtimesetup.h"
#include "core/cpuruntime.h"
// SPDX-License-Identifier: AGPL-3.0-only
#include "mainwindow.h"
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

namespace
{
QString identity(const QString &path)
{
    const QFileInfo f(path);
    return f.isFile() ? f.absoluteFilePath() + "|" + QString::number(f.size()) + "|" +
                           QString::number(f.lastModified().toMSecsSinceEpoch()) : QString();
}
QString sha256(const QString &path)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) return {};
    QCryptographicHash hash(QCryptographicHash::Sha256);
    while (!file.atEnd()) {
        auto chunk = file.read(1024 * 1024);
        if (chunk.isEmpty() && file.error() != QFile::NoError) return {};
        hash.addData(chunk);
    }
    return QString::fromLatin1(hash.result().toHex());
}
QJsonObject jsonFile(const QString &path)
{
    QFile f(path);
    if (!f.open(QIODevice::ReadOnly)) return {};
    return QJsonDocument::fromJson(f.readAll()).object();
}
QRectF boxOf(const QJsonObject &value)
{
    const auto b = value.value("box").toArray();
    return b.size() == 4 ? QRectF(QPointF(b[0].toDouble(), b[1].toDouble()),
                                  QPointF(b[2].toDouble(), b[3].toDouble())) : QRectF();
}
vision::ModelConfig configOf(const QJsonObject &o)
{
    vision::ModelConfig c;
    c.modelPath = o.value("model_path").toString();
    c.task = o.value("task").toString() == "yolov5" ? vision::ModelTask::YoloV5 :
             o.value("task").toString() == "classification" ? vision::ModelTask::Classification : vision::ModelTask::YoloV8;
    c.inputSize = o.value("input_size").toInt(640);
    c.inputChannels = o.value("input_channels").toInt();
    c.modelClassCount = o.value("model_class_count").toInt();
    c.confidence = float(o.value("confidence").toDouble(.25));
    c.iou = float(o.value("nms_iou").toDouble(.45));
    c.swapRB = o.value("swap_rb").toBool(true);
    c.colorMode = o.value("color_mode").toString() == "grayscale" ? vision::InputColorMode::Grayscale : vision::InputColorMode::Color;
    c.scale = o.value("scale").toDouble(1. / 255);
    c.meanR = o.value("mean_r").toDouble(); c.meanG = o.value("mean_g").toDouble(); c.meanB = o.value("mean_b").toDouble();
    c.device = o.value("device").toString() == "cuda" ? vision::ComputeDevice::CUDA :
               o.value("device").toString() == "auto" ? vision::ComputeDevice::Auto : vision::ComputeDevice::CPU;
    c.resolvedDevice = o.value("resolved_device").toString() == "cuda" ? vision::ComputeDevice::CUDA : vision::ComputeDevice::CPU;
    c.deviceIndex = o.value("device_index").toInt();
    c.deviceName = o.value("device_name").toString();
    c.deviceNotice = o.value("device_notice").toString();
    c.maxDetections = o.value("max_detections").toInt(300);
    c.cpuThreads = o.value("cpu_threads").toInt();
    c.resolvedCpuThreads = o.value("resolved_cpu_threads").toInt();
    for (const auto &v : o.value("labels").toArray()) c.labels.append(v.toString());
    for (const auto &v : o.value("class_filter").toArray()) c.classFilter.append(v.toInt());
    return c;
}
QString datasetName(const QJsonObject &manifest, int id)
{
    const auto names = manifest.value("names").toArray();
    return id >= 0 && id < names.size() ? names[id].toString() : QString::number(id);
}
// Smoke-only observations use the real widget tree and clipping ancestors.
// They do not run a nested event loop or alter ordinary workbench behavior.
QJsonObject smokeWidgetGeometry(QWidget *widget, QWidget *window)
{
    if (!widget) return {{"exists", false}, {"fully_visible", false}};
    const QRect bounds(widget->mapTo(window, QPoint()), widget->size());
    bool visible = widget->isVisibleTo(window) && !widget->size().isEmpty() && window->rect().contains(bounds);
    QJsonArray clips;
    for (auto *parent = widget->parentWidget(); parent && parent != window; parent = parent->parentWidget()) {
        const QRect clipping(widget->mapTo(parent, QPoint()), widget->size());
        clips.append(QJsonObject{{"ancestor", parent->objectName()}, {"width", parent->width()}, {"height", parent->height()},
            {"child_x", clipping.x()}, {"child_y", clipping.y()}, {"child_width", clipping.width()}, {"child_height", clipping.height()}});
        visible = visible && parent->rect().contains(clipping);
    }
    QJsonObject result{{"exists", true}, {"object_name", widget->objectName()}, {"fully_visible", visible},
        {"x", bounds.x()}, {"y", bounds.y()}, {"width", bounds.width()}, {"height", bounds.height()},
        {"enabled", widget->isEnabled()}, {"clipping_ancestors", clips}};
    if (auto *label = qobject_cast<QLabel *>(widget)) {
        const auto content = label->contentsRect();
        const int required = label->fontMetrics().boundingRect(QRect(0, 0, qMax(1, content.width()), 100000),
            label->wordWrap() ? Qt::TextWordWrap : 0, label->text()).height();
        const int layoutRequired = label->heightForWidth(label->width());
        const bool textFits = content.height() >= required && (layoutRequired < 0 || label->height() >= layoutRequired) &&
            (label->wordWrap() || label->fontMetrics().horizontalAdvance(label->text()) <= content.width());
        result.insert("text_geometry", QJsonObject{{"text", label->text()}, {"word_wrap", label->wordWrap()},
            {"contents_width", content.width()}, {"contents_height", content.height()}, {"font_line_height", label->fontMetrics().height()},
            {"required_text_height", required}, {"layout_height_for_width", layoutRequired}, {"actual_height", label->height()},
            {"unwrapped_text_width", label->fontMetrics().horizontalAdvance(label->text())}, {"text_fits", textFits}});
    }
    return result;
}
QJsonArray smokeSplitterSizes(QSplitter *splitter)
{
    QJsonArray sizes;
    if (splitter) for (int size : splitter->sizes()) sizes.append(size);
    return sizes;
}
}

void MainWindow::connectEvaluation()
{
    connect(evaluation_, &EvaluationPanel::datasetSelectionStarted, this, &MainWindow::cancelPendingDatasetSessionRestore);
    connect(evaluation_, &EvaluationPanel::datasetRequested, this, &MainWindow::loadDataset);
    connect(evaluation_, &EvaluationPanel::configChanged, this, [this] {
        if (busy_) return;
        if (evaluationReviewActive_ && evaluationRunFingerprint_ != evaluationFingerprint()) {
            evaluationReviewActive_ = false;
            resetEvaluationReviewNavigation();
            if (evaluationPreviewCancellation_) evaluationPreviewCancellation_->store(true);
            if (evaluationFilterCancellation_) evaluationFilterCancellation_->store(true);
            ++evaluationPreviewGeneration_; ++evaluationFilterGeneration_;
            for (int row = 0; row < queue_->count(); ++row) queue_->item(row)->setHidden(false);
        }
        updateStartEligibility();
        selectQueueImage(queue_->currentRow());
    });
    connect(evaluation_, &EvaluationPanel::modeChanged, this, [this](bool mode) {
        if (busy_) return;
        if (evaluationPreviewCancellation_) evaluationPreviewCancellation_->store(true);
        if (evaluationFilterCancellation_) evaluationFilterCancellation_->store(true);
        ++evaluationPreviewGeneration_; ++evaluationFilterGeneration_;
        if (!mode) for (int row = 0; row < queue_->count(); ++row) queue_->item(row)->setHidden(false);
        evaluationReviewActive_ = mode && !evaluationReport_.isEmpty();
        if (!evaluationReviewActive_) resetEvaluationReviewNavigation();
        if (mode) workbenchTabs_->setCurrentIndex(1);
        else workbenchTabs_->setCurrentIndex(0);
        updateStartEligibility();
        selectQueueImage(queue_->currentRow());
        if (evaluationReviewActive_) applyEvaluationFilter(evaluationReviewFilter_);
    });
    connect(workbenchTabs_, &QTabWidget::currentChanged, this, [this](int index) {
        evaluation_->hideDatasetSettings();
        updateStartEligibility();
        // Navigation is presentation only: retain task mode, parameters and caches.
        if (!busy_ && index == 0) selectQueueImage(queue_->currentRow());
    });
    connect(evaluation_, &EvaluationPanel::thresholdChanged, this, [this](double) {
        if (!busy_ && !evaluationReport_.isEmpty()) {
            evaluationReviewActive_ = true;
            applyEvaluationFilter(evaluationReviewFilter_);
        }
    });
    connect(evaluation_, &EvaluationPanel::imageFilterChanged, this, &MainWindow::applyEvaluationFilter);
    connect(evaluation_, &EvaluationPanel::errorReviewRequested, this, &MainWindow::showEvaluationErrors);
    connect(evaluation_, &EvaluationPanel::applyRequested, this, &MainWindow::applyEvaluationProfile);
    connect(evaluation_, &EvaluationPanel::reanalysisRequested, this, &MainWindow::reanalyzeEvaluation);
    connect(evaluation_, &EvaluationPanel::sceneReviewRequested, this, &MainWindow::showEvaluationSceneErrors);
    connect(evaluation_, &EvaluationPanel::externalReportImportRequested, this, &MainWindow::importEvaluationReport);
    connect(historyTable_, &QTableWidget::cellDoubleClicked, this, [this](int row, int) {
        if (busy_ || row < 0 || row >= history_.size()) return;
        auto entry = history_[row].toObject();
        if (entry.value("kind").toString() == "evaluation") loadEvaluationReport(entry.value("report_directory").toString());
    });
}

void MainWindow::updateEvaluationModel()
{
    if (!evaluation_ || !taskBox_) return;
    auto c = currentConfig();
    if (c.labels.isEmpty()) c.labels = nativeLabels_;
    if (c.labels.isEmpty() && c.modelClassCount == 80) c.labels = vision::cocoLabels();
    evaluation_->setModel(c);
}

QString MainWindow::evaluationFingerprint() const
{
    auto config = vision::EvaluationWorker::modelConfigJson(currentConfig());
    for (const auto &key : {"confidence", "class_filter", "labels", "resolved_device", "device_name", "device_notice"}) config.remove(key);
    config.insert("model_identity", identity(modelPath_));
    config.insert("model_class_id", evaluation_ ? evaluation_->modelClassId() : 0);
    config.insert("dataset_class_id", evaluation_ ? evaluation_->datasetClassId() : 0);
    config.insert("dataset_fingerprint", datasetManifest_.value("fingerprint"));
    if (evaluation_) {
        config.insert("max_detections", evaluation_->maxDetections());
        config.insert("score_floor", evaluation_->scoreFloor());
        config.insert("cpu_threads", evaluation_->cpuThreads());
    }
    return QString::fromLatin1(QCryptographicHash::hash(QJsonDocument(config).toJson(QJsonDocument::Compact), QCryptographicHash::Sha256).toHex());
}

void MainWindow::clearDatasetContext()
{
    if (loadingDataset_ || !evaluation_) return;
    resetEvaluationReviewNavigation();
    ++queuePopulationGeneration_;
    if (queuePopulationTimer_) { queuePopulationTimer_->stop(); queuePopulationTimer_->deleteLater(); queuePopulationTimer_ = nullptr; }
    queuePopulating_ = false;
    ++imagePreviewGeneration_;
    if (imagePreviewLoader_) imagePreviewLoader_->cancel();
    if (evaluationPreviewCancellation_) evaluationPreviewCancellation_->store(true);
    if (evaluationFilterCancellation_) evaluationFilterCancellation_->store(true);
    ++evaluationPreviewGeneration_; ++evaluationFilterGeneration_;
    datasetManifest_ = {}; datasetSamples_.clear(); evaluationSampleIndices_.clear();
    evaluationReport_ = {}; evaluationDirectory_.clear(); evaluationReviewActive_ = false;
    evaluation_->clearDataset(); evaluation_->clearReport();
    workbenchTabs_->setCurrentIndex(0);
    showGroundTruth_->hide();
    for (int row = 0; queue_ && row < queue_->count(); ++row) queue_->item(row)->setHidden(false);
}

void MainWindow::loadDataset(const QString &yaml, const QString &split, const QString &metadata)
{
    cancelPendingDatasetSessionRestore();
    beginDatasetImport(yaml, split, metadata);
}

QString MainWindow::datasetSessionFilePath() const
{
    return dataRoot_ + "/dataset-session.json";
}

void MainWindow::cancelPendingDatasetSessionRestore()
{
    datasetSessionUserInteracted_ = true;
    ++datasetSessionRestoreGeneration_;
}

void MainWindow::restoreDatasetSession()
{
    if (datasetSessionRestoreRequested_ || vision::cpuRuntimeDownloadNeeded()) return;
    datasetSessionRestoreRequested_ = true;
    if (!settings_->value("restoreDatasetOnStartup", true).toBool()) return;
    const quint64 generation = datasetSessionRestoreGeneration_;
    QTimer::singleShot(0, this, [this, generation] {
        // A startup convenience must never replace an explicit input choice or
        // begin while a native/modal chooser owns a nested event loop.
        if (!settings_->value("restoreDatasetOnStartup", true).toBool() ||
            generation != datasetSessionRestoreGeneration_ || datasetSessionUserInteracted_ ||
            closing_ || busy_ || queuePopulating_ || evaluationWorker_ || gpuService_->isPreparing() ||
            (modelConversion_ && modelConversion_->isBusy()) || QApplication::activeModalWidget() ||
            sourceKind_ != vision::SourceKind::Images || !files_.isEmpty() || !datasetManifest_.isEmpty()) return;
        const auto result = vision::DatasetSessionStore::load(datasetSessionFilePath());
        if (result.status == vision::DatasetSessionStatus::Missing) return;
        if (result.status != vision::DatasetSessionStatus::Loaded) {
            showNotice("未能恢复上次数据集：" + result.error, true);
            return;
        }
        beginDatasetImport(result.session.sourcePath, result.session.split, result.session.metadataPath);
    });
}

void MainWindow::beginDatasetImport(const QString &yaml, const QString &split, const QString &metadata)
{
    if (busy_ || evaluationWorker_ || gpuService_->isPreparing() || (modelConversion_ && modelConversion_->isBusy())) return;
    if (closing_) return;
    if (!ui::ensureCpuRuntime(this, [this, yaml, split, metadata] { beginDatasetImport(yaml, split, metadata); })) return;
    const quint64 generation = ++datasetImportGeneration_;
    const QString source = QDir::cleanPath(QFileInfo(yaml).absoluteFilePath());
    auto *worker = new vision::EvaluationWorker(this);
    const auto delivered = std::make_shared<bool>(false);
    const auto cancelled = std::make_shared<std::atomic_bool>(false);
    datasetImportCancellation_ = cancelled;
    const auto importError = std::make_shared<QString>();
    evaluation_->setDatasetSource(yaml, split, metadata);
    evaluationWorker_ = worker;
    worker->configureDataset(yaml, split, metadata, projectRoot_);
    taskRangeFiles_.clear(); completed_ = 0;
    setBusy(true);
    updateSourceUi();
    connect(worker, &vision::EvaluationWorker::progress, this, [this, worker, generation, cancelled](int, int, const QString &stage) {
        if (generation == datasetImportGeneration_ && evaluationWorker_ == worker && !cancelled->load() && !closing_)
            showNotice(stage);
    });
    connect(worker, &vision::EvaluationWorker::failed, this, [this, worker, generation, cancelled, importError](const QString &message) {
        if (generation != datasetImportGeneration_ || evaluationWorker_ != worker || cancelled->load() || closing_) return;
        *importError = message;
        if (datasetManifest_.isEmpty()) evaluation_->showDatasetImportError(message);
        else evaluation_->setDataset(datasetManifest_);
        showNotice(message, true);
    });
    connect(worker, &vision::EvaluationWorker::datasetReady, this, [this, worker, generation, source, delivered, cancelled](const QJsonObject &manifest) {
        if (generation != datasetImportGeneration_ || evaluationWorker_ != worker || cancelled->load() || closing_) return;
        *delivered = true;
        clearDatasetContext();
        datasetManifest_ = manifest; datasetSamples_.clear();
        evaluationReport_ = {}; evaluationDirectory_.clear(); evaluationReviewActive_ = false;
        evaluation_->clearReport(); evaluation_->setDataset(manifest);
        clearImageResults(); files_.clear(); queue_->clear();
        canvas_->clear(); predictionTable_->setRowCount(0); exportButton_->setEnabled(false);
        sourceKind_ = vision::SourceKind::Images;
        showGroundTruth_->show();
        detectionRange_->setCurrentIndex(0);
        workbenchTabs_->setCurrentIndex(0);
        lastResult_ = {};
        populateDatasetQueue(manifest, [this, generation, source, cancelled, manifest] {
            if (generation != datasetImportGeneration_ || cancelled->load() || closing_) return;
            if (!evaluationWorker_) setBusy(false);
            updateEvaluationModel(); updateSourceUi();
            queue_->setCurrentRow(files_.isEmpty() ? -1 : 0);
            selectQueueImage(queue_->currentRow());
            QString notice = QString("数据集预检查完成 · %1 张有效图片 · %2").arg(files_.size()).arg(manifest.value("split").toString());
            if (manifest.value("root_relocated").toBool()) notice += " · 旧路径已按当前文件夹读取，原文件未修改";
            showNotice(notice);
            // Only a completely accepted import becomes the next startup
            // source. Keep the directory/YAML entry point, not data_yaml.
            vision::DatasetSession session;
            session.sourcePath = source;
            session.split = manifest.value("split").toString("val");
            session.metadataPath = manifest.value("metadata_path").toString();
            QString error;
            if (!vision::DatasetSessionStore::save(datasetSessionFilePath(), session, &error))
                showNotice("数据集已载入，但无法保存下次启动记录：" + error, true);
        });
    });
    connect(worker, &QThread::finished, this, [this, worker, generation, delivered, importError, cancelled] {
        if (generation != datasetImportGeneration_ || evaluationWorker_ != worker) {
            worker->deleteLater();
            return;
        }
        if (!*delivered) {
            if (datasetManifest_.isEmpty()) evaluation_->showDatasetImportError(importError->isEmpty() ? QStringLiteral("数据集检查已取消，可重新检查。") : *importError);
            else evaluation_->setDataset(datasetManifest_);
        }
        evaluationWorker_ = nullptr; worker->deleteLater();
        if (datasetImportCancellation_ == cancelled) datasetImportCancellation_.reset();
        if (!queuePopulating_) {
            setBusy(false); updateEvaluationModel(); updateSourceUi();
            selectQueueImage(queue_->currentRow());
        }
        if (closing_) QTimer::singleShot(0, this, &QWidget::close);
    });
    worker->start();
}

void MainWindow::populateDatasetQueue(const QJsonObject &manifest, const std::function<void()> &completed)
{
    const quint64 generation = ++queuePopulationGeneration_;
    queuePopulating_ = true;
    if (queuePopulationTimer_) { queuePopulationTimer_->stop(); queuePopulationTimer_->deleteLater(); }
    queuePopulationTimer_ = new QTimer(this);
    queuePopulationTimer_->setInterval(0);
    auto samples = std::make_shared<QJsonArray>(manifest.value("samples").toArray());
    auto offset = std::make_shared<int>(0);
    auto *timer = queuePopulationTimer_;
    const QIcon placeholder = ui::icon("image");
    static_cast<LazyImageQueue *>(queue_)->setColorMode(currentConfig().colorMode);
    connect(timer, &QTimer::timeout, this, [this, generation, samples, offset, timer, placeholder, completed] {
        if (generation != queuePopulationGeneration_) { timer->stop(); timer->deleteLater(); return; }
        QElapsedTimer budget; budget.start();
        {
            const QSignalBlocker queueSignals(queue_);
            queue_->setUpdatesEnabled(false);
            int added = 0;
            while (*offset < samples->size() && added < 96 && budget.elapsed() < 5) {
                const auto sample = samples->at((*offset)++).toObject();
                const QString path = sample.value("image").toString();
                files_.append(path); datasetSamples_.insert(path, sample);
                auto *item = new QListWidgetItem(placeholder, QFileInfo(path).fileName());
                item->setData(Qt::UserRole, path);
                item->setData(Qt::UserRole + 11, false);
                item->setToolTip(path + "\nGT · " + QString::number(sample.value("gt").toArray().size()) + " 个标注");
                queue_->addItem(item); ++added;
            }
            queue_->setUpdatesEnabled(true);
        }
        progress_->setRange(0, samples->size()); progress_->setValue(*offset);
        showNotice(QString("正在载入队列 · %1 / %2 张").arg(*offset).arg(samples->size()));
        if (*offset < samples->size()) return;
        timer->stop(); timer->deleteLater(); queuePopulationTimer_ = nullptr; queuePopulating_ = false;
        static_cast<LazyImageQueue *>(queue_)->scheduleVisibleThumbnails();
        completed();
    });
    timer->start();
}

void MainWindow::startEvaluation()
{
    if (!ui::ensureCpuRuntime(this, [this] { startEvaluation(); })) return;
    if (busy_ || evaluationWorker_ || resultIoPending_ || !evaluation_->canEvaluate()) return;
    if (modelInspectionPending_ || inspectedModelIdentity_ != modelFileIdentity()) {
        inspectSelectedModel(); showNotice("请等待当前模型类别信息加载完成后开始评估"); return;
    }
    QString error;
    const auto range = selectedEvaluationRange(&error);
    if (!error.isEmpty()) { showNotice(error, true); return; }
    vision::EvaluationRequest request;
    request.config = currentConfig(); request.manifest = datasetManifest_; request.files = range;
    request.modelClassId = evaluation_->modelClassId(); request.datasetClassId = evaluation_->datasetClassId();
    request.confidence = confidence_->value(); request.matchIou = evaluation_->matchIou();
    request.scanStart = evaluation_->scanStart(); request.scanEnd = evaluation_->scanEnd(); request.scanStep = evaluation_->scanStep();
    request.scoreFloor = evaluation_->scoreFloor(); request.batchSize = evaluation_->batchSize();
    request.config.maxDetections = evaluation_->maxDetections(); request.config.cpuThreads = evaluation_->cpuThreads();
    const auto outputOptions = evaluation_->analysisOptions();
    request.figureFormats.clear();
    for (const auto &format : outputOptions.value("figure_formats").toArray()) request.figureFormats.append(format.toString());
    request.figureDpi = outputOptions.value("figure_dpi").toInt(300);
    request.outputRoot = dataRoot_; request.projectRoot = projectRoot_;
    evaluationRunConfig_ = request.config;
    evaluationRunFingerprint_ = evaluationFingerprint();
    if (evaluationPreviewCancellation_) evaluationPreviewCancellation_->store(true);
    if (evaluationFilterCancellation_) evaluationFilterCancellation_->store(true);
    ++evaluationPreviewGeneration_; ++evaluationFilterGeneration_;
    for (int row = 0; row < queue_->count(); ++row) queue_->item(row)->setHidden(false);
    evaluationReviewActive_ = false;
    resetEvaluationReviewNavigation();
    taskRangeFiles_ = range; taskRangeFirst_ = files_.indexOf(range.first()) + 1; taskRangeLast_ = files_.indexOf(range.last()) + 1;
    completed_ = 0;
    auto *worker = new vision::EvaluationWorker(this); evaluationWorker_ = worker;
    worker->configureEvaluation(request);
    setBusy(true); workbenchTabs_->setCurrentIndex(1);
    connect(worker, &vision::EvaluationWorker::progress, this, [this](int completed, int total, const QString &stage) {
        completed_ = completed;
        progress_->setRange(0, total > 0 ? total : 0); progress_->setValue(completed);
        updateRangeUi(); showNotice(stage + QString(" · %1/%2").arg(completed).arg(total));
    });
    connect(worker, &vision::EvaluationWorker::failed, this, [this](const QString &message) { showNotice(message, true); });
    connect(worker, &vision::EvaluationWorker::reportReady, this, [this](const QJsonObject &report, const QString &directory) {
        acceptEvaluationReport(report, directory, true);
    });
    connect(worker, &QThread::finished, this, [this, worker] {
        evaluationWorker_ = nullptr; worker->deleteLater(); setBusy(false);
        if (evaluationReviewActive_ && !evaluationReviewPending_ && !evaluationReviewAvailable_)
            applyEvaluationFilter(evaluationReviewFilter_);
        refreshHistory();
        if (closing_) QTimer::singleShot(0, this, &QWidget::close);
    });
    worker->start();
}

void MainWindow::reanalyzeEvaluation(const QJsonObject &options)
{
    if (!ui::ensureCpuRuntime(this, [this, options] { reanalyzeEvaluation(options); })) return;
    if (busy_ || evaluationWorker_ || evaluationReport_.isEmpty() || evaluationDirectory_.isEmpty()) return;
    if (evaluationPreviewCancellation_) evaluationPreviewCancellation_->store(true);
    if (evaluationFilterCancellation_) evaluationFilterCancellation_->store(true);
    ++evaluationPreviewGeneration_; ++evaluationFilterGeneration_;
    auto *worker = new vision::EvaluationWorker(this);
    evaluationWorker_ = worker;
    worker->configureReanalysis(evaluationDirectory_, options, dataRoot_, projectRoot_);
    setBusy(true);
    connect(worker, &vision::EvaluationWorker::progress, this, [this](int, int, const QString &stage) {
        progress_->setRange(0, 0); showNotice(stage);
    });
    connect(worker, &vision::EvaluationWorker::failed, this, [this](const QString &message) { showNotice(message, true); });
    connect(worker, &vision::EvaluationWorker::reportReady, this, [this](const QJsonObject &report, const QString &directory) {
        acceptEvaluationReport(report, directory, true);
    });
    connect(worker, &QThread::finished, this, [this, worker] {
        evaluationWorker_ = nullptr; worker->deleteLater(); setBusy(false); refreshHistory();
        if (evaluationReviewActive_) applyEvaluationFilter(evaluationReviewFilter_);
        if (closing_) QTimer::singleShot(0, this, &QWidget::close);
    });
    worker->start();
}

void MainWindow::showEvaluationSceneErrors(const QString &group)
{
    if (busy_ || evaluationReport_.isEmpty()) return;
    const auto details = evaluationReport_.value("groups").toObject().value(group).toObject();
    if (!details.value("available").toBool() || !details.value("sample_images").isArray()) {
        showNotice("该报告没有可用的已审核场景名单，请生成新版评估或重新分析。", true); return;
    }
    const QString filter = "scene:" + group;
    if (evaluationPreviewFilter_->findData(filter) < 0)
        evaluationPreviewFilter_->addItem("场景错例 · " + group, filter);
    evaluationReviewSelectFirst_ = true;
    applyEvaluationFilter(filter);
    workbenchTabs_->setCurrentIndex(0);
}

void MainWindow::importEvaluationReport(const QString &source, const QJsonObject &options)
{
    if (!ui::ensureCpuRuntime(this, [this, source, options] { importEvaluationReport(source, options); })) return;
    if (busy_ || evaluationWorker_ || source.isEmpty()) return;
    if (evaluationPreviewCancellation_) evaluationPreviewCancellation_->store(true);
    if (evaluationFilterCancellation_) evaluationFilterCancellation_->store(true);
    ++evaluationPreviewGeneration_; ++evaluationFilterGeneration_;
    auto *worker = new vision::EvaluationWorker(this);
    evaluationWorker_ = worker;
    const auto imported = std::make_shared<QString>();
    worker->configureExternalImport(source, options, dataRoot_, projectRoot_);
    setBusy(true);
    connect(worker, &vision::EvaluationWorker::progress, this, [this](int, int, const QString &) {
        progress_->setRange(0, 0); showNotice("正在校验并导入脚本评估快照，不执行模型推理…");
    });
    connect(worker, &vision::EvaluationWorker::failed, this, [this](const QString &error) { showNotice(error, true); });
    connect(worker, &vision::EvaluationWorker::reportReady, this, [imported](const QJsonObject &, const QString &path) {
        *imported = path;
    });
    connect(worker, &QThread::finished, this, [this, worker, imported] {
        evaluationWorker_ = nullptr; worker->deleteLater(); setBusy(false);
        if (!closing_ && !imported->isEmpty()) loadEvaluationReport(*imported, true);
        else if (!closing_ && evaluationReviewActive_) applyEvaluationFilter(evaluationReviewFilter_);
        if (closing_) QTimer::singleShot(0, this, &QWidget::close);
    });
    worker->start();
}

void MainWindow::acceptEvaluationReport(const QJsonObject &report, const QString &directory, bool record)
{
    resetEvaluationReviewNavigation();
    if (evaluationPreviewCancellation_) evaluationPreviewCancellation_->store(true);
    if (evaluationFilterCancellation_) evaluationFilterCancellation_->store(true);
    ++evaluationPreviewGeneration_; ++evaluationFilterGeneration_;
    evaluationReviewFilter_ = "all";
    evaluationReport_ = report; evaluationDirectory_ = directory;
    evaluation_->setReport(report, directory);
    evaluationRunConfig_ = configOf(report.value("config").toObject());
    evaluationSampleIndices_.clear();
    const auto selected = report.value("request").toObject().value("sample_images").toArray();
    const int completed = report.value("per_image").toArray().size();
    for (int i = 0; i < std::min(int(selected.size()), completed); ++i)
        evaluationSampleIndices_.insert(selected[i].toString(), i);
    evaluationReviewActive_ = true;
    workbenchTabs_->setTabEnabled(1, true);
    workbenchTabs_->setCurrentIndex(1);
    if (record) {
        const auto request = report.value("request").toObject();
        QJsonObject entry{{"time", QDateTime::currentDateTime().toString(Qt::ISODateWithMs)},
            {"kind", "evaluation"}, {"source", datasetManifest_.value("data_yaml")},
            {"model", modelDisplayName(request.value("model_path").toString())},
            {"model_path", request.value("model_path")}, {"report_directory", directory},
            {"backend", report.value("config").toObject().value("resolved_device")},
            {"objects", report.value("metrics").toObject().value("gt_boxes")},
            {"task", report.value("status").toString() == "cancelled" ? "评估 · 已停止" :
                report.value("full_split").toBool() ? "评估 · 完整集合" : "评估 · 子集"},
            {"split", datasetManifest_.value("split")}, {"metrics", report.value("metrics")}};
        history_.prepend(entry); while (history_.size() > 200) history_.removeLast();
        QSaveFile file(dataRoot_ + "/history.json");
        const auto bytes = QJsonDocument(history_).toJson();
        if (!file.open(QIODevice::WriteOnly) || file.write(bytes) != bytes.size() || !file.commit())
            showNotice("评估已完成，但运行记录保存失败，请从评估目录打开报告。", true);
        auto association = modelEvaluationProfiles_.value(request.value("model_path").toString()).toMap();
        if (association.value("model_sha256").toString() != request.value("model_sha256").toString()) association.clear();
        association.insert("report_directory", directory);
        if (report.value("allow_apply_profile").toBool(true) && !request.value("model_identity").toString().isEmpty())
            association.insert("model_identity", request.value("model_identity").toString());
        association.insert("model_sha256", request.value("model_sha256").toString());
        association.insert("split", datasetManifest_.value("split").toString());
        association.insert("full_split", report.value("full_split").toBool());
        association.insert("metrics", report.value("metrics").toObject().toVariantMap());
        modelEvaluationProfiles_.insert(request.value("model_path").toString(), association);
        persist(); refreshModelLibrary(); refreshHistory();
    }
    showNotice(QString("%1 · %2 张已完成样本 · 可分析阈值、回看错例与导出报告")
        .arg(report.value("status").toString() == "cancelled" ? "评估已停止" :
             report.value("full_split").toBool() ? "完整集合评估完成" : "子集评估完成").arg(completed));
    updateEvaluationReviewUi();
    if (!busy_) applyEvaluationFilter("all");
}

void MainWindow::loadEvaluationReport(const QString &directory, bool record)
{
    if (busy_ || directory.isEmpty()) return;
    cancelPendingDatasetSessionRestore();
    auto report = jsonFile(directory + "/report.json");
    auto status = jsonFile(directory + "/status.json");
    if (report.value("schema_version").toInt() != 1 ||
        (status.value("status").toString() != "complete" && status.value("status").toString() != "cancelled")) {
        showNotice("该评估记录缺失、损坏或尚未完成保存。", true); return;
    }
    auto manifest = report.value("manifest").toObject();
    if (manifest.isEmpty()) manifest = jsonFile(directory + "/manifest.json");
    if (manifest.value("samples").toArray().isEmpty()) { showNotice("评估报告没有有效数据集快照。", true); return; }
    clearImageResults(); datasetManifest_ = manifest; datasetSamples_.clear();
    files_.clear(); queue_->clear(); sourceKind_ = vision::SourceKind::Images;
    evaluation_->setDataset(manifest); showGroundTruth_->show();
    setBusy(true); selectRoute(0);
    populateDatasetQueue(manifest, [this, report, directory, record] {
        setBusy(false);
        acceptEvaluationReport(report, directory, record);
        evaluationRunFingerprint_.clear(); // Historical snapshot remains explicit.
        queue_->setCurrentRow(files_.isEmpty() ? -1 : 0);
        updateSourceUi();
    });
}

void MainWindow::attachGroundTruth(vision::InferenceResult &result) const
{
    result.groundTruth.clear(); result.matchedPredictions.clear(); result.matchedGroundTruth.clear(); result.ignoredPredictions.clear();
    result.hasGroundTruth = false;
    result.hasEvaluationMatches = false;
    if (!evaluation_ || datasetManifest_.isEmpty() || result.task == vision::ModelTask::Classification ||
        !datasetSamples_.contains(result.source)) return;
    const auto sample = datasetSamples_.value(result.source);
    const int datasetId = evaluation_->datasetClassId(), modelId = evaluation_->modelClassId();
    if (datasetId < 0 || modelId < 0) return;
    result.hasGroundTruth = true;
    for (const auto &value : sample.value("gt").toArray()) {
        auto gt = value.toObject(); if (gt.value("class_id").toInt(-1) != datasetId) continue;
        result.groundTruth.append({datasetId, datasetName(datasetManifest_, datasetId), 1, boxOf(gt)});
    }
    for (int i = 0; i < result.predictions.size(); ++i)
        if (result.predictions[i].classId != modelId) result.ignoredPredictions.append(i);
    if (!result.modelName.isEmpty() && modelId < modelClassCount_ &&
        datasetId < datasetManifest_.value("names").toArray().size())
        vision::matchEvaluation(result, evaluation_->matchIou());
}

void MainWindow::reviewEvaluationImage(int row)
{
    if (busy_ || evaluationReviewPending_ || row < 0 || row >= files_.size()) return;
    updateEvaluationReviewUi();
    if (evaluationPreviewCancellation_) evaluationPreviewCancellation_->store(true);
    const auto cancelled = evaluationPreviewCancellation_ = std::make_shared<std::atomic_bool>(false);
    const quint64 generation = ++evaluationPreviewGeneration_;
    const auto path = files_[row];
    vision::EvaluationPreviewRequest request;
    request.directory = evaluationDirectory_;
    request.index = evaluationSampleIndices_.value(path, -1);
    request.sample = datasetSamples_.value(path);
    request.config = evaluationRunConfig_;
    const auto parameters = evaluationReport_.value("request").toObject();
    request.modelClassId = parameters.value("model_class_id").toInt();
    request.datasetClassId = parameters.value("dataset_class_id").toInt();
    request.datasetClassName = datasetName(datasetManifest_, request.datasetClassId);
    request.confidence = evaluation_->analysisThreshold();
    request.matchIou = parameters.value("match_iou").toDouble(.5);
    lastResult_ = {}; canvas_->clear(); predictionTable_->setRowCount(0); exportButton_->setEnabled(false);
    resultInfo_->setText("正在读取并校验评估图像…");
    const auto output = std::make_shared<vision::EvaluationPreview>();
    auto *thread = QThread::create([request, cancelled, output] {
        *output = vision::loadEvaluationPreview(request, [cancelled] { return cancelled->load(); });
    });
    connect(thread, &QThread::finished, this, [this, generation, cancelled, output, path, row, request] {
        if (cancelled->load() || generation != evaluationPreviewGeneration_ || busy_ || !evaluationReviewActive_ ||
            request.directory != evaluationDirectory_ || row != queue_->currentRow() || row >= files_.size() || files_[row] != path) return;
        if (!output->error.isEmpty() || output->result.image.isNull()) {
            lastResult_ = {}; canvas_->clear(); predictionTable_->setRowCount(0); exportButton_->setEnabled(false);
            resultInfo_->setText("原图已移动、改变或缓存损坏。评估数值快照仍保留；为避免错配，暂不叠加预测。");
            showNotice(output->error, true); return;
        }
        lastConfig_ = request.config; lastConfig_.confidence = float(request.confidence); lastConfig_.classFilter = {request.modelClassId};
        displayResult(output->result);
        if (output->completed) queue_->item(row)->setForeground(QColor("#60CDFF"));
        else {
            lastResult_ = {}; exportButton_->setEnabled(false); countMetric_->setText("—"); latencyMetric_->setText("—"); classMetric_->setText("—");
            resultInfo_->setText("该图片未在当前评估任务中完成推理，GT 仅用于标注预览。");
        }
    });
    connect(thread, &QThread::finished, thread, &QObject::deleteLater);
    thread->start();
}

void MainWindow::resetEvaluationReviewNavigation()
{
    if (evaluationFilterCancellation_) evaluationFilterCancellation_->store(true);
    ++evaluationFilterGeneration_;
    evaluationReviewPending_ = evaluationReviewAvailable_ = evaluationReviewSelectFirst_ = false;
    evaluationReviewRows_.clear(); evaluationErrorRows_.clear(); evaluationReviewError_.clear();
    evaluationEvaluatedCount_ = evaluationFalsePositiveCount_ = evaluationFalseNegativeCount_ = evaluationNegativeFalsePositiveCount_ = 0;
    evaluationReviewFilter_ = "all";
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
    const bool report = !evaluationReport_.isEmpty() && sourceKind_ == vision::SourceKind::Images;
    evaluationReviewBar_->setVisible(report && evaluationReviewActive_ && !busy_);
    evaluationReviewButton_->setEnabled(report && !busy_ && !files_.isEmpty());
    evaluationPreviewFilter_->setEnabled(!busy_ && report);
    const int position = evaluationReviewRows_.indexOf(queue_->currentRow());
    const bool ready = !busy_ && evaluationReviewActive_ && evaluationReviewAvailable_ && !evaluationReviewPending_;
    evaluationReviewPreviousButton_->setEnabled(ready && position > 0);
    evaluationReviewNextButton_->setEnabled(ready && position >= 0 && position + 1 < evaluationReviewRows_.size());
    QString summary = QString("阈值 %1 · ").arg(evaluation_->analysisThreshold(), 0, 'f', 3);
    if (evaluationReviewPending_) summary += "正在整理错例…";
    else if (!evaluationReviewAvailable_) summary += evaluationReviewError_.isEmpty() ? "错例统计尚未就绪" : "错例统计不可用";
    else summary += QString("错图 %1 / 已评估 %2 / 集合 %3 · 当前 %4/%5")
        .arg(evaluationErrorRows_.size()).arg(evaluationEvaluatedCount_).arg(datasetManifest_.value("samples").toArray().size())
        .arg(position < 0 ? 0 : position + 1).arg(evaluationReviewRows_.size());
    evaluationReviewSummaryLabel_->setText(summary);
    evaluationReviewSummaryLabel_->setToolTip(evaluationReviewError_.isEmpty() ? summary +
        "\n错图按图片去重，包含当前阈值下的误检或漏检；导航按当前筛选的队列顺序。" : evaluationReviewError_);
    evaluation_->setErrorSummary(evaluationEvaluatedCount_, evaluationErrorRows_.size(), evaluationFalsePositiveCount_,
        evaluationFalseNegativeCount_, evaluationNegativeFalsePositiveCount_, datasetManifest_.value("samples").toArray().size(), evaluationReviewPending_,
        evaluationReviewAvailable_, evaluationReviewError_);
}

void MainWindow::showEvaluationErrors()
{
    if (busy_ || evaluationReport_.isEmpty() || files_.isEmpty() || sourceKind_ != vision::SourceKind::Images) return;
    evaluationReviewActive_ = true;
    evaluationReviewSelectFirst_ = true;
    applyEvaluationFilter("errors");
    workbenchTabs_->setCurrentIndex(0);
}

void MainWindow::navigateEvaluationReview(int step)
{
    if (busy_ || evaluationReviewPending_ || !evaluationReviewAvailable_ || !evaluationReviewActive_) return;
    const int position = evaluationReviewRows_.indexOf(queue_->currentRow());
    const int next = position + step;
    if (position < 0 || next < 0 || next >= evaluationReviewRows_.size()) return;
    const int row = evaluationReviewRows_[next];
    queue_->setCurrentRow(row);
    queue_->scrollToItem(queue_->item(row), QAbstractItemView::EnsureVisible);
    updateEvaluationReviewUi();
}

void MainWindow::applyEvaluationFilter(const QString &filter)
{
    if (busy_ || evaluationReport_.isEmpty()) return;
    const bool sceneFilter = filter.startsWith("scene:");
    QStringList sceneImages;
    if (sceneFilter) {
        const auto group = evaluationReport_.value("groups").toObject().value(filter.mid(6)).toObject();
        if (!group.value("available").toBool() || !group.value("sample_images").isArray()) return;
        for (const auto &value : group.value("sample_images").toArray()) sceneImages.append(value.toString());
    } else if (!QStringList{"all", "errors", "fp", "fn", "negative_fp"}.contains(filter)) return;
    evaluationReviewFilter_ = filter;
    evaluationReviewActive_ = true;
    evaluation_->setImageFilter(sceneFilter ? "errors" : filter, false);
    {
        QSignalBlocker blocker(evaluationPreviewFilter_);
        evaluationPreviewFilter_->setCurrentIndex(evaluationPreviewFilter_->findData(filter));
    }
    if (evaluationFilterCancellation_) evaluationFilterCancellation_->store(true);
    if (evaluationPreviewCancellation_) evaluationPreviewCancellation_->store(true);
    ++evaluationPreviewGeneration_;
    const auto cancelled = evaluationFilterCancellation_ = std::make_shared<std::atomic_bool>(false);
    const quint64 generation = ++evaluationFilterGeneration_;
    const bool selectFirst = evaluationReviewSelectFirst_;
    evaluationReviewSelectFirst_ = false;
    evaluationReviewPending_ = true; evaluationReviewAvailable_ = false; evaluationReviewError_.clear();
    evaluationReviewRows_.clear(); evaluationErrorRows_.clear();
    lastResult_ = {}; canvas_->clear(); predictionTable_->setRowCount(0); exportButton_->setEnabled(false);
    resultInfo_->setText("正在整理当前阈值的评估图片与错例…");
    updateEvaluationReviewUi();
    const auto parameters = evaluationReport_.value("request").toObject();
    const auto files = files_; const auto indices = evaluationSampleIndices_; const auto directory = evaluationDirectory_;
    const auto threshold = evaluation_->analysisThreshold();
    const auto result = std::make_shared<vision::EvaluationReviewSummary>();
    auto *thread = QThread::create([result, files, indices, directory, threshold, parameters, filter, cancelled, sceneImages] {
        *result = vision::summarizeEvaluationSamples(files, indices, directory,
            parameters.value("model_class_id").toInt(), parameters.value("dataset_class_id").toInt(), threshold,
            parameters.value("match_iou").toDouble(.5), filter, [cancelled] { return cancelled->load(); }, sceneImages);
    });
    connect(thread, &QThread::finished, this, [this, result, generation, cancelled, files, directory, selectFirst] {
        if (cancelled->load() || generation != evaluationFilterGeneration_ || busy_ || !evaluationReviewActive_ ||
            directory != evaluationDirectory_ || files != files_) return;
        evaluationReviewPending_ = false;
        if (!result->valid || result->visible.size() != files_.size()) {
            evaluationReviewAvailable_ = false;
            evaluationReviewError_ = result->error.isEmpty() ? QStringLiteral("评估缓存校验失败，无法整理错例。") : result->error;
            resultInfo_->setText("错例统计不可用，请检查报告缓存；不能将校验失败视作零错例。");
            showNotice(evaluationReviewError_, true);
            updateEvaluationReviewUi();
            return;
        }
        evaluationReviewAvailable_ = true;
        evaluationErrorRows_ = result->errorRows;
        evaluationEvaluatedCount_ = result->evaluatedRows.size();
        evaluationFalsePositiveCount_ = result->falsePositiveRows.size();
        evaluationFalseNegativeCount_ = result->falseNegativeRows.size();
        evaluationNegativeFalsePositiveCount_ = result->negativeFalsePositiveRows.size();
        evaluationReviewRows_.clear();
        for (int row = 0; row < files_.size(); ++row) {
            const bool datasetImage = datasetSamples_.contains(files_.at(row));
            const bool visible = result->visible[row] && (datasetImage || evaluationReviewFilter_ == "all");
            queue_->item(row)->setHidden(!visible);
            if (visible && datasetImage) evaluationReviewRows_.append(row);
        }
        int row = queue_->currentRow();
        const bool temporarySelection = evaluationReviewFilter_ == "all" && row >= 0 && row < files_.size() &&
            !datasetSamples_.contains(files_.at(row));
        if (selectFirst || (!temporarySelection && !evaluationReviewRows_.contains(row)))
            row = evaluationReviewRows_.isEmpty() ? -1 : evaluationReviewRows_.first();
        const bool changed = row != queue_->currentRow();
        queue_->setCurrentRow(row);
        if (row < 0) {
            queue_->clearSelection(); lastResult_ = {}; canvas_->clear(); predictionTable_->setRowCount(0);
            exportButton_->setEnabled(false);
            resultInfo_->setText("当前阈值下没有符合此筛选的已完成图片。评估范围保持不变。");
        } else {
            queue_->scrollToItem(queue_->item(row), QAbstractItemView::EnsureVisible);
            if (!changed) selectQueueImage(row);
        }
        updateRangeUi(); updateEvaluationReviewUi();
    });
    connect(thread, &QThread::finished, thread, &QObject::deleteLater);
    thread->start();
}

void MainWindow::applyEvaluationProfile(double threshold)
{
    if (busy_ || evaluationReport_.isEmpty()) return;
    if (!evaluationReport_.value("allow_apply_profile").toBool(true)) {
        showNotice("导入报告缺少完整的原始后端配置，仅供分析，不能自动采用检测配置。", true); return;
    }
    const auto request = evaluationReport_.value("request").toObject();
    const QString path = request.value("model_path").toString();
    if (QFileInfo(path).absoluteFilePath() != QFileInfo(modelPath_).absoluteFilePath() ||
        sha256(path) != request.value("model_sha256").toString()) {
        showNotice("请先在模型库选用报告对应的同一模型文件，再采用该检测配置。", true); return;
    }
    const auto c = configOf(evaluationReport_.value("config").toObject());
    taskBox_->setCurrentIndex(int(c.task)); inputSize_->setValue(c.inputSize); iou_->setValue(c.iou);
    const auto color = c.colorMode == vision::InputColorMode::Grayscale ? "grayscale" : c.swapRB ? "rgb" : "bgr";
    inputColorMode_->setCurrentIndex(inputColorMode_->findData(color));
    scale_->setValue(c.scale); meanR_->setValue(c.meanR); meanG_->setValue(c.meanG); meanB_->setValue(c.meanB);
    confidence_->setValue(threshold); classFilter_->setText(QString::number(request.value("model_class_id").toInt()));
    setComputeDevice(c.device, c.deviceIndex);
    auto profile = modelEvaluationProfiles_.value(modelPath_).toMap();
    profile.insert("config", evaluationReport_.value("config").toObject().toVariantMap());
    profile.insert("model_class_id", request.value("model_class_id").toInt());
    profile.insert("adopted_threshold", threshold); profile.insert("adopted", true); profile.insert("report_directory", evaluationDirectory_);
    profile.insert("adopted_report_directory", evaluationDirectory_);
    profile.insert("model_identity", identity(modelPath_)); profile.insert("model_sha256", request.value("model_sha256").toString());
    modelEvaluationProfiles_.insert(modelPath_, profile);
    evaluationReviewActive_ = true; // The immutable evaluation view remains available after adopting parameters.
    selectQueueImage(queue_->currentRow());
    persist(); refreshModelLibrary();
    showNotice(QString("已采用检测配置 · 置信度 %1 · 原始类别 ID %2 · 每图最多 %3 框 · 尺寸/颜色/NMS/设备与报告一致")
        .arg(threshold, 0, 'f', 3).arg(request.value("model_class_id").toInt()).arg(c.maxDetections));
}

void MainWindow::runEvaluationSmoke(const QString &directory, const QString &requestPath)
{
    const QString output = QFileInfo(directory).absoluteFilePath();
    const auto request = jsonFile(requestPath);
    if (!QDir().mkpath(output) || request.isEmpty()) {
        qCritical("Invalid evaluation smoke request/output"); QCoreApplication::exit(2); return;
    }
    const auto done = std::make_shared<bool>(false);
    const bool verifyDecision = request.value("verify_decision_layout").toBool(false);
    const auto decision = std::make_shared<QJsonObject>();
    const auto inferenceStarts = std::make_shared<int>(0);
    const auto finish = [this, output, request, done, decision, verifyDecision, inferenceStarts](bool success, const QString &error) {
        if (*done) return;
        *done = true;
        QJsonObject summary{{"schema_version", 1}, {"success", success}, {"error", error},
            {"version", QCoreApplication::applicationVersion()}, {"request", request},
            {"report_directory", evaluationDirectory_}, {"report", evaluationReport_}};
        if (verifyDecision) {
            decision->insert("completed", success);
            decision->insert("ordinary_inference_start_requests", *inferenceStarts);
            summary.insert("decision_layout", *decision);
        }
        QJsonArray exports;
        for (const auto &file : QDir(output + "/selected-result").entryInfoList(QDir::Files))
            exports.append(QJsonObject{{"path", file.absoluteFilePath()}, {"sha256", sha256(file.absoluteFilePath())}, {"size", file.size()}});
        summary.insert("exports", exports);
        QSaveFile f(output + "/evaluation-smoke-report.json");
        const auto bytes = QJsonDocument(summary).toJson();
        const bool saved = f.open(QIODevice::WriteOnly) && f.write(bytes) == bytes.size() && f.commit();
        if (!success) saveScreenshot(output + "/failure.png");
        QCoreApplication::exit(success && saved ? 0 : 1);
    };
    if (!QFileInfo::exists(request.value("model_path").toString()) ||
        !QFileInfo::exists(request.value("dataset_yaml").toString())) {
        finish(false, "Missing model or dataset YAML"); return;
    }
    labels_.clear(); labelsPath_.clear(); autoExport_->setChecked(false);
    setModel(request.value("model_path").toString());
    taskBox_->setCurrentIndex(request.value("task").toString() == "yolov5" ? 0 : 1);
    inputSize_->setValue(request.value("input_size").toInt(640));
    confidence_->setValue(request.value("confidence").toDouble(.25));
    iou_->setValue(request.value("nms_iou").toDouble(.45));
    const QString color = request.value("color_mode").toString("rgb");
    inputColorMode_->setCurrentIndex(inputColorMode_->findData(color == "grayscale" ? "grayscale" : color == "bgr" ? "bgr" : "rgb"));
    setComputeDevice(request.value("device").toString() == "cuda" ? vision::ComputeDevice::CUDA : vision::ComputeDevice::CPU,
                     request.value("device_index").toInt());
    loadDataset(request.value("dataset_yaml").toString(), request.value("split").toString("val"), request.value("metadata_path").toString());
    auto *timer = new QTimer(this); timer->setInterval(40);
    if (verifyDecision) connect(this, &MainWindow::startRequested, timer,
        [inferenceStarts](const vision::JobRequest &) { ++*inferenceStarts; });
    const auto stage = std::make_shared<int>(0);
    connect(timer, &QTimer::timeout, this, [this, timer, stage, request, output, finish, done, decision, verifyDecision, inferenceStarts] {
        if (*done) { timer->stop(); return; }
        if (busy_ || modelInspectionPending_) return;
        const auto check = [decision, finish](const QString &name, bool value) {
            auto checks = decision->value("checks").toObject(); checks.insert(name, value); decision->insert("checks", checks);
            if (!value) finish(false, "Decision-layout acceptance failed: " + name);
            return value;
        };
        auto *tabs = evaluation_->findChild<QTabWidget *>("evaluationAnalysisTabs");
        auto *table = evaluation_->findChild<QTableWidget *>("evaluationThresholdTable");
        auto *curve = evaluation_->findChild<QWidget *>("evaluationThresholdCurve");
        auto *recommend = evaluation_->findChild<QPushButton *>("evaluationUseRecommendationButton");
        auto *review = evaluation_->findChild<QPushButton *>("evaluationReviewErrorsButton");
        auto *apply = evaluation_->findChild<QPushButton *>("evaluationApplyThresholdButton");
        auto *splitter = evaluation_->findChild<QSplitter *>("evaluationDecisionSplit");
        auto *strategy = evaluation_->findChild<QComboBox *>("evaluationRecommendationStrategyCombo");
        auto *constraint = evaluation_->findChild<QDoubleSpinBox *>("evaluationRecommendationConstraintSpin");
        const auto point = [this, table, curve, tabs] {
            QJsonArray metrics;
            for (int index = 2; index < 6; ++index) {
                const auto *value = evaluation_->findChild<QLabel *>(QString("evaluationMetric%1").arg(index));
                metrics.append(value ? value->text() : QString());
            }
            const auto *item = table ? table->item(table->currentRow(), 0) : nullptr;
            const QRect selected = item ? table->visualItemRect(item) : QRect();
            return QJsonObject{{"threshold", evaluation_->analysisThreshold()},
                {"curve_threshold", curve ? curve->property("analysisThreshold").toDouble() : -1},
                {"recommended_threshold", curve ? curve->property("recommendedThreshold").toDouble() : -1},
                {"selected_row", table ? table->currentRow() : -1}, {"table_confidence", item ? item->data(Qt::UserRole).toDouble() : -1},
                {"selected_row_visible", item && table->viewport()->rect().contains(selected)},
                {"metrics", metrics}, {"analysis_tab", tabs ? tabs->currentIndex() : -1}};
        };
        const auto unchanged = [this, decision, inferenceStarts] {
            return vision::EvaluationWorker::modelConfigJson(currentConfig()) == decision->value("config_before").toObject() &&
                sha256(evaluationDirectory_ + "/report.json") == decision->value("report_sha256_before").toString() &&
                QJsonArray::fromStringList(QDir(dataRoot_ + "/evaluations").entryList(QDir::Dirs | QDir::NoDotAndDotDot, QDir::Name)) ==
                    decision->value("evaluation_directories_before").toArray() &&
                QString::fromLatin1(QCryptographicHash::hash(QJsonDocument(history_).toJson(QJsonDocument::Compact), QCryptographicHash::Sha256).toHex()) ==
                    decision->value("history_sha256_before").toString() && *inferenceStarts == 0 &&
                !busy_ && !evaluationWorker_;
        };
        if (*stage == 0) {
            if (datasetManifest_.isEmpty() || modelClassCount_ <= 0) {
                finish(false, "Dataset prepare or model inspection failed: " + statusLabel_->text()); return;
            }
            auto *controls = evaluation_->controlsWidget();
            controls->findChild<QSpinBox *>("evaluationModelClassIdSpin")->setValue(request.value("model_class_id").toInt());
            controls->findChild<QSpinBox *>("evaluationDatasetClassIdSpin")->setValue(request.value("dataset_class_id").toInt());
            controls->findChild<QDoubleSpinBox *>("evaluationMatchIouSpin")->setValue(request.value("match_iou").toDouble(.5));
            controls->findChild<QDoubleSpinBox *>("evaluationScanStartSpin")->setValue(request.value("scan_start").toDouble(.05));
            controls->findChild<QDoubleSpinBox *>("evaluationScanEndSpin")->setValue(request.value("scan_end").toDouble(.95));
            controls->findChild<QDoubleSpinBox *>("evaluationScanStepSpin")->setValue(request.value("scan_step").toDouble(.01));
            controls->findChild<QSpinBox *>("evaluationBatchSizeSpin")->setValue(request.value("batch_size").toInt(1));
            controls->findChild<QSpinBox *>("evaluationCpuThreadsSpin")->setValue(request.value("cpu_threads").toInt());
            controls->findChild<QSpinBox *>("evaluationMaxDetectionsSpin")->setValue(request.value("max_detections").toInt(300));
            controls->findChild<QDoubleSpinBox *>("evaluationScoreFloorSpin")->setValue(request.value("score_floor").toDouble(.001));
            if (request.value("figure_formats").isArray()) {
                const auto formats = request.value("figure_formats").toArray();
                evaluation_->findChild<QCheckBox *>("evaluationFigurePngCheck")->setChecked(formats.contains("png"));
                evaluation_->findChild<QCheckBox *>("evaluationFigureSvgCheck")->setChecked(formats.contains("svg"));
                evaluation_->findChild<QCheckBox *>("evaluationFigurePdfCheck")->setChecked(formats.contains("pdf"));
            }
            evaluation_->findChild<QSpinBox *>("evaluationFigureDpiSpin")->setValue(request.value("figure_dpi").toInt(300));
            evaluation_->setMode(true); detectionRange_->setCurrentIndex(0); queue_->setCurrentRow(0);
            if (!evaluation_->canEvaluate() || !runButton_->isEnabled()) {
                finish(false, "Cannot start evaluation: " + evaluation_->validationMessage()); return;
            }
            startEvaluation();
            if (!busy_ || !evaluationWorker_) { finish(false, "Workbench failed to start actual evaluation"); return; }
            *stage = 1;
        } else if (*stage == 1) {
            if (evaluationReport_.isEmpty() || evaluationReport_.value("status").toString() != "complete" ||
                evaluationReport_.value("completed_images").toInt() != files_.size()) {
                finish(false, "Incomplete evaluation: " + statusLabel_->text()); return;
            }
            if (verifyDecision) {
                if (!check("required_widgets_exist", tabs && table && curve && recommend && review && apply && splitter && strategy && constraint)) return;
                decision->insert("schema_version", 1);
                decision->insert("config_before", vision::EvaluationWorker::modelConfigJson(currentConfig()));
                decision->insert("report_sha256_before", sha256(evaluationDirectory_ + "/report.json"));
                decision->insert("evaluation_directories_before", QJsonArray::fromStringList(QDir(dataRoot_ + "/evaluations").entryList(QDir::Dirs | QDir::NoDotAndDotDot, QDir::Name)));
                decision->insert("history_sha256_before", QString::fromLatin1(QCryptographicHash::hash(QJsonDocument(history_).toJson(QJsonDocument::Compact), QCryptographicHash::Sha256).toHex()));
                decision->insert("initial_threshold", evaluation_->analysisThreshold());
                decision->insert("initial_filter", evaluation_->imageFilter());
                decision->insert("initial_point", point());
                QJsonArray titles;
                for (int index = 0; index < tabs->count(); ++index) titles.append(tabs->tabText(index));
                decision->insert("tabs", titles);
                if (!check("four_pages_and_order", titles == QJsonArray{"阈值调优", "能力诊断", "场景分析", "报告管理"})) return;
                auto *page = evaluation_->findChild<QWidget *>("evaluationThresholdPage");
                if (!check("decision_controls_share_threshold_page", page && page->isAncestorOf(table) && page->isAncestorOf(curve) &&
                    page->isAncestorOf(recommend) && page->isAncestorOf(review) && page->isAncestorOf(apply) &&
                    page->isAncestorOf(evaluation_->findChild<QWidget *>("evaluationCurrentPointCard")))) return;
                tabs->setCurrentIndex(0); workbenchTabs_->setCurrentIndex(1); resize(1260, 820); *stage = 10; return;
            }
            saveScreenshot(output + "/evaluation-analysis.png");
            saveScreenshot(output + "/evaluation-thresholds.png");
            if (auto *tabs = evaluation_->findChild<QTabWidget *>("evaluationAnalysisTabs")) {
                const QStringList names{"thresholds", "diagnostics", "scenes", "report"};
                for (int page = 1; page < tabs->count(); ++page) {
                    tabs->setCurrentIndex(page);
                    saveScreenshot(output + "/evaluation-" + names.value(page, QString::number(page)) + ".png");
                }
                tabs->setCurrentIndex(0);
            }
            workbenchTabs_->setCurrentIndex(0); queue_->setCurrentRow(0); reviewEvaluationImage(0); *stage = 2;
        } else if (*stage == 10 || *stage == 11) {
            const int expectedWidth = *stage == 10 ? 1260 : 1366, expectedHeight = *stage == 10 ? 820 : 900;
            const QString sizeName = QString::number(expectedWidth) + "x" + QString::number(expectedHeight);
            const QStringList geometryStrategies{"f1", "precision", "fpr"};
            const int strategyIndex = decision->value("geometry_strategy_index").toInt();
            const QString strategyName = geometryStrategies.value(strategyIndex);
            const QString suffix = strategyName == "f1" ? QString() : "_" + strategyName;
            QJsonObject widgets;
            bool visible = width() == expectedWidth && height() == expectedHeight && strategy->currentData() == strategyName;
            QStringList geometryWidgets{"evaluationAnalysisThresholdSpin", "evaluationThresholdCurve", "evaluationThresholdTable",
                "evaluationDecisionSidebar", "evaluationRecommendationStrategyCombo", "evaluationUseRecommendationButton",
                "evaluationReviewErrorsButton", "evaluationApplyThresholdButton", "evaluationRecommendationHint", "evaluationCounts",
                "evaluationMetric2", "evaluationMetric3", "evaluationMetric4", "evaluationMetric5"};
            if (strategyName != "f1") geometryWidgets.append("evaluationRecommendationConstraintSpin");
            for (const auto &name : geometryWidgets) {
                const auto geometry = smokeWidgetGeometry(evaluation_->findChild<QWidget *>(name), this);
                widgets.insert(name, geometry); visible = visible && geometry.value("fully_visible").toBool() &&
                    (!geometry.contains("text_geometry") || geometry.value("text_geometry").toObject().value("text_fits").toBool());
            }
            widgets.insert("window", QJsonObject{{"width", width()}, {"height", height()}});
            const auto *selectedItem = table->item(table->currentRow(), 0);
            const QRect rowRect = selectedItem ? table->visualItemRect(selectedItem) : QRect();
            const bool rowVisible = selectedItem && table->viewport()->rect().contains(rowRect);
            widgets.insert("selected_row", QJsonObject{{"row", table->currentRow()},
                {"confidence", selectedItem ? selectedItem->data(Qt::UserRole).toDouble() : -1}, {"fully_visible", rowVisible},
                {"x", rowRect.x()}, {"y", rowRect.y()}, {"width", rowRect.width()}, {"height", rowRect.height()},
                {"viewport_width", table->viewport()->width()}, {"viewport_height", table->viewport()->height()}});
            visible = visible && rowVisible;
            // Observe settled asynchronous Qt layouts; never scroll/select on
            // behalf of the UI under test. Two identical consecutive timer
            // observations are required, with all originals retained.
            auto waits = decision->value("geometry_waits").toObject();
            const QString geometryKey = sizeName + suffix;
            auto waiting = waits.value(geometryKey).toObject();
            auto observations = waiting.value("observations").toArray(); observations.append(widgets);
            waiting.insert("observations", observations); waiting.insert("ticks", observations.size());
            const bool stable = observations.size() >= 2 && observations.at(observations.size() - 2).toObject() == widgets;
            waiting.insert("stable", stable); waits.insert(geometryKey, waiting); decision->insert("geometry_waits", waits);
            if (!stable) {
                if (observations.size() >= 10) { finish(false, "Decision layout did not stabilize within 10 GUI timer observations: " + geometryKey); }
                return;
            }
            auto geometries = decision->value("window_geometries").toObject(); geometries.insert(sizeName + suffix, widgets); decision->insert("window_geometries", geometries);
            saveScreenshot(output + "/decision-" + sizeName + suffix + ".png");
            if (!check("visible_at_" + sizeName + suffix, visible)) return;
            if (strategyIndex < 2) {
                decision->insert("geometry_strategy_index", strategyIndex + 1);
                strategy->setCurrentIndex(strategy->findData(geometryStrategies[strategyIndex + 1])); return;
            }
            decision->insert("geometry_strategy_index", 0); strategy->setCurrentIndex(strategy->findData("f1"));
            if (*stage == 10) { resize(1366, 900); *stage = 11; return; }
            decision->insert("splitter_before", smokeSplitterSizes(splitter));
            const auto sizes = splitter->sizes();
            if (!check("splitter_has_two_noncollapsed_sections", sizes.size() == 2 && sizes[0] > 30 && sizes[1] > 60)) return;
            splitter->setSizes({sizes[0] - 20, sizes[1] + 20}); *stage = 12;
        } else if (*stage == 12) {
            decision->insert("splitter_resized", smokeSplitterSizes(splitter));
            if (!check("splitter_resizes_chart_and_table", decision->value("splitter_before") != decision->value("splitter_resized"))) return;
            const double recommended = curve->property("recommendedThreshold").toDouble();
            if (!check("real_full_val_recommendation_available", recommend->isEnabled() && recommended >= 0 && strategy->currentData() == "f1")) return;
            // Display both distinct markers before the actual button selects it.
            const int other = std::abs(table->item(0, 0)->data(Qt::UserRole).toDouble() - recommended) < 1e-8 ? table->rowCount() - 1 : 0;
            table->setCurrentCell(other, 0); *stage = 13;
        } else if (*stage == 13) {
            if (evaluationReviewPending_) return;
            decision->insert("distinct_markers", point());
            if (!check("current_and_recommended_markers_are_distinct", std::abs(evaluation_->analysisThreshold() - curve->property("recommendedThreshold").toDouble()) > 1e-8 &&
                std::abs(evaluation_->analysisThreshold() - curve->property("analysisThreshold").toDouble()) < 1e-8)) return;
            if (!check("configuration_unchanged_before_recommendation", unchanged())) return;
            saveScreenshot(output + "/decision-distinct-markers.png"); recommend->click(); *stage = 14;
        } else if (*stage == 14) {
            if (evaluationReviewPending_) return;
            const auto observed = point(); decision->insert("recommended_point", observed);
            const double threshold = observed.value("threshold").toDouble();
            if (!check("recommendation_click_syncs_curve_table_and_visible_row", observed.value("selected_row_visible").toBool() &&
                std::abs(threshold - observed.value("recommended_threshold").toDouble()) < 1e-8 &&
                std::abs(threshold - observed.value("table_confidence").toDouble()) < 1e-8 &&
                std::abs(threshold - observed.value("curve_threshold").toDouble()) < 1e-8)) return;
            if (!check("recommendation_does_not_adopt_or_reinfer", unchanged())) return;
            saveScreenshot(output + "/decision-recommended-point.png");
            table->setCurrentCell(table->rowCount() - 1, 0); *stage = 15;
        } else if (*stage == 15) {
            if (evaluationReviewPending_) return;
            const auto observed = point(); decision->insert("table_selected_point", observed);
            if (!check("table_selection_syncs_current_point_and_visible_row", observed.value("selected_row_visible").toBool() &&
                std::abs(observed.value("threshold").toDouble() - observed.value("table_confidence").toDouble()) < 1e-8 &&
                std::abs(observed.value("threshold").toDouble() - observed.value("curve_threshold").toDouble()) < 1e-8)) return;
            decision->insert("review_threshold", evaluation_->analysisThreshold()); decision->insert("review_tab", tabs->currentIndex());
            decision->insert("review_strategy", strategy->currentData().toString()); decision->insert("review_constraint", constraint->value());
            decision->insert("review_splitter", smokeSplitterSizes(splitter));
            if (!check("real_review_button_enabled", review->isEnabled())) return;
            review->click(); *stage = 16;
        } else if (*stage == 16) {
            if (evaluationReviewPending_) return;
            if (!check("review_button_opens_real_error_queue", workbenchTabs_->currentIndex() == 0 && evaluationReviewActive_ &&
                evaluationReviewAvailable_ && evaluationReviewFilter_ == "errors" && evaluationEvaluatedCount_ == files_.size() &&
                evaluationReviewRows_ == evaluationErrorRows_)) return;
            if (!evaluationReviewRows_.isEmpty() && (lastResult_.image.isNull() || !exportButton_->isEnabled())) return;
            decision->insert("review", QJsonObject{{"errors", evaluationErrorRows_.size()}, {"evaluated", evaluationEvaluatedCount_},
                {"collection", datasetManifest_.value("samples").toArray().size()}, {"queue_row", queue_->currentRow()},
                {"visible_rows", QJsonArray::fromVariantList([this] { QVariantList rows; for (int row : evaluationReviewRows_) rows.append(row); return rows; }())},
                {"preview_available", !lastResult_.image.isNull()}, {"filter", evaluationReviewFilter_}});
            saveScreenshot(output + "/decision-error-review.png");
            auto *back = findChild<QPushButton *>("evaluationReviewAnalysisButton");
            if (!check("real_return_button_visible", back && back->isVisibleTo(this) && back->isEnabled())) return;
            back->click(); *stage = 17;
        } else if (*stage == 17) {
            const auto observed = point(); decision->insert("after_review_point", observed);
            if (!check("return_preserves_threshold_page_strategy_and_splitter", workbenchTabs_->currentIndex() == 1 &&
                tabs->currentIndex() == decision->value("review_tab").toInt() &&
                std::abs(evaluation_->analysisThreshold() - decision->value("review_threshold").toDouble()) < 1e-8 &&
                strategy->currentData().toString() == decision->value("review_strategy").toString() &&
                std::abs(constraint->value() - decision->value("review_constraint").toDouble()) < 1e-8 &&
                smokeSplitterSizes(splitter) == decision->value("review_splitter").toArray() && evaluation_->imageFilter() == "errors")) return;
            if (!check("review_does_not_adopt_reinfer_or_rewrite_report", unchanged())) return;
            decision->insert("returned_review", QJsonObject{{"threshold", evaluation_->analysisThreshold()}, {"analysis_tab", tabs->currentIndex()},
                {"strategy", strategy->currentData().toString()}, {"constraint", constraint->value()},
                {"splitter", smokeSplitterSizes(splitter)}, {"filter", evaluation_->imageFilter()}});
            decision->insert("history_sha256_before_apply", QString::fromLatin1(QCryptographicHash::hash(QJsonDocument(history_).toJson(QJsonDocument::Compact), QCryptographicHash::Sha256).toHex()));
            decision->insert("evaluation_directories_before_apply", QJsonArray::fromStringList(QDir(dataRoot_ + "/evaluations").entryList(QDir::Dirs | QDir::NoDotAndDotDot, QDir::Name)));
            decision->insert("config_before_explicit_apply", vision::EvaluationWorker::modelConfigJson(currentConfig()));
            decision->insert("apply_button_enabled", apply->isEnabled());
            if (!check("explicit_apply_allowed_for_verified_native_report", apply->isEnabled() && evaluationReport_.value("allow_apply_profile").toBool(true))) return;
            apply->click(); *stage = 18;
        } else if (*stage == 18) {
            const auto profile = modelEvaluationProfiles_.value(modelPath_).toMap();
            decision->insert("config_after_explicit_apply", vision::EvaluationWorker::modelConfigJson(currentConfig()));
            decision->insert("adopted_threshold", confidence_->value());
            if (!check("only_explicit_apply_updates_detection_profile", profile.value("adopted").toBool() &&
                std::abs(profile.value("adopted_threshold").toDouble() - decision->value("review_threshold").toDouble()) < 1e-7 &&
                std::abs(confidence_->value() - decision->value("review_threshold").toDouble()) < 1e-7 &&
                classFilter_->text() == QString::number(request.value("model_class_id").toInt()) &&
                profile.value("adopted_report_directory").toString() == evaluationDirectory_ &&
                sha256(evaluationDirectory_ + "/report.json") == decision->value("report_sha256_before").toString() && !busy_ && !evaluationWorker_)) return;
            // Return to the original smoke export contract: first image at the
            // request's original analysis threshold, irrespective of review.
            evaluation_->setAnalysisThreshold(decision->value("initial_threshold").toDouble());
            evaluation_->setImageFilter("all"); *stage = 19;
        } else if (*stage == 19) {
            if (evaluationReviewPending_) return;
            decision->insert("final_point", point());
            decision->insert("report_sha256_after", sha256(evaluationDirectory_ + "/report.json"));
            if (!check("original_report_preserved_and_initial_export_threshold_restored", decision->value("report_sha256_before") == decision->value("report_sha256_after") &&
                std::abs(evaluation_->analysisThreshold() - decision->value("initial_threshold").toDouble()) < 1e-8 && evaluation_->imageFilter() == "all")) return;
            saveScreenshot(output + "/evaluation-analysis.png"); saveScreenshot(output + "/evaluation-thresholds.png");
            const QStringList names{"thresholds", "diagnostics", "scenes", "report"};
            for (int index = 1; index < tabs->count(); ++index) { tabs->setCurrentIndex(index); saveScreenshot(output + "/evaluation-" + names[index] + ".png"); }
            tabs->setCurrentIndex(0); workbenchTabs_->setCurrentIndex(0); queue_->setCurrentRow(0); reviewEvaluationImage(0); *stage = 2;
        } else if (*stage == 2 && !lastResult_.image.isNull() && exportButton_->isEnabled()) {
            QString error;
            if (!writeResult(lastResult_, output + "/selected-result", &error)) { finish(false, error); return; }
            saveScreenshot(output + "/evaluation-preview.png");
            finish(true, {});
        }
    });
    timer->start();
    QTimer::singleShot(240000, this, [this, finish, done] {
        if (*done) return;
        if (evaluationWorker_) evaluationWorker_->requestStop();
        finish(false, "Evaluation smoke timed out: " + statusLabel_->text());
    });
}

void MainWindow::runDatasetSessionSmoke(const QString &directory)
{
    const QString output = QFileInfo(directory).absoluteFilePath();
    if (!QDir().mkpath(output)) {
        qCritical("Cannot create dataset session smoke output directory");
        QCoreApplication::exit(2);
        return;
    }
    const QString sessionPath = datasetSessionFilePath();
    const auto expected = vision::DatasetSessionStore::load(sessionPath);
    const auto initialSession = jsonFile(sessionPath);
    const auto initialHistory = history_;
    const QString initialHistoryHash = sha256(dataRoot_ + "/history.json");
    const auto evaluationDirectories = [this] {
        return QDir(dataRoot_ + "/evaluations").entryList(QDir::Dirs | QDir::NoDotAndDotDot, QDir::Name);
    };
    const auto initialDirectories = evaluationDirectories();
    const auto starts = std::make_shared<int>(0);
    const auto done = std::make_shared<bool>(false);
    const auto elapsed = std::make_shared<QElapsedTimer>();
    elapsed->start();
    connect(this, &MainWindow::startRequested, this, [starts](const vision::JobRequest &) { ++*starts; });
    auto *timer = new QTimer(this);
    timer->setInterval(25);
    const auto finish = [this, output, sessionPath, initialSession, initialHistory, initialHistoryHash,
                         initialDirectories, evaluationDirectories, starts, done, elapsed, timer]
                        (bool success, const QString &error, const QJsonObject &checks) {
        if (*done) return;
        *done = true;
        timer->stop();
        const auto first = canvas_->result();
        QJsonArray paths, queueDetails;
        for (int index = 0; index < queue_->count(); ++index) {
            const QString path = queue_->item(index)->data(Qt::UserRole).toString();
            paths.append(path);
            queueDetails.append(QJsonObject{{"index", index}, {"image", path},
                {"ground_truth_boxes", datasetSamples_.value(path).value("gt").toArray().size()}});
        }
        const auto finalDirectories = evaluationDirectories();
        QJsonObject report{{"schema_version", 1}, {"success", success}, {"error", error},
            {"version", QCoreApplication::applicationVersion()}, {"qt", qVersion()},
            {"scope", "Actual startup restoration through restoreDatasetSession; no detection or evaluation is requested."},
            {"session_file", sessionPath}, {"initial_session_json", initialSession},
            {"session_json", jsonFile(sessionPath)}, {"session_sha256", sha256(sessionPath)},
            {"manifest", QJsonObject{{"dataset_source", datasetManifest_.value("dataset_source")},
                {"data_yaml", datasetManifest_.value("data_yaml")}, {"split", datasetManifest_.value("split")},
                {"metadata_path", datasetManifest_.value("metadata_path")},
                {"fingerprint", datasetManifest_.value("fingerprint")},
                {"sample_count", datasetManifest_.value("samples").toArray().size()}}},
            {"queue_count", queue_->count()}, {"queue_paths", paths}, {"queue_ground_truth", queueDetails},
            {"first_image", QJsonObject{{"source", first.source}, {"width", first.image.width()},
                {"height", first.image.height()}, {"non_null", !first.image.isNull()},
                {"ground_truth_boxes", first.groundTruth.size()}, {"prediction_count", first.predictions.size()},
                {"inference_ms", first.inferenceMs}}},
            {"start_requested_count", *starts}, {"history_before_count", initialHistory.size()},
            {"history_after_count", history_.size()}, {"history_unchanged", history_ == initialHistory},
            {"history_file_sha256_before", initialHistoryHash},
            {"history_file_sha256_after", sha256(dataRoot_ + "/history.json")},
            {"evaluation_directories_before", QJsonArray::fromStringList(initialDirectories)},
            {"evaluation_directories_after", QJsonArray::fromStringList(finalDirectories)},
            {"checks", checks}, {"elapsed_ms", double(elapsed->elapsed())},
            {"status", statusLabel_->toolTip()}};
        if (!success) saveScreenshot(output + "/dataset-session-failure.png");
        QSaveFile file(output + "/dataset-session-smoke-report.json");
        file.setDirectWriteFallback(false);
        const auto bytes = QJsonDocument(report).toJson(QJsonDocument::Indented);
        const bool saved = file.open(QIODevice::WriteOnly) && file.write(bytes) == bytes.size() && file.commit();
        if (!saved) qCritical("Cannot save dataset session smoke report");
        if (evaluationWorker_) evaluationWorker_->requestStop();
        QCoreApplication::exit(success && saved ? 0 : 1);
    };
    connect(timer, &QTimer::timeout, this, [this, output, expected, starts, done, initialHistory,
                                          initialHistoryHash, initialDirectories, evaluationDirectories, finish] {
        if (*done || busy_ || evaluationWorker_ || queuePopulating_) return;
        if (expected.status != vision::DatasetSessionStatus::Loaded || datasetManifest_.isEmpty() || files_.isEmpty()) {
            finish(false, "Dataset session did not restore: " + statusLabel_->toolTip(), {});
            return;
        }
        const auto first = canvas_->result();
        if (first.image.isNull() || first.source != files_.first()) return;
        const auto samples = datasetManifest_.value("samples").toArray();
        if (samples.isEmpty()) {
            finish(false, "Restored queue has no dataset samples", {});
            return;
        }
        bool orderMatches = queue_->count() == samples.size() && files_.size() == samples.size();
        for (int index = 0; orderMatches && index < samples.size(); ++index)
            orderMatches = files_[index] == samples[index].toObject().value("image").toString() &&
                queue_->item(index)->data(Qt::UserRole).toString() == files_[index] && datasetSamples_.contains(files_[index]);
        const auto currentSession = vision::DatasetSessionStore::load(datasetSessionFilePath());
        const QString requestedSource = QDir::cleanPath(QFileInfo(expected.session.sourcePath).absoluteFilePath());
        const QString canonicalSource = QFileInfo(requestedSource).canonicalFilePath();
        const QString expectedMetadata = expected.session.metadataPath.isEmpty() ? QString() :
            QFileInfo(expected.session.metadataPath).canonicalFilePath();
        const auto firstSample = samples.first().toObject();
        QJsonObject checks{
            {"session_loaded", currentSession.status == vision::DatasetSessionStatus::Loaded},
            {"source_entry_preserved", currentSession.session.sourcePath == requestedSource},
            {"manifest_source_matches", datasetManifest_.value("dataset_source").toString() == canonicalSource},
            {"split_matches", datasetManifest_.value("split").toString() == expected.session.split &&
                currentSession.session.split == expected.session.split},
            {"metadata_matches", datasetManifest_.value("metadata_path").toString() == expectedMetadata &&
                currentSession.session.metadataPath == expectedMetadata},
            {"queue_complete_in_manifest_order", orderMatches},
            {"first_source_and_pixels_loaded", first.source == files_.first() && !first.image.isNull() &&
                first.image.width() == firstSample.value("width").toInt() && first.image.height() == firstSample.value("height").toInt()},
            {"no_detection_start", *starts == 0},
            {"no_predictions", first.predictions.isEmpty() && first.inferenceMs == 0},
            {"history_unchanged", history_ == initialHistory && sha256(dataRoot_ + "/history.json") == initialHistoryHash},
            {"no_evaluation_run", evaluationDirectories() == initialDirectories && evaluationReport_.isEmpty() && evaluationDirectory_.isEmpty() && !evaluation_->evaluationMode()}};
        const bool screenshotSaved = grab().save(output + "/dataset-session-restored.png");
        QImageReader screenshot(output + "/dataset-session-restored.png");
        checks.insert("screenshot_written", screenshotSaved && screenshot.canRead() && screenshot.size().width() > 0 && screenshot.size().height() > 0);
        bool success = true;
        for (auto it = checks.constBegin(); it != checks.constEnd(); ++it) success = success && it.value().toBool();
        finish(success, success ? QString() : QStringLiteral("Restored state failed one or more actual checks"), checks);
    });
    restoreDatasetSession();
    timer->start();
    QTimer::singleShot(45000, this, [this, done, finish] {
        if (*done) return;
        stopInference();
        finish(false, "Dataset session restoration timed out: " + statusLabel_->toolTip(), {});
    });
}
