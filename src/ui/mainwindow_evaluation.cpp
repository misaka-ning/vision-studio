// SPDX-License-Identifier: AGPL-3.0-only
#include "mainwindow.h"
#include "evaluationpanel.h"
#include "imagecanvas.h"
#include "lazyimagequeue.h"
#include "asyncimagepreview.h"
#include "modelconversionpage.h"
#include "icons.h"
#include "core/evaluationworker.h"
#include "core/evaluationmatch.h"
#include "core/evaluationreview.h"
#include "core/gpuruntime.h"
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
#include <QSaveFile>
#include <QSettings>
#include <QSignalBlocker>
#include <QSpinBox>
#include <QTabWidget>
#include <QTableWidget>
#include <QTimer>

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
    for (const auto &v : o.value("labels").toArray()) c.labels.append(v.toString());
    for (const auto &v : o.value("class_filter").toArray()) c.classFilter.append(v.toInt());
    return c;
}
QString datasetName(const QJsonObject &manifest, int id)
{
    const auto names = manifest.value("names").toArray();
    return id >= 0 && id < names.size() ? names[id].toString() : QString::number(id);
}
}

void MainWindow::connectEvaluation()
{
    connect(evaluation_, &EvaluationPanel::datasetRequested, this, &MainWindow::loadDataset);
    connect(evaluation_, &EvaluationPanel::configChanged, this, [this] {
        if (busy_) return;
        if (evaluationReviewActive_ && evaluationRunFingerprint_ != evaluationFingerprint()) {
            evaluationReviewActive_ = false;
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
        if (mode) workbenchTabs_->setCurrentIndex(1);
        else workbenchTabs_->setCurrentIndex(0);
        updateStartEligibility();
        selectQueueImage(queue_->currentRow());
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
            reviewEvaluationImage(queue_->currentRow());
            applyEvaluationFilter(evaluationReviewFilter_);
        }
    });
    connect(evaluation_, &EvaluationPanel::imageFilterChanged, this, &MainWindow::applyEvaluationFilter);
    connect(evaluation_, &EvaluationPanel::applyRequested, this, &MainWindow::applyEvaluationProfile);
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
    return QString::fromLatin1(QCryptographicHash::hash(QJsonDocument(config).toJson(QJsonDocument::Compact), QCryptographicHash::Sha256).toHex());
}

void MainWindow::clearDatasetContext()
{
    if (loadingDataset_ || !evaluation_) return;
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
    if (busy_ || evaluationWorker_ || gpuSetupProcess_ || (modelConversion_ && modelConversion_->isBusy())) return;
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
    connect(worker, &vision::EvaluationWorker::progress, this, [this](int, int, const QString &stage) { showNotice(stage); });
    connect(worker, &vision::EvaluationWorker::failed, this, [this, importError](const QString &message) {
        *importError = message;
        if (datasetManifest_.isEmpty()) evaluation_->showDatasetImportError(message);
        else evaluation_->setDataset(datasetManifest_);
        showNotice(message, true);
    });
    connect(worker, &vision::EvaluationWorker::datasetReady, this, [this, delivered, cancelled](const QJsonObject &manifest) {
        if (cancelled->load()) return;
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
        populateDatasetQueue(manifest, [this, manifest] {
            if (!evaluationWorker_) setBusy(false);
            updateEvaluationModel(); updateSourceUi();
            queue_->setCurrentRow(files_.isEmpty() ? -1 : 0);
            selectQueueImage(queue_->currentRow());
            QString notice = QString("数据集预检查完成 · %1 张有效图片 · %2").arg(files_.size()).arg(manifest.value("split").toString());
            if (manifest.value("root_relocated").toBool()) notice += " · 旧路径已按当前文件夹读取，原文件未修改";
            showNotice(notice);
        });
    });
    connect(worker, &QThread::finished, this, [this, worker, delivered, importError, cancelled] {
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
    if (busy_ || evaluationWorker_ || resultIoPending_ || !evaluation_->canEvaluate()) return;
    if (modelInspectionPending_ || inspectedModelIdentity_ != modelFileIdentity()) {
        inspectSelectedModel(); showNotice("请等待当前模型类别信息加载完成后开始评估"); return;
    }
    QString error;
    const auto range = selectedImageRange(&error);
    if (!error.isEmpty()) { showNotice(error, true); return; }
    vision::EvaluationRequest request;
    request.config = currentConfig(); request.manifest = datasetManifest_; request.files = range;
    request.modelClassId = evaluation_->modelClassId(); request.datasetClassId = evaluation_->datasetClassId();
    request.confidence = confidence_->value(); request.matchIou = evaluation_->matchIou();
    request.scanStart = evaluation_->scanStart(); request.scanEnd = evaluation_->scanEnd(); request.scanStep = evaluation_->scanStep();
    request.outputRoot = dataRoot_; request.projectRoot = projectRoot_;
    evaluationRunConfig_ = request.config;
    evaluationRunFingerprint_ = evaluationFingerprint();
    if (evaluationPreviewCancellation_) evaluationPreviewCancellation_->store(true);
    if (evaluationFilterCancellation_) evaluationFilterCancellation_->store(true);
    ++evaluationPreviewGeneration_; ++evaluationFilterGeneration_;
    for (int row = 0; row < queue_->count(); ++row) queue_->item(row)->setHidden(false);
    evaluationReviewActive_ = false;
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
        if (evaluationReviewActive_) reviewEvaluationImage(queue_->currentRow());
        refreshHistory();
        if (closing_) QTimer::singleShot(0, this, &QWidget::close);
    });
    worker->start();
}

void MainWindow::acceptEvaluationReport(const QJsonObject &report, const QString &directory, bool record)
{
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
}

void MainWindow::loadEvaluationReport(const QString &directory)
{
    if (busy_ || directory.isEmpty()) return;
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
    populateDatasetQueue(manifest, [this, report, directory] {
        setBusy(false);
        acceptEvaluationReport(report, directory, false);
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
    if (busy_ || row < 0 || row >= files_.size()) return;
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
    lastResult_ = {}; exportButton_->setEnabled(false);
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

void MainWindow::applyEvaluationFilter(const QString &filter)
{
    evaluationReviewFilter_ = filter;
    if (busy_ || evaluationReport_.isEmpty()) return;
    if (evaluationFilterCancellation_) evaluationFilterCancellation_->store(true);
    const auto cancelled = evaluationFilterCancellation_ = std::make_shared<std::atomic_bool>(false);
    const quint64 generation = ++evaluationFilterGeneration_;
    const auto parameters = evaluationReport_.value("request").toObject();
    const auto files = files_; const auto indices = evaluationSampleIndices_; const auto directory = evaluationDirectory_;
    const auto threshold = evaluation_->analysisThreshold();
    struct Filter { QVector<bool> visible; QString error; };
    const auto result = std::make_shared<Filter>();
    auto *thread = QThread::create([result, files, indices, directory, threshold, parameters, filter, cancelled] {
        result->visible = vision::filterEvaluationSamples(files, indices, directory,
            parameters.value("model_class_id").toInt(), parameters.value("dataset_class_id").toInt(), threshold,
            parameters.value("match_iou").toDouble(.5), filter, [cancelled] { return cancelled->load(); }, &result->error);
    });
    connect(thread, &QThread::finished, this, [this, result, generation, cancelled, files, directory] {
        if (cancelled->load() || generation != evaluationFilterGeneration_ || busy_ || !evaluationReviewActive_ ||
            directory != evaluationDirectory_ || files != files_) return;
        if (result->visible.size() != files_.size()) {
            if (!result->error.isEmpty()) showNotice(result->error, true);
            return;
        }
        for (int row = 0; row < files_.size(); ++row) queue_->item(row)->setHidden(!result->visible[row]);
        if (queue_->currentRow() < 0 || queue_->item(queue_->currentRow())->isHidden()) {
            int first = -1;
            for (int row = 0; row < files_.size(); ++row) if (result->visible[row]) { first = row; break; }
            queue_->setCurrentRow(first);
            if (first < 0) {
                queue_->clearSelection(); lastResult_ = {}; canvas_->clear(); predictionTable_->setRowCount(0);
                exportButton_->setEnabled(false); resultInfo_->setText("当前阈值下没有符合此筛选的已完成图片。评估范围保持不变。");
            }
        }
        updateRangeUi();
    });
    connect(thread, &QThread::finished, thread, &QObject::deleteLater);
    thread->start();
}

void MainWindow::applyEvaluationProfile(double threshold)
{
    if (busy_ || evaluationReport_.isEmpty()) return;
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
    reviewEvaluationImage(queue_->currentRow());
    persist(); refreshModelLibrary();
    showNotice(QString("已采用检测配置 · 置信度 %1 · 原始类别 ID %2 · 尺寸/颜色/NMS/设备与报告一致")
        .arg(threshold, 0, 'f', 3).arg(request.value("model_class_id").toInt()));
}

void MainWindow::runEvaluationSmoke(const QString &directory, const QString &requestPath)
{
    const QString output = QFileInfo(directory).absoluteFilePath();
    const auto request = jsonFile(requestPath);
    if (!QDir().mkpath(output) || request.isEmpty()) {
        qCritical("Invalid evaluation smoke request/output"); QCoreApplication::exit(2); return;
    }
    const auto done = std::make_shared<bool>(false);
    const auto finish = [this, output, request, done](bool success, const QString &error) {
        if (*done) return;
        *done = true;
        QJsonObject summary{{"schema_version", 1}, {"success", success}, {"error", error},
            {"version", QCoreApplication::applicationVersion()}, {"request", request},
            {"report_directory", evaluationDirectory_}, {"report", evaluationReport_}};
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
    const auto stage = std::make_shared<int>(0);
    connect(timer, &QTimer::timeout, this, [this, timer, stage, request, output, finish, done] {
        if (*done) { timer->stop(); return; }
        if (busy_ || modelInspectionPending_) return;
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
            saveScreenshot(output + "/evaluation-analysis.png");
            workbenchTabs_->setCurrentIndex(0); queue_->setCurrentRow(0); reviewEvaluationImage(0); *stage = 2;
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
