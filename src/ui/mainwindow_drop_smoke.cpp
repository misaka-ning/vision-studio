// SPDX-License-Identifier: AGPL-3.0-only
#include "mainwindow.h"
#include "evaluationpanel.h"
#include "imagecanvas.h"
#include "core/gpuruntime.h"
#include <QApplication>
#include <QCheckBox>
#include <QComboBox>
#include <QCryptographicHash>
#include <QDialog>
#include <QDebug>
#include <QDir>
#include <QDoubleSpinBox>
#include <QDragEnterEvent>
#include <QDropEvent>
#include <QElapsedTimer>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QMimeData>
#include <QPushButton>
#include <QSaveFile>
#include <QSpinBox>
#include <QTabWidget>
#include <QTimer>
#include <QUrl>

namespace
{
QByteArray contents(const QString &path)
{
    QFile file(path);
    return file.open(QIODevice::ReadOnly) ? file.readAll() : QByteArray();
}
QString localPath(const QJsonValue &value)
{
    return value.isString() ? QFileInfo(value.toString()).canonicalFilePath() : QString();
}
struct DropSmoke
{
    int stage = 0, starts = 0;
    bool done = false, dispatching = false;
    QString dialogError;
    QStringList originals, temporary;
    QJsonObject manifest, checks, inference, evaluation;
    QByteArray session;
};
} // namespace

void MainWindow::runPreviewDropSmoke(const QString &directory, const QString &requestPath)
{
    const QString output = QFileInfo(directory).absoluteFilePath();
    const auto raw = contents(requestPath);
    const QJsonDocument document = raw.size() <= 65536 ? QJsonDocument::fromJson(raw) : QJsonDocument();
    const auto request = document.object();
    const QString dataset = localPath(request.value("dataset_directory"));
    const QString yaml = localPath(request.value("dataset_yaml"));
    const QString video = localPath(request.value("video"));
    const QString model = localPath(request.value("model"));
    auto state = std::make_shared<DropSmoke>();
    for (const auto &value : request.value("images").toArray()) state->temporary.append(localPath(value));
    const QStringList required{
        "directory_drop_loads_dataset", "mixed_types_rejected", "cancel_preserves_dataset",
        "append_preserves_originals_gt_and_session", "busy_drop_does_not_mutate_queue",
        "temporary_real_cpu_inference_without_gt", "temporary_selected_export",
        "full_evaluation_excludes_temporaries", "temporary_suffix_and_single_evaluation_blocked",
        "replace_clears_dataset_context_without_saving_temporary_session", "video_drop_selected",
        "yaml_drop_loads_dataset"};
    auto finish = [this, state, output, required](bool success, const QString &error) {
        if (state->done) return;
        state->done = true;
        for (const auto &key : required) success = success && state->checks.value(key).toBool();
        saveScreenshot(output + (success ? "/final-preview.png" : "/failed-preview.png"));
        QJsonArray paths;
        for (const auto &path : files_) paths.append(path);
        QJsonObject report{{"schema_version", 1}, {"version", QCoreApplication::applicationVersion()},
            {"success", success}, {"error", error}, {"checks", state->checks}, {"phase", state->stage},
            {"inference_start_requests", state->starts}, {"inference", state->inference},
            {"evaluation", state->evaluation}, {"queue_paths", paths},
            {"original_dataset_images", state->originals.size()},
            {"temporary_images", state->temporary.size()}, {"source_status", sourceLabel_->text()},
            {"session_json", QJsonDocument::fromJson(contents(datasetSessionFilePath())).object()}};
        QSaveFile file(output + "/preview-drop-smoke-report.json");
        const auto encoded = QJsonDocument(report).toJson();
        const bool saved = file.open(QIODevice::WriteOnly) && file.write(encoded) == encoded.size() && file.commit();
        qInfo().noquote() << QJsonDocument(report).toJson(QJsonDocument::Compact);
        QCoreApplication::exit(success && saved ? 0 : 2);
    };
    if (!QDir().mkpath(output) || !document.isObject() || dataset.isEmpty() || yaml.isEmpty() ||
        video.isEmpty() || model.isEmpty() || state->temporary.size() != 2 ||
        state->temporary.contains(QString()) || state->temporary[0] == state->temporary[1]) {
        finish(false, "Invalid preview-drop smoke request"); return;
    }
    connect(this, &MainWindow::startRequested, this, [state](const vision::JobRequest &) { ++state->starts; });
    // These are real QApplication drag/drop events, not a direct call to the import slot.
    const auto drop = [this, state, output](const QStringList &paths, const QString &choice = QString()) {
        state->dispatching = true;
        if (!choice.isEmpty()) {
            auto *answer = new QTimer(this);
            auto elapsed = std::make_shared<QElapsedTimer>(); elapsed->start();
            answer->setInterval(10);
            connect(answer, &QTimer::timeout, this, [this, answer, elapsed, state, choice, output] {
                auto *dialog = qobject_cast<QDialog *>(QApplication::activeModalWidget());
                if (dialog && dialog->objectName() == "imageDropDatasetConflictDialog") {
                    auto *button = dialog->findChild<QPushButton *>(choice);
                    if (button) {
                        if (choice == "imageDropAppendButton") {
                            saveScreenshot(output + "/append-decision.png");
                            dialog->grab().save(output + "/append-decision-dialog.png");
                        }
                        answer->stop(); answer->deleteLater(); button->click(); return;
                    }
                }
                if (elapsed->elapsed() > 3000) {
                    state->dialogError = "Expected dataset conflict choice was not available";
                    answer->stop(); answer->deleteLater();
                    if (dialog) dialog->reject();
                }
            });
            answer->start();
        }
        QMimeData mime;
        QList<QUrl> urls;
        for (const auto &path : paths) urls.append(QUrl::fromLocalFile(path));
        mime.setUrls(urls);
        QDragEnterEvent enter(canvas_->rect().center(), Qt::CopyAction, &mime, Qt::LeftButton, Qt::NoModifier);
        QApplication::sendEvent(canvas_, &enter);
        bool accepted = false;
        if (enter.isAccepted()) {
            QDropEvent event(canvas_->rect().center(), Qt::CopyAction, &mime, Qt::LeftButton, Qt::NoModifier);
            QApplication::sendEvent(canvas_, &event);
            accepted = event.isAccepted();
        }
        state->dispatching = false;
        return accepted;
    };
    setComputeDevice(vision::ComputeDevice::CPU);
    setModel(model);
    taskBox_->setCurrentIndex(int(vision::ModelTask::YoloV5));
    inputSize_->setValue(640); confidence_->setValue(.25); iou_->setValue(.45);
    autoExport_->setChecked(false); classFilter_->clear();
    selectRoute(0); workbenchTabs_->setCurrentIndex(0);
    auto *timer = new QTimer(this); timer->setInterval(25);
    connect(timer, &QTimer::timeout, this, [this, state, output, dataset, yaml, video, drop, finish] {
        if (state->done || state->dispatching) return;
        if (!state->dialogError.isEmpty()) { finish(false, state->dialogError); return; }
        if (busy_ || queuePopulating_ || droppedImageImportActive_ || evaluationWorker_ || modelInspectionPending_ || resultIoPending_) return;
        if (state->stage == 0) {
            if (modelClassCount_ <= 0) { finish(false, "Model class inspection failed"); return; }
            if (!drop({dataset})) { finish(false, "Directory drag/drop was rejected"); return; }
            state->stage = 1;
        } else if (state->stage == 1) {
            if (datasetManifest_.isEmpty() || files_.size() < 2) { finish(false, "Dropped dataset did not load"); return; }
            state->originals = files_; state->manifest = datasetManifest_;
            state->session = contents(datasetSessionFilePath());
            state->checks.insert("directory_drop_loads_dataset", !state->session.isEmpty() && state->starts == 0);
            const bool mixed = drop({state->temporary[0], video});
            state->checks.insert("mixed_types_rejected", !mixed && files_ == state->originals && datasetManifest_ == state->manifest);
            drop(state->temporary, "imageDropCancelButton");
            state->checks.insert("cancel_preserves_dataset", files_ == state->originals &&
                datasetManifest_ == state->manifest && contents(datasetSessionFilePath()) == state->session);
            if (!drop(state->temporary, "imageDropAppendButton")) { finish(false, "Append drag/drop was rejected"); return; }
            state->stage = 2;
        } else if (state->stage == 2) {
            const auto expected = state->originals + state->temporary;
            bool gt = datasetSamples_.size() == state->originals.size();
            for (const auto &path : state->originals) gt = gt && datasetSamples_.contains(path);
            for (const auto &path : state->temporary) gt = gt && !datasetSamples_.contains(path);
            state->checks.insert("append_preserves_originals_gt_and_session", files_ == expected && gt &&
                datasetManifest_ == state->manifest && contents(datasetSessionFilePath()) == state->session);
            if (!state->checks.value("append_preserves_originals_gt_and_session").toBool()) { finish(false, "Append provenance/session mismatch"); return; }
            queue_->setCurrentRow(state->originals.size()); detectionRange_->setCurrentIndex(2);
            startInference();
            if (!busy_) { finish(false, "Temporary image inference did not start"); return; }
            drop({video});
            state->checks.insert("busy_drop_does_not_mutate_queue", files_ == expected && sourceKind_ == vision::SourceKind::Images);
            state->stage = 3;
        } else if (state->stage == 3) {
            if (lastResult_.source != state->temporary[0] || lastResult_.image.isNull()) {
                if (failed_) finish(false, "Temporary inference failed: " + lastError_);
                return;
            }
            const bool inferred = state->starts == 1 && !lastResult_.hasGroundTruth &&
                !lastResult_.hasEvaluationMatches && !lastResult_.demonstration && lastResult_.device == vision::ComputeDevice::CPU;
            state->checks.insert("temporary_real_cpu_inference_without_gt", inferred);
            state->inference = QJsonObject{{"actual_device", vision::computeDeviceKey(lastResult_.device)},
                {"backend", lastResult_.backend}, {"predictions", lastResult_.predictions.size()},
                {"has_ground_truth", lastResult_.hasGroundTruth}, {"source", lastResult_.source}};
            const auto prepared = prepareResultExport(lastResult_, output + "/temporary-result", lastConfig_);
            QString error;
            const bool exported = writePreparedResult(prepared, &error);
            const auto json = QJsonDocument::fromJson(contents(prepared.base + ".json")).object();
            state->checks.insert("temporary_selected_export", exported && json.value("source").toString() == state->temporary[0] &&
                !json.contains("ground_truth") && !json.contains("evaluation"));
            saveScreenshot(output + "/temporary-detection.png");
            auto *controls = evaluation_->controlsWidget();
            controls->findChild<QSpinBox *>("evaluationModelClassIdSpin")->setValue(0);
            controls->findChild<QSpinBox *>("evaluationDatasetClassIdSpin")->setValue(0);
            detectionRange_->setCurrentIndex(0); queue_->setCurrentRow(0);
            QString rangeError;
            if (selectedEvaluationRange(&rangeError) != state->originals || !rangeError.isEmpty()) { finish(false, "Evaluation range contains temporary inputs"); return; }
            workbenchTabs_->setCurrentIndex(1); evaluation_->setMode(true); startEvaluation();
            if (!busy_ || !evaluationWorker_) { finish(false, "Original dataset evaluation did not start"); return; }
            state->stage = 4;
        } else if (state->stage == 4) {
            if (evaluationReviewPending_) return;
            QJsonArray expected;
            for (const auto &path : state->originals) expected.append(path);
            const auto request = evaluationReport_.value("request").toObject();
            const bool evaluated = evaluationReport_.value("status").toString() == "complete" &&
                evaluationReport_.value("completed_images").toInt() == state->originals.size() &&
                evaluationReport_.value("full_split").toBool() && request.value("sample_images").toArray() == expected &&
                evaluationEvaluatedCount_ == state->originals.size();
            state->checks.insert("full_evaluation_excludes_temporaries", evaluated);
            state->evaluation = QJsonObject{{"report_directory", evaluationDirectory_},
                {"completed_images", evaluationReport_.value("completed_images")},
                {"full_split", evaluationReport_.value("full_split")}, {"sample_images", request.value("sample_images")}};
            saveScreenshot(output + "/evaluation-with-temporaries.png");
            workbenchTabs_->setCurrentIndex(0); queue_->setCurrentRow(state->originals.size());
            bool blocked = true;
            for (int mode : {1, 2}) {
                detectionRange_->setCurrentIndex(mode);
                QString error;
                blocked = blocked && selectedEvaluationRange(&error).isEmpty() && !error.isEmpty() &&
                    !evaluationStartButton_->isEnabled();
            }
            state->checks.insert("temporary_suffix_and_single_evaluation_blocked", blocked);
            detectionRange_->setCurrentIndex(0);
            if (!drop(state->temporary, "imageDropReplaceButton")) { finish(false, "Replace drag/drop was rejected"); return; }
            state->stage = 5;
        } else if (state->stage == 5) {
            if (canvas_->result().image.isNull() || canvas_->result().source != state->temporary[0]) return;
            state->checks.insert("replace_clears_dataset_context_without_saving_temporary_session",
                datasetManifest_.isEmpty() && datasetSamples_.isEmpty() && files_ == state->temporary &&
                contents(datasetSessionFilePath()) == state->session && !canvas_->result().hasGroundTruth);
            saveScreenshot(output + "/image-replacement.png");
            if (!drop({video})) { finish(false, "Video drag/drop was rejected"); return; }
            state->checks.insert("video_drop_selected", sourceKind_ == vision::SourceKind::Video && streamPath_ == video);
            if (!drop({yaml})) { finish(false, "YAML drag/drop was rejected"); return; }
            state->stage = 6;
        } else if (state->stage == 6) {
            const auto saved = QJsonDocument::fromJson(contents(datasetSessionFilePath())).object();
            state->checks.insert("yaml_drop_loads_dataset", !datasetManifest_.isEmpty() && files_ == state->originals &&
                saved.value("source_path").toString() == yaml && state->starts == 1);
            finish(true, {});
        }
    });
    timer->start();
    QTimer::singleShot(120000, this, [this, state, finish] {
        if (state->done) return;
        if (auto *modal = qobject_cast<QDialog *>(QApplication::activeModalWidget())) modal->reject();
        stopInference();
        finish(false, "Preview-drop smoke timed out: " + statusLabel_->text());
    });
}
