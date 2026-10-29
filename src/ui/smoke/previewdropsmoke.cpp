// SPDX-License-Identifier: AGPL-3.0-only
#include "smokedrivers.h"
#include "smokecontext.h"
#include "ui/evaluationpanel.h"
#include "ui/imagecanvas.h"
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

namespace vision::ui::smoke
{
namespace
{
class PreviewDropDriver final : public SmokeContext
{
  public:
    using SmokeContext::SmokeContext;
    void run(const QString &directory, const QString &requestPath);
};
}
void PreviewDropDriver::run(const QString &directory, const QString &requestPath)
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
        window_.saveScreenshot(output + (success ? "/final-preview.png" : "/failed-preview.png"));
        QJsonArray paths;
        for (const auto &path : window_.inputState().files) paths.append(path);
        QJsonObject report{{"schema_version", 1}, {"version", QCoreApplication::applicationVersion()},
            {"success", success}, {"error", error}, {"checks", state->checks}, {"phase", state->stage},
            {"inference_start_requests", state->starts}, {"inference", state->inference},
            {"evaluation", state->evaluation}, {"queue_paths", paths},
            {"original_dataset_images", state->originals.size()},
            {"temporary_images", state->temporary.size()}, {"source_status", ui_.sourceLabel->text()},
            {"session_json", QJsonDocument::fromJson(contents(window_.datasetSessionFilePath())).object()}};
        QSaveFile file(output + "/preview-drop-smoke-report.json");
        const auto encoded = QJsonDocument(report).toJson();
        const bool saved = file.open(QIODevice::WriteOnly) && file.write(encoded) == encoded.size() && file.commit();
        qInfo().noquote() << QJsonDocument(report).toJson(QJsonDocument::Compact);
        window_.requestExit(success && saved ? 0 : 2);
    };
    if (!QDir().mkpath(output) || !document.isObject() || dataset.isEmpty() || yaml.isEmpty() ||
        video.isEmpty() || model.isEmpty() || state->temporary.size() != 2 ||
        state->temporary.contains(QString()) || state->temporary[0] == state->temporary[1]) {
        finish(false, "Invalid preview-drop smoke request"); return;
    }
    connect(&window_, &MainWindow::startRequested, this, [state](const vision::JobRequest &) { ++state->starts; });
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
                            window_.saveScreenshot(output + "/append-decision.png");
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
        QDragEnterEvent enter(ui_.canvas->rect().center(), Qt::CopyAction, &mime, Qt::LeftButton, Qt::NoModifier);
        QApplication::sendEvent(ui_.canvas, &enter);
        bool accepted = false;
        if (enter.isAccepted()) {
            QDropEvent event(ui_.canvas->rect().center(), Qt::CopyAction, &mime, Qt::LeftButton, Qt::NoModifier);
            QApplication::sendEvent(ui_.canvas, &event);
            accepted = event.isAccepted();
        }
        state->dispatching = false;
        return accepted;
    };
    window_.setComputeDevice(vision::ComputeDevice::CPU);
    window_.selectModel(model);
    ui_.taskBox->setCurrentIndex(int(vision::ModelTask::YoloV5));
    ui_.inputSize->setValue(640); ui_.confidence->setValue(.25); ui_.iou->setValue(.45);
    ui_.autoExport->setChecked(false); ui_.classFilter->clear();
    selectRoute(0); ui_.workbenchTabs->setCurrentIndex(0);
    auto *timer = new QTimer(this); timer->setInterval(25);
    connect(timer, &QTimer::timeout, this, [this, state, output, dataset, yaml, video, drop, finish] {
        if (state->done || state->dispatching) return;
        if (!state->dialogError.isEmpty()) { finish(false, state->dialogError); return; }
        if (window_.isWorkbenchBusy() || window_.inputState().queuePopulating || window_.inputState().importing || window_.isEvaluationBusy() || window_.modelState().inspectionPending || window_.pendingResultWrites()) return;
        if (state->stage == 0) {
            if (window_.modelState().classCount <= 0) { finish(false, "Model class inspection failed"); return; }
            if (!drop({dataset})) { finish(false, "Directory drag/drop was rejected"); return; }
            state->stage = 1;
        } else if (state->stage == 1) {
            if (window_.inputState().datasetManifest.isEmpty() || window_.inputState().files.size() < 2) { finish(false, "Dropped dataset did not load"); return; }
            state->originals = window_.inputState().files; state->manifest = window_.inputState().datasetManifest;
            state->session = contents(window_.datasetSessionFilePath());
            state->checks.insert("directory_drop_loads_dataset", !state->session.isEmpty() && state->starts == 0);
            const bool mixed = drop({state->temporary[0], video});
            state->checks.insert("mixed_types_rejected", !mixed && window_.inputState().files == state->originals && window_.inputState().datasetManifest == state->manifest);
            drop(state->temporary, "imageDropCancelButton");
            state->checks.insert("cancel_preserves_dataset", window_.inputState().files == state->originals &&
                window_.inputState().datasetManifest == state->manifest && contents(window_.datasetSessionFilePath()) == state->session);
            if (!drop(state->temporary, "imageDropAppendButton")) { finish(false, "Append drag/drop was rejected"); return; }
            state->stage = 2;
        } else if (state->stage == 2) {
            const auto expected = state->originals + state->temporary;
            bool gt = window_.inputState().datasetSamples.size() == state->originals.size();
            for (const auto &path : state->originals) gt = gt && window_.inputState().datasetSamples.contains(path);
            for (const auto &path : state->temporary) gt = gt && !window_.inputState().datasetSamples.contains(path);
            state->checks.insert("append_preserves_originals_gt_and_session", window_.inputState().files == expected && gt &&
                window_.inputState().datasetManifest == state->manifest && contents(window_.datasetSessionFilePath()) == state->session);
            if (!state->checks.value("append_preserves_originals_gt_and_session").toBool()) { finish(false, "Append provenance/session mismatch"); return; }
            ui_.queue->setCurrentRow(state->originals.size()); ui_.detectionRange->setCurrentIndex(2);
            ui_.runButton->click();
            if (!window_.isWorkbenchBusy()) { finish(false, "Temporary image inference did not start"); return; }
            drop({video});
            state->checks.insert("busy_drop_does_not_mutate_queue", window_.inputState().files == expected && window_.inputState().sourceKind == vision::SourceKind::Images);
            state->stage = 3;
        } else if (state->stage == 3) {
            if (window_.displayedResult().source != state->temporary[0] || window_.displayedResult().image.isNull()) {
                if (window_.detectionState().failed) finish(false, "Temporary inference failed: " + window_.detectionState().error);
                return;
            }
            const bool inferred = state->starts == 1 && !window_.displayedResult().hasGroundTruth &&
                !window_.displayedResult().hasEvaluationMatches && !window_.displayedResult().demonstration && window_.displayedResult().device == vision::ComputeDevice::CPU;
            state->checks.insert("temporary_real_cpu_inference_without_gt", inferred);
            state->inference = QJsonObject{{"actual_device", vision::computeDeviceKey(window_.displayedResult().device)},
                {"backend", window_.displayedResult().backend}, {"predictions", window_.displayedResult().predictions.size()},
                {"has_ground_truth", window_.displayedResult().hasGroundTruth}, {"source", window_.displayedResult().source}};
            QString base, error;
            const bool exported = window_.exportDisplayedResult(output + "/temporary-result", &base, &error);
            const auto json = QJsonDocument::fromJson(contents(base + ".json")).object();
            state->checks.insert("temporary_selected_export", exported && json.value("source").toString() == state->temporary[0] &&
                !json.contains("ground_truth") && !json.contains("evaluation"));
            window_.saveScreenshot(output + "/temporary-detection.png");
            auto *controls = ui_.evaluation->controlsWidget();
            controls->findChild<QSpinBox *>("evaluationModelClassIdSpin")->setValue(0);
            controls->findChild<QSpinBox *>("evaluationDatasetClassIdSpin")->setValue(0);
            ui_.detectionRange->setCurrentIndex(0); ui_.queue->setCurrentRow(0);
            QString rangeError;
            if (window_.inputs()->range(static_cast<vision::app::ImageRange>(ui_.detectionRange->currentIndex()), true, &rangeError) != state->originals || !rangeError.isEmpty()) { finish(false, "Evaluation range contains temporary inputs"); return; }
            ui_.workbenchTabs->setCurrentIndex(1); ui_.evaluation->setMode(true); ui_.evaluationStartButton->click();
            if (!window_.isWorkbenchBusy() || !window_.isEvaluationBusy()) { finish(false, "Original dataset evaluation did not start"); return; }
            state->stage = 4;
        } else if (state->stage == 4) {
            if (window_.evaluationState().review.pending) return;
            QJsonArray expected;
            for (const auto &path : state->originals) expected.append(path);
            const auto request = window_.evaluationState().report.value("request").toObject();
            const bool evaluated = window_.evaluationState().report.value("status").toString() == "complete" &&
                window_.evaluationState().report.value("completed_images").toInt() == state->originals.size() &&
                window_.evaluationState().report.value("full_split").toBool() && request.value("sample_images").toArray() == expected &&
                window_.evaluationState().review.evaluated == state->originals.size();
            state->checks.insert("full_evaluation_excludes_temporaries", evaluated);
            state->evaluation = QJsonObject{{"report_directory", window_.evaluationState().directory},
                {"completed_images", window_.evaluationState().report.value("completed_images")},
                {"full_split", window_.evaluationState().report.value("full_split")}, {"sample_images", request.value("sample_images")}};
            window_.saveScreenshot(output + "/evaluation-with-temporaries.png");
            ui_.workbenchTabs->setCurrentIndex(0); ui_.queue->setCurrentRow(state->originals.size());
            bool blocked = true;
            for (int mode : {1, 2}) {
                ui_.detectionRange->setCurrentIndex(mode);
                QString error;
                blocked = blocked && window_.inputs()->range(static_cast<vision::app::ImageRange>(ui_.detectionRange->currentIndex()), true, &error).isEmpty() && !error.isEmpty() &&
                    !ui_.evaluationStartButton->isEnabled();
            }
            state->checks.insert("temporary_suffix_and_single_evaluation_blocked", blocked);
            ui_.detectionRange->setCurrentIndex(0);
            if (!drop(state->temporary, "imageDropReplaceButton")) { finish(false, "Replace drag/drop was rejected"); return; }
            state->stage = 5;
        } else if (state->stage == 5) {
            if (ui_.canvas->result().image.isNull() || ui_.canvas->result().source != state->temporary[0]) return;
            state->checks.insert("replace_clears_dataset_context_without_saving_temporary_session",
                window_.inputState().datasetManifest.isEmpty() && window_.inputState().datasetSamples.isEmpty() && window_.inputState().files == state->temporary &&
                contents(window_.datasetSessionFilePath()) == state->session && !ui_.canvas->result().hasGroundTruth);
            window_.saveScreenshot(output + "/image-replacement.png");
            if (!drop({video})) { finish(false, "Video drag/drop was rejected"); return; }
            state->checks.insert("video_drop_selected", window_.inputState().sourceKind == vision::SourceKind::Video && window_.inputState().streamPath == video);
            if (!drop({yaml})) { finish(false, "YAML drag/drop was rejected"); return; }
            state->stage = 6;
        } else if (state->stage == 6) {
            const auto saved = QJsonDocument::fromJson(contents(window_.datasetSessionFilePath())).object();
            state->checks.insert("yaml_drop_loads_dataset", !window_.inputState().datasetManifest.isEmpty() && window_.inputState().files == state->originals &&
                saved.value("source_path").toString() == yaml && state->starts == 1);
            finish(true, {});
        }
    });
    timer->start();
    QTimer::singleShot(120000, this, [this, state, finish] {
        if (state->done) return;
        if (auto *modal = qobject_cast<QDialog *>(QApplication::activeModalWidget())) modal->reject();
        ui_.stopButton->click();
        finish(false, "Preview-drop smoke timed out: " + status_->text());
    });
}

void runPreviewDropSmoke(MainWindow &window, const QString &directory, const QString &requestPath)
{
    auto *driver = new PreviewDropDriver(window);
    if (!driver->hasWorkbench()) {
        qCritical("Missing actual workbench widgets for preview-drop smoke");
        window.requestExit(2); return;
    }
    driver->run(directory, requestPath);
}
} // namespace vision::ui::smoke
