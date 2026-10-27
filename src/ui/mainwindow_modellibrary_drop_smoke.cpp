// SPDX-License-Identifier: AGPL-3.0-only
#include "mainwindow.h"
#include "modelviewer.h"
#include <QAbstractItemModel>
#include <QAction>
#include <QApplication>
#include <QCheckBox>
#include <QComboBox>
#include <QContextMenuEvent>
#include <QCryptographicHash>
#include <QDebug>
#include <QDialog>
#include <QDir>
#include <QDoubleSpinBox>
#include <QDragEnterEvent>
#include <QDragMoveEvent>
#include <QDropEvent>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QMenu>
#include <QMimeData>
#include <QPointer>
#include <QProcess>
#include <QSaveFile>
#include <QSettings>
#include <QSpinBox>
#include <QStackedWidget>
#include <QTimer>
#include <QUrl>

namespace
{
QString digest(const QString &path)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) return {};
    QCryptographicHash hash(QCryptographicHash::Sha256);
    return hash.addData(&file) ? QString::fromLatin1(hash.result().toHex()) : QString();
}
QJsonObject jsonFile(const QString &path)
{
    QFile file(path);
    return file.open(QIODevice::ReadOnly) ? QJsonDocument::fromJson(file.readAll()).object() : QJsonObject{};
}
struct ModelDropSmoke
{
    int stage = 0, starts = 0;
    bool done = false;
    QStringList baseline, expected, protectedPaths;
    QJsonObject beforeHashes, checks, inference;
    QJsonArray events, fixtures;
    QVariantMap names, notes, conversions, evaluations;
    QPointer<QObject> browser;
    QPointer<QProcess> service;
    qint64 servicePid = 0;
    quint64 inspectionGeneration = 0;
};
const QStringList requiredChecks{
    "single_page_drop_selects_global_model", "real_onnx_metadata_loaded",
    "multi_child_drop_preserves_url_order_and_last_selection", "real_pt_metadata_and_structure_loaded",
    "viewport_copy_drop_accepted", "canonical_duplicates_preserve_order_and_metadata",
    "duplicate_preserves_structure_and_inspection_cache", "mixed_invalid_batch_is_atomic",
    "remote_urls_rejected", "non_url_drag_ignored", "move_and_link_only_rejected",
    "directory_empty_and_missing_files_rejected", "context_menu_four_actions_preserved",
    "internal_reorder_preserves_selection_and_metadata", "real_torchscript_drop_and_metadata_loaded",
    "actual_busy_drop_preserves_library_and_active_model", "selected_onnx_real_cpu_inference",
    "selected_result_export_matches_global_model", "preferences_persist_actual_state",
    "original_and_external_files_unchanged", "external_files_referenced_without_copy_or_move"};
} // namespace

void MainWindow::runModelLibraryDropSmoke(const QString &directory)
{
    const QString output = QFileInfo(directory).absoluteFilePath();
    const QString fixtures = qEnvironmentVariable("VISION_STUDIO_SMOKE_INPUT_ROOT", projectRoot_);
    const QString external = output + "/external-models";
    const QString first = external + "/首个-yolov5n.onnx", third = external + "/第三个-yolov5n.ONNX";
    const QString pt = external + "/第二个-yolov8n.pt", script = external + "/第四个-yolov8n.torchscript";
    const QString blocked = external + "/busy-only-yolov5n.onnx", invalid = external + "/invalid.txt";
    const QString empty = external + "/empty.onnx", alias = external + "/alias/../" + QFileInfo(pt).fileName();
    const QString image = fixtures + "/assets/bus.jpg";
    const auto state = std::make_shared<ModelDropSmoke>();
    const auto metadataMatches = [this, state] {
        return modelDisplayNames_ == state->names && modelNotes_ == state->notes &&
            convertedModelProfiles_ == state->conversions && modelEvaluationProfiles_ == state->evaluations;
    };
    const auto finish = [this, state, output, first, metadataMatches](bool success, const QString &error) {
        if (state->done) return;
        state->done = true;
        QJsonObject after;
        for (const auto &path : state->protectedPaths) after.insert(path, digest(path));
        state->checks.insert("original_and_external_files_unchanged", !after.isEmpty() && after == state->beforeHashes);
        bool referenced = true;
        for (const auto &value : state->fixtures) {
            const auto item = value.toObject();
            const QString path = item.value("external").toString();
            referenced = referenced && QFileInfo(path).isFile() &&
                !QFileInfo::exists(dataRoot_ + "/models/" + QFileInfo(path).fileName());
        }
        state->checks.insert("external_files_referenced_without_copy_or_move", referenced);
        persist();
        const bool persisted = settings_->status() == QSettings::NoError &&
            settings_->value("models").toStringList() == models_ && settings_->value("activeModel").toString() == modelPath_ &&
            settings_->value("modelDisplayNames").toMap() == modelDisplayNames_ &&
            settings_->value("modelNotes").toMap() == modelNotes_ &&
            settings_->value("convertedModelProfiles").toMap() == convertedModelProfiles_ &&
            settings_->value("evaluationProfiles").toMap() == modelEvaluationProfiles_;
        state->checks.insert("preferences_persist_actual_state", persisted && metadataMatches());
        QJsonArray failedChecks;
        for (const auto &name : requiredChecks) {
            if (!state->checks.value(name).toBool()) failedChecks.append(name);
            success = success && state->checks.value(name).toBool();
        }
        saveScreenshot(output + (success ? "/model-library-drop-final.png" : "/model-library-drop-failed.png"));
        QJsonObject preferences;
        for (const auto &key : {"models", "activeModel", "modelDisplayNames", "modelNotes", "convertedModelProfiles", "evaluationProfiles"})
            preferences.insert(key, QJsonValue::fromVariant(settings_->value(key)));
        const QJsonObject report{{"schema_version", 1}, {"version", QCoreApplication::applicationVersion()},
            {"qt", qVersion()}, {"success", success}, {"error", error}, {"checks", state->checks},
            {"failed_checks", failedChecks},
            {"required_checks", QJsonArray::fromStringList(requiredChecks)}, {"phase", state->stage},
            {"events", state->events}, {"fixtures", state->fixtures}, {"source_hashes_before", state->beforeHashes},
            {"source_hashes_after", after}, {"inference", state->inference}, {"inference_start_requests", state->starts},
            {"model_order", QJsonArray::fromStringList(models_)}, {"active_model", modelPath_},
            {"expected_inference_model", first}, {"settings_file", settings_->fileName()},
            {"persisted_preferences", preferences}, {"restart_executed", false},
            {"scope", "Actual QApplication drag events, real model inspection/cache and selected-model ONNX CPU inference; process restart is not executed by this smoke."}};
        QSaveFile file(output + "/model-library-drop-smoke-report.json");
        const QByteArray bytes = QJsonDocument(report).toJson();
        const bool saved = file.open(QIODevice::WriteOnly) && file.write(bytes) == bytes.size() && file.commit();
        qInfo().noquote() << QJsonDocument(report).toJson(QJsonDocument::Compact);
        QCoreApplication::exit(success && saved ? 0 : 2);
    };
    if (!QDir().mkpath(external) || !QDir().mkpath(external + "/alias") ||
        qEnvironmentVariable("VISION_STUDIO_DATA_DIR").isEmpty()) {
        finish(false, "Use an isolated VISION_STUDIO_DATA_DIR and writable smoke output"); return;
    }
    const QList<QPair<QString, QString>> copies{
        {fixtures + "/models/yolov5n.onnx", first}, {fixtures + "/models/yolov5n.onnx", third},
        {fixtures + "/models/yolov8n.pt", pt}, {fixtures + "/models/yolov8n.torchscript", script},
        {fixtures + "/models/yolov5n.onnx", blocked}};
    for (const auto &copy : copies) {
        const QString sourceHash = digest(copy.first);
        if (sourceHash.isEmpty() || (!QFileInfo::exists(copy.second) && !QFile::copy(copy.first, copy.second)) ||
            digest(copy.second) != sourceHash) {
            finish(false, "Missing/invalid public smoke fixture; provide ONNX, PT, current TorchScript export and bus image through --smoke-input-root"); return;
        }
        state->fixtures.append(QJsonObject{{"source", copy.first}, {"external", copy.second}, {"sha256", sourceHash}});
        state->protectedPaths.append(copy.first); state->protectedPaths.append(copy.second);
    }
    if (digest(image).isEmpty()) { finish(false, "Missing public bus image fixture"); return; }
    state->protectedPaths.append(image); state->protectedPaths.removeDuplicates();
    for (const auto &path : state->protectedPaths) state->beforeHashes.insert(path, digest(path));
    for (const auto &path : {invalid, empty}) {
        QFile file(path);
        if (!file.open(QIODevice::WriteOnly) || (path == invalid && file.write("not a model\n") <= 0)) {
            finish(false, "Cannot prepare isolated invalid-file fixtures"); return;
        }
    }
    auto *page = findChild<QWidget *>("modelLibraryPage");
    auto *hint = findChild<QLabel *>("modelLibraryDropHint");
    if (!page || !hint || !modelList_ || !modelViewer_) { finish(false, "Missing actual model-library widgets"); return; }
    const auto dispatch = [state](QWidget *target, const QMimeData &mime, Qt::DropActions actions, const QString &name) {
        QDragEnterEvent enter(target->rect().center(), actions, &mime, Qt::LeftButton, Qt::NoModifier);
        QApplication::sendEvent(target, &enter);
        bool moved = false, accepted = false;
        Qt::DropAction result = Qt::IgnoreAction;
        if (enter.isAccepted()) {
            QDragMoveEvent move(target->rect().center(), actions, &mime, Qt::LeftButton, Qt::NoModifier);
            QApplication::sendEvent(target, &move); moved = move.isAccepted();
            if (moved) {
                QDropEvent drop(target->rect().center(), actions, &mime, Qt::LeftButton, Qt::NoModifier);
                QApplication::sendEvent(target, &drop); accepted = drop.isAccepted(); result = drop.dropAction();
            }
        }
        state->events.append(QJsonObject{{"case", name}, {"target", target->objectName()},
            {"enter", enter.isAccepted()}, {"move", moved}, {"drop", accepted}, {"result_action", int(result)}});
        return accepted && result == Qt::CopyAction;
    };
    const auto drop = [dispatch](QWidget *target, const QList<QUrl> &urls, const QString &name,
                                 Qt::DropActions actions = Qt::CopyAction | Qt::MoveAction) {
        QMimeData mime; mime.setUrls(urls); return dispatch(target, mime, actions, name);
    };
    const auto urls = [](const QStringList &paths) {
        QList<QUrl> result; for (const auto &path : paths) result.append(QUrl::fromLocalFile(path)); return result;
    };
    connect(this, &MainWindow::startRequested, this, [state](const vision::JobRequest &) { ++state->starts; });
    cancelPendingDatasetSessionRestore();
    state->baseline = models_; state->expected = state->baseline + QStringList{first};
    setComputeDevice(vision::ComputeDevice::CPU); autoExport_->setChecked(false); classFilter_->clear();
    selectRoute(1);
    if (!drop(page, urls({first}), "single-page")) { finish(false, "Actual page drag/drop rejected valid ONNX"); return; }
    auto *timer = new QTimer(this); timer->setInterval(40);
    connect(timer, &QTimer::timeout, this,
        [this, state, output, first, third, pt, script, blocked, invalid, empty, alias, image, external,
         page, hint, drop, dispatch, urls, metadataMatches, finish] {
        if (state->done) return;
        if (busy_ || modelInspectionPending_ || resultIoPending_) return;
        if (failed_) { finish(false, "Actual inference failed: " + lastError_); return; }
        const auto selected = [this](const QString &path) {
            const auto *conversion = findChild<QLabel *>("conversionSourcePath");
            return modelPath_ == path && modelName_->toolTip() == path && modelList_->currentItem() &&
                modelList_->currentItem()->data(Qt::UserRole).toString() == path &&
                conversion && conversion->text() == path && modelViewer_->modelPath() == path;
        };
        const auto ready = [this] {
            return modelInspectionError_.isEmpty() && modelClassCount_ == 80 && modelInputChannels_ == 3 &&
                modelViewer_->state() == ModelViewer::State::Ready && modelViewer_->graphNodeCount() > 0 &&
                modelViewer_->hierarchyItemCount() > 0 && modelViewer_->parameterCount() > 0;
        };
        if (modelViewer_->state() == ModelViewer::State::Error) {
            finish(false, "Actual structure loading failed: " + modelViewer_->errorString()); return;
        }
        if (state->stage == 0) {
            if (!ready()) return;
            state->checks.insert("single_page_drop_selects_global_model", selected(first) && models_ == state->expected);
            state->checks.insert("real_onnx_metadata_loaded", ready());
            state->expected += QStringList{third, script, pt};
            if (!drop(hint, urls({third, script, pt}), "multi-child")) { finish(false, "Child drag/drop rejected valid batch"); return; }
            state->stage = 1;
        } else if (state->stage == 1) {
            if (!ready()) return;
            state->checks.insert("multi_child_drop_preserves_url_order_and_last_selection", selected(pt) && models_ == state->expected);
            state->checks.insert("real_pt_metadata_and_structure_loaded", ready() && nativeLabels_.size() == 80);
            modelDisplayNames_.insert(pt, "拖入验收 · 显示名称"); modelNotes_.insert(pt, "仅用于验收的备注");
            // Data markers test preservation only; they do not pretend to be an
            // actual conversion/evaluation and are never used for inference.
            convertedModelProfiles_.insert(third, QVariantMap{{"smoke_marker", "preserve-conversion-record"}});
            modelEvaluationProfiles_.insert(third, QVariantMap{{"smoke_marker", "preserve-evaluation-record"}});
            state->names = modelDisplayNames_; state->notes = modelNotes_;
            state->conversions = convertedModelProfiles_; state->evaluations = modelEvaluationProfiles_;
            refreshModelLibrary(); persist();
            state->browser = modelViewer_->findChild<QObject *>("netronWebView");
            state->service = modelViewer_->findChild<QProcess *>();
            state->servicePid = state->service ? state->service->processId() : 0;
            state->inspectionGeneration = modelInspectionGeneration_;
            state->checks.insert("viewport_copy_drop_accepted", drop(modelList_->viewport(), urls({alias, pt}), "canonical-viewport"));
            state->stage = 2;
        } else if (state->stage == 2) {
            if (!ready()) return;
            state->checks.insert("canonical_duplicates_preserve_order_and_metadata", selected(pt) && models_ == state->expected && metadataMatches());
            state->checks.insert("duplicate_preserves_structure_and_inspection_cache", state->browser && state->service &&
                state->servicePid > 0 && modelViewer_->findChild<QObject *>("netronWebView") == state->browser &&
                modelViewer_->findChild<QProcess *>() == state->service && state->service->processId() == state->servicePid &&
                modelInspectionGeneration_ == state->inspectionGeneration);
            const auto unchanged = [this, state, pt, metadataMatches] { return models_ == state->expected && modelPath_ == pt && metadataMatches(); };
            const bool mixed = drop(page, urls({blocked, invalid}), "mixed-invalid");
            state->checks.insert("mixed_invalid_batch_is_atomic", !mixed && unchanged());
            const bool remote = drop(hint, {QUrl::fromLocalFile(blocked), QUrl("https://example.invalid/network.pt")}, "remote-url");
            state->checks.insert("remote_urls_rejected", !remote && unchanged());
            QMimeData text; text.setText("not a model URL");
            state->checks.insert("non_url_drag_ignored", !dispatch(page, text, Qt::CopyAction, "non-url") && unchanged());
            const bool move = drop(page, urls({blocked}), "move-only", Qt::MoveAction);
            const bool link = drop(page, urls({blocked}), "link-only", Qt::LinkAction);
            state->checks.insert("move_and_link_only_rejected", !move && !link && unchanged());
            bool invalidFiles = true;
            for (const auto &path : {external, empty, external + "/missing.pt"})
                invalidFiles = !drop(hint, urls({blocked, path}), "invalid-file") && unchanged() && invalidFiles;
            state->checks.insert("directory_empty_and_missing_files_rejected", invalidFiles);
            modelList_->scrollToItem(modelList_->currentItem());
            const QPoint pos = modelList_->visualItemRect(modelList_->currentItem()).center();
            QContextMenuEvent context(QContextMenuEvent::Mouse, pos, modelList_->viewport()->mapToGlobal(pos));
            QApplication::sendEvent(modelList_->viewport(), &context);
            auto *menu = modelList_->findChild<QMenu *>("modelContextMenu");
            bool contextOk = menu && menu->actions().size() == 4;
            for (const auto &name : {"modelOpenLocationAction", "modelNoteAction", "modelRenameAction", "modelRemoveAction"}) {
                const auto *action = menu ? menu->findChild<QAction *>(name) : nullptr;
                contextOk = contextOk && action && action->isEnabled() && action->data().toString() == pt;
            }
            state->checks.insert("context_menu_four_actions_preserved", contextOk && unchanged());
            if (menu) menu->close();
            const QString selectedPath = modelPath_;
            QStringList reordered = state->expected; reordered.append(reordered.takeFirst());
            const bool moved = modelList_->model()->moveRows(QModelIndex(), 0, 1, QModelIndex(), modelList_->count());
            state->checks.insert("internal_reorder_preserves_selection_and_metadata", moved && models_ == reordered &&
                modelPath_ == selectedPath && metadataMatches() && modelList_->currentItem()->data(Qt::UserRole).toString() == selectedPath);
            state->expected = reordered;
            saveScreenshot(output + "/model-library-imported.png");
            if (!drop(modelList_->viewport(), urls({script}), "torchscript-viewport")) { finish(false, "Viewport rejected actual TorchScript"); return; }
            state->stage = 3;
        } else if (state->stage == 3) {
            if (!ready()) return;
            state->checks.insert("real_torchscript_drop_and_metadata_loaded", selected(script) && models_ == state->expected && ready());
            if (!drop(page, urls({first}), "select-inference-onnx")) { finish(false, "Could not select dropped ONNX for actual inference"); return; }
            state->stage = 4;
        } else if (state->stage == 4) {
            if (!ready()) return;
            if (!selected(first)) { finish(false, "Global model differs before inference"); return; }
            inputSize_->setValue(640); confidence_->setValue(.25); iou_->setValue(.45);
            scale_->setValue(1.0 / 255.0); meanR_->setValue(0); meanG_->setValue(0); meanB_->setValue(0);
            inputColorMode_->setCurrentIndex(inputColorMode_->findData("rgb"));
            labels_.clear(); labelsPath_.clear(); classFilter_->clear();
            clearImageResults(); files_.clear(); queue_->clear(); sourceKind_ = vision::SourceKind::Images;
            addFiles({image}); detectionRange_->setCurrentIndex(0);
            startInference();
            const bool wasBusy = busy_;
            const bool accepted = drop(page, urls({blocked}), "actual-inference-busy");
            state->checks.insert("actual_busy_drop_preserves_library_and_active_model", wasBusy && !accepted &&
                models_ == state->expected && modelPath_ == first && metadataMatches());
            state->stage = 5;
        } else if (state->stage == 5) {
            if (lastResult_.image.isNull()) return;
            const bool inferred = state->starts == 1 && completed_ == 1 && !lastResult_.predictions.isEmpty() &&
                lastResult_.source == image && !lastResult_.demonstration && lastResult_.device == vision::ComputeDevice::CPU &&
                lastResult_.backend.contains("OpenCV") && lastConfig_.modelPath == first && modelPath_ == first;
            state->checks.insert("selected_onnx_real_cpu_inference", inferred);
            const auto prepared = prepareResultExport(lastResult_, output + "/selected-result", lastConfig_);
            QString error;
            const bool exported = writePreparedResult(prepared, &error);
            const auto document = jsonFile(prepared.base + ".json");
            state->checks.insert("selected_result_export_matches_global_model", exported &&
                document.value("model_file").toString() == first && document.value("source").toString() == image &&
                document.value("predictions").toArray().size() == lastResult_.predictions.size());
            state->inference = QJsonObject{{"model", first}, {"source", image}, {"actual_device", "cpu"},
                {"backend", lastResult_.backend}, {"predictions", lastResult_.predictions.size()},
                {"export_json", prepared.base + ".json"}, {"export_sha256", digest(prepared.base + ".json")}};
            selectRoute(0); saveScreenshot(output + "/model-library-selected-inference.png");
            selectRoute(1); finish(true, error);
        }
    });
    timer->start();
    QTimer::singleShot(180000, this, [this, state, finish] {
        if (state->done) return;
        if (auto *modal = qobject_cast<QDialog *>(QApplication::activeModalWidget())) modal->reject();
        stopInference(); finish(false, "Model-library drop smoke timed out: " + statusLabel_->text());
    });
}
