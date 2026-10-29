// SPDX-License-Identifier: AGPL-3.0-only
#include "smokedrivers.h"
#include "smokecontext.h"
#include "ui/pages/modelslibrarypage.h"
#include "ui/pages/modeldisplaypage.h"
#include "ui/modelviewer.h"
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
#include <QPushButton>
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

namespace vision::ui::smoke
{
namespace
{
class ModelLibraryDropDriver final : public SmokeContext
{
  public:
    explicit ModelLibraryDropDriver(MainWindow &window) : SmokeContext(window),
        preferences_(window.preferencesFilePath(), QSettings::IniFormat)
    {
        if (auto *page = window.findChild<ModelsLibraryPage *>()) library_ = page->controls();
        if (auto *page = window.findChild<ModelDisplayPage *>()) viewer_ = page->controls().viewer;
    }
    void run(const QString &directory);
  private:
    ModelsLibraryPage::Controls library_;
    ModelViewer *viewer_ = nullptr;
    QSettings preferences_;
};
}
void ModelLibraryDropDriver::run(const QString &directory)
{
    const QString output = QFileInfo(directory).absoluteFilePath();
    const QString fixtures = qEnvironmentVariable("VISION_STUDIO_SMOKE_INPUT_ROOT", window_.projectRoot());
    const QString external = output + "/external-models";
    const QString first = external + "/首个-yolov5n.onnx", third = external + "/第三个-yolov5n.ONNX";
    const QString pt = external + "/第二个-yolov8n.pt", script = external + "/第四个-yolov8n.torchscript";
    const QString blocked = external + "/busy-only-yolov5n.onnx", invalid = external + "/invalid.txt";
    const QString empty = external + "/empty.onnx", alias = external + "/alias/../" + QFileInfo(pt).fileName();
    const QString image = fixtures + "/assets/bus.jpg";
    const auto state = std::make_shared<ModelDropSmoke>();
    const auto metadataMatches = [this, state] {
        return window_.modelState().displayNames == state->names && window_.modelState().notes == state->notes &&
            window_.modelState().convertedProfiles == state->conversions && window_.modelState().evaluationProfiles == state->evaluations;
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
                !QFileInfo::exists(window_.dataRoot() + "/models/" + QFileInfo(path).fileName());
        }
        state->checks.insert("external_files_referenced_without_copy_or_move", referenced);
        window_.persistPreferences(); preferences_.sync();
        const bool persisted = preferences_.status() == QSettings::NoError &&
            preferences_.value("models").toStringList() == window_.modelState().models && preferences_.value("activeModel").toString() == window_.modelState().activeModel &&
            preferences_.value("modelDisplayNames").toMap() == window_.modelState().displayNames &&
            preferences_.value("modelNotes").toMap() == window_.modelState().notes &&
            preferences_.value("convertedModelProfiles").toMap() == window_.modelState().convertedProfiles &&
            preferences_.value("evaluationProfiles").toMap() == window_.modelState().evaluationProfiles;
        state->checks.insert("preferences_persist_actual_state", persisted && metadataMatches());
        QJsonArray failedChecks;
        for (const auto &name : requiredChecks) {
            if (!state->checks.value(name).toBool()) failedChecks.append(name);
            success = success && state->checks.value(name).toBool();
        }
        window_.saveScreenshot(output + (success ? "/model-library-drop-final.png" : "/model-library-drop-failed.png"));
        QJsonObject preferences;
        for (const auto &key : {"models", "activeModel", "modelDisplayNames", "modelNotes", "convertedModelProfiles", "evaluationProfiles"})
            preferences.insert(key, QJsonValue::fromVariant(preferences_.value(key)));
        const QJsonObject report{{"schema_version", 1}, {"version", QCoreApplication::applicationVersion()},
            {"qt", qVersion()}, {"success", success}, {"error", error}, {"checks", state->checks},
            {"failed_checks", failedChecks},
            {"required_checks", QJsonArray::fromStringList(requiredChecks)}, {"phase", state->stage},
            {"events", state->events}, {"fixtures", state->fixtures}, {"source_hashes_before", state->beforeHashes},
            {"source_hashes_after", after}, {"inference", state->inference}, {"inference_start_requests", state->starts},
            {"model_order", QJsonArray::fromStringList(window_.modelState().models)}, {"active_model", window_.modelState().activeModel},
            {"expected_inference_model", first}, {"settings_file", preferences_.fileName()},
            {"persisted_preferences", preferences}, {"restart_executed", false},
            {"scope", "Actual QApplication drag events, real model inspection/cache and selected-model ONNX CPU inference; process restart is not executed by this smoke."}};
        QSaveFile file(output + "/model-library-drop-smoke-report.json");
        const QByteArray bytes = QJsonDocument(report).toJson();
        const bool saved = file.open(QIODevice::WriteOnly) && file.write(bytes) == bytes.size() && file.commit();
        qInfo().noquote() << QJsonDocument(report).toJson(QJsonDocument::Compact);
        window_.requestExit(success && saved ? 0 : 2);
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
    auto *page = window_.findChild<QWidget *>("modelLibraryPage");
    auto *hint = window_.findChild<QLabel *>("modelLibraryDropHint");
    if (!page || !hint || !library_.list || !viewer_) { finish(false, "Missing actual model-library widgets"); return; }
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
    connect(&window_, &MainWindow::startRequested, this, [state](const vision::JobRequest &) { ++state->starts; });
    window_.inputs()->noteUserInteraction();
    state->baseline = window_.modelState().models; state->expected = state->baseline + QStringList{first};
    window_.setComputeDevice(vision::ComputeDevice::CPU); ui_.autoExport->setChecked(false); ui_.classFilter->clear();
    selectRoute(1);
    if (!drop(page, urls({first}), "single-page")) { finish(false, "Actual page drag/drop rejected valid ONNX"); return; }
    auto *timer = new QTimer(this); timer->setInterval(40);
    connect(timer, &QTimer::timeout, this,
        [this, state, output, first, third, pt, script, blocked, invalid, empty, alias, image, external,
         page, hint, drop, dispatch, urls, metadataMatches, finish] {
        if (state->done) return;
        if (window_.isWorkbenchBusy() || window_.modelState().inspectionPending || window_.pendingResultWrites()) return;
        if (window_.detectionState().failed) { finish(false, "Actual inference failed: " + window_.detectionState().error); return; }
        const auto selected = [this](const QString &path) {
            const auto *conversion = window_.findChild<QLabel *>("conversionSourcePath");
            return window_.modelState().activeModel == path && ui_.modelName->toolTip() == path && library_.list->currentItem() &&
                library_.list->currentItem()->data(Qt::UserRole).toString() == path &&
                conversion && conversion->text() == path && viewer_->modelPath() == path;
        };
        const auto ready = [this] {
            return window_.modelState().inspectionError.isEmpty() && window_.modelState().classCount == 80 && window_.modelState().inputChannels == 3 &&
                viewer_->state() == ModelViewer::State::Ready && viewer_->graphNodeCount() > 0 &&
                viewer_->hierarchyItemCount() > 0 && viewer_->parameterCount() > 0;
        };
        if (viewer_->state() == ModelViewer::State::Error) {
            finish(false, "Actual structure loading failed: " + viewer_->errorString()); return;
        }
        if (state->stage == 0) {
            if (!ready()) return;
            state->checks.insert("single_page_drop_selects_global_model", selected(first) && window_.modelState().models == state->expected);
            state->checks.insert("real_onnx_metadata_loaded", ready());
            state->expected += QStringList{third, script, pt};
            if (!drop(hint, urls({third, script, pt}), "multi-child")) { finish(false, "Child drag/drop rejected valid batch"); return; }
            state->stage = 1;
        } else if (state->stage == 1) {
            if (!ready()) return;
            state->checks.insert("multi_child_drop_preserves_url_order_and_last_selection", selected(pt) && window_.modelState().models == state->expected);
            state->checks.insert("real_pt_metadata_and_structure_loaded", ready() && window_.modelState().nativeLabels.size() == 80);
            window_.models()->rename(pt, "拖入验收 · 显示名称"); window_.models()->setNote(pt, "仅用于验收的备注");
            // Data markers test preservation only; they do not pretend to be an
            // actual conversion/evaluation and are never used for inference.
            window_.models()->setConvertedProfileRecord(third, QVariantMap{{"smoke_marker", "preserve-conversion-record"}});
            window_.models()->setEvaluationProfile(third, QVariantMap{{"smoke_marker", "preserve-evaluation-record"}});
            state->names = window_.modelState().displayNames; state->notes = window_.modelState().notes;
            state->conversions = window_.modelState().convertedProfiles; state->evaluations = window_.modelState().evaluationProfiles;
            window_.refreshModelPresentation(); window_.persistPreferences(); preferences_.sync();
            state->browser = viewer_->findChild<QObject *>("netronWebView");
            state->service = viewer_->findChild<QProcess *>();
            state->servicePid = state->service ? state->service->processId() : 0;
            state->inspectionGeneration = window_.modelState().inspectionGeneration;
            state->checks.insert("viewport_copy_drop_accepted", drop(library_.list->viewport(), urls({alias, pt}), "canonical-viewport"));
            state->stage = 2;
        } else if (state->stage == 2) {
            if (!ready()) return;
            state->checks.insert("canonical_duplicates_preserve_order_and_metadata", selected(pt) && window_.modelState().models == state->expected && metadataMatches());
            state->checks.insert("duplicate_preserves_structure_and_inspection_cache", state->browser && state->service &&
                state->servicePid > 0 && viewer_->findChild<QObject *>("netronWebView") == state->browser &&
                viewer_->findChild<QProcess *>() == state->service && state->service->processId() == state->servicePid &&
                window_.modelState().inspectionGeneration == state->inspectionGeneration);
            const auto unchanged = [this, state, pt, metadataMatches] { return window_.modelState().models == state->expected && window_.modelState().activeModel == pt && metadataMatches(); };
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
            library_.list->scrollToItem(library_.list->currentItem());
            const QPoint pos = library_.list->visualItemRect(library_.list->currentItem()).center();
            QContextMenuEvent context(QContextMenuEvent::Mouse, pos, library_.list->viewport()->mapToGlobal(pos));
            QApplication::sendEvent(library_.list->viewport(), &context);
            auto *menu = library_.list->findChild<QMenu *>("modelContextMenu");
            bool contextOk = menu && menu->actions().size() == 4;
            for (const auto &name : {"modelOpenLocationAction", "modelNoteAction", "modelRenameAction", "modelRemoveAction"}) {
                const auto *action = menu ? menu->findChild<QAction *>(name) : nullptr;
                contextOk = contextOk && action && action->isEnabled() && action->data().toString() == pt;
            }
            state->checks.insert("context_menu_four_actions_preserved", contextOk && unchanged());
            if (menu) menu->close();
            const QString selectedPath = window_.modelState().activeModel;
            QStringList reordered = state->expected; reordered.append(reordered.takeFirst());
            const bool moved = library_.list->model()->moveRows(QModelIndex(), 0, 1, QModelIndex(), library_.list->count());
            state->checks.insert("internal_reorder_preserves_selection_and_metadata", moved && window_.modelState().models == reordered &&
                window_.modelState().activeModel == selectedPath && metadataMatches() && library_.list->currentItem()->data(Qt::UserRole).toString() == selectedPath);
            state->expected = reordered;
            window_.saveScreenshot(output + "/model-library-imported.png");
            if (!drop(library_.list->viewport(), urls({script}), "torchscript-viewport")) { finish(false, "Viewport rejected actual TorchScript"); return; }
            state->stage = 3;
        } else if (state->stage == 3) {
            if (!ready()) return;
            state->checks.insert("real_torchscript_drop_and_metadata_loaded", selected(script) && window_.modelState().models == state->expected && ready());
            if (!drop(page, urls({first}), "select-inference-onnx")) { finish(false, "Could not select dropped ONNX for actual inference"); return; }
            state->stage = 4;
        } else if (state->stage == 4) {
            if (!ready()) return;
            if (!selected(first)) { finish(false, "Global model differs before inference"); return; }
            ui_.inputSize->setValue(640); ui_.confidence->setValue(.25); ui_.iou->setValue(.45);
            ui_.scale->setValue(1.0 / 255.0); ui_.meanR->setValue(0); ui_.meanG->setValue(0); ui_.meanB->setValue(0);
            ui_.inputColorMode->setCurrentIndex(ui_.inputColorMode->findData("rgb"));
            window_.configuration()->setLabels({}, {}); ui_.classFilter->clear();
            window_.inputs()->clear();
            window_.inputs()->requestImages({image}, false); ui_.detectionRange->setCurrentIndex(0);
            state->stage = 41;
        } else if (state->stage == 41) {
            if (window_.isWorkbenchBusy() || window_.inputState().importing || window_.inputState().queuePopulating) return;
            ui_.runButton->click();
            const bool wasBusy = window_.isWorkbenchBusy();
            const bool accepted = drop(page, urls({blocked}), "actual-inference-busy");
            state->checks.insert("actual_busy_drop_preserves_library_and_active_model", wasBusy && !accepted &&
                window_.modelState().models == state->expected && window_.modelState().activeModel == first && metadataMatches());
            state->stage = 5;
        } else if (state->stage == 5) {
            if (window_.isWorkbenchBusy() || window_.displayedResult().image.isNull()) return;
            const bool inferred = state->starts == 1 && window_.completedCount() == 1 && !window_.displayedResult().predictions.isEmpty() &&
                window_.displayedResult().source == image && !window_.displayedResult().demonstration && window_.displayedResult().device == vision::ComputeDevice::CPU &&
                window_.displayedResult().backend.contains("OpenCV") && window_.displayedConfig().modelPath == first && window_.modelState().activeModel == first;
            state->checks.insert("selected_onnx_real_cpu_inference", inferred);
            QString base, error;
            const bool exported = window_.exportDisplayedResult(output + "/selected-result", &base, &error);
            const auto document = jsonFile(base + ".json");
            state->checks.insert("selected_result_export_matches_global_model", exported &&
                document.value("model_file").toString() == first && document.value("source").toString() == image &&
                document.value("predictions").toArray().size() == window_.displayedResult().predictions.size());
            state->inference = QJsonObject{{"model", first}, {"source", image}, {"actual_device", "cpu"},
                {"backend", window_.displayedResult().backend}, {"predictions", window_.displayedResult().predictions.size()},
                {"export_json", base + ".json"}, {"export_sha256", digest(base + ".json")}};
            selectRoute(0); window_.saveScreenshot(output + "/model-library-selected-inference.png");
            selectRoute(1); finish(true, error);
        }
    });
    timer->start();
    QTimer::singleShot(180000, this, [this, state, finish] {
        if (state->done) return;
        if (auto *modal = qobject_cast<QDialog *>(QApplication::activeModalWidget())) modal->reject();
        ui_.stopButton->click(); finish(false, "Model-library drop smoke timed out: " + status_->text());
    });
}

void runModelLibraryDropSmoke(MainWindow &window, const QString &directory)
{
    auto *driver = new ModelLibraryDropDriver(window);
    if (!driver->hasWorkbench()) {
        qCritical("Missing actual workbench widgets for model-library drop smoke");
        window.requestExit(2); return;
    }
    driver->run(directory);
}
} // namespace vision::ui::smoke
