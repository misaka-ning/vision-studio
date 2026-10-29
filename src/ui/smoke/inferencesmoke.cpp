// SPDX-License-Identifier: AGPL-3.0-only
#include "smokedrivers.h"
#include "smokecontext.h"
#include "ui/pages/modeldisplaypage.h"
#include "ui/modelviewer.h"
#include "core/gpuruntime.h"
#include <QCryptographicHash>
#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QProcess>
#include <QTimer>

namespace vision::ui::smoke {
namespace {
class ModelDisplaySmoke final : public SmokeContext
{
public:
    explicit ModelDisplaySmoke(MainWindow &window) : SmokeContext(window)
    {
        const auto &controls = window.findChild<ModelDisplayPage *>()->controls();
        modelViewer_ = controls.viewer; structureHint_ = controls.hint;
    }
    void run(const QString &dir, const QString &model);
private:
    ModelViewer *modelViewer_ = nullptr;
    QLabel *structureHint_ = nullptr;
};
void ModelDisplaySmoke::run(const QString &dir, const QString &model)
{
    const QString output = QFileInfo(dir).absoluteFilePath();
    if (!QDir().mkpath(output)) { window_.requestExit(2); return; }
    const auto completed = std::make_shared<bool>(false);
    const auto capturing = std::make_shared<bool>(false);
    const auto views = std::make_shared<QJsonObject>();
    const auto yaml = std::make_shared<QJsonObject>();
    const auto cachePreserved = std::make_shared<bool>(false);
    const auto fileDigest = [](const QString &path) {
        QFile file(path);
        if (!file.open(QIODevice::ReadOnly)) return QString();
        QCryptographicHash hash(QCryptographicHash::Sha256);
        return hash.addData(&file) ? QString::fromLatin1(hash.result().toHex()) : QString();
    };
    const QString sourceHash = fileDigest(model);
    const auto writeReport = [this, output, completed, views, yaml, cachePreserved](bool success, const QString &error) {
        if (*completed) return;
        *completed = true;
        if (!success) window_.saveScreenshot(output + "/model-display.png");
        const QJsonObject report{{"success", success}, {"error", error}, {"application", "Vision Studio"},
            {"version", QCoreApplication::applicationVersion()}, {"qt", qVersion()}, {"netron", "9.3.1"},
            {"model", modelViewer_->modelPath()}, {"nodes", modelViewer_->graphNodeCount()},
            {"hierarchy_items", modelViewer_->hierarchyItemCount()}, {"parameter_rows", modelViewer_->parameterCount()},
            {"display_modes", *views}, {"yaml", *yaml}, {"cache_preserved", *cachePreserved},
            {"onnx_hint_visible", structureHint_->isVisible()}, {"onnx_hint", structureHint_->text()}};
        QString errorText;
        const bool saved = vision::app::ResultWriter::writeAtomically(output + "/model-display-report.json", QJsonDocument(report).toJson(), &errorText);
        qInfo().noquote() << QJsonDocument(report).toJson(QJsonDocument::Compact);
        window_.requestExit(success && saved ? 0 : 2);
    };
    const auto captureViews = [this, output, capturing, views, yaml, cachePreserved, writeReport, fileDigest, sourceHash] {
        if (*capturing) return;
        *capturing = true;
        if (modelViewer_->state() != ModelViewer::State::Ready || modelViewer_->graphNodeCount() <= 0) {
            writeReport(false, "模型没有生成可显示的结构。"); return;
        }
        const QPointer<QObject> browser = modelViewer_->findChild<QObject *>("netronWebView");
        const QPointer<QProcess> service = modelViewer_->findChild<QProcess *>();
        const qint64 processId = service ? service->processId() : 0;
        const QString path = modelViewer_->modelPath();
        const int items = modelViewer_->hierarchyItemCount(), parameters = modelViewer_->parameterCount();
        const QString text = modelViewer_->yamlText();
        const bool available = modelViewer_->yamlAvailable();
        const auto *yamlModeButton = modelViewer_->findChild<QPushButton *>("modelYamlModeButton");
        const bool yamlModeVisible = yamlModeButton && yamlModeButton->isVisible();
        QString exportError;
        const QString destination = output + "/model-architecture.yaml";
        const bool exported = available && modelViewer_->saveYaml(destination, &exportError);
        QString refusedError;
        const bool existingRefused = exported && !modelViewer_->saveYaml(destination, &refusedError) && !refusedError.isEmpty();
        *yaml = QJsonObject{{"available", available}, {"mode_visible", yamlModeVisible}, {"reason", modelViewer_->yamlReason()},
            {"bytes", text.toUtf8().size()}, {"sha256", QString::fromLatin1(QCryptographicHash::hash(text.toUtf8(), QCryptographicHash::Sha256).toHex())},
            {"exported", exported}, {"export_error", exportError}, {"existing_export_refused", existingRefused},
            {"export_sha256", exported ? fileDigest(destination) : QString()}, {"source_sha256_before", sourceHash}};
        auto *timer = new QTimer(this);
        auto step = std::make_shared<int>(-1);
        QList<ModelViewer::DisplayMode> modes{ModelViewer::DisplayMode::Graph, ModelViewer::DisplayMode::Hierarchy,
            ModelViewer::DisplayMode::Parameters};
        QStringList names{"graph", "hierarchy", "parameters"};
        QStringList screenshots{"model-display.png", "model-hierarchy.png", "model-parameters.png"};
        if (available) {
            modes.append(ModelViewer::DisplayMode::YoloYaml);
            names.append("yolo_yaml");
            screenshots.append("model-yaml.png");
        }
        connect(timer, &QTimer::timeout, this, [this, output, views, yaml, cachePreserved, writeReport, fileDigest,
            sourceHash, timer, step, modes, names, screenshots, browser, service, processId, path, items, parameters, text, available] {
            if (*step >= 0) {
                window_.saveScreenshot(output + '/' + screenshots[*step]);
                views->insert(names[*step], modelViewer_->displayMode() == modes[*step]);
            }
            if (++*step < modes.size()) { modelViewer_->setDisplayMode(modes[*step]); return; }
            timer->stop(); timer->deleteLater();
            modelViewer_->setDisplayMode(ModelViewer::DisplayMode::Graph);
            *cachePreserved = browser && service && processId > 0 &&
                modelViewer_->findChild<QObject *>("netronWebView") == browser &&
                modelViewer_->findChild<QProcess *>() == service && service->processId() == processId &&
                modelViewer_->state() == ModelViewer::State::Ready && modelViewer_->modelPath() == path &&
                modelViewer_->hierarchyItemCount() == items && modelViewer_->parameterCount() == parameters &&
                modelViewer_->yamlAvailable() == available && modelViewer_->yamlText() == text;
            yaml->insert("source_sha256_after", fileDigest(path));
            bool success = *cachePreserved && items > 0 && !sourceHash.isEmpty() && fileDigest(path) == sourceHash &&
                yaml->value("mode_visible").toBool() == available;
            for (const auto &name : names) success = success && views->value(name).toBool();
            success = success && (available ? yaml->value("exported").toBool() && yaml->value("existing_export_refused").toBool() &&
                yaml->value("export_sha256") == yaml->value("sha256") : !modelViewer_->yamlReason().isEmpty());
            writeReport(success, success ? QString() : "模型视图、架构另存或解析缓存自检失败。");
        });
        timer->start(300);
    };
    connect(modelViewer_, &ModelViewer::modelLoaded, this, [this, captureViews](const QString &) {
        QTimer::singleShot(300, this, captureViews); }, Qt::SingleShotConnection);
    connect(modelViewer_, &ModelViewer::loadFailed, this, [writeReport](const QString &error) {
        writeReport(false, error); }, Qt::SingleShotConnection);
    window_.showModelStructure(model);
    if (modelViewer_->state() == ModelViewer::State::Ready) QTimer::singleShot(300, this, captureViews);
    else if (modelViewer_->state() == ModelViewer::State::Error)
        QTimer::singleShot(0, this, [this, writeReport] { writeReport(false, modelViewer_->errorString()); });
}
class InferenceSmoke final : public SmokeContext
{
public:
    using SmokeContext::SmokeContext;
    void run(const QString &directory, bool pt)
    {
        output_ = QFileInfo(directory).absoluteFilePath();
        if (!QDir().mkpath(output_)) { window_.requestExit(2); return; }
        window_.enableDiagnostics();
        connect(window_.detections(), &vision::app::DetectionController::finished, this,
            [this](bool) { complete(); }, Qt::SingleShotConnection);
        if (pt) window_.runPtDemo(); else window_.runDemo();
    }
private:
    void complete()
    {
        const auto result = window_.displayedResult();
        const auto state = window_.detectionState();
        const bool success = !state.failed && !result.image.isNull() && !result.predictions.isEmpty();
        QString error = state.error;
        const bool exported = success && window_.exportDisplayedResult(output_, nullptr, &error);
        QTimer::singleShot(100, this, [this, success, exported, error] {
            window_.saveScreenshot(output_ + "/workbench.png");
            const auto result = window_.displayedResult();
            const bool hasResult = !result.image.isNull();
            QJsonObject j{
                {"success", success && exported}, {"qt", qVersion()},
                {"predictions", result.predictions.size()}, {"inference_ms", result.inferenceMs},
                {"error", error}, {"backend", hasResult ? result.backend : QString()},
                {"requested_device", vision::computeDeviceKey(hasResult ? result.requestedDevice : window_.displayedConfig().device)},
                {"actual_device", hasResult ? QJsonValue(vision::computeDeviceKey(result.device)) : QJsonValue(QJsonValue::Null)},
                {"device_index", hasResult ? result.deviceIndex : -1},
                {"device_name", hasResult ? result.deviceName : QString()},
                {"device_notice", hasResult ? result.deviceNotice : QString()}};
            vision::app::ResultWriter::writeAtomically(output_ + "/smoke-report.json", QJsonDocument(j).toJson());
            qInfo().noquote() << QJsonDocument(j).toJson(QJsonDocument::Compact);
            window_.requestExit(success && exported ? 0 : 2);
        });
    }
    QString output_;
};
}
void runModelDisplaySmoke(MainWindow &window, const QString &directory, const QString &model)
{ (new ModelDisplaySmoke(window))->run(directory, model); }
void runInferenceSmoke(MainWindow &window, const QString &directory, bool pt)
{ (new InferenceSmoke(window))->run(directory, pt); }
}
