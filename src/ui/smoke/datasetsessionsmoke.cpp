// SPDX-License-Identifier: AGPL-3.0-only
#include "smokedrivers.h"
#include "core/datasetsessionstore.h"
#include "ui/evaluationpanel.h"
#include "ui/imagecanvas.h"
#include "ui/mainwindow.h"
#include "ui/pages/workbenchpage.h"
#include <QCoreApplication>
#include <QCryptographicHash>
#include <QDir>
#include <QElapsedTimer>
#include <QFile>
#include <QFileInfo>
#include <QImageReader>
#include <QJsonArray>
#include <QJsonDocument>
#include <QLabel>
#include <QListWidget>
#include <QPixmap>
#include <QPushButton>
#include <QSaveFile>
#include <QTimer>
#include <memory>

namespace vision::ui::smoke
{
namespace
{
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
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) return {};
    return QJsonDocument::fromJson(file.readAll()).object();
}
}

void runDatasetSessionSmoke(MainWindow &window, const QString &directory)
{
    const QString output = QFileInfo(directory).absoluteFilePath();
    if (!QDir().mkpath(output)) {
        qCritical("Cannot create dataset session smoke output directory");
        window.requestExit(2);
        return;
    }
    auto *page = window.findChild<WorkbenchPage *>();
    auto *status = window.findChild<QLabel *>("statusText");
    if (!page || !status) {
        qCritical("Cannot find dataset session smoke workbench controls");
        window.requestExit(2);
        return;
    }
    auto *context = new QObject(&window);
    auto *canvas = page->controls().canvas;
    auto *queue = page->controls().queue;
    auto *evaluation = page->controls().evaluation;
    auto *stopButton = page->controls().stopButton;
    const QString sessionPath = window.datasetSessionFilePath();
    const auto expected = vision::DatasetSessionStore::load(sessionPath);
    const auto initialSession = jsonFile(sessionPath);
    const auto initialHistory = window.historySnapshot();
    const QString initialHistoryHash = sha256(window.dataRoot() + "/history.json");
    const auto evaluationDirectories = [&window] {
        return QDir(window.dataRoot() + "/evaluations").entryList(QDir::Dirs | QDir::NoDotAndDotDot, QDir::Name);
    };
    const auto initialDirectories = evaluationDirectories();
    const auto starts = std::make_shared<int>(0);
    const auto done = std::make_shared<bool>(false);
    const auto elapsed = std::make_shared<QElapsedTimer>();
    elapsed->start();
    QObject::connect(&window, &MainWindow::startRequested, context, [starts](const vision::JobRequest &) { ++*starts; });
    auto *timer = new QTimer(context);
    timer->setInterval(25);
    const auto finish = [&window, canvas, queue, status, output, sessionPath, initialSession, initialHistory, initialHistoryHash,
                         initialDirectories, evaluationDirectories, starts, done, elapsed, timer]
                        (bool success, const QString &error, const QJsonObject &checks) {
        if (*done) return;
        *done = true;
        timer->stop();
        const auto first = canvas->result();
        QJsonArray paths, queueDetails;
        for (int index = 0; index < queue->count(); ++index) {
            const QString path = queue->item(index)->data(Qt::UserRole).toString();
            paths.append(path);
            queueDetails.append(QJsonObject{{"index", index}, {"image", path},
                {"ground_truth_boxes", window.inputState().datasetSamples.value(path).value("gt").toArray().size()}});
        }
        const auto finalDirectories = evaluationDirectories();
        QJsonObject report{{"schema_version", 1}, {"success", success}, {"error", error},
            {"version", QCoreApplication::applicationVersion()}, {"qt", qVersion()},
            {"scope", "Actual startup restoration through restoreDatasetSession; no detection or evaluation is requested."},
            {"session_file", sessionPath}, {"initial_session_json", initialSession},
            {"session_json", jsonFile(sessionPath)}, {"session_sha256", sha256(sessionPath)},
            {"manifest", QJsonObject{{"dataset_source", window.inputState().datasetManifest.value("dataset_source")},
                {"data_yaml", window.inputState().datasetManifest.value("data_yaml")}, {"split", window.inputState().datasetManifest.value("split")},
                {"metadata_path", window.inputState().datasetManifest.value("metadata_path")},
                {"fingerprint", window.inputState().datasetManifest.value("fingerprint")},
                {"sample_count", window.inputState().datasetManifest.value("samples").toArray().size()}}},
            {"queue_count", queue->count()}, {"queue_paths", paths}, {"queue_ground_truth", queueDetails},
            {"first_image", QJsonObject{{"source", first.source}, {"width", first.image.width()},
                {"height", first.image.height()}, {"non_null", !first.image.isNull()},
                {"ground_truth_boxes", first.groundTruth.size()}, {"prediction_count", first.predictions.size()},
                {"inference_ms", first.inferenceMs}}},
            {"start_requested_count", *starts}, {"history_before_count", initialHistory.size()},
            {"history_after_count", window.historySnapshot().size()}, {"history_unchanged", window.historySnapshot() == initialHistory},
            {"history_file_sha256_before", initialHistoryHash},
            {"history_file_sha256_after", sha256(window.dataRoot() + "/history.json")},
            {"evaluation_directories_before", QJsonArray::fromStringList(initialDirectories)},
            {"evaluation_directories_after", QJsonArray::fromStringList(finalDirectories)},
            {"checks", checks}, {"elapsed_ms", double(elapsed->elapsed())},
            {"status", status->toolTip()}};
        if (!success) window.saveScreenshot(output + "/dataset-session-failure.png");
        QSaveFile file(output + "/dataset-session-smoke-report.json");
        file.setDirectWriteFallback(false);
        const auto bytes = QJsonDocument(report).toJson(QJsonDocument::Indented);
        const bool saved = file.open(QIODevice::WriteOnly) && file.write(bytes) == bytes.size() && file.commit();
        if (!saved) qCritical("Cannot save dataset session smoke report");
        if (window.evaluations()->isBusy()) window.evaluations()->requestStop();
        window.requestExit(success && saved ? 0 : 1);
    };
    QObject::connect(timer, &QTimer::timeout, context,
        [&window, canvas, queue, evaluation, status, output, expected, starts, done, initialHistory,
         initialHistoryHash, initialDirectories, evaluationDirectories, finish] {
        if (*done || window.isWorkbenchBusy() || window.evaluations()->isBusy() || window.inputState().queuePopulating) return;
        if (expected.status != vision::DatasetSessionStatus::Loaded || window.inputState().datasetManifest.isEmpty() || window.inputState().files.isEmpty()) {
            finish(false, "Dataset session did not restore: " + status->toolTip(), {});
            return;
        }
        const auto first = canvas->result();
        if (first.image.isNull() || first.source != window.inputState().files.first()) return;
        const auto samples = window.inputState().datasetManifest.value("samples").toArray();
        if (samples.isEmpty()) {
            finish(false, "Restored queue has no dataset samples", {});
            return;
        }
        bool orderMatches = queue->count() == samples.size() && window.inputState().files.size() == samples.size();
        for (int index = 0; orderMatches && index < samples.size(); ++index)
            orderMatches = window.inputState().files[index] == samples[index].toObject().value("image").toString() &&
                queue->item(index)->data(Qt::UserRole).toString() == window.inputState().files[index] && window.inputState().datasetSamples.contains(window.inputState().files[index]);
        const auto currentSession = vision::DatasetSessionStore::load(window.datasetSessionFilePath());
        const QString requestedSource = QDir::cleanPath(QFileInfo(expected.session.sourcePath).absoluteFilePath());
        const QString canonicalSource = QFileInfo(requestedSource).canonicalFilePath();
        const QString expectedMetadata = expected.session.metadataPath.isEmpty() ? QString() :
            QFileInfo(expected.session.metadataPath).canonicalFilePath();
        const auto firstSample = samples.first().toObject();
        QJsonObject checks{
            {"session_loaded", currentSession.status == vision::DatasetSessionStatus::Loaded},
            {"source_entry_preserved", currentSession.session.sourcePath == requestedSource},
            {"manifest_source_matches", window.inputState().datasetManifest.value("dataset_source").toString() == canonicalSource},
            {"split_matches", window.inputState().datasetManifest.value("split").toString() == expected.session.split &&
                currentSession.session.split == expected.session.split},
            {"metadata_matches", window.inputState().datasetManifest.value("metadata_path").toString() == expectedMetadata &&
                currentSession.session.metadataPath == expectedMetadata},
            {"queue_complete_in_manifest_order", orderMatches},
            {"first_source_and_pixels_loaded", first.source == window.inputState().files.first() && !first.image.isNull() &&
                first.image.width() == firstSample.value("width").toInt() && first.image.height() == firstSample.value("height").toInt()},
            {"no_detection_start", *starts == 0},
            {"no_predictions", first.predictions.isEmpty() && first.inferenceMs == 0},
            {"history_unchanged", window.historySnapshot() == initialHistory && sha256(window.dataRoot() + "/history.json") == initialHistoryHash},
            {"no_evaluation_run", evaluationDirectories() == initialDirectories && window.evaluationState().report.isEmpty() && window.evaluationState().directory.isEmpty() && !evaluation->evaluationMode()}};
        const bool screenshotSaved = window.grab().save(output + "/dataset-session-restored.png");
        QImageReader screenshot(output + "/dataset-session-restored.png");
        checks.insert("screenshot_written", screenshotSaved && screenshot.canRead() && screenshot.size().width() > 0 && screenshot.size().height() > 0);
        bool success = true;
        for (auto it = checks.constBegin(); it != checks.constEnd(); ++it) success = success && it.value().toBool();
        finish(success, success ? QString() : QStringLiteral("Restored state failed one or more actual checks"), checks);
    });
    window.restoreDatasetSession();
    timer->start();
    QTimer::singleShot(45000, context, [stopButton, status, done, finish] {
        if (*done) return;
        stopButton->click();
        finish(false, "Dataset session restoration timed out: " + status->toolTip(), {});
    });
}
} // namespace vision::ui::smoke
