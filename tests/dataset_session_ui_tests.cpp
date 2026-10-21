// SPDX-License-Identifier: AGPL-3.0-only
#include "core/datasetsessionstore.h"
#include "core/evaluationworker.h"
#include "ui/evaluationpanel.h"
#include "ui/imagecanvas.h"
#include "ui/mainwindow.h"
#include "ui/modelviewer.h"

#include <QApplication>
#include <QComboBox>
#include <QDir>
#include <QDoubleSpinBox>
#include <QElapsedTimer>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QFont>
#include <QImage>
#include <QJsonArray>
#include <QJsonDocument>
#include <QListWidget>
#include <QMessageBox>
#include <QSettings>
#include <QSignalSpy>
#include <QSpinBox>
#include <QStandardPaths>
#include <QTemporaryDir>
#include <QTimer>
#include <QtTest>
#include <functional>
#include <memory>
#include <stdexcept>

namespace
{
void writeFile(const QString &path, const QByteArray &data)
{
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly) || file.write(data) != data.size())
        throw std::runtime_error(("Cannot write dataset session fixture: " + path).toStdString());
}
QByteArray contents(const QString &path)
{
    QFile file(path);
    return file.open(QIODevice::ReadOnly) ? file.readAll() : QByteArray();
}
struct DatasetFixture
{
    QString root, yaml, metadata;
    QHash<QString, QStringList> images;
};
}

// These tests run the actual dataset preparation Python worker and native queue
// population. No model predictions are injected or advertised as accuracy data.
class DatasetSessionUiTests final : public QObject
{
    Q_OBJECT
    std::unique_ptr<QTemporaryDir> area_;
    std::unique_ptr<MainWindow> window_;
    QHash<QByteArray, QByteArray> previous_;
    QHash<QByteArray, bool> had_;
    QString python_, unexpectedDialog_;
    QTimer dialogGuard_;

    QString sessionPath() const { return area_->path() + "/data/dataset-session.json"; }
    bool waitUntil(const std::function<bool()> &condition, int timeout = 45000)
    {
        QElapsedTimer elapsed;
        elapsed.start();
        while (!condition() && elapsed.elapsed() < timeout && unexpectedDialog_.isEmpty()) {
            QApplication::processEvents();
            QTest::qWait(5);
        }
        QApplication::processEvents();
        return condition() && unexpectedDialog_.isEmpty();
    }
    bool idle() const { return !window_->busy_ && !window_->evaluationWorker_ && !window_->queuePopulating_; }
    QString state() const
    {
        if (!unexpectedDialog_.isEmpty()) return unexpectedDialog_;
        return window_->evaluation_->validationMessage() + " | " + window_->lastError_;
    }
    void showWindow()
    {
        window_ = std::make_unique<MainWindow>();
        window_->resize(1380, 940);
        window_->show();
        window_->modelViewer_->stop();
    }
    DatasetFixture fixture(const QString &name, int largeValCount = 0)
    {
        DatasetFixture result;
        result.root = area_->path() + "/数据集 with spaces/" + name;
        result.yaml = result.root + "/data.yaml";
        result.metadata = result.root + "/人工审核.csv";
        QByteArray metadata = QStringLiteral("图片,split,annotation_status,label_review_required,scene_review_required,Small,Occluded,Blurred,No Ball\n").toUtf8();
        const QStringList splits{"train", "val", "test"};
        for (int splitIndex = 0; splitIndex < splits.size(); ++splitIndex) {
            const QString split = splits.at(splitIndex);
            const int count = largeValCount > 0 && split == "val" ? largeValCount : splitIndex + 2;
            QDir().mkpath(result.root + "/images/" + split);
            QDir().mkpath(result.root + "/labels/" + split);
            QString first;
            for (int index = 0; index < count; ++index) {
                const QString stem = split + "-" + QString::number(index).rightJustified(5, '0');
                const QString path = result.root + "/images/" + split + "/" + stem + ".png";
                if (largeValCount > 0 && index > 0) {
                    if (!QFile::copy(first, path)) throw std::runtime_error("Cannot copy session queue image");
                } else {
                    QImage image(64, 48, QImage::Format_RGB888);
                    image.fill(QColor(35 + splitIndex * 60, 40 + index % 100, 140));
                    if (!image.save(path)) throw std::runtime_error("Cannot save session queue image");
                    first = path;
                }
                result.images[split].append(path);
                writeFile(result.root + "/labels/" + split + "/" + stem + ".txt", "0 0.5 0.5 0.25 0.25\n");
                const QString review = index == count - 1 ? "reviewed_excluded" : "reviewed_annotation";
                metadata += QString("%1.png,%2,%3,0,0,0,0,0,0\n").arg(stem, split, review).toUtf8();
            }
        }
        QString quotedRoot = result.root;
        quotedRoot.replace('\\', '/');
        quotedRoot.replace("'", "''");
        writeFile(result.yaml, ("path: '" + quotedRoot + "'\ntrain: images/train\nval: images/val\ntest: images/test\nnames: [football]\n").toUtf8());
        writeFile(result.metadata, metadata);
        return result;
    }
    bool importDataset(const DatasetFixture &data, const QString &source, const QString &split, bool metadata = false)
    {
        if (!idle()) return false;
        window_->evaluation_->datasetRequested(source, split, metadata ? data.metadata : QString());
        return ready(data, source, split, metadata);
    }
    bool ready(const DatasetFixture &data, const QString &source, const QString &split, bool metadata = false)
    {
        QStringList expected = data.images.value(split);
        if (metadata) expected.removeLast();
        return waitUntil([&, expected] {
            return idle() && window_->files_ == expected && window_->queue_->count() == expected.size() &&
                   window_->evaluation_->datasetYaml() == source && window_->evaluation_->datasetSplit() == split &&
                   window_->evaluation_->metadataPath() == (metadata ? data.metadata : QString()) &&
                   !expected.isEmpty() && window_->canvas_->result().source == expected.first() &&
                   !window_->canvas_->result().image.isNull() && window_->canvas_->result().hasGroundTruth;
        });
    }
    void restart()
    {
        window_->close();
        window_.reset();
        showWindow();
    }

private slots:
    void initTestCase()
    {
        QApplication::setStyle("Fusion");
        QApplication::setFont(QFont("Noto Sans CJK SC", 10));
        QApplication::setAttribute(Qt::AA_DontUseNativeDialogs);
        for (QString candidate : {qEnvironmentVariable("VISION_STUDIO_EVALUATION_PYTHON"), qEnvironmentVariable("VISION_STUDIO_PYTHON"),
                                 QStringLiteral(VISION_PROJECT_DIR) + "/runtime/bin/python", QStringLiteral(VISION_PROJECT_DIR) + "/runtime/python.exe",
                                 QStringLiteral("/opt/VisionStudio/runtime/bin/python")}) {
            if (candidate.isEmpty()) continue;
            if (!QFileInfo(candidate).isAbsolute()) candidate = QStandardPaths::findExecutable(candidate);
            if (QFileInfo(candidate).isFile() && QFileInfo(candidate).isExecutable()) { python_ = QFileInfo(candidate).absoluteFilePath(); break; }
        }
        QVERIFY2(!python_.isEmpty(), "A real evaluation Python runtime is required; missing runtime is not a passed acceptance check.");
        dialogGuard_.setInterval(10);
        connect(&dialogGuard_, &QTimer::timeout, this, [this] {
            for (auto *widget : QApplication::topLevelWidgets())
                if (auto *dialog = qobject_cast<QMessageBox *>(widget); dialog && dialog->isVisible()) {
                    unexpectedDialog_ = dialog->text();
                    dialog->reject();
                }
        });
    }
    void init()
    {
        for (const QByteArray &name : {QByteArray("HOME"), QByteArray("XDG_CONFIG_HOME"), QByteArray("XDG_DATA_HOME"),
             QByteArray("VISION_STUDIO_HOME"), QByteArray("VISION_STUDIO_DATA_DIR"), QByteArray("VISION_STUDIO_GPU_RUNTIME_DIR"),
             QByteArray("VISION_STUDIO_PYTHON"), QByteArray("VISION_STUDIO_PT_WORKER")}) {
            previous_[name] = qgetenv(name.constData());
            had_[name] = qEnvironmentVariableIsSet(name.constData());
        }
        area_ = std::make_unique<QTemporaryDir>();
        QVERIFY(area_->isValid());
        QDir().mkpath(area_->path() + "/home");
        QDir().mkpath(area_->path() + "/data");
        qputenv("HOME", (area_->path() + "/home").toUtf8());
        qputenv("XDG_CONFIG_HOME", (area_->path() + "/home/config").toUtf8());
        qputenv("XDG_DATA_HOME", (area_->path() + "/home/data").toUtf8());
        qputenv("VISION_STUDIO_HOME", QStringLiteral(VISION_PROJECT_DIR).toUtf8());
        qputenv("VISION_STUDIO_DATA_DIR", (area_->path() + "/data").toUtf8());
        qputenv("VISION_STUDIO_GPU_RUNTIME_DIR", (area_->path() + "/absent-gpu").toUtf8());
        qputenv("VISION_STUDIO_PYTHON", python_.toUtf8());
        qputenv("VISION_STUDIO_PT_WORKER", (QStringLiteral(VISION_PROJECT_DIR) + "/scripts/pt_worker.py").toUtf8());
        QSettings preferences(area_->path() + "/data/preferences.ini", QSettings::IniFormat);
        preferences.setValue("models", QStringList{});
        preferences.setValue("activeModel", QString());
        preferences.setValue("computeDevice", "cpu");
        preferences.setValue("autoExport", false);
        preferences.setValue("inputSize", 320);
        preferences.setValue("confidence", .37);
        preferences.setValue("colorMode", "grayscale");
        preferences.setValue("labels", QStringList{"football"});
        preferences.sync();
        unexpectedDialog_.clear();
        dialogGuard_.start();
        showWindow();
    }
    void cleanup()
    {
        if (window_ && window_->busy_) {
            window_->stopInference();
            waitUntil([this] { return idle(); });
        }
        window_.reset();
        dialogGuard_.stop();
        area_.reset();
        for (auto it = previous_.cbegin(); it != previous_.cend(); ++it)
            if (had_.value(it.key())) qputenv(it.key().constData(), it.value());
            else qunsetenv(it.key().constData());
        previous_.clear(); had_.clear();
    }

    void successfulImportRestoresOriginalSourceSplitAndMetadata_data()
    {
        QTest::addColumn<bool>("yamlSource");
        QTest::addColumn<QString>("split");
        for (bool yaml : {false, true})
            for (const auto &split : {"train", "val", "test"})
                QTest::newRow(qPrintable(QString("%1-%2").arg(yaml ? "yaml" : "directory", split))) << yaml << QString::fromLatin1(split);
    }
    void successfulImportRestoresOriginalSourceSplitAndMetadata()
    {
        QFETCH(bool, yamlSource);
        QFETCH(QString, split);
        const auto data = fixture("round-trip");
        const QString source = yamlSource ? data.yaml : data.root;
        QSignalSpy firstStarts(window_.get(), &MainWindow::startRequested);
        QVERIFY2(importDataset(data, source, split, true), qPrintable(state()));
        const auto saved = vision::DatasetSessionStore::load(sessionPath());
        QCOMPARE(saved.status, vision::DatasetSessionStatus::Loaded);
        QCOMPARE(saved.session.sourcePath, source);
        QCOMPARE(saved.session.split, split);
        QCOMPARE(saved.session.metadataPath, data.metadata);
        QCOMPARE(firstStarts.count(), 0);
        const auto previousHistory = window_->history_;
        restart();
        QVERIFY(window_->datasetManifest_.isEmpty()); // Construction alone is safe for explicit CLI tasks.
        QSignalSpy restoredStarts(window_.get(), &MainWindow::startRequested);
        QTimer heartbeat;
        QElapsedTimer clock;
        clock.start();
        qint64 lastTick = 0, maxGap = 0;
        int ticks = 0;
        connect(&heartbeat, &QTimer::timeout, this, [&] {
            const qint64 now = clock.elapsed();
            maxGap = qMax(maxGap, now - lastTick); lastTick = now; ++ticks;
        });
        heartbeat.start(10);
        window_->restoreDatasetSession();
        QVERIFY2(ready(data, source, split, true), qPrintable(state()));
        heartbeat.stop();
        QCOMPARE(restoredStarts.count(), 0);
        QVERIFY(ticks > 0);
        QVERIFY2(maxGap < 1000, qPrintable(QString("Dataset restore starved the UI event loop for %1 ms").arg(maxGap)));
        QCOMPARE(window_->history_, previousHistory);
        QVERIFY(window_->evaluationReport_.isEmpty());
        QVERIFY(!QFileInfo::exists(area_->path() + "/data/results"));
        QCOMPARE(window_->inputSize_->value(), 320);
        QCOMPARE(window_->confidence_->value(), .37);
        QCOMPARE(window_->inputColorMode_->currentData().toString(), QString("grayscale"));
        QVERIFY(window_->modelPath_.isEmpty());
        QCOMPARE(window_->canvas_->result().groundTruth.size(), 1);
        qInfo().noquote() << "dataset_session_restore_evidence" << (yamlSource ? "yaml" : "directory") << split
                         << "images" << window_->files_.size() << "metadata_excluded" << 1
                         << "inference_requests" << restoredStarts.count() << "heartbeat_ticks" << ticks << "max_gap_ms" << maxGap;
    }

    void failedImportKeepsLastSuccessfulRecord()
    {
        const auto first = fixture("valid");
        QVERIFY2(importDataset(first, first.root, "test"), qPrintable(state()));
        const QByteArray original = contents(sessionPath());
        const auto paths = window_->files_;
        window_->evaluation_->datasetRequested(area_->path() + "/absent/data.yaml", "train", {});
        QVERIFY2(waitUntil([this] { return idle(); }), qPrintable(state()));
        QCOMPARE(contents(sessionPath()), original);
        QCOMPARE(window_->files_, paths);
        QCOMPARE(window_->evaluation_->datasetYaml(), first.root);
        restart();
        window_->restoreDatasetSession();
        QVERIFY2(ready(first, first.root, "test"), qPrintable(state()));
    }

    void cancelledQueuePopulationKeepsLastSuccessfulRecord()
    {
        const auto first = fixture("valid");
        QVERIFY2(importDataset(first, first.yaml, "val"), qPrintable(state()));
        const QByteArray original = contents(sessionPath());
        const auto pending = fixture("cancelled", 1500);
        QTimer stopper;
        int partialCount = 0;
        qint64 stopDuration = -1;
        connect(&stopper, &QTimer::timeout, this, [&] {
            if (!window_->queuePopulating_ || window_->queue_->count() == 0 || window_->queue_->count() >= 1500) return;
            partialCount = window_->queue_->count();
            stopper.stop();
            QElapsedTimer elapsed;
            elapsed.start();
            window_->stopInference();
            stopDuration = elapsed.elapsed();
        });
        stopper.start(0);
        window_->evaluation_->datasetRequested(pending.root, "val", {});
        QVERIFY2(waitUntil([&] { return stopDuration >= 0 && idle(); }), qPrintable(state()));
        stopper.stop();
        QVERIFY(partialCount > 0 && partialCount < 1500);
        QVERIFY(stopDuration < 500);
        QCOMPARE(contents(sessionPath()), original);
        QTest::qWait(50);
        QVERIFY(window_->files_.isEmpty());
        QCOMPARE(contents(sessionPath()), original);
        restart();
        window_->restoreDatasetSession();
        QVERIFY2(ready(first, first.yaml, "val"), qPrintable(state()));
        qInfo() << "dataset_session_cancel_evidence" << "partial_count" << partialCount << "stop_return_ms" << stopDuration;
    }

    void missingCorruptAndUnavailableSessionsRecoverSafely_data()
    {
        QTest::addColumn<QString>("kind");
        QTest::newRow("missing-json") << QString("missing");
        QTest::newRow("corrupt-json") << QString("corrupt");
        QTest::newRow("unknown-schema") << QString("schema");
        QTest::newRow("missing-source") << QString("source");
    }
    void missingCorruptAndUnavailableSessionsRecoverSafely()
    {
        QFETCH(QString, kind);
        if (kind == "corrupt") writeFile(sessionPath(), "{\"schema_version\":1,");
        else if (kind == "schema") writeFile(sessionPath(), "{\"schema_version\":99,\"source_path\":\"/absent\",\"split\":\"val\"}");
        else if (kind == "source") {
            QString error;
            QVERIFY2(vision::DatasetSessionStore::save(sessionPath(), {area_->path() + "/unmounted/data.yaml", "val", {}}, &error), qPrintable(error));
        }
        const QByteArray original = contents(sessionPath());
        QSignalSpy starts(window_.get(), &MainWindow::startRequested);
        window_->restoreDatasetSession();
        QTest::qWait(50); // Process the deferred entry before observing its final idle state.
        QVERIFY2(waitUntil([this] { return idle(); }), qPrintable(state()));
        QVERIFY(window_->datasetManifest_.isEmpty());
        QVERIFY(window_->files_.isEmpty());
        QVERIFY(window_->canvas_->result().image.isNull());
        QCOMPARE(starts.count(), 0);
        QCOMPARE(contents(sessionPath()), original);
        QVERIFY2(unexpectedDialog_.isEmpty(), qPrintable(unexpectedDialog_));
        const auto manual = fixture("manual-recovery");
        QVERIFY2(importDataset(manual, manual.root, "train"), qPrintable(state()));
        const auto recovered = vision::DatasetSessionStore::load(sessionPath());
        QCOMPARE(recovered.status, vision::DatasetSessionStatus::Loaded);
        QCOMPARE(recovered.session.sourcePath, manual.root);
        QCOMPARE(recovered.session.split, QString("train"));
    }

    void explicitManualSelectionWinsBeforeDeferredRestore_data()
    {
        QTest::addColumn<bool>("restoreFirst");
        QTest::newRow("manual-then-restore") << false;
        QTest::newRow("restore-then-manual-without-event-loop") << true;
    }
    void explicitManualSelectionWinsBeforeDeferredRestore()
    {
        QFETCH(bool, restoreFirst);
        const auto first = fixture("saved");
        const auto manual = fixture("manual");
        QVERIFY2(importDataset(first, first.root, "val"), qPrintable(state()));
        restart();
        QSignalSpy starts(window_.get(), &MainWindow::startRequested);
        if (restoreFirst) window_->restoreDatasetSession();
        window_->evaluation_->datasetRequested(manual.yaml, "test", manual.metadata);
        if (!restoreFirst) window_->restoreDatasetSession();
        QVERIFY2(ready(manual, manual.yaml, "test", true), qPrintable(state()));
        QTest::qWait(60);
        QCOMPARE(window_->evaluation_->datasetYaml(), manual.yaml);
        const auto saved = vision::DatasetSessionStore::load(sessionPath());
        QCOMPARE(saved.session.sourcePath, manual.yaml);
        QCOMPARE(saved.session.split, QString("test"));
        QCOMPARE(starts.count(), 0);
        // A repeated call must not replace the already selected active context.
        const quint64 acceptedImport = window_->datasetImportGeneration_;
        window_->restoreDatasetSession();
        QTest::qWait(60);
        QVERIFY(idle());
        QCOMPARE(window_->datasetImportGeneration_, acceptedImport);
        QCOMPARE(window_->evaluation_->datasetYaml(), manual.yaml);
        QCOMPARE(window_->files_.size(), manual.images.value("test").size() - 1);
    }

    void openingAndCancellingPickerSuppressesPendingRestore()
    {
        const auto first = fixture("saved");
        QVERIFY2(importDataset(first, first.root, "val"), qPrintable(state()));
        const QByteArray original = contents(sessionPath());
        restart();
        QSignalSpy starts(window_.get(), &MainWindow::startRequested);
        bool sawPicker = false, workerStartedInsidePicker = false;
        QTimer closePicker;
        connect(&closePicker, &QTimer::timeout, this, [&] {
            for (auto *widget : QApplication::topLevelWidgets())
                if (auto *dialog = qobject_cast<QFileDialog *>(widget); dialog && dialog->isVisible()) {
                    sawPicker = dialog->objectName() == "yoloDatasetSourceDialog";
                    workerStartedInsidePicker = window_->busy_ || window_->evaluationWorker_;
                    dialog->reject();
                    closePicker.stop();
                }
        });
        closePicker.start(20);
        window_->restoreDatasetSession();
        window_->evaluation_->chooseDataset(); // exec() runs a nested event loop containing the pending restore timer.
        closePicker.stop();
        QVERIFY(sawPicker);
        QVERIFY(!workerStartedInsidePicker);
        QTest::qWait(60);
        QVERIFY(idle());
        QVERIFY(window_->datasetManifest_.isEmpty());
        QVERIFY(window_->files_.isEmpty());
        QCOMPARE(contents(sessionPath()), original);
        QCOMPARE(starts.count(), 0);
    }
};

QTEST_MAIN(DatasetSessionUiTests)
#include "dataset_session_ui_tests.moc"
