// SPDX-License-Identifier: AGPL-3.0-only
#include "ui/mainwindow.h"
#include "ui/evaluationpanel.h"
#include "ui/imagecanvas.h"
#include "ui/modelviewer.h"
#include "core/evaluationworker.h"

#include <QApplication>
#include <QCheckBox>
#include <QComboBox>
#include <QDateTime>
#include <QDir>
#include <QDoubleSpinBox>
#include <QDragEnterEvent>
#include <QDropEvent>
#include <QElapsedTimer>
#include <QFile>
#include <QFileDialog>
#include <QJsonArray>
#include <QJsonDocument>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QMenu>
#include <QMessageBox>
#include <QMimeData>
#include <QProgressBar>
#include <QPushButton>
#include <QSettings>
#include <QStandardPaths>
#include <QSignalSpy>
#include <QSpinBox>
#include <QTabWidget>
#include <QTableWidget>
#include <QTemporaryDir>
#include <QTimer>
#include <QToolButton>
#include <QtTest>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <functional>
#include <memory>
#include <stdexcept>

namespace
{
QByteArray varint(quint64 value)
{
    QByteArray result;
    do { quint8 byte = value & 127; value >>= 7; result.append(char(byte | (value ? 128 : 0))); } while (value);
    return result;
}
QByteArray integer(int field, quint64 value) { return varint(quint64(field) << 3) + varint(value); }
QByteArray bytes(int field, const QByteArray &value) { return varint((quint64(field) << 3) | 2) + varint(value.size()) + value; }
QByteArray text(int field, const char *value) { return bytes(field, QByteArray(value)); }
QByteArray information(const char *name, const QVector<qint64> &dimensions)
{
    QByteArray shape; for (auto dimension : dimensions) shape += bytes(1, integer(1, dimension));
    return text(1, name) + bytes(2, bytes(1, integer(1, 1) + bytes(2, shape)));
}
template<class T> QByteArray tensor(const char *name, const QVector<qint64> &dimensions, int type, const QVector<T> &values)
{
    QByteArray data, result;
    for (const T &value : values) {
        quint8 encoded[sizeof(T)]; std::memcpy(encoded, &value, sizeof(T));
#if Q_BYTE_ORDER == Q_BIG_ENDIAN
        std::reverse(encoded, encoded + sizeof(T));
#endif
        data.append(reinterpret_cast<const char *>(encoded), sizeof(T));
    }
    for (auto dimension : dimensions) result += integer(1, dimension);
    return result + integer(2, type) + text(8, name) + bytes(9, data);
}
QByteArray node(const char *op, const QVector<const char *> &inputs, const char *output, const QByteArray &attributes = {})
{
    QByteArray result; for (const char *input : inputs) result += text(1, input);
    return result + text(2, output) + text(4, op) + attributes;
}
void writeFile(const QString &path, const QByteArray &value)
{
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly) || file.write(value) != value.size())
        throw std::runtime_error(("Cannot write integration fixture: " + path).toStdString());
}
QString writeOnnx(const QString &root, int size = 64, const QString &name = "known.onnx", int classes = 1)
{
    // A genuine Conv -> GlobalAveragePool -> Reshape network executed by
    // OpenCV, not injected inference results. Its fixed score/geometry makes
    // the end-to-end matching and threshold assertions independently known.
    const int outputs = (4 + classes) * 8;
    QVector<float> values(outputs, 0);
    values[0] = values[8] = size * .5f; values[16] = values[24] = size * .3125f; values[32] = .8f;
    QByteArray graph;
    const auto kernel = bytes(5, text(1, "kernel_shape") + integer(8, 1) + integer(8, 1) + integer(20, 7));
    graph += bytes(1, node("Conv", {"images", "weights", "bias"}, "conv", kernel));
    graph += bytes(1, node("GlobalAveragePool", {"conv"}, "average"));
    graph += bytes(1, node("Reshape", {"average", "shape"}, "output"));
    graph += text(2, "vision_studio_native_integration_fixture");
    graph += bytes(5, tensor("weights", {outputs, 3, 1, 1}, 1, QVector<float>(outputs * 3, 0)));
    graph += bytes(5, tensor("bias", {outputs}, 1, values));
    graph += bytes(5, tensor("shape", {3}, 7, QVector<qint64>{1, 4 + classes, 8}));
    graph += bytes(11, information("images", {1, 3, size, size})); graph += bytes(12, information("output", {1, 4 + classes, 8}));
    const QString path = root + "/" + name;
    writeFile(path, integer(1, 7) + text(2, "Vision Studio native integration tests") + bytes(7, graph) + bytes(8, integer(2, 11)));
    return path;
}
QJsonObject readJson(const QString &path)
{
    QFile file(path); if (!file.open(QIODevice::ReadOnly)) return {};
    return QJsonDocument::fromJson(file.readAll()).object();
}
}

class EvaluationIntegrationTests final : public QObject
{
    Q_OBJECT
    std::unique_ptr<QTemporaryDir> area_;
    std::unique_ptr<MainWindow> window_;
    QHash<QByteArray, QByteArray> previous_;
    QHash<QByteArray, bool> had_;
    QString model_, yaml_, unexpectedDialog_, python_;
    QStringList images_;
    QTimer dialogGuard_;

    bool waitUntil(const std::function<bool()> &condition, int timeout = 60000)
    {
        QElapsedTimer timer; timer.start();
        while (!condition() && timer.elapsed() < timeout && unexpectedDialog_.isEmpty()) {
            QApplication::processEvents(); QTest::qWait(5);
        }
        QApplication::processEvents();
        return condition() && unexpectedDialog_.isEmpty();
    }
    QString state() const
    {
        return unexpectedDialog_.isEmpty() ? window_->statusLabel_->text() + " | " + window_->lastError_ : unexpectedDialog_;
    }
    QString dataset(int count = 4)
    {
        const QString directory = area_->path() + "/dataset-" + QString::number(count);
        QDir().mkpath(directory + "/images/val"); QDir().mkpath(directory + "/labels/val");
        images_.clear();
        for (int index = 0; index < count; ++index) {
            const QString stem = QString::number(index).rightJustified(4, '0');
            const QString path = directory + "/images/val/" + stem + ".png";
            QImage image(64, 64, QImage::Format_RGB888); image.fill(QColor((index * 37) % 255, (index * 23 + 50) % 255, (index * 11 + 90) % 255));
            if (!image.save(path)) throw std::runtime_error("Cannot save integration dataset image");
            images_.append(path);
            const QByteArray label = index % 4 == 2 ? QByteArray() : index % 4 == 1 ? QByteArray("0 0.125 0.125 0.125 0.125\n") : QByteArray("0 0.5 0.5 0.3125 0.3125\n");
            writeFile(directory + "/labels/val/" + stem + ".txt", label);
        }
        yaml_ = directory + "/data.yaml";
        writeFile(yaml_, ("path: " + directory + "\nval: images/val\nnames: [football]\n").toUtf8());
        return yaml_;
    }
    bool selectDatasetPath(bool directYaml = false, bool keyboardShortcut = false)
    {
        QTimer chooser; QElapsedTimer pickerTimeout; pickerTimeout.start();
        bool pickerSeen = false, correctMode = false;
        const QString selected = directYaml ? yaml_ : QFileInfo(yaml_).absolutePath();
        connect(&chooser, &QTimer::timeout, this, [this, &pickerTimeout, &pickerSeen, &correctMode, selected, directYaml] {
            for (auto *widget : QApplication::topLevelWidgets())
                if (auto *dialog = qobject_cast<QFileDialog *>(widget); dialog && dialog->isVisible()) {
                    correctMode = dialog->fileMode() == (directYaml ? QFileDialog::ExistingFile : QFileDialog::Directory);
                    if (pickerTimeout.elapsed() > 15000) { unexpectedDialog_ = "YOLO dataset chooser did not accept the existing selection"; dialog->reject(); return; }
                    if (!pickerSeen) {
                        pickerSeen = true; dialog->setDirectory(QFileInfo(selected).absolutePath());
                    }
                    // QFileSystemModel may finish loading after setDirectory and
                    // replace an early selection. Keep the filename editor in
                    // sync without restarting the directory model every tick.
                    if (auto *filename = dialog->findChild<QLineEdit *>("fileNameEdit")) filename->setText(selected);
                    if (!dialog->selectedFiles().isEmpty() && QFileInfo(dialog->selectedFiles().first()).absoluteFilePath() == selected)
                        QMetaObject::invokeMethod(dialog, "accept", Qt::QueuedConnection);
                }
        });
        chooser.start(10);
        if (directYaml) {
            auto *options = window_->findChild<QToolButton *>("openYoloDatasetOptionsButton");
            if (!options || !options->menu()) return false;
            QAction *yamlAction = nullptr;
            for (auto *action : options->menu()->actions()) if (action->text().contains("YAML")) yamlAction = action;
            if (!yamlAction) return false;
            yamlAction->trigger();
        } else if (keyboardShortcut) {
            window_->activateWindow(); window_->canvas_->setFocus();
            QTest::keyClick(window_->canvas_, Qt::Key_O, Qt::ControlModifier);
        } else {
            auto *button = window_->findChild<QPushButton *>("openYoloDatasetButton");
            if (!button) return false;
            button->click();
        }
        chooser.stop();
        return pickerSeen && correctMode && unexpectedDialog_.isEmpty();
    }
    bool loadDataset(int count = 4, bool publicFileDialog = false)
    {
        dataset(count);
        if (publicFileDialog) {
            if (!selectDatasetPath()) return false;
        } else window_->loadDataset(yaml_, "val");
        if (!waitUntil([this] { return !window_->busy_ && !window_->evaluationWorker_; })) return false;
        return window_->files_ == images_ && window_->datasetManifest_.value("samples").toArray().size() == count;
    }
    void configureEvaluation()
    {
        window_->evaluation_->controlsWidget()->findChild<QDoubleSpinBox *>("evaluationScanStartSpin")->setValue(.1);
        window_->evaluation_->controlsWidget()->findChild<QDoubleSpinBox *>("evaluationScanEndSpin")->setValue(.9);
        window_->evaluation_->controlsWidget()->findChild<QDoubleSpinBox *>("evaluationScanStepSpin")->setValue(.2);
        window_->evaluation_->setMode(true);
    }
    bool runEvaluation(int range, int selection)
    {
        configureEvaluation(); window_->queue_->setCurrentRow(selection); window_->detectionRange_->setCurrentIndex(range);
        if (!window_->runButton_->isEnabled()) return false;
        window_->runButton_->click();
        if (!window_->busy_ || !window_->evaluationWorker_) return false;
        return waitUntil([this] { return !window_->busy_ && !window_->evaluationWorker_; }) && !window_->evaluationReport_.isEmpty();
    }
    void evidence(const QString &scenario)
    {
        const auto report = window_->evaluationReport_;
        qInfo().noquote() << "evaluation_integration_evidence:" << QJsonDocument(QJsonObject{{"scenario", scenario},
            {"status", report.value("status")}, {"full_split", report.value("full_split")},
            {"completed_images", report.value("completed_images")}, {"metrics", report.value("metrics")},
            {"backend", window_->lastResult_.backend}, {"report_directory", window_->evaluationDirectory_}}).toJson(QJsonDocument::Compact);
        const QString destination = qEnvironmentVariable("VISION_EVALUATION_INTEGRATION_EVIDENCE");
        if (destination.isEmpty() || report.isEmpty()) return;
        const QString prefix = destination + "/" + scenario;
        QDir().mkpath(prefix);
        for (const auto &name : {"report.json", "status.json", "REPORT.md", "thresholds.csv", "per-image.csv", "errors.csv"}) {
            const QString source = window_->evaluationDirectory_ + "/" + name;
            if (QFileInfo::exists(source)) QFile::copy(source, prefix + "/" + name);
        }
        window_->saveScreenshot(prefix + "/native-window.png");
        if (scenario == "threshold-filter-adopt") {
            const QSize before = window_->size();
            window_->resize(1260, 820); QTest::qWait(40);
            window_->saveScreenshot(prefix + "/minimum-window.png");
            window_->resize(before);
        }
    }
private slots:
    void initTestCase()
    {
        QApplication::setStyle("Fusion"); QApplication::setFont(QFont("Noto Sans CJK SC", 10));
        QApplication::setAttribute(Qt::AA_DontUseNativeDialogs);
        QCoreApplication::setApplicationVersion("2.4.0");
        for (QString candidate : {qEnvironmentVariable("VISION_STUDIO_EVALUATION_PYTHON"), qEnvironmentVariable("VISION_STUDIO_PYTHON"),
            QStringLiteral(VISION_PROJECT_DIR) + "/runtime/bin/python", QStringLiteral(VISION_PROJECT_DIR) + "/runtime/python.exe",
            QStringLiteral("/opt/VisionStudio/runtime/bin/python")}) {
            if (candidate.isEmpty()) continue;
            if (!QFileInfo(candidate).isAbsolute()) candidate = QStandardPaths::findExecutable(candidate);
            if (QFileInfo(candidate).isFile() && QFileInfo(candidate).isExecutable()) { python_ = QFileInfo(candidate).absoluteFilePath(); break; }
        }
        QVERIFY2(!python_.isEmpty(), "Evaluation Python runtime is required; set VISION_STUDIO_EVALUATION_PYTHON or VISION_STUDIO_PYTHON. Acceptance must not silently skip.");
        connect(&dialogGuard_, &QTimer::timeout, this, [this] {
            for (QWidget *widget : QApplication::topLevelWidgets())
                if (auto *dialog = qobject_cast<QMessageBox *>(widget); dialog && dialog->isVisible()) {
                    unexpectedDialog_ = dialog->text(); dialog->accept();
                }
        }); dialogGuard_.setInterval(10);
    }
    void init()
    {
        for (const QByteArray &name : {QByteArray("HOME"), QByteArray("XDG_CONFIG_HOME"), QByteArray("XDG_DATA_HOME"),
            QByteArray("VISION_STUDIO_HOME"), QByteArray("VISION_STUDIO_DATA_DIR"), QByteArray("VISION_STUDIO_GPU_RUNTIME_DIR"),
            QByteArray("VISION_STUDIO_PYTHON"), QByteArray("VISION_STUDIO_PT_WORKER")}) {
            previous_[name] = qgetenv(name.constData()); had_[name] = qEnvironmentVariableIsSet(name.constData());
        }
        area_ = std::make_unique<QTemporaryDir>(); QVERIFY(area_->isValid());
        QDir().mkpath(area_->path() + "/home"); QDir().mkpath(area_->path() + "/data");
        qputenv("HOME", (area_->path() + "/home").toUtf8()); qputenv("XDG_CONFIG_HOME", (area_->path() + "/home/config").toUtf8());
        qputenv("XDG_DATA_HOME", (area_->path() + "/home/data").toUtf8());
        qputenv("VISION_STUDIO_HOME", QStringLiteral(VISION_PROJECT_DIR).toUtf8());
        qputenv("VISION_STUDIO_DATA_DIR", (area_->path() + "/data").toUtf8());
        qputenv("VISION_STUDIO_GPU_RUNTIME_DIR", (area_->path() + "/absent-gpu").toUtf8());
        qputenv("VISION_STUDIO_PYTHON", python_.toUtf8());
        qputenv("VISION_STUDIO_PT_WORKER", (QStringLiteral(VISION_PROJECT_DIR) + "/scripts/pt_worker.py").toUtf8());
        model_ = writeOnnx(area_->path());
        QSettings settings(area_->path() + "/data/preferences.ini", QSettings::IniFormat);
        settings.setValue("models", QStringList{model_}); settings.setValue("activeModel", model_);
        settings.setValue("computeDevice", "cpu"); settings.setValue("autoExport", false);
        settings.setValue("task", int(vision::ModelTask::YoloV8)); settings.setValue("inputSize", 64);
        settings.setValue("labels", QStringList{"football"}); settings.setValue("confidence", .3); settings.sync();
        unexpectedDialog_.clear(); dialogGuard_.start();
        window_ = std::make_unique<MainWindow>(); window_->resize(1380, 940); window_->show(); window_->modelViewer_->stop();
        QVERIFY2(waitUntil([this] { return !window_->modelInspectionPending_ && window_->modelClassCount_ == 1; }, 30000), qPrintable(state()));
        QCOMPARE(window_->modelPath_, model_); QCOMPARE(window_->inputSize_->value(), 64);
    }
    void cleanup()
    {
        if (window_ && window_->busy_) { window_->stopInference(); waitUntil([this] { return !window_->busy_; }); }
        window_.reset(); dialogGuard_.stop(); area_.reset();
        for (auto it = previous_.cbegin(); it != previous_.cend(); ++it)
            if (had_.value(it.key())) qputenv(it.key().constData(), it.value()); else qunsetenv(it.key().constData());
        previous_.clear(); had_.clear();
    }
    void datasetInputSharesQueueGroundTruthAndPreviewMode()
    {
        QVERIFY2(loadDataset(4, true), qPrintable(state()));
        QCOMPARE(window_->queue_->count(), 4); QVERIFY(window_->showGroundTruth_->isVisible());
        QVERIFY(!window_->evaluation_->evaluationMode());
        QCOMPARE(window_->canvas_->result().source, images_.first());
        QVERIFY(window_->canvas_->result().hasGroundTruth); QCOMPARE(window_->canvas_->result().groundTruth.size(), 1);
        QCOMPARE(window_->canvas_->result().predictions.size(), 0); QVERIFY(!window_->exportButton_->isEnabled());
        window_->queue_->setCurrentRow(2); QVERIFY(window_->canvas_->result().hasGroundTruth);
        QCOMPARE(window_->canvas_->result().groundTruth.size(), 0);
        configureEvaluation(); QVERIFY(window_->evaluation_->evaluationMode());
        QCOMPARE(window_->workbenchTabs_->currentWidget(), static_cast<QWidget *>(window_->evaluation_));
        window_->evaluation_->setMode(false); QCOMPARE(window_->workbenchTabs_->currentIndex(), 0);
        QCOMPARE(window_->files_, images_);
    }
    void shortcutOpensDatasetDirectoryInsteadOfLooseImagePicker()
    {
        dataset();
        QVERIFY2(selectDatasetPath(false, true), qPrintable(state()));
        QVERIFY2(waitUntil([this] { return !window_->busy_ && !window_->evaluationWorker_; }), qPrintable(state()));
        QCOMPARE(window_->files_, images_);
        QCOMPARE(window_->datasetManifest_.value("samples").toArray().size(), 4);
        QVERIFY(!window_->datasetManifest_.isEmpty());
    }
    void canvasAcceptsDatasetsAndRejectsLooseImagesWithoutChangingQueue()
    {
        const auto drop = [this](const QString &path) {
            QMimeData mime; mime.setUrls({QUrl::fromLocalFile(path)});
            const QPoint point = window_->canvas_->rect().center();
            QDragEnterEvent enter(point, Qt::CopyAction, &mime, Qt::LeftButton, Qt::NoModifier);
            QApplication::sendEvent(window_->canvas_, &enter);
            if (!enter.isAccepted()) return false;
            QDropEvent event(point, Qt::CopyAction, &mime, Qt::LeftButton, Qt::NoModifier);
            QApplication::sendEvent(window_->canvas_, &event);
            return event.isAccepted();
        };
        dataset();
        QVERIFY(drop(QFileInfo(yaml_).absolutePath()));
        QVERIFY2(waitUntil([this] { return !window_->busy_ && !window_->evaluationWorker_; }), qPrintable(state()));
        QCOMPARE(window_->files_, images_);
        const QJsonObject initialManifest = window_->datasetManifest_;
        const auto initialSource = window_->lastResult_.source;
        QVERIFY(!drop(images_.last()));
        QCOMPARE(window_->files_, images_); QCOMPARE(window_->datasetManifest_, initialManifest);
        QCOMPARE(window_->lastResult_.source, initialSource);
        QCOMPARE(window_->queue_->currentRow(), 0);
        dataset(3);
        QVERIFY(drop(yaml_));
        QVERIFY2(waitUntil([this] { return !window_->busy_ && !window_->evaluationWorker_; }), qPrintable(state()));
        QCOMPARE(window_->files_, images_); QCOMPARE(window_->queue_->count(), 3);
        QCOMPARE(window_->datasetManifest_.value("data_yaml").toString(), yaml_);
    }

    void directoryImportFindsConventionalDatasetBesideTrainingConfiguration()
    {
        dataset();
        const QString directory = QFileInfo(yaml_).absolutePath();
        const QString conventional = directory + "/dataset.yaml";
        QVERIFY(QFile::rename(yaml_, conventional)); yaml_ = conventional;
        writeFile(directory + "/train_from_scratch.example.yaml", "epochs: 100\nmodel: yolo.pt\n");
        QVERIFY2(selectDatasetPath(), qPrintable(state()));
        QVERIFY2(waitUntil([this] { return !window_->busy_ && !window_->evaluationWorker_; }), qPrintable(state()));
        QCOMPARE(window_->datasetManifest_.value("data_yaml").toString(), conventional);
        QCOMPARE(window_->evaluation_->datasetYaml(), directory);
        QCOMPARE(window_->files_, images_); QCOMPARE(window_->queue_->count(), 4);
        QVERIFY(window_->canvas_->result().hasGroundTruth); QVERIFY(window_->evaluation_->canEvaluate());
    }
    void explicitYamlOptionDisambiguatesDirectoryWithoutGuessing()
    {
        dataset();
        const auto previousFiles = window_->files_;
        const QString directory = QFileInfo(yaml_).absolutePath();
        QVERIFY(QFile::copy(yaml_, directory + "/dataset.yaml"));
        QVERIFY2(selectDatasetPath(), qPrintable(state()));
        QVERIFY2(waitUntil([this] { return !window_->busy_ && !window_->evaluationWorker_; }), qPrintable(state()));
        QVERIFY(window_->datasetManifest_.isEmpty()); QCOMPARE(window_->files_, previousFiles);
        QCOMPARE(window_->evaluation_->datasetYaml(), directory); QVERIFY(!window_->evaluation_->canEvaluate());
        QVERIFY(window_->evaluation_->controlsWidget()->isVisible());
        const auto *hint = window_->evaluation_->controlsWidget()->findChild<QLabel *>("evaluationDatasetHint"); QVERIFY(hint);
        QVERIFY(hint->toolTip().contains("YAML", Qt::CaseInsensitive));
        QVERIFY2(selectDatasetPath(true), qPrintable(state()));
        QVERIFY2(waitUntil([this] { return !window_->busy_ && !window_->evaluationWorker_; }), qPrintable(state()));
        QCOMPARE(window_->datasetManifest_.value("data_yaml").toString(), yaml_);
        QCOMPARE(window_->files_, images_); QVERIFY(window_->evaluation_->canEvaluate());
    }
    void firstImportMissingValKeepsDirectoryForExplicitTrainRetry()
    {
        dataset();
        const auto previousFiles = window_->files_;
        const QString directory = QFileInfo(yaml_).absolutePath();
        writeFile(yaml_, ("path: " + directory + "\ntrain: images/val\nnames: [football]\n").toUtf8());
        QVERIFY2(selectDatasetPath(), qPrintable(state()));
        QVERIFY2(waitUntil([this] { return !window_->busy_ && !window_->evaluationWorker_; }), qPrintable(state()));
        QVERIFY(window_->datasetManifest_.isEmpty()); QCOMPARE(window_->files_, previousFiles);
        QCOMPARE(window_->evaluation_->datasetYaml(), directory); QCOMPARE(window_->evaluation_->datasetSplit(), QString("val"));
        QVERIFY(window_->evaluation_->controlsWidget()->isVisible());
        auto *split = window_->evaluation_->controlsWidget()->findChild<QComboBox *>("evaluationSplitCombo"); QVERIFY(split);
        QVERIFY(split->isVisible()); QVERIFY(split->isEnabled());
        auto *retry = window_->evaluation_->controlsWidget()->findChild<QPushButton *>("evaluationReloadDatasetButton");
        QVERIFY(retry); QVERIFY(retry->isVisible()); QVERIFY(retry->isEnabled());
        QVERIFY(!window_->evaluation_->canEvaluate()); QVERIFY(!window_->evaluation_->evaluationMode());
        split->setCurrentIndex(split->findData("train"));
        QVERIFY2(waitUntil([this] { return !window_->busy_ && !window_->evaluationWorker_; }), qPrintable(state()));
        QCOMPARE(window_->datasetManifest_.value("split").toString(), QString("train"));
        QCOMPARE(window_->evaluation_->datasetSplit(), QString("train"));
        QCOMPARE(window_->evaluation_->datasetYaml(), directory); QCOMPARE(window_->files_, images_);
        QVERIFY(window_->evaluation_->canEvaluate()); QVERIFY(window_->canvas_->result().hasGroundTruth);
    }
    void relocatedDatasetFolderRemainsUsableWhenSwitchingSplit()
    {
        dataset();
        const QString directory = QFileInfo(yaml_).absolutePath();
        const QString absent = area_->path() + "/old-export-location";
        writeFile(yaml_, ("path: " + absent + "\nval: images/val\ntrain: images/val\nnames: [football]\n").toUtf8());
        QFile original(yaml_); QVERIFY(original.open(QIODevice::ReadOnly)); const auto bytes = original.readAll(); original.close();
        QVERIFY2(selectDatasetPath(), qPrintable(state()));
        QVERIFY2(waitUntil([this] { return !window_->busy_ && !window_->evaluationWorker_; }), qPrintable(state()));
        QCOMPARE(window_->files_, images_); QCOMPARE(window_->evaluation_->datasetYaml(), directory);
        QVERIFY(!window_->datasetManifest_.value("relocation_notice").toString().isEmpty());
        QVERIFY(!QFileInfo::exists(absent));
        auto *split = window_->evaluation_->controlsWidget()->findChild<QComboBox *>("evaluationSplitCombo"); QVERIFY(split);
        split->setCurrentIndex(split->findData("train"));
        QVERIFY2(waitUntil([this] { return !window_->busy_ && !window_->evaluationWorker_; }), qPrintable(state()));
        QCOMPARE(window_->datasetManifest_.value("split").toString(), QString("train"));
        QCOMPARE(window_->files_, images_); QCOMPARE(window_->evaluation_->datasetYaml(), directory);
        QVERIFY(window_->evaluation_->canEvaluate());
        QVERIFY2(runEvaluation(0, 0), qPrintable(state()));
        QCOMPARE(window_->evaluationReport_.value("status").toString(), QString("complete"));
        QCOMPARE(window_->evaluationReport_.value("completed_images").toInt(), 4);
        QCOMPARE(window_->evaluationReport_.value("manifest").toObject().value("dataset_source").toString(), directory);
        evidence("relocated-folder-evaluation");
        QVERIFY(original.open(QIODevice::ReadOnly)); QCOMPARE(original.readAll(), bytes);
    }
    void minimumWindowContainsEntireCanvas()
    {
        window_->resize(1260, 820); QTest::qWait(40);
        const auto fullyContained = [this] {
            return window_->canvas_->parentWidget()->rect().contains(QRect(window_->canvas_->pos(), window_->canvas_->size()));
        };
        QVERIFY(fullyContained()); QVERIFY(window_->canvas_->height() >= 280);
        QVERIFY(!window_->recordButton_->isVisible()); QVERIFY(!window_->countMetric_->parentWidget()->isVisible());
        QVERIFY2(loadDataset(), qPrintable(state())); QTest::qWait(40);
        QVERIFY(fullyContained()); QVERIFY(window_->canvas_->height() >= 280);
        QVERIFY(!window_->countMetric_->parentWidget()->isVisible());
        QVERIFY(window_->evaluation_->controlsWidget()->isVisible());
        QPushButton *clear = nullptr;
        for (auto *button : window_->findChildren<QPushButton *>()) if (button->text() == "清空") clear = button;
        QVERIFY(clear); clear->click(); QTest::qWait(40);
        QVERIFY(window_->files_.isEmpty()); QVERIFY(window_->datasetManifest_.isEmpty());
        QVERIFY(fullyContained()); QVERIFY(!window_->countMetric_->parentWidget()->isVisible());
        QVERIFY(!window_->evaluation_->controlsWidget()->isVisible());
        auto *camera = window_->findChild<QPushButton *>("openCameraButton"); QVERIFY(camera); camera->click(); QTest::qWait(40);
        QCOMPARE(window_->sourceKind_, vision::SourceKind::Camera);
        QVERIFY(window_->countMetric_->parentWidget()->isVisible()); QVERIFY(window_->recordButton_->isVisible());
        QVERIFY(fullyContained()); QVERIFY(window_->canvas_->height() >= 280);
    }
    void compactCommandBarPreservesCanvasAndCollapsesDatasetDetails_data()
    {
        QTest::addColumn<int>("width"); QTest::addColumn<QString>("device");
        for (int width : {1260, 1380})
            for (const QString &device : {QStringLiteral("cpu"), QStringLiteral("cuda")})
                QTest::newRow(qPrintable(QString("%1-%2-layout-only").arg(width).arg(device))) << width << device;
    }
    void compactCommandBarPreservesCanvasAndCollapsesDatasetDetails()
    {
        QFETCH(int, width); QFETCH(QString, device);
        window_->resize(width, 820);
        // This checks device-selector layout only; no CUDA inference is claimed.
        window_->computeDevice_->setCurrentIndex(window_->computeDevice_->findData(device));
        QTest::qWait(40);
        auto *bar = window_->findChild<QWidget *>("commandBar"); QVERIFY(bar && bar->isVisible());
        QVERIFY2(bar->height() <= 112, qPrintable(QString("empty command bar: %1 px").arg(bar->height())));
        QVERIFY2(loadDataset(), qPrintable(state())); QTest::qWait(40);
        auto *settings = window_->evaluation_->controlsWidget()->findChild<QWidget *>("evaluationDatasetSettings");
        auto *settingsButton = window_->evaluation_->controlsWidget()->findChild<QToolButton *>("evaluationDatasetSettingsButton");
        auto *advanced = window_->evaluation_->controlsWidget()->findChild<QWidget *>("evaluationAdvancedControls");
        auto *advancedButton = window_->evaluation_->controlsWidget()->findChild<QToolButton *>("evaluationAdvancedButton");
        QVERIFY(settings && settingsButton && advanced && advancedButton);
        QVERIFY(!settings->isVisible()); QVERIFY(!advanced->isVisible());
        QVERIFY2(bar->height() <= 112, qPrintable(QString("dataset command bar: %1 px").arg(bar->height())));
        QVERIFY(window_->canvas_->height() >= 280);
        QVERIFY(window_->canvas_->parentWidget()->rect().contains(QRect(window_->canvas_->pos(), window_->canvas_->size())));
        for (const char *name : {"openYoloDatasetButton", "openYoloDatasetOptionsButton", "openVideoButton", "openCameraButton", "computeDevice", "evaluationModeCombo", "evaluationSplitCombo", "evaluationReloadDatasetButton", "evaluationDatasetSettingsButton"}) {
            auto *control = bar->findChild<QWidget *>(name); QVERIFY2(control && control->isVisible(), name);
            QVERIFY2(bar->rect().contains(QRect(control->mapTo(bar, QPoint()), control->size())), name);
        }
        if (device == "cuda") {
            auto *gpu = bar->findChild<QComboBox *>("gpuDeviceIndex"); QVERIFY(gpu && gpu->isVisible());
            QVERIFY(bar->rect().contains(QRect(gpu->mapTo(bar, QPoint()), gpu->size())));
        }
        const QString screenshotRoot = qEnvironmentVariable("VISION_EVALUATION_INTEGRATION_EVIDENCE");
        const auto screenshot = [this, screenshotRoot, width, device](const QString &state) {
            if (screenshotRoot.isEmpty()) return true;
            if (!QDir().mkpath(screenshotRoot)) return false;
            const QString path = screenshotRoot + QString("/compact-%1-%2-%3.png").arg(width).arg(device, state);
            window_->saveScreenshot(path);
            return QImage(path).size() == window_->size();
        };
        QVERIFY(screenshot("collapsed"));
        settingsButton->click(); QTest::qWait(30);
        QVERIFY(settings->isVisible());
        QVERIFY(bar->findChild<QSpinBox *>("evaluationModelClassIdSpin")->isVisible());
        QVERIFY(bar->findChild<QSpinBox *>("evaluationDatasetClassIdSpin")->isVisible());
        advancedButton->click(); QTest::qWait(30); QVERIFY(advanced->isVisible());
        for (const char *name : {"evaluationMetadataEdit", "evaluationMatchIouSpin", "evaluationScanStartSpin", "evaluationScanEndSpin", "evaluationScanStepSpin"}) {
            auto *control = bar->findChild<QWidget *>(name); QVERIFY2(control && control->isVisible(), name);
            QVERIFY2(bar->rect().contains(QRect(control->mapTo(bar, QPoint()), control->size())), name);
        }
        const QRect expandedCanvas(window_->canvas_->mapTo(window_.get(), QPoint()), window_->canvas_->size());
        qInfo().noquote() << "expanded_workbench_geometry" << width << device << "command_bar_px" << bar->height()
                         << "canvas" << QRect(window_->canvas_->pos(), window_->canvas_->size())
                         << "canvas_parent" << window_->canvas_->parentWidget()->rect()
                         << "canvas_in_window" << expandedCanvas << "window" << window_->rect();
        QVERIFY(screenshot("expanded"));
        QVERIFY(window_->rect().contains(expandedCanvas));
        QVERIFY(window_->canvas_->parentWidget()->rect().contains(QRect(window_->canvas_->pos(), window_->canvas_->size())));
        QVERIFY(window_->canvas_->height() >= 280);
        settingsButton->click(); QTest::qWait(30);
        QVERIFY(!settings->isVisible());
        QVERIFY2(bar->height() <= 112, qPrintable(QString("recollapsed command bar: %1 px").arg(bar->height())));
        QVERIFY(window_->canvas_->height() >= 280);
        qInfo().noquote() << "compact_workbench_evidence" << width << device << "command_bar_px" << bar->height()
                         << "canvas" << window_->canvas_->size();
    }

    void missingCurrentSelectionDoesNotExpandTask()
    {
        QVERIFY2(loadDataset(), qPrintable(state())); configureEvaluation();
        window_->queue_->setCurrentRow(-1);
        for (int mode : {1, 2}) {
            window_->detectionRange_->setCurrentIndex(mode);
            QVERIFY(!window_->runButton_->isEnabled());
            window_->runButton_->click(); QVERIFY(!window_->busy_); QVERIFY(!window_->evaluationWorker_);
            QVERIFY(window_->selectedImageRange().isEmpty());
        }
        window_->detectionRange_->setCurrentIndex(0); QVERIFY(window_->runButton_->isEnabled());
    }
    void missingLabelsNeverBecomeNegativeSamples()
    {
        dataset();
        QVERIFY(QFile::remove(QFileInfo(yaml_).absolutePath() + "/labels/val/0001.txt"));
        window_->loadDataset(yaml_, "val");
        QVERIFY2(waitUntil([this] { return !window_->busy_ && !window_->evaluationWorker_; }), qPrintable(state()));
        QVERIFY(window_->datasetManifest_.value("requires_attention").toBool());
        QCOMPARE(window_->datasetManifest_.value("validation").toObject().value("missing_labels").toInt(), 1);
        QVERIFY(!window_->evaluation_->canEvaluate()); window_->evaluation_->setMode(true);
        QVERIFY(!window_->evaluation_->evaluationMode());
        QCOMPARE(window_->datasetManifest_.value("validation").toObject().value("negative_images").toInt(), 1);
    }
    void failedSplitChangeDoesNotMislabelPreviousDataset()
    {
        QVERIFY2(loadDataset(), qPrintable(state()));
        const auto manifest = window_->datasetManifest_;
        const auto files = window_->files_;
        auto *split = window_->evaluation_->controlsWidget()->findChild<QComboBox *>("evaluationSplitCombo"); QVERIFY(split);
        split->setCurrentIndex(split->findData("test"));
        QVERIFY2(waitUntil([this] { return !window_->busy_ && !window_->evaluationWorker_; }), qPrintable(state()));
        QVERIFY(window_->evaluation_->canEvaluate()); QCOMPARE(window_->datasetManifest_, manifest); QCOMPARE(window_->files_, files);
        QCOMPARE(window_->evaluation_->datasetSplit(), window_->datasetManifest_.value("split").toString());
        QCOMPARE(window_->evaluation_->datasetYaml(), window_->datasetManifest_.value("data_yaml").toString());
        QVERIFY(!window_->busy_);
    }
    void rangesAndMetrics_data()
    {
        QTest::addColumn<int>("count"); QTest::addColumn<int>("range"); QTest::addColumn<int>("selection");
        QTest::newRow("all-starts-at-first") << 4 << 0 << 2;
        QTest::newRow("tail-includes-selected") << 4 << 1 << 1;
        QTest::newRow("current-only") << 4 << 2 << 1;
        QTest::newRow("single-current") << 1 << 2 << 0;
        QTest::newRow("sixty-one-all") << 61 << 0 << 50;
        QTest::newRow("sixty-one-tail") << 61 << 1 << 50;
        QTest::newRow("sixty-one-current-last") << 61 << 2 << 60;
    }
    void rangesAndMetrics()
    {
        QFETCH(int, count); QFETCH(int, range); QFETCH(int, selection);
        QVERIFY2(loadDataset(count), qPrintable(state()));
        const QStringList expected = range == 0 ? images_ : range == 1 ? images_.mid(selection) : QStringList{images_[selection]};
        QVERIFY2(runEvaluation(range, selection), qPrintable(state()));
        const auto report = window_->evaluationReport_;
        QCOMPARE(report.value("status").toString(), "complete");
        QCOMPARE(report.value("completed_images").toInt(), expected.size());
        QStringList actual; for (const auto &entry : report.value("request").toObject().value("sample_images").toArray()) actual.append(entry.toString());
        QCOMPARE(actual, expected); QCOMPARE(window_->taskRangeFiles_, expected);
        QCOMPARE(window_->progress_->maximum(), expected.size()); QCOMPARE(window_->progress_->value(), expected.size());
        QCOMPARE(window_->evaluationSampleIndices_.size(), expected.size()); QVERIFY(window_->imageResults_.isEmpty());
        int tp = 0, fp = 0, fn = 0, negative = 0;
        for (const auto &path : expected) {
            const int index = images_.indexOf(path);
            if (index % 4 == 2) { ++fp; ++negative; } else if (index % 4 == 1) { ++fp; ++fn; } else ++tp;
        }
        const auto metrics = report.value("metrics").toObject();
        QCOMPARE(metrics.value("TP").toInt(), tp); QCOMPARE(metrics.value("FP").toInt(), fp); QCOMPARE(metrics.value("FN").toInt(), fn);
        QCOMPARE(metrics.value("no_ball_images").toInt(), negative);
        if (count > 1 && range == 0) QVERIFY(metrics.value("mAP50").isDouble()); else QVERIFY(metrics.value("mAP50").isNull());
        for (int index = 0; index < images_.size(); ++index) {
            window_->queue_->setCurrentRow(index);
            if (expected.contains(images_[index])) {
                QTRY_COMPARE_WITH_TIMEOUT(window_->lastResult_.source, images_[index], 10000);
                QCOMPARE(window_->lastResult_.predictions.size(), 1);
                QCOMPARE(window_->predictionTable_->rowCount(), index % 4 == 1 ? 2 : 1); QVERIFY(window_->lastResult_.hasGroundTruth);
                QVERIFY(window_->lastResult_.hasEvaluationMatches);
                QVERIFY(window_->exportButton_->isEnabled());
            } else {
                QTRY_COMPARE_WITH_TIMEOUT(window_->canvas_->result().source, images_[index], 10000);
                QTRY_VERIFY_WITH_TIMEOUT(window_->canvas_->result().predictions.isEmpty(), 10000); QVERIFY(!window_->exportButton_->isEnabled());
            }
        }
        evidence(QString("range-%1-count-%2-selection-%3").arg(range).arg(count).arg(selection));
    }
    void thresholdCurveFiltersAndAdoptedConfiguration()
    {
        QVERIFY2(loadDataset(), qPrintable(state())); QVERIFY2(runEvaluation(0, 0), qPrintable(state()));
        const QStringList frozen = window_->taskRangeFiles_;
        const auto report = window_->evaluationReport_;
        auto *filter = window_->evaluation_->findChild<QComboBox *>("evaluationImageFilterCombo"); QVERIFY(filter);
        filter->setCurrentIndex(filter->findData("fp"));
        QTRY_VERIFY_WITH_TIMEOUT(window_->queue_->item(0)->isHidden() && !window_->queue_->item(1)->isHidden() && !window_->queue_->item(2)->isHidden() && window_->queue_->item(3)->isHidden(), 10000);
        filter->setCurrentIndex(filter->findData("fn"));
        QTRY_VERIFY_WITH_TIMEOUT(!window_->queue_->item(1)->isHidden() && window_->queue_->item(2)->isHidden(), 10000);
        filter->setCurrentIndex(filter->findData("negative_fp"));
        QTRY_COMPARE_WITH_TIMEOUT(window_->queue_->currentRow(), 2, 10000);
        QTRY_COMPARE_WITH_TIMEOUT(window_->lastResult_.source, images_[2], 10000);
        QCOMPARE(window_->lastResult_.predictions.size(), 1); QCOMPARE(window_->lastResult_.groundTruth.size(), 0);
        window_->evaluation_->setAnalysisThreshold(.9);
        for (int i = 0; i < 4; ++i) QTRY_VERIFY_WITH_TIMEOUT(window_->queue_->item(i)->isHidden(), 10000);
        QTRY_VERIFY_WITH_TIMEOUT(window_->lastResult_.predictions.isEmpty(), 10000);
        QCOMPARE(window_->evaluation_->findChild<QLabel *>("evaluationMetric5")->text(), "0.0%");
        QCOMPARE(window_->taskRangeFiles_, frozen); QCOMPARE(window_->evaluationReport_, report);
        window_->evaluation_->setAnalysisThreshold(.7); filter->setCurrentIndex(filter->findData("all")); window_->queue_->setCurrentRow(0);
        QTRY_COMPARE_WITH_TIMEOUT(window_->lastResult_.source, images_[0], 10000);
        QTRY_COMPARE_WITH_TIMEOUT(window_->lastResult_.matchedPredictions.size(), 1, 10000); QCOMPARE(window_->lastResult_.matchedGroundTruth.size(), 1);
        const double confidence = window_->confidence_->value();
        window_->evaluation_->setAnalysisThreshold(.9); QCOMPARE(window_->confidence_->value(), confidence);
        window_->evaluation_->setAnalysisThreshold(.7);
        window_->inputSize_->setValue(128); window_->iou_->setValue(.7);
        auto *adopt = window_->evaluation_->findChild<QPushButton *>("evaluationApplyThresholdButton"); QVERIFY(adopt); adopt->click();
        QCOMPARE(window_->inputSize_->value(), 64); QVERIFY(std::abs(window_->iou_->value() - .45) < .001);
        QCOMPARE(window_->confidence_->value(), .7); QCOMPARE(window_->classFilter_->text(), "0");
        QVERIFY(window_->modelEvaluationProfiles_.value(model_).toMap().value("adopted").toBool());
        QCOMPARE(window_->evaluationReport_, report);
        QTRY_COMPARE_WITH_TIMEOUT(window_->lastResult_.source, images_[0], 10000);
        QTRY_COMPARE_WITH_TIMEOUT(window_->lastResult_.predictions.size(), 1, 10000);
        evidence("threshold-filter-adopt");
    }
    void adoptedConfidencePreservesThreeDecimalWorkingPoints()
    {
        QVERIFY2(loadDataset(), qPrintable(state()));
        window_->resize(1260, 820); QTest::qWait(40);
        const auto canvasBounds = QRect(window_->canvas_->pos(), window_->canvas_->size());
        qInfo() << "minimum_canvas_geometry" << canvasBounds << "parent_rect" << window_->canvas_->parentWidget()->rect()
                << "fully_in_parent" << window_->canvas_->parentWidget()->rect().contains(canvasBounds);
        auto *controls = window_->evaluation_->controlsWidget();
        controls->findChild<QDoubleSpinBox *>("evaluationScanStartSpin")->setValue(.005);
        controls->findChild<QDoubleSpinBox *>("evaluationScanEndSpin")->setValue(.255);
        controls->findChild<QDoubleSpinBox *>("evaluationScanStepSpin")->setValue(.25);
        window_->evaluation_->setMode(true); window_->detectionRange_->setCurrentIndex(0);
        QVERIFY(window_->runButton_->isEnabled()); window_->runButton_->click();
        QVERIFY2(waitUntil([this] { return !window_->busy_ && !window_->evaluationWorker_; }), qPrintable(state()));
        QVERIFY(!window_->evaluationReport_.isEmpty()); QCOMPARE(window_->evaluationReport_.value("rows").toArray().size(), 2);
        const QJsonObject frozen = window_->evaluationReport_;
        auto *adopt = window_->evaluation_->findChild<QPushButton *>("evaluationApplyThresholdButton"); QVERIFY(adopt);
        for (const double threshold : {.005, .255}) {
            window_->evaluation_->setAnalysisThreshold(threshold); adopt->click();
            QVERIFY(std::abs(window_->confidence_->value() - threshold) < 1e-9);
            const auto profile = window_->modelEvaluationProfiles_.value(model_).toMap();
            QVERIFY(profile.value("adopted").toBool());
            QVERIFY(std::abs(profile.value("adopted_threshold").toDouble() - threshold) < 1e-9);
            QTRY_COMPARE_WITH_TIMEOUT(window_->lastResult_.source, images_[0], 10000);
            QVERIFY(std::abs(window_->lastConfig_.confidence - threshold) < 1e-7);
        }
        QCOMPARE(window_->evaluationReport_, frozen); evidence("precise-adopted-working-points");
    }
    void selectedResultExportUsesCurrentThresholdAndTruth()
    {
        QVERIFY2(loadDataset(), qPrintable(state())); QVERIFY2(runEvaluation(0, 0), qPrintable(state()));
        const QString output = area_->path() + "/exports";
        window_->evaluation_->setAnalysisThreshold(.7); window_->queue_->setCurrentRow(1);
        QTRY_COMPARE_WITH_TIMEOUT(window_->lastResult_.source, images_[1], 10000);
        const auto result = window_->lastResult_; QVERIFY(result.hasGroundTruth);
        QCOMPARE(result.predictions.size(), 1); QCOMPARE(result.groundTruth.size(), 1); QVERIFY(result.matchedPredictions.isEmpty());
        QString error; QVERIFY2(window_->writeResult(result, output, &error), qPrintable(error));
        const auto names = QDir(output).entryList({"*.json"}, QDir::Files); QCOMPARE(names.size(), 1);
        const QString base = output + "/" + QFileInfo(names.first()).completeBaseName();
        const auto json = readJson(base + ".json"); QCOMPARE(json.value("source").toString(), images_[1]);
        QCOMPARE(json.value("predictions").toArray().size(), 1);
        QVERIFY(std::abs(json.value("config").toObject().value("confidence_threshold").toDouble() - .7) < 1e-5);
        QCOMPARE(json.value("ground_truth").toArray().size(), 1);
        QVERIFY(json.value("evaluation").toObject().value("matching_available").toBool());
        QVERIFY(json.value("evaluation").toObject().value("matched_predictions").toArray().isEmpty());
        QCOMPARE(json.value("predictions").toArray().first().toObject().value("match_status").toString(), "FP");
        const QImage image(base + ".png"); QVERIFY(!image.isNull());
        QCOMPARE(image.convertToFormat(QImage::Format_ARGB32), ImageCanvas::annotatedResultImage(result).convertToFormat(QImage::Format_ARGB32));
        QFile csv(base + ".csv"); QVERIFY(csv.open(QIODevice::ReadOnly));
        const QByteArray values = csv.readAll(); QVERIFY(values.contains("class_id")); QVERIFY(values.contains("0,"));
        QVERIFY(values.contains("FP"));
        const QString evidenceRoot = qEnvironmentVariable("VISION_EVALUATION_INTEGRATION_EVIDENCE");
        if (!evidenceRoot.isEmpty()) {
            QDir().mkpath(evidenceRoot + "/selected-export");
            for (const QString &suffix : {".png", ".json", ".csv"}) QFile::copy(base + suffix, evidenceRoot + "/selected-export/selected" + suffix);
        }
        window_->evaluation_->setAnalysisThreshold(.9); window_->queue_->setCurrentRow(0);
        QTRY_COMPARE_WITH_TIMEOUT(window_->lastResult_.source, images_[0], 10000);
        QTRY_VERIFY_WITH_TIMEOUT(window_->lastResult_.predictions.isEmpty(), 10000); QCOMPARE(window_->lastResult_.groundTruth.size(), 1);
        QVERIFY2(window_->writeResult(window_->lastResult_, output + "/zero", &error), qPrintable(error));
        const auto zeroNames = QDir(output + "/zero").entryList({"*.json"}, QDir::Files); QCOMPARE(zeroNames.size(), 1);
        const auto zero = readJson(output + "/zero/" + zeroNames.first());
        QVERIFY(zero.value("predictions").toArray().isEmpty()); QCOMPARE(zero.value("ground_truth").toArray().size(), 1);
        evidence("selected-result-export");
    }
    void historyReopenDoesNotSelectAnotherGlobalModel()
    {
        QVERIFY2(loadDataset(), qPrintable(state())); QVERIFY2(runEvaluation(0, 0), qPrintable(state()));
        const QString report = window_->evaluationDirectory_;
        QVERIFY(!window_->history_.isEmpty()); QCOMPARE(window_->history_.first().toObject().value("kind").toString(), "evaluation");
        const QString another = writeOnnx(area_->path(), 64, "another.onnx");
        window_->models_.append(another); window_->setModel(another);
        QVERIFY2(waitUntil([this] { return !window_->modelInspectionPending_ && window_->modelClassCount_ == 1; }, 30000), qPrintable(state()));
        window_->loadEvaluationReport(report);
        QCOMPARE(window_->modelPath_, another); QCOMPARE(window_->evaluationDirectory_, report);
        QCOMPARE(window_->queue_->count(), 4); QTRY_COMPARE_WITH_TIMEOUT(window_->lastResult_.source, images_[0], 10000);
        QCOMPARE(window_->lastResult_.predictions.size(), 1); QCOMPARE(window_->lastResult_.matchedPredictions.size(), 1);
        window_->confidence_->setValue(.4); window_->evaluation_->findChild<QPushButton *>("evaluationApplyThresholdButton")->click();
        QCOMPARE(window_->modelPath_, another); QCOMPARE(window_->confidence_->value(), .4);
        QVERIFY(window_->statusLabel_->text().contains("同一模型"));
        evidence("history-other-global-model");
    }
    void evaluationScopeDoesNotBorrowOrdinaryPreviewCache()
    {
        QVERIFY2(loadDataset(), qPrintable(state()));
        QVERIFY(!window_->evaluation_->evaluationMode());
        window_->runButton_->click();
        QVERIFY2(waitUntil([this] { return !window_->busy_ && !window_->resultIoPending_; }), qPrintable(state()));
        QCOMPARE(window_->imageResults_.size(), 4);
        window_->queue_->setCurrentRow(0); QCOMPARE(window_->lastResult_.predictions.size(), 1);
        QVERIFY2(runEvaluation(2, 1), qPrintable(state()));
        QCOMPARE(window_->evaluationSampleIndices_.size(), 1);
        window_->queue_->setCurrentRow(0);
        QTRY_COMPARE_WITH_TIMEOUT(window_->canvas_->result().source, images_[0], 10000);
        QVERIFY(window_->canvas_->result().hasGroundTruth);
        QTRY_VERIFY_WITH_TIMEOUT(window_->canvas_->result().predictions.isEmpty(), 10000);
        QVERIFY(!window_->exportButton_->isEnabled());
        window_->queue_->setCurrentRow(1); QTRY_COMPARE_WITH_TIMEOUT(window_->lastResult_.source, images_[1], 10000);
        QCOMPARE(window_->lastResult_.predictions.size(), 1); QVERIFY(window_->exportButton_->isEnabled());
        evidence("evaluation-scope-versus-ordinary-cache");
    }
    void switchingToPreviewLeavesEvaluationReview()
    {
        QVERIFY2(loadDataset(), qPrintable(state())); QVERIFY2(runEvaluation(0, 0), qPrintable(state()));
        QVERIFY(window_->evaluationReviewActive_); window_->evaluation_->setMode(false);
        QVERIFY(!window_->evaluationReviewActive_); window_->queue_->setCurrentRow(1);
        QTRY_VERIFY_WITH_TIMEOUT(window_->canvas_->result().predictions.isEmpty(), 10000); QVERIFY(window_->canvas_->result().hasGroundTruth);
        QCOMPARE(window_->workbenchTabs_->currentIndex(), 0);
        window_->evaluation_->setMode(true); window_->evaluation_->setAnalysisThreshold(.7);
        QVERIFY(window_->evaluationReviewActive_); QTRY_COMPARE_WITH_TIMEOUT(window_->lastResult_.predictions.size(), 1, 10000);
    }
    void changedPreprocessingRestoresPureGroundTruthPreview()
    {
        QVERIFY2(loadDataset(), qPrintable(state())); QVERIFY2(runEvaluation(0, 0), qPrintable(state()));
        window_->queue_->setCurrentRow(0);
        QTRY_COMPARE_WITH_TIMEOUT(window_->lastResult_.source, images_[0], 10000);
        QTRY_VERIFY_WITH_TIMEOUT(window_->canvas_->result().hasEvaluationMatches, 10000);
        QVERIFY(!window_->canvas_->result().evaluationReport.isEmpty());
        const auto frozen = window_->evaluationReport_;
        window_->inputColorMode_->setCurrentIndex(window_->inputColorMode_->findData("grayscale"));
        const auto preview = window_->canvas_->result();
        QCOMPARE(preview.source, images_[0]); QVERIFY(preview.hasGroundTruth);
        QCOMPARE(preview.groundTruth.size(), 1); QVERIFY(preview.predictions.isEmpty());
        QVERIFY(!preview.hasEvaluationMatches); QVERIFY(preview.matchedPredictions.isEmpty());
        QVERIFY(preview.matchedGroundTruth.isEmpty()); QVERIFY(preview.ignoredPredictions.isEmpty());
        QVERIFY(preview.modelName.isEmpty()); QVERIFY(preview.evaluationReport.isEmpty());
        QVERIFY(window_->lastResult_.image.isNull()); QVERIFY(!window_->exportButton_->isEnabled());
        QCOMPARE(window_->evaluationReport_, frozen);
    }
    void smallerModelInvalidatesMappingWithoutInventingFalseNegatives()
    {
        QVERIFY2(loadDataset(), qPrintable(state()));
        const auto larger = writeOnnx(area_->path(), 64, "two-class.onnx", 2);
        window_->models_.append(larger); window_->setModel(larger);
        QVERIFY2(waitUntil([this] { return !window_->modelInspectionPending_ && window_->modelClassCount_ == 2; }, 30000), qPrintable(state()));
        auto *modelId = window_->evaluation_->controlsWidget()->findChild<QSpinBox *>("evaluationModelClassIdSpin"); QVERIFY(modelId);
        modelId->setValue(1); QCOMPARE(modelId->value(), 1);
        window_->setModel(model_);
        QVERIFY2(waitUntil([this] { return !window_->modelInspectionPending_ && window_->modelClassCount_ == 1; }, 30000), qPrintable(state()));
        QCOMPARE(modelId->value(), 1); QVERIFY(!window_->evaluation_->canEvaluate());
        QVERIFY(!window_->evaluation_->evaluationMode());
        window_->queue_->setCurrentRow(0); window_->detectionRange_->setCurrentIndex(2);
        QVERIFY(window_->runButton_->isEnabled()); window_->runButton_->click();
        QVERIFY2(waitUntil([this] { return !window_->busy_ && !window_->resultIoPending_; }), qPrintable(state()));
        QCOMPARE(window_->lastResult_.source, images_[0]); QCOMPARE(window_->lastResult_.predictions.size(), 1);
        QVERIFY(window_->lastResult_.hasGroundTruth); QCOMPARE(window_->lastResult_.groundTruth.size(), 1);
        QVERIFY(!window_->lastResult_.hasEvaluationMatches);
        QVERIFY(window_->lastResult_.matchedPredictions.isEmpty()); QVERIFY(window_->lastResult_.matchedGroundTruth.isEmpty());
        QVERIFY(window_->evaluation_->validationMessage().contains("超出"));
    }
    void stoppingPreservesCompletedPrefixAndReview()
    {
        QVERIFY2(loadDataset(64), qPrintable(state()));
        const QString slower = writeOnnx(area_->path(), 640, "larger.onnx");
        window_->models_.append(slower); window_->setModel(slower); window_->inputSize_->setValue(640);
        QVERIFY2(waitUntil([this] { return !window_->modelInspectionPending_ && window_->modelClassCount_ == 1; }, 30000), qPrintable(state()));
        configureEvaluation(); window_->detectionRange_->setCurrentIndex(0); window_->queue_->setCurrentRow(0);
        QVERIFY(window_->runButton_->isEnabled()); window_->runButton_->click(); QVERIFY(window_->evaluationWorker_);
        bool clicked = false; qint64 clickDuration = -1; int firstCompleted = 0; int heartbeats = 0;
        QTimer heartbeat; connect(&heartbeat, &QTimer::timeout, this, [&] { ++heartbeats; }); heartbeat.start(10);
        connect(window_->evaluationWorker_, &vision::EvaluationWorker::progress, this,
                [&](int completed, int, const QString &) {
            if (!clicked && completed >= 1 && window_->busy_) {
                firstCompleted = completed; clicked = true;
                QElapsedTimer timer; timer.start(); window_->stopButton_->click(); clickDuration = timer.elapsed();
            }
        });
        QVERIFY2(waitUntil([this] { return !window_->busy_ && !window_->evaluationWorker_; }, 90000), qPrintable(state()));
        heartbeat.stop(); QVERIFY(clicked); QVERIFY(firstCompleted >= 1); QVERIFY(clickDuration >= 0 && clickDuration < 500);
        const auto report = window_->evaluationReport_; QCOMPARE(report.value("status").toString(), "cancelled");
        const int done = report.value("completed_images").toInt(); QVERIFY(done >= 1 && done < 64); QVERIFY(heartbeats > 1);
        QCOMPARE(window_->evaluationSampleIndices_.size(), done);
        const auto countBefore = QDir(window_->evaluationDirectory_ + "/samples").entryList({"*.json"}, QDir::Files).size();
        QTest::qWait(200);
        QCOMPARE(QDir(window_->evaluationDirectory_ + "/samples").entryList({"*.json"}, QDir::Files).size(), countBefore);
        for (int row = 0; row < done; ++row) {
            window_->queue_->setCurrentRow(row); QTRY_COMPARE_WITH_TIMEOUT(window_->lastResult_.source, images_[row], 10000);
            QCOMPARE(window_->lastResult_.predictions.size(), 1); QVERIFY(window_->exportButton_->isEnabled());
        }
        window_->queue_->setCurrentRow(done); QTRY_COMPARE_WITH_TIMEOUT(window_->canvas_->result().source, images_[done], 10000);
        QTRY_VERIFY_WITH_TIMEOUT(window_->canvas_->result().predictions.isEmpty(), 10000); QVERIFY(!window_->exportButton_->isEnabled());
        QVERIFY(report.value("metrics").toObject().value("mAP50").isNull());
        qInfo() << "stop_button_return_ms" << clickDuration << "completed_prefix" << done << "heartbeats" << heartbeats;
        evidence("stop-prefix-review");
    }
};
QTEST_MAIN(EvaluationIntegrationTests)
#include "evaluation_integration_tests.moc"
