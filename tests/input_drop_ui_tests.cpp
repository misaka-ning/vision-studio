// SPDX-License-Identifier: AGPL-3.0-only
#include "core/datasetsessionstore.h"
#include "core/evaluationworker.h"
#include "ui/mainwindow.h"
#include "ui/evaluationpanel.h"
#include "ui/imagecanvas.h"
#include "ui/lazyimagequeue.h"
#include "ui/modelviewer.h"
#include <QApplication>
#include <QCheckBox>
#include <QComboBox>
#include <QCryptographicHash>
#include <QDir>
#include <QDirIterator>
#include <QDialog>
#include <QDoubleSpinBox>
#include <QDragEnterEvent>
#include <QDropEvent>
#include <QElapsedTimer>
#include <QFile>
#include <QFileInfo>
#include <QFont>
#include <QJsonArray>
#include <QJsonDocument>
#include <QLabel>
#include <QListWidget>
#include <QMessageBox>
#include <QMimeData>
#include <QPointer>
#include <QPushButton>
#include <QSettings>
#include <QSignalSpy>
#include <QSpinBox>
#include <QStandardPaths>
#include <QTabWidget>
#include <QTableWidget>
#include <QTemporaryDir>
#include <QTimer>
#include <QUrl>
#include <QtTest>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <algorithm>
#include <cstring>
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
QByteArray readBytes(const QString &path)
{
    QFile file(path);
    return file.open(QIODevice::ReadOnly) ? file.readAll() : QByteArray();
}
QJsonObject readJson(const QString &path) { return QJsonDocument::fromJson(readBytes(path)).object(); }
QMap<QString, QByteArray> inputHashes(const QString &root)
{
    QMap<QString, QByteArray> result;
    QDirIterator it(root, QDir::Files, QDirIterator::Subdirectories);
    while (it.hasNext()) {
        const QString path = it.next();
        result.insert(QDir(root).relativeFilePath(path), QCryptographicHash::hash(readBytes(path), QCryptographicHash::Sha256));
    }
    return result;
}
}

class InputDropUiTests final : public QObject
{
    Q_OBJECT
    std::unique_ptr<QTemporaryDir> area_;
    std::unique_ptr<MainWindow> window_;
    QHash<QByteArray, QByteArray> oldEnvironment_;
    QHash<QByteArray, bool> hadEnvironment_;
    QString python_, model_, datasetRoot_, datasetYaml_, unexpectedDialog_;
    QStringList datasetImages_, looseImages_;
    QTimer dialogGuard_;
    bool expectingChoice_ = false;

    QString sessionPath() const { return area_->path() + "/data/dataset-session.json"; }
    QString state() const { return unexpectedDialog_.isEmpty() ? window_->statusLabel_->text() + " | " + window_->lastError_ : unexpectedDialog_; }
    bool idle() const { return !window_->busy_ && !window_->evaluationWorker_ && !window_->queuePopulating_ && !window_->resultIoPending_; }
    bool waitUntil(const std::function<bool()> &condition, int timeout = 45000)
    {
        QElapsedTimer elapsed; elapsed.start();
        while (!condition() && elapsed.elapsed() < timeout && unexpectedDialog_.isEmpty()) {
            QApplication::processEvents(); QTest::qWait(5);
        }
        QApplication::processEvents();
        return condition() && unexpectedDialog_.isEmpty();
    }
    bool previewReady(const QString &path, bool processed = false)
    {
        return waitUntil([&, path, processed] {
            const auto &value = window_->canvas_->result();
            return idle() && value.source == path && !value.image.isNull() &&
                (value.inferenceMs > 0) == processed && (!processed ||
                (window_->lastResult_.source == path && !window_->lastResult_.image.isNull() && window_->exportButton_->isEnabled()));
        }, 15000);
    }
    void showWindow()
    {
        window_ = std::make_unique<MainWindow>(); window_->resize(1380, 940); window_->show(); window_->modelViewer_->stop();
    }
    bool drop(QWidget *target, const QList<QUrl> &urls)
    {
        QMimeData mime; mime.setUrls(urls);
        const QPoint point = target->rect().center();
        QDragEnterEvent enter(point, Qt::CopyAction, &mime, Qt::LeftButton, Qt::NoModifier);
        QApplication::sendEvent(target, &enter);
        if (!enter.isAccepted()) return false;
        QDropEvent event(QPointF(point), Qt::CopyAction, &mime, Qt::LeftButton, Qt::NoModifier);
        QApplication::sendEvent(target, &event);
        return event.isAccepted();
    }
    bool dropPaths(QWidget *target, const QStringList &paths)
    {
        QList<QUrl> urls; for (const auto &path : paths) urls.append(QUrl::fromLocalFile(path));
        return drop(target, urls);
    }
    bool dropChoosing(const QStringList &paths, const QString &choice)
    {
        bool observed = false;
        QTimer chooser; QElapsedTimer elapsed; elapsed.start();
        expectingChoice_ = true;
        connect(&chooser, &QTimer::timeout, this, [&] {
            for (auto *widget : QApplication::topLevelWidgets())
                if (auto *dialog = qobject_cast<QDialog *>(widget); dialog && dialog->isVisible() &&
                    dialog->objectName() == "imageDropDatasetConflictDialog") {
                    observed = true;
                    auto *button = dialog->findChild<QPushButton *>(choice);
                    if (button) QTest::mouseClick(button, Qt::LeftButton);
                    else { unexpectedDialog_ = "Drop conflict dialog is missing " + choice; dialog->reject(); }
                    chooser.stop();
                }
            if (elapsed.elapsed() > 10000) {
                unexpectedDialog_ = "Expected dataset/image drop choice did not complete";
                if (auto *dialog = qobject_cast<QDialog *>(QApplication::activeModalWidget())) dialog->reject();
                chooser.stop();
            }
        });
        chooser.start(10);
        dropPaths(window_->canvas_, paths);
        chooser.stop(); expectingChoice_ = false;
        return observed && unexpectedDialog_.isEmpty();
    }
    void createDataset(int count = 4)
    {
        datasetRoot_ = area_->path() + "/inputs/标注数据集";
        datasetYaml_ = datasetRoot_ + "/data.yaml";
        QDir().mkpath(datasetRoot_ + "/images/val"); QDir().mkpath(datasetRoot_ + "/labels/val");
        datasetImages_.clear();
        for (int index = 0; index < count; ++index) {
            const QString stem = QString::number(index).rightJustified(5, '0');
            const QString path = datasetRoot_ + "/images/val/" + stem + ".png";
            QImage image(64, 64, QImage::Format_RGB888); image.fill(QColor(20 + index % 160, 70, 130));
            if (!image.save(path)) throw std::runtime_error("Cannot create dataset drop image");
            writeFile(datasetRoot_ + "/labels/val/" + stem + ".txt", "0 0.5 0.5 0.3125 0.3125\n");
            datasetImages_.append(path);
        }
        QString root = datasetRoot_; root.replace('\\', '/'); root.replace("'", "''");
        writeFile(datasetYaml_, ("path: '" + root + "'\nval: images/val\nnames: [football]\n").toUtf8());
    }
    QStringList createLoose(int count = 3)
    {
        const QString root = area_->path() + "/inputs/临时 images"; QDir().mkpath(root);
        QStringList paths;
        for (int index = 0; index < count; ++index) {
            const QString path = root + "/" + QString::number(index).rightJustified(5, '0') + ".png";
            QImage image(index == count - 1 ? QSize(73, 51) : QSize(64, 64), QImage::Format_RGB888);
            image.fill(index == count - 1 ? QColor(10, 210, 190) : QColor(150 + index % 80, 25, 80));
            if (!image.save(path)) throw std::runtime_error("Cannot create temporary drop image");
            paths.append(path);
        }
        looseImages_ = paths; return paths;
    }
    bool importDataset()
    {
        return dropPaths(window_->canvas_, {datasetRoot_}) && waitUntil([this] {
            return idle() && window_->files_ == datasetImages_ && window_->queue_->count() == datasetImages_.size() &&
                window_->datasetManifest_.value("samples").toArray().size() == datasetImages_.size();
        }) && previewReady(datasetImages_.first());
    }

private slots:
    void initTestCase()
    {
        QApplication::setStyle("Fusion"); QApplication::setFont(QFont("Noto Sans CJK SC", 10));
        QApplication::setAttribute(Qt::AA_DontUseNativeDialogs);
        for (QString candidate : {qEnvironmentVariable("VISION_STUDIO_EVALUATION_PYTHON"), qEnvironmentVariable("VISION_STUDIO_PYTHON"),
                                 QStringLiteral(VISION_PROJECT_DIR) + "/runtime/bin/python", QStringLiteral(VISION_PROJECT_DIR) + "/runtime/python.exe",
                                 QStringLiteral("/opt/VisionStudio/runtime/bin/python")}) {
            if (candidate.isEmpty()) continue;
            if (!QFileInfo(candidate).isAbsolute()) candidate = QStandardPaths::findExecutable(candidate);
            if (QFileInfo(candidate).isFile() && QFileInfo(candidate).isExecutable()) { python_ = QFileInfo(candidate).absoluteFilePath(); break; }
        }
        QVERIFY2(!python_.isEmpty(), "A real inference/evaluation runtime is required; missing runtime cannot count as passing QA.");
        dialogGuard_.setInterval(10);
        connect(&dialogGuard_, &QTimer::timeout, this, [this] {
            for (QWidget *widget : QApplication::topLevelWidgets())
                if (auto *dialog = qobject_cast<QMessageBox *>(widget); dialog && dialog->isVisible() && !expectingChoice_) {
                    unexpectedDialog_ = dialog->text(); dialog->reject();
                }
        });
    }
    void init()
    {
        for (const QByteArray &key : {QByteArray("HOME"), QByteArray("XDG_CONFIG_HOME"), QByteArray("XDG_DATA_HOME"),
            QByteArray("VISION_STUDIO_HOME"), QByteArray("VISION_STUDIO_DATA_DIR"), QByteArray("VISION_STUDIO_GPU_RUNTIME_DIR"),
            QByteArray("VISION_STUDIO_PYTHON"), QByteArray("VISION_STUDIO_PT_WORKER")}) {
            oldEnvironment_[key] = qgetenv(key.constData()); hadEnvironment_[key] = qEnvironmentVariableIsSet(key.constData());
        }
        area_ = std::make_unique<QTemporaryDir>(); QVERIFY(area_->isValid());
        QDir().mkpath(area_->path() + "/data"); QDir().mkpath(area_->path() + "/home");
        qputenv("HOME", (area_->path() + "/home").toUtf8());
        qputenv("XDG_CONFIG_HOME", (area_->path() + "/home/config").toUtf8());
        qputenv("XDG_DATA_HOME", (area_->path() + "/home/data").toUtf8());
        qputenv("VISION_STUDIO_HOME", QStringLiteral(VISION_PROJECT_DIR).toUtf8());
        qputenv("VISION_STUDIO_DATA_DIR", (area_->path() + "/data").toUtf8());
        qputenv("VISION_STUDIO_GPU_RUNTIME_DIR", (area_->path() + "/absent-gpu").toUtf8());
        qputenv("VISION_STUDIO_PYTHON", python_.toUtf8());
        qputenv("VISION_STUDIO_PT_WORKER", (QStringLiteral(VISION_PROJECT_DIR) + "/scripts/pt_worker.py").toUtf8());
        model_ = writeOnnx(area_->path()); createDataset(); createLoose();
        QSettings settings(area_->path() + "/data/preferences.ini", QSettings::IniFormat);
        settings.setValue("models", QStringList{model_}); settings.setValue("activeModel", model_);
        settings.setValue("computeDevice", "cpu"); settings.setValue("autoExport", false);
        settings.setValue("task", int(vision::ModelTask::YoloV8)); settings.setValue("inputSize", 64);
        settings.setValue("labels", QStringList{"football"}); settings.setValue("confidence", .3); settings.sync();
        unexpectedDialog_.clear(); expectingChoice_ = false; dialogGuard_.start(); showWindow();
        QVERIFY2(waitUntil([this] { return !window_->modelInspectionPending_ && window_->modelClassCount_ == 1; }, 30000), qPrintable(state()));
    }
    void cleanup()
    {
        if (window_ && window_->busy_) { window_->stopInference(); waitUntil([this] { return idle(); }); }
        window_.reset(); dialogGuard_.stop(); area_.reset();
        for (auto it = oldEnvironment_.cbegin(); it != oldEnvironment_.cend(); ++it)
            if (hadEnvironment_.value(it.key())) qputenv(it.key().constData(), it.value()); else qunsetenv(it.key().constData());
        oldEnvironment_.clear(); hadEnvironment_.clear();
    }
    void nativeDropAcceptsDatasetAndVideoSources_data()
    {
        QTest::addColumn<QString>("kind");
        for (const auto &kind : {"directory", "yaml", "yml", "video"})
            QTest::newRow(kind) << QString::fromLatin1(kind);
    }
    void nativeDropAcceptsDatasetAndVideoSources()
    {
        QFETCH(QString, kind);
        QString path = kind == "directory" ? datasetRoot_ : datasetYaml_;
        if (kind == "yml") {
            path = datasetRoot_ + "/explicit-source.yml";
            QVERIFY(QFile::copy(datasetYaml_, path));
        } else if (kind == "video") {
            path = area_->path() + "/inputs/本地 video.avi";
            cv::VideoWriter writer(path.toStdString(), cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 10, cv::Size(64, 48));
            QVERIFY(writer.isOpened());
            for (int frame = 0; frame < 6; ++frame) writer.write(cv::Mat(48, 64, CV_8UC3, cv::Scalar(20 + frame * 15, 80, 170)));
            writer.release();
        }
        const auto original = inputHashes(area_->path() + "/inputs");
        QSignalSpy starts(window_.get(), &MainWindow::startRequested);
        QVERIFY(dropPaths(window_->canvas_, {path}));
        if (kind == "video") {
            QTRY_COMPARE_WITH_TIMEOUT(window_->sourceKind_, vision::SourceKind::Video, 10000);
            QCOMPARE(QFileInfo(window_->streamPath_).canonicalFilePath(), QFileInfo(path).canonicalFilePath());
            QVERIFY(window_->datasetManifest_.isEmpty());
            QVERIFY(!QFileInfo::exists(sessionPath()));
        } else {
            QVERIFY2(waitUntil([this] { return idle() && window_->files_ == datasetImages_; }), qPrintable(state()));
            QVERIFY2(previewReady(datasetImages_.first()), qPrintable(state()));
            QVERIFY(window_->canvas_->result().hasGroundTruth);
            QCOMPARE(window_->canvas_->result().groundTruth.size(), 1);
            QCOMPARE(vision::DatasetSessionStore::load(sessionPath()).session.sourcePath, QFileInfo(path).absoluteFilePath());
        }
        QCOMPARE(starts.count(), 0);
        QCOMPARE(inputHashes(area_->path() + "/inputs"), original);
    }
    void looseImagesKeepMimeOrderAndOnlyLiveInTheQueue()
    {
        const QStringList incoming{looseImages_[1], looseImages_[0], looseImages_[1], looseImages_[2]};
        const QStringList expected{looseImages_[1], looseImages_[0], looseImages_[2]};
        const auto original = inputHashes(area_->path() + "/inputs");
        QSignalSpy starts(window_.get(), &MainWindow::startRequested);
        QVERIFY(dropPaths(window_->canvas_, incoming));
        QVERIFY2(waitUntil([&] { return idle() && window_->files_ == expected && window_->queue_->count() == expected.size(); }), qPrintable(state()));
        QVERIFY2(previewReady(expected.first()), qPrintable(state()));
        QVERIFY(window_->datasetManifest_.isEmpty());
        QVERIFY(window_->datasetSamples_.isEmpty());
        for (int row = 0; row < expected.size(); ++row) {
            QCOMPARE(window_->queue_->item(row)->data(Qt::UserRole).toString(), expected[row]);
            QVERIFY(window_->queue_->item(row)->data(Qt::UserRole + 11).toBool());
        }
        QVERIFY(!window_->canvas_->result().hasGroundTruth);
        QVERIFY(!window_->canvas_->result().hasEvaluationMatches);
        QVERIFY(!QFileInfo::exists(sessionPath()));
        QCOMPARE(starts.count(), 0);
        QCOMPARE(inputHashes(area_->path() + "/inputs"), original);
    }
    void datasetConflictHonorsReplaceAppendAndCancel_data()
    {
        QTest::addColumn<QString>("choice");
        QTest::newRow("replace") << QString("imageDropReplaceButton");
        QTest::newRow("append") << QString("imageDropAppendButton");
        QTest::newRow("cancel") << QString("imageDropCancelButton");
    }
    void datasetConflictHonorsReplaceAppendAndCancel()
    {
        QFETCH(QString, choice);
        QVERIFY2(importDataset(), qPrintable(state()));
        const auto manifest = window_->datasetManifest_;
        const QByteArray savedSession = readBytes(sessionPath());
        const auto original = inputHashes(area_->path() + "/inputs");
        const QString initialSource = window_->canvas_->result().source;
        QSignalSpy starts(window_.get(), &MainWindow::startRequested);
        QVERIFY2(dropChoosing(looseImages_, choice), qPrintable(state()));
        const bool append = choice == "imageDropAppendButton", cancel = choice == "imageDropCancelButton";
        const QStringList expected = append ? datasetImages_ + looseImages_ : cancel ? datasetImages_ : looseImages_;
        QVERIFY2(waitUntil([&] { return idle() && window_->files_ == expected && window_->queue_->count() == expected.size(); }), qPrintable(state()));
        if (cancel) {
            QCOMPARE(window_->canvas_->result().source, initialSource);
            QCOMPARE(window_->datasetManifest_, manifest);
        } else if (append) {
            QCOMPARE(window_->datasetManifest_, manifest);
            QCOMPARE(window_->datasetSamples_.size(), datasetImages_.size());
            for (int row = 0; row < expected.size(); ++row)
                QCOMPARE(window_->queue_->item(row)->data(Qt::UserRole + 11).toBool(), row >= datasetImages_.size());
            window_->queue_->setCurrentRow(expected.size() - 1);
            QVERIFY2(previewReady(expected.last()), qPrintable(state()));
            QVERIFY(!window_->canvas_->result().hasGroundTruth);
        } else {
            QVERIFY(window_->datasetManifest_.isEmpty());
            QVERIFY(window_->datasetSamples_.isEmpty());
            QVERIFY2(previewReady(expected.first()), qPrintable(state()));
            QVERIFY(!window_->canvas_->result().hasGroundTruth);
        }
        QCOMPARE(readBytes(sessionPath()), savedSession);
        QCOMPARE(inputHashes(area_->path() + "/inputs"), original);
        QCOMPARE(starts.count(), 0);
    }
    void nativePreviewChildDropCancelsPendingAutomaticSessionRestore_data()
    {
        QTest::addColumn<QString>("surface");
        QTest::newRow("preview-page") << QString("page");
        QTest::newRow("toolbar-label") << QString("label");
        QTest::newRow("queue-viewport") << QString("queue");
    }
    void nativePreviewChildDropCancelsPendingAutomaticSessionRestore()
    {
        QFETCH(QString, surface);
        QVERIFY2(importDataset(), qPrintable(state()));
        const QByteArray savedSession = readBytes(sessionPath());
        window_->close(); window_.reset(); showWindow();
        QSignalSpy starts(window_.get(), &MainWindow::startRequested);
        window_->restoreDatasetSession();
        auto *previewPage = window_->findChild<QWidget *>("imagePreviewPage"); QVERIFY(previewPage);
        QWidget *target = surface == "label" ? window_->canvasTitle_ : surface == "queue" ? window_->queue_->viewport() : previewPage;
        QVERIFY(target && target->isVisible());
        QVERIFY(dropPaths(target, {looseImages_.last()}));
        QVERIFY2(waitUntil([this] { return idle() && window_->files_ == QStringList{looseImages_.last()}; }), qPrintable(state()));
        QVERIFY2(previewReady(looseImages_.last()), qPrintable(state()));
        QTest::qWait(80);
        QVERIFY(window_->datasetManifest_.isEmpty());
        QCOMPARE(readBytes(sessionPath()), savedSession);
        QCOMPARE(starts.count(), 0);
    }
    void invalidMixedDropDoesNotPartiallyReplaceValidInput()
    {
        const QString corrupt = area_->path() + "/inputs/broken.png";
        writeFile(corrupt, "This is not an encoded image.\n");
        const auto inputs = inputHashes(area_->path() + "/inputs");
        drop(window_->canvas_, {QUrl::fromLocalFile(looseImages_.first()), QUrl("https://example.invalid/remote.png")});
        QTest::qWait(50);
        QVERIFY(window_->files_.isEmpty());
        QVERIFY(window_->datasetManifest_.isEmpty());
        dropPaths(window_->canvas_, {looseImages_.first(), corrupt});
        QVERIFY2(waitUntil([this] { return idle(); }), qPrintable(state()));
        QVERIFY(window_->files_.isEmpty());
        QCOMPARE(window_->queue_->count(), 0);
        QVERIFY(window_->canvas_->result().image.isNull());
        QVERIFY(!QFileInfo::exists(sessionPath()));
        QCOMPARE(inputHashes(area_->path() + "/inputs"), inputs);
    }
    void runningDetectionRejectsDropWithoutReplacingItsQueue()
    {
        QVERIFY2(importDataset(), qPrintable(state()));
        const QByteArray savedSession = readBytes(sessionPath());
        const auto inputSnapshot = inputHashes(area_->path() + "/inputs");
        QSignalSpy starts(window_.get(), &MainWindow::startRequested);
        window_->workbenchTabs_->setCurrentIndex(0);
        window_->runButton_->click();
        QVERIFY(window_->busy_);
        QCOMPARE(starts.count(), 1);
        dropPaths(window_->canvas_, {looseImages_.first()});
        QCOMPARE(window_->files_, datasetImages_);
        QCOMPARE(window_->queue_->count(), datasetImages_.size());
        QCOMPARE(readBytes(sessionPath()), savedSession);
        QVERIFY2(unexpectedDialog_.isEmpty(), qPrintable(unexpectedDialog_));
        window_->stopInference();
        QVERIFY2(waitUntil([this] { return idle(); }), qPrintable(state()));
        QCOMPARE(window_->files_, datasetImages_);
        QCOMPARE(inputHashes(area_->path() + "/inputs"), inputSnapshot);
    }
    void appendedTemporaryImagesPreserveDatasetCacheTruthAndSelectedExports()
    {
        QVERIFY2(importDataset(), qPrintable(state()));
        const auto manifest = window_->datasetManifest_;
        const QByteArray savedSession = readBytes(sessionPath());
        const auto inputs = inputHashes(area_->path() + "/inputs");
        window_->workbenchTabs_->setCurrentIndex(0);
        window_->runButton_->click();
        QVERIFY2(waitUntil([this] { return idle() && window_->completed_ == datasetImages_.size(); }), qPrintable(state()));
        QCOMPARE(window_->imageResults_.size(), datasetImages_.size());
        QVERIFY2(dropChoosing(looseImages_, "imageDropAppendButton"), qPrintable(state()));
        QVERIFY2(waitUntil([this] { return idle() && window_->files_ == datasetImages_ + looseImages_; }), qPrintable(state()));
        QCOMPARE(window_->datasetManifest_, manifest);
        QCOMPARE(window_->imageResults_.size(), datasetImages_.size());
        window_->queue_->setCurrentRow(0);
        QVERIFY2(previewReady(datasetImages_.first(), true), qPrintable(state()));
        QVERIFY(window_->lastResult_.hasGroundTruth);
        QCOMPARE(window_->lastResult_.groundTruth.size(), 1);
        QCOMPARE(window_->lastResult_.predictions.size(), 1);
        QVERIFY(window_->lastResult_.hasEvaluationMatches);
        QCOMPARE(window_->lastResult_.matchedPredictions.size(), 1);
        QString error;
        const QString originalExport = area_->path() + "/export-original";
        QVERIFY2(window_->writeResult(window_->lastResult_, originalExport, &error), qPrintable(error));
        const auto originals = QDir(originalExport).entryList({"*.json"}, QDir::Files);
        QCOMPARE(originals.size(), 1);
        const auto originalJson = readJson(originalExport + "/" + originals.first());
        QCOMPARE(originalJson.value("source").toString(), datasetImages_.first());
        QCOMPARE(originalJson.value("ground_truth").toArray().size(), 1);
        const int temporaryRow = datasetImages_.size();
        window_->queue_->setCurrentRow(temporaryRow);
        QVERIFY2(previewReady(looseImages_.first()), qPrintable(state()));
        QVERIFY(!window_->canvas_->result().hasGroundTruth);
        window_->detectionRange_->setCurrentIndex(2);
        QSignalSpy starts(window_.get(), &MainWindow::startRequested);
        window_->runButton_->click();
        QCOMPARE(starts.count(), 1);
        QCOMPARE(starts.first().first().value<vision::JobRequest>().files, QStringList{looseImages_.first()});
        QVERIFY2(waitUntil([this] { return idle() && window_->completed_ == 1; }), qPrintable(state()));
        QVERIFY2(previewReady(looseImages_.first(), true), qPrintable(state()));
        QCOMPARE(window_->lastResult_.predictions.size(), 1);
        QVERIFY(!window_->lastResult_.hasGroundTruth);
        QVERIFY(!window_->lastResult_.hasEvaluationMatches);
        QVERIFY(window_->lastResult_.groundTruth.isEmpty());
        const QString temporaryExport = area_->path() + "/export-temporary";
        QVERIFY2(window_->writeResult(window_->lastResult_, temporaryExport, &error), qPrintable(error));
        const auto temporaries = QDir(temporaryExport).entryList({"*.json"}, QDir::Files);
        QCOMPARE(temporaries.size(), 1);
        const auto temporaryJson = readJson(temporaryExport + "/" + temporaries.first());
        QCOMPARE(temporaryJson.value("source").toString(), looseImages_.first());
        QVERIFY(temporaryJson.value("ground_truth").toArray().isEmpty());
        const QImage exported(temporaryExport + "/" + QFileInfo(temporaries.first()).completeBaseName() + ".png");
        QVERIFY(!exported.isNull());
        QCOMPARE(exported.convertToFormat(QImage::Format_ARGB32), ImageCanvas::annotatedResultImage(window_->lastResult_).convertToFormat(QImage::Format_ARGB32));
        QCOMPARE(window_->datasetManifest_, manifest);
        QCOMPARE(readBytes(sessionPath()), savedSession);
        QCOMPARE(inputHashes(area_->path() + "/inputs"), inputs);
    }
    void evaluationRangesOnlyIncludeOriginalLabeledSamples_data()
    {
        QTest::addColumn<int>("range"); QTest::addColumn<int>("selection");
        QTest::newRow("whole-dataset") << 0 << 2;
        QTest::newRow("dataset-tail") << 1 << 1;
        QTest::newRow("one-original-sample") << 2 << 2;
    }
    void evaluationRangesOnlyIncludeOriginalLabeledSamples()
    {
        QFETCH(int, range); QFETCH(int, selection);
        QVERIFY2(importDataset(), qPrintable(state()));
        QVERIFY2(dropChoosing(looseImages_, "imageDropAppendButton"), qPrintable(state()));
        QVERIFY2(waitUntil([this] { return idle() && window_->queue_->count() == datasetImages_.size() + looseImages_.size(); }), qPrintable(state()));
        const auto inputs = inputHashes(area_->path() + "/inputs");
        const QByteArray savedSession = readBytes(sessionPath());
        const auto expected = range == 0 ? datasetImages_ : range == 1 ? datasetImages_.mid(selection) : QStringList{datasetImages_[selection]};
        auto *controls = window_->evaluation_->controlsWidget();
        controls->findChild<QDoubleSpinBox *>("evaluationScanStartSpin")->setValue(.7);
        controls->findChild<QDoubleSpinBox *>("evaluationScanEndSpin")->setValue(.9);
        controls->findChild<QDoubleSpinBox *>("evaluationScanStepSpin")->setValue(.2);
        window_->workbenchTabs_->setCurrentIndex(1);
        window_->detectionRange_->setCurrentIndex(range);
        window_->queue_->setCurrentRow(selection);
        auto *start = window_->findChild<QPushButton *>("evaluationStartButton");
        QVERIFY(start && start->isVisible() && start->isEnabled());
        QSignalSpy detections(window_.get(), &MainWindow::startRequested);
        QTest::mouseClick(start, Qt::LeftButton);
        QVERIFY(window_->busy_ && window_->evaluationWorker_);
        QVERIFY2(waitUntil([this] { return idle() && !window_->evaluationReport_.isEmpty(); }, 60000), qPrintable(state()));
        const auto report = window_->evaluationReport_;
        QStringList actual;
        for (const auto &path : report.value("request").toObject().value("sample_images").toArray()) actual.append(path.toString());
        QCOMPARE(actual, expected);
        QCOMPARE(report.value("completed_images").toInt(), expected.size());
        QCOMPARE(report.value("manifest").toObject().value("samples").toArray().size(), datasetImages_.size());
        QCOMPARE(report.value("metrics").toObject().value("no_ball_images").toInt(), 0);
        QCOMPARE(report.value("metrics").toObject().value("TP").toInt(), expected.size());
        QCOMPARE(window_->evaluationSampleIndices_.size(), expected.size());
        for (const auto &temporary : looseImages_) QVERIFY(!window_->evaluationSampleIndices_.contains(temporary));
        QCOMPARE(detections.count(), 0);
        QCOMPARE(readBytes(sessionPath()), savedSession);
        QCOMPARE(inputHashes(area_->path() + "/inputs"), inputs);
        qInfo() << "temporary_drop_evaluation_scope_evidence" << "range" << range << "selection" << selection
                << "labeled_samples" << expected.size() << "temporary_excluded" << looseImages_.size();
    }
    void currentAndTailOfTemporaryItemDoNotStartEvaluation()
    {
        QVERIFY2(importDataset(), qPrintable(state()));
        QVERIFY2(dropChoosing(looseImages_, "imageDropAppendButton"), qPrintable(state()));
        QVERIFY2(waitUntil([this] { return idle() && window_->queue_->count() == datasetImages_.size() + looseImages_.size(); }), qPrintable(state()));
        window_->workbenchTabs_->setCurrentIndex(1);
        window_->queue_->setCurrentRow(datasetImages_.size());
        auto *start = window_->findChild<QPushButton *>("evaluationStartButton"); QVERIFY(start);
        const QByteArray savedSession = readBytes(sessionPath());
        for (int range : {1, 2}) {
            window_->detectionRange_->setCurrentIndex(range);
            QVERIFY(!start->isEnabled());
            QVERIFY(window_->evaluationRangeHint_->isVisible());
            QVERIFY(window_->evaluationRangeHint_->text().contains("临时") || window_->evaluationRangeHint_->text().contains("标注"));
            start->click();
            QVERIFY(idle());
            QVERIFY(window_->evaluationReport_.isEmpty());
        }
        QCOMPARE(readBytes(sessionPath()), savedSession);
    }
    void largeTemporaryDropStaysResponsiveAndDiscardsStaleImageReplies()
    {
        createLoose(1500);
        const auto inputs = inputHashes(area_->path() + "/inputs");
        QTimer heartbeat; QElapsedTimer elapsed; elapsed.start();
        qint64 last = 0, gap = 0; int ticks = 0;
        connect(&heartbeat, &QTimer::timeout, this, [&] { const qint64 now = elapsed.elapsed(); gap = qMax(gap, now - last); last = now; ++ticks; });
        heartbeat.start(10);
        QVERIFY(dropPaths(window_->canvas_, looseImages_));
        QVERIFY2(waitUntil([this] { return idle() && window_->files_ == looseImages_ && window_->queue_->count() == looseImages_.size(); }), qPrintable(state()));
        heartbeat.stop();
        QVERIFY(ticks > 0);
        QVERIFY2(gap < 1000, qPrintable(QString("Large drop blocked event handling for %1 ms").arg(gap)));
        auto *queue = static_cast<LazyImageQueue *>(window_->queue_);
        QVERIFY(queue->cachedThumbnailCount() <= 128);
        for (int row : {0, 700, 3, 1499, 2, 1499}) window_->queue_->setCurrentRow(row);
        QVERIFY2(previewReady(looseImages_.last()), qPrintable(state()));
        QCOMPARE(window_->canvas_->result().image.size(), QSize(73, 51));
        QCOMPARE(window_->canvas_->result().image.pixelColor(0, 0), QColor(10, 210, 190));
        QVERIFY(!window_->canvas_->result().hasGroundTruth);
        window_->queue_->setCurrentRow(0);
        const auto buttons = window_->findChildren<QPushButton *>();
        auto clear = std::find_if(buttons.begin(), buttons.end(), [](auto *button) { return button->text() == "清空"; });
        QVERIFY(clear != buttons.end());
        (*clear)->click();
        QTest::qWait(150);
        QVERIFY(window_->files_.isEmpty());
        QCOMPARE(window_->queue_->count(), 0);
        QVERIFY(window_->canvas_->result().image.isNull());
        QTRY_COMPARE_WITH_TIMEOUT(queue->activeThumbnailJobs(), 0, 10000);
        QVERIFY(!QFileInfo::exists(sessionPath()));
        QCOMPARE(inputHashes(area_->path() + "/inputs"), inputs);
        qInfo() << "temporary_drop_lazy_queue_evidence" << "images" << 1500 << "heartbeat_ticks" << ticks
                << "max_gap_ms" << gap << "cached_thumbnails" << queue->cachedThumbnailCount();
    }
    void stoppingPartialTemporaryAppendRestoresOriginalQueueAndSession()
    {
        QVERIFY2(importDataset(), qPrintable(state()));
        const auto manifest = window_->datasetManifest_;
        const QByteArray savedSession = readBytes(sessionPath());
        createLoose(1500);
        QTimer stopper; int partial = 0; qint64 duration = -1;
        connect(&stopper, &QTimer::timeout, this, [&] {
            const int count = window_->queue_->count();
            if (!window_->queuePopulating_ || count <= datasetImages_.size() || count >= datasetImages_.size() + looseImages_.size()) return;
            partial = count; stopper.stop();
            QElapsedTimer elapsed; elapsed.start(); window_->stopInference(); duration = elapsed.elapsed();
        });
        stopper.start(0);
        QVERIFY2(dropChoosing(looseImages_, "imageDropAppendButton"), qPrintable(state()));
        QVERIFY2(waitUntil([&] { return duration >= 0 && idle(); }), qPrintable(state()));
        stopper.stop();
        QVERIFY(partial > datasetImages_.size() && partial < datasetImages_.size() + looseImages_.size());
        QVERIFY(duration < 500);
        QCOMPARE(window_->files_, datasetImages_);
        QCOMPARE(window_->queue_->count(), datasetImages_.size());
        QCOMPARE(window_->datasetManifest_, manifest);
        QCOMPARE(readBytes(sessionPath()), savedSession);
        QTest::qWait(100);
        QCOMPARE(window_->files_, datasetImages_);
        QCOMPARE(readBytes(sessionPath()), savedSession);
        window_->queue_->setCurrentRow(0);
        QVERIFY2(previewReady(datasetImages_.first()), qPrintable(state()));
        QVERIFY(window_->canvas_->result().hasGroundTruth);
        qInfo() << "temporary_append_stop_evidence" << "partial_queue" << partial << "original_count" << datasetImages_.size()
                << "stop_return_ms" << duration;
    }
};
QTEST_MAIN(InputDropUiTests)
#include "input_drop_ui_tests.moc"
