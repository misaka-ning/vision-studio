#include "ui/imagecanvas.h"
#include "ui/lazyimagequeue.h"
#include "ui/mainwindow.h"
#include "ui/modelviewer.h"

#include <QApplication>
#include <QAbstractItemModel>
#include <QCheckBox>
#include <QComboBox>
#include <QDir>
#include <QElapsedTimer>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QImageWriter>
#include <QJsonArray>
#include <QJsonDocument>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QMessageBox>
#include <QPointer>
#include <QPushButton>
#include <QSettings>
#include <QSignalSpy>
#include <QTemporaryDir>
#include <QTimer>
#include <QtTest>
#include <algorithm>
#include <cmath>
#include <memory>

namespace
{
QPushButton *findButton(QWidget *parent, const QString &title)
{
    for (auto *button : parent->findChildren<QPushButton *>())
        if (button->text() == title)
            return button;
    return nullptr;
}

QImage texturedImage(const QSize &size)
{
    QImage image(size, QImage::Format_RGB888);
    quint32 state = 0xA53189F1;
    for (int y = 0; y < size.height(); ++y)
        for (int x = 0; x < size.width() * 3; ++x)
        {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            image.scanLine(y)[x] = uchar(state);
        }
    return image;
}

QByteArray contents(const QString &path)
{
    QFile file(path);
    return file.open(QIODevice::ReadOnly) ? file.readAll() : QByteArray();
}
} // namespace

class BatchResponsivenessTests final : public QObject
{
    Q_OBJECT
    std::unique_ptr<QTemporaryDir> fixture_;
    std::unique_ptr<MainWindow> window_;
    std::unique_ptr<QTimer> dialogGuard_;
    QHash<QByteArray, QByteArray> oldEnvironment_;
    QHash<QByteArray, bool> hadEnvironment_;
    QString unexpectedDialog_;

    QStringList prepareFolder(const QSize &size)
    {
        const QString folder = fixture_->path() + QStringLiteral("/batch");
        if (!QDir().mkpath(folder))
            return {};
        const QString first = folder + QStringLiteral("/01.png");
        QImageWriter writer(first, "PNG");
        writer.setCompression(11);
        if (!writer.write(texturedImage(size)))
            return {};
        writer.device()->close();
        QStringList files{first};
        for (int row = 2; row <= 3; ++row)
        {
            const QString file = folder + QStringLiteral("/%1.png").arg(row, 2, 10, QChar('0'));
            if (!QFile::copy(first, file))
                return {};
            files.append(file);
        }
        window_->clearImageResults();
        window_->files_.clear();
        window_->queue_->clear();
        window_->addFiles(files);
        return files;
    }

    QListWidget *queue() const
    {
        for (auto *list : window_->findChildren<QListWidget *>())
            if (list->viewMode() == QListView::IconMode)
                return list;
        return nullptr;
    }

    ImageCanvas *canvas() const
    {
        auto *page = window_->findChild<QWidget *>(QStringLiteral("workbenchPage"));
        return page ? page->findChild<ImageCanvas *>() : nullptr;
    }

  private slots:
    void initTestCase()
    {
        QApplication::setStyle(QStringLiteral("Fusion"));
        QApplication::setFont(QFont(QStringLiteral("Noto Sans CJK SC"), 10));
        QApplication::setAttribute(Qt::AA_DontUseNativeDialogs);
        QCoreApplication::setApplicationVersion(QStringLiteral("2.0.0"));
    }

    void init()
    {
        for (const QByteArray &name : {QByteArray("VISION_STUDIO_HOME"), QByteArray("VISION_STUDIO_DATA_DIR"),
                                       QByteArray("VISION_STUDIO_GPU_RUNTIME_DIR")})
        {
            oldEnvironment_.insert(name, qgetenv(name.constData()));
            hadEnvironment_.insert(name, qEnvironmentVariableIsSet(name.constData()));
        }
        fixture_ = std::make_unique<QTemporaryDir>();
        QVERIFY(fixture_->isValid());
        for (const QString &directory : {QStringLiteral("models"), QStringLiteral("assets"),
                                         QStringLiteral("scripts"), QStringLiteral("output")})
            QVERIFY(QDir().mkpath(fixture_->path() + "/" + directory));
        const QString project = QStringLiteral(VISION_PROJECT_DIR);
        for (const QString &path : {QStringLiteral("models/yolov5n.onnx"), QStringLiteral("assets/bus.jpg"),
                                    QStringLiteral("scripts/netron_server.py"), QStringLiteral("scripts/evaluation_worker.py"),
                                    QStringLiteral("scripts/evaluation_data.py"), QStringLiteral("scripts/evaluation_metrics.py"),
                                    QStringLiteral("scripts/evaluation_reporting.py")})
            QVERIFY(QFile::copy(project + "/" + path, fixture_->path() + "/" + path));
        qputenv("VISION_STUDIO_HOME", fixture_->path().toUtf8());
        qputenv("VISION_STUDIO_DATA_DIR", (fixture_->path() + QStringLiteral("/output")).toUtf8());
        qputenv("VISION_STUDIO_GPU_RUNTIME_DIR", (fixture_->path() + QStringLiteral("/absent-gpu")).toUtf8());
        QSettings settings(fixture_->path() + QStringLiteral("/output/preferences.ini"), QSettings::IniFormat);
        settings.setValue(QStringLiteral("computeDevice"), QStringLiteral("cpu"));
        settings.setValue(QStringLiteral("confidence"), 1.0);
        settings.setValue(QStringLiteral("autoExport"), true);
        settings.sync();
        unexpectedDialog_.clear();
        dialogGuard_ = std::make_unique<QTimer>();
        connect(dialogGuard_.get(), &QTimer::timeout, this,
            [this]
            {
                for (QWidget *widget : QApplication::topLevelWidgets())
                    if (auto *dialog = qobject_cast<QMessageBox *>(widget); dialog && dialog->isVisible())
                    {
                        unexpectedDialog_ = dialog->text();
                        dialog->accept();
                    }
            });
        dialogGuard_->start(20);
        window_ = std::make_unique<MainWindow>();
        window_->resize(1260, 820);
        window_->show();
        auto *viewer = window_->findChild<ModelViewer *>(QStringLiteral("modelStructureViewer"));
        QVERIFY(viewer);
        QTRY_VERIFY_WITH_TIMEOUT(viewer->state() != ModelViewer::State::Loading, 30000);
        QVERIFY2(viewer->state() == ModelViewer::State::Ready, qPrintable(viewer->errorString()));
    }

    void cleanup()
    {
        window_.reset();
        dialogGuard_.reset();
        fixture_.reset();
        for (auto iterator = oldEnvironment_.cbegin(); iterator != oldEnvironment_.cend(); ++iterator)
            if (hadEnvironment_.value(iterator.key()))
                qputenv(iterator.key().constData(), iterator.value());
            else
                qunsetenv(iterator.key().constData());
        oldEnvironment_.clear();
        hadEnvironment_.clear();
    }

    void largeQueueLoadsOnlyVisibleThumbnailsAndCancelsStalePreviews()
    {
        constexpr int count = 2000;
        const QString directory = fixture_->path() + "/large-lazy-queue";
        QVERIFY(QDir().mkpath(directory + "/images/val")); QVERIFY(QDir().mkpath(directory + "/labels/val"));
        QStringList files;
        for (int index = 0; index < count; ++index) {
            QImage image(48 + index % 3, 32 + index % 5, QImage::Format_RGB888);
            image.fill(QColor((index * 37) % 255, (index * 71 + 20) % 255, (index * 19 + 90) % 255));
            const QString stem = QString("%1").arg(index, 4, 10, QChar('0'));
            const QString path = directory + "/images/val/" + stem + ".png";
            QVERIFY(image.save(path)); files.append(path);
            QFile label(directory + "/labels/val/" + stem + ".txt"); QVERIFY(label.open(QIODevice::WriteOnly)); label.close();
        }
        const QString yaml = directory + "/data.yaml"; QFile configuration(yaml); QVERIFY(configuration.open(QIODevice::WriteOnly));
        const QByteArray yamlContents("path: .\nval: images/val\nnames: [queue_fixture]\n");
        QCOMPARE(configuration.write(yamlContents), qint64(yamlContents.size())); configuration.close();
        auto *images = window_->findChild<LazyImageQueue *>("inputQueue"); auto *view = canvas();
        auto *clear = findButton(window_.get(), "清空"); auto *exportButton = findButton(window_.get(), "导出结果");
        auto *color = window_->findChild<QComboBox *>("inputColorMode");
        QVERIFY(images && view && clear && exportButton && color);
        int ticks = 0; qint64 maximumGap = 0, previous = 0; QElapsedTimer elapsed; elapsed.start(); QTimer heartbeat;
        connect(&heartbeat, &QTimer::timeout, this, [&] {
            const qint64 now = elapsed.elapsed(); maximumGap = std::max(maximumGap, now - previous); previous = now; ++ticks;
        }); heartbeat.start(10);
        QSignalSpy starts(window_.get(), &MainWindow::startRequested);
        QObject populationObserver; bool probeScheduled = false, sawPartialPopulation = false, controlsLocked = false; int partialCount = -1;
        const auto partialProbe = connect(images->model(), &QAbstractItemModel::rowsInserted, &populationObserver,
            [&](const QModelIndex &, int, int) {
                if (probeScheduled || !window_->queuePopulating_) return;
                probeScheduled = true;
                QTimer::singleShot(0, &populationObserver, [&] {
                    partialCount = images->count();
                    sawPartialPopulation = window_->queuePopulating_ && partialCount > 0 && partialCount < count;
                    auto *datasetButton = window_->findChild<QPushButton *>("openYoloDatasetButton");
                    controlsLocked = window_->busy_ && !clear->isEnabled() && !window_->runButton_->isEnabled() &&
                                     datasetButton && !datasetButton->isEnabled();
                });
            });
        window_->loadDataset(yaml, "val");
        QVERIFY(window_->busy_); QVERIFY(!clear->isEnabled());
        QTRY_VERIFY_WITH_TIMEOUT(!window_->busy_ && !window_->evaluationWorker_ && !window_->queuePopulating_, 30000);
        disconnect(partialProbe);
        QVERIFY2(images->count() == count, qPrintable(window_->statusLabel_->toolTip() + " | " + window_->lastError_));
        QCOMPARE(images->count(), count); for (int i = 0; i < files.size(); ++i) QCOMPARE(QDir::fromNativeSeparators(window_->files_[i]), QDir::fromNativeSeparators(files[i]));
        QVERIFY(probeScheduled); QVERIFY(sawPartialPopulation); QVERIFY(controlsLocked);
        QCOMPARE(window_->datasetManifest_.value("samples").toArray().size(), count);
        for (int row : {0, 1000, 1999}) QCOMPARE(QDir::fromNativeSeparators(images->item(row)->data(Qt::UserRole).toString()), QDir::fromNativeSeparators(files[row]));
        images->setCurrentRow(0); images->scrollToItem(images->item(0));
        QTRY_COMPARE_WITH_TIMEOUT(view->result().source, files[0], 10000);
        QTRY_VERIFY_WITH_TIMEOUT(!view->result().image.isNull(), 10000);
        QTRY_VERIFY_WITH_TIMEOUT(images->cachedThumbnailCount() > 0, 10000);
        QVERIFY(images->cachedThumbnailCount() <= 128); QVERIFY(images->activeThumbnailJobs() <= 4);
        int loaded = 0;
        for (int row = 0; row < count; ++row)
            if (!qvariant_cast<QImage>(images->item(row)->data(Qt::UserRole + 1)).isNull()) ++loaded;
        QVERIFY(loaded > 0 && loaded <= 128); QVERIFY(loaded < count);
        for (int row : {1999, 750, 25, 1300, 3, 1998}) images->setCurrentRow(row);
        QTRY_COMPARE_WITH_TIMEOUT(view->result().source, files[1998], 10000);
        QTRY_COMPARE_WITH_TIMEOUT(view->result().originalImage.convertToFormat(QImage::Format_RGB888),
                                  QImage(files[1998]).convertToFormat(QImage::Format_RGB888), 10000);
        QTest::qWait(120); QCOMPARE(view->result().source, files[1998]);
        color->setCurrentIndex(color->findData("grayscale"));
        QTRY_VERIFY_WITH_TIMEOUT(view->result().source == files[1998] && view->result().image.isGrayscale() &&
                                  !view->result().image.isNull(), 10000);
        QCOMPARE(view->result().image.format(), QImage::Format_Grayscale8);
        const QColor originalPixel = QImage(files[1998]).pixelColor(0, 0), grayPixel = view->result().image.pixelColor(0, 0);
        QCOMPARE(grayPixel.red(), grayPixel.green()); QCOMPARE(grayPixel.green(), grayPixel.blue());
        QVERIFY(std::abs(grayPixel.red() - (.299 * originalPixel.red() + .587 * originalPixel.green() + .114 * originalPixel.blue())) <= 1.0);
        const QString capture = qEnvironmentVariable("VISION_STUDIO_UI_CAPTURE_DIR");
        if (!capture.isEmpty()) { QVERIFY(QDir().mkpath(capture)); window_->saveScreenshot(capture + "/lazy-queue-grayscale-preview.png"); }
        images->setCurrentRow(17); images->setCurrentRow(500); QTest::mouseClick(clear, Qt::LeftButton);
        QTRY_COMPARE(images->count(), 0); QTRY_VERIFY(view->result().image.isNull());
        QTRY_COMPARE_WITH_TIMEOUT(images->activeThumbnailJobs(), 0, 10000);
        QTest::qWait(120); QVERIFY(view->result().image.isNull()); QVERIFY(!exportButton->isEnabled());
        QCOMPARE(images->cachedThumbnailCount(), 0); QCOMPARE(starts.count(), 0);
        // Reuse the same real dataset and cancel after an actual partial chunk.
        bool stopScheduled = false, stoppedPopulation = false; int countAtStop = -1; qint64 stopDispatchMs = -1;
        const auto cancelProbe = connect(images->model(), &QAbstractItemModel::rowsInserted, &populationObserver,
            [&](const QModelIndex &, int, int) {
                if (stopScheduled || !window_->queuePopulating_) return;
                stopScheduled = true;
                QTimer::singleShot(0, &populationObserver, [&] {
                    countAtStop = images->count();
                    if (!window_->queuePopulating_ || countAtStop <= 0 || countAtStop >= count) return;
                    stoppedPopulation = true; QElapsedTimer stopTime; stopTime.start();
                    QTest::mouseClick(window_->stopButton_, Qt::LeftButton); stopDispatchMs = stopTime.elapsed();
                });
            });
        window_->loadDataset(yaml, "val");
        QTRY_VERIFY_WITH_TIMEOUT(stoppedPopulation && !window_->busy_ && !window_->evaluationWorker_ && !window_->queuePopulating_, 30000);
        disconnect(cancelProbe);
        QVERIFY(stopDispatchMs >= 0 && stopDispatchMs < 250);
        QCOMPARE(images->count(), 0); QVERIFY(window_->files_.isEmpty()); QVERIFY(view->result().image.isNull());
        QTest::qWait(120); QCOMPARE(images->count(), 0); QVERIFY(window_->files_.isEmpty());
        heartbeat.stop(); QVERIFY(ticks >= 10); QVERIFY2(maximumGap < 500, qPrintable(QString("GUI heartbeat gap: %1 ms").arg(maximumGap)));
        qInfo().noquote() << "lazy_queue_evidence" << "standard_yolo_dataset" << count << "partial_count" << partialCount
                         << "population_controls_locked" << controlsLocked << "stop_at_count" << countAtStop << "stop_dispatch_ms" << stopDispatchMs
                         << "initial_loaded_thumbnails" << loaded
                         << "heartbeat_ticks" << ticks << "maximum_gap_ms" << maximumGap << "rapid_selection_final_index" << 1998
                         << "clear_cancelled_jobs" << images->activeThumbnailJobs();
    }

    void highResolutionAutomaticExportCanStopWithoutBlockingUi_data()
    {
        QTest::addColumn<QSize>("imageSize");
        QTest::newRow("1920x1080-rgb-random") << QSize(1920, 1080);
        QTest::newRow("3840x2160-rgb-random") << QSize(3840, 2160);
    }

    void highResolutionAutomaticExportCanStopWithoutBlockingUi()
    {
        QFETCH(QSize, imageSize);
        const auto files = prepareFolder(imageSize);
        QCOMPARE(files.size(), 3);
        auto *images = queue();
        auto *view = canvas();
        auto *run = findButton(window_.get(), QStringLiteral("开始检测"));
        auto *stop = findButton(window_.get(), QStringLiteral("停止运行"));
        auto *exportButton = findButton(window_.get(), QStringLiteral("导出结果"));
        QVERIFY(images && view && run && stop && exportButton);
        images->setCurrentRow(2);
        QVector<vision::InferenceResult> accepted;
        bool scheduled = false, stopped = false;
        int acceptedAtStop = -1, ticks = 0, ticksAtStop = 0;
        qint64 maxGap = 0, lastTick = 0, dispatchDelay = -1;
        QElapsedTimer heartbeatTime, deliveryTime, stopTime;
        QTimer heartbeat;
        heartbeatTime.start();
        connect(&heartbeat, &QTimer::timeout, this,
            [&] { const qint64 now = heartbeatTime.elapsed(); maxGap = std::max(maxGap, now - lastTick);
                  lastTick = now; ++ticks; });
        heartbeat.start(10);
        QObject observer;
        const auto observation = connect(images, &QListWidget::currentRowChanged, &observer,
            [&](int row)
            {
                if (row < 0 || row >= files.size() || view->result().inferenceMs <= 0 ||
                    view->result().source != files[row])
                    return;
                accepted.append(view->result());
                if (!scheduled)
                {
                    scheduled = true;
                    deliveryTime.start();
                    QTimer::singleShot(30, &observer,
                        [&]
                        {
                            dispatchDelay = deliveryTime.elapsed();
                            acceptedAtStop = accepted.size();
                            ticksAtStop = ticks;
                            stopTime.start();
                            stopped = true;
                            QTest::mouseClick(stop, Qt::LeftButton);
                        });
                }
            });
        QSignalSpy starts(window_.get(), &MainWindow::startRequested);
        QTest::mouseClick(run, Qt::LeftButton);
        QTRY_VERIFY_WITH_TIMEOUT(stopped && run->isVisible(), 45000);
        const qint64 observedStopElapsed = stopTime.elapsed();
        QVERIFY(observedStopElapsed < 2000);
        QVERIFY(dispatchDelay < 500);
        QVERIFY(maxGap < 500);
        QVERIFY(accepted.size() > 0 && accepted.size() < files.size());
        QCOMPARE(accepted.size(), acceptedAtStop);
        QCOMPARE(starts.size(), 1);
        QVERIFY2(unexpectedDialog_.isEmpty(), qPrintable(unexpectedDialog_));
        // Saving the accepted frame continues in the background after cancellation.
        QTRY_VERIFY_WITH_TIMEOUT(run->isEnabled(), 15000);
        QVERIFY(ticks > ticksAtStop);
        heartbeat.stop();
        disconnect(observation);
        const QByteArray savedHistory = contents(fixture_->path() + QStringLiteral("/output/history.json"));
        QCOMPARE(QJsonDocument::fromJson(savedHistory).array().size(), accepted.size());
        const QDir exported(fixture_->path() + QStringLiteral("/output/results"));
        QCOMPARE(exported.entryList({QStringLiteral("*.json")}, QDir::Files).size(), accepted.size());
        QCOMPARE(exported.entryList({QStringLiteral("*.png")}, QDir::Files).size(), accepted.size());
        QCOMPARE(exported.entryList({QStringLiteral("*.csv")}, QDir::Files).size(), accepted.size());
        for (const auto &result : accepted)
        {
            const auto metadataFiles = exported.entryList({QFileInfo(result.source).completeBaseName() +
                                                            QStringLiteral("-*.json")}, QDir::Files);
            QCOMPARE(metadataFiles.size(), 1);
            const QString base = metadataFiles[0].left(metadataFiles[0].size() - 5);
            const auto metadata = QJsonDocument::fromJson(contents(exported.filePath(metadataFiles[0]))).object();
            QCOMPARE(metadata.value(QStringLiteral("source")).toString(), result.source);
            QCOMPARE(metadata.value(QStringLiteral("actual_device")).toString(), QStringLiteral("cpu"));
            QCOMPARE(metadata.value(QStringLiteral("version")).toString(), QStringLiteral("2.0.0"));
            QCOMPARE(QImage(exported.filePath(base + QStringLiteral(".png"))).size(), imageSize);
            QVERIFY(!contents(exported.filePath(base + QStringLiteral(".csv"))).isEmpty());
            QVERIFY(QFile::remove(result.source));
        }
        images->setCurrentRow(2); // This image was not processed.
        QTRY_COMPARE(view->result().source, files[2]);
        QTRY_COMPARE(view->result().inferenceMs, 0.0);
        QVERIFY(!exportButton->isEnabled());
        for (int row = 0; row < accepted.size(); ++row)
        {
            images->setCurrentRow(row);
            QTRY_COMPARE(view->result().source, accepted[row].source);
            QTRY_COMPARE(view->result().image, accepted[row].image);
            QTRY_COMPARE(view->result().originalImage, accepted[row].originalImage);
            QTRY_COMPARE(view->result().inferenceMs, accepted[row].inferenceMs);
            QVERIFY(exportButton->isEnabled());
        }
        QCOMPARE(contents(fixture_->path() + QStringLiteral("/output/history.json")), savedHistory);
        const QJsonObject evidence{{"width", imageSize.width()}, {"height", imageSize.height()},
            {"backend", accepted.first().backend}, {"device", "cpu"}, {"automatic_export", true},
            {"heartbeat_ticks", ticks}, {"maximum_heartbeat_gap_ms", maxGap},
            {"scheduled_stop_dispatch_ms", dispatchDelay}, {"observed_stop_elapsed_ms", observedStopElapsed},
            {"accepted_at_stop", acceptedAtStop}, {"accepted_after_stop", accepted.size()}, {"total_images", files.size()},
            {"cache_survives_removed_sources", true}, {"complete_exports", true}};
        qInfo().noquote() << "batch_responsiveness_evidence:" << QJsonDocument(evidence).toJson(QJsonDocument::Compact);
    }

    void automaticExportDiskFailureStopsWithoutDeadlockAndBatchCanRecover()
    {
        const auto files = prepareFolder(QSize(320, 240));
        QCOMPARE(files.size(), 3);
        auto *images = queue();
        auto *view = canvas();
        auto *run = findButton(window_.get(), QStringLiteral("开始检测"));
        auto *exportButton = findButton(window_.get(), QStringLiteral("导出结果"));
        auto *status = window_->findChild<QLabel *>(QStringLiteral("statusText"));
        QVERIFY(images && view && run && exportButton && status);
        images->setCurrentRow(2);
        bool blocked = false;
        vision::InferenceResult accepted;
        QObject observer;
        const QString destination = fixture_->path() + QStringLiteral("/output/results");
        const auto observation = connect(images, &QListWidget::currentRowChanged, &observer,
            [&](int row)
            {
                if (row != 0 || blocked || view->result().inferenceMs <= 0)
                    return;
                accepted = view->result();
                // Start passes the directory check; replace it only after the
                // actual first result, immediately before its async export begins.
                if (!QDir().rmdir(destination))
                    return;
                QFile blocker(destination);
                blocked = blocker.open(QIODevice::WriteOnly) && blocker.write("owned-test-blocker") == 18;
                blocker.close();
            });
        QSignalSpy starts(window_.get(), &MainWindow::startRequested);
        QTest::mouseClick(run, Qt::LeftButton);
        QTRY_VERIFY_WITH_TIMEOUT(run->isVisible() && run->isEnabled(), 15000);
        disconnect(observation);
        QVERIFY(blocked);
        QCOMPARE(accepted.source, files.first());
        QVERIFY(status->toolTip().contains(QStringLiteral("自动导出失败")));
        QCOMPARE(QJsonDocument::fromJson(contents(fixture_->path() + QStringLiteral("/output/history.json"))).array().size(), 1);
        QCOMPARE(contents(destination), QByteArray("owned-test-blocker"));
        images->setCurrentRow(2);
        QVERIFY(!exportButton->isEnabled());
        images->setCurrentRow(0);
        QTRY_COMPARE(view->result().image, accepted.image);
        QVERIFY(exportButton->isEnabled());
        QCheckBox *automatic = nullptr;
        for (auto *check : window_->findChildren<QCheckBox *>())
            if (check->text().contains(QStringLiteral("自动保存")))
                automatic = check;
        QVERIFY(automatic);
        automatic->setChecked(false);
        QTest::mouseClick(run, Qt::LeftButton);
        QTRY_VERIFY_WITH_TIMEOUT(run->isVisible() && run->isEnabled(), 15000);
        QCOMPARE(starts.size(), 2);
        QCOMPARE(QJsonDocument::fromJson(contents(fixture_->path() + QStringLiteral("/output/history.json"))).array().size(), 4);
        for (int row : {0, 1, 2})
        {
            images->setCurrentRow(row);
            QTRY_COMPARE(view->result().source, files[row]);
            QVERIFY(view->result().inferenceMs > 0);
            QVERIFY(exportButton->isEnabled());
        }
        QCOMPARE(contents(destination), QByteArray("owned-test-blocker"));
        QVERIFY2(unexpectedDialog_.isEmpty(), qPrintable(unexpectedDialog_));
        qInfo().noquote() << "batch_disk_failure_evidence: export_error_reported=true no_deadlock=true completed_result_retained=true recovered_images=3";
    }
};

QTEST_MAIN(BatchResponsivenessTests)
#include "batch_responsiveness_tests.moc"
