// SPDX-License-Identifier: AGPL-3.0-only
#include "app/detectioncontroller.h"
#include "app/historyrepository.h"
#include "app/asyncthreadretirement.h"
#include "thread_retirement_fixtures.h"
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QSignalSpy>
#include <QTemporaryDir>
#include <QThread>
#include <QTimer>
#include <QtTest>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

class DetectionControllerTests : public QObject
{
    Q_OBJECT
  private slots:
    void fullNativeTailGatesReadinessAndEarlyDestruction_data()
    {
        QTest::addColumn<bool>("earlyOwnerDestruction");
        QTest::newRow("normal-shutdown-barrier") << false;
        QTest::newRow("early-owner-destruction") << true;
    }
    void fullNativeTailGatesReadinessAndEarlyDestruction()
    {
        QFETCH(bool, earlyOwnerDestruction);
        const int originalCount = vision::app::AsyncThreadRetirement::activeCount();
        auto controller = std::make_unique<vision::app::DetectionController>();
        const auto threads = vision::app::AsyncThreadRetirement::watchedThreads(controller.get());
        QCOMPARE(threads.size(), 1);
        auto *thread = threads.first(); QPointer<QThread> guardedThread(thread);
        bool destroyedReenteredShutdown = false;
        connect(thread, &QObject::destroyed, this, [&] {
            if (!controller) return;
            controller->requestShutdown();
            destroyedReenteredShutdown = true;
        }, Qt::DirectConnection);
        const auto state = std::make_shared<vision::test::ThreadTailState>();
        vision::test::ThreadTailRelease release{state};
        auto *marker = new QObject; marker->moveToThread(thread);
        connect(thread, &QThread::finished, marker, &QObject::deleteLater);
        std::atomic_bool prepared{false};
        QMetaObject::invokeMethod(marker, [state, &prepared] {
            vision::test::establishThreadLocalTail(state); prepared.store(true);
        }, Qt::QueuedConnection);
        QTRY_VERIFY_WITH_TIMEOUT(prepared.load(), 3000);
        int beats = 0;
        QTimer heartbeat; heartbeat.setInterval(0);
        connect(&heartbeat, &QTimer::timeout, this, [&] { ++beats; }); heartbeat.start();
        QElapsedTimer elapsed; elapsed.start(); controller->requestShutdown();
        QVERIFY(elapsed.elapsed() < 100);
        QTRY_VERIFY_WITH_TIMEOUT(state->entered.load(), 3000);
        QVERIFY(thread->isFinished()); QVERIFY(!controller->isShutdownReady());
        const int before = beats; QTest::qWait(30); QVERIFY(beats > before);
        if (earlyOwnerDestruction) {
            elapsed.restart(); controller.reset(); QVERIFY(elapsed.elapsed() < 100);
            QVERIFY(guardedThread); QCOMPARE(vision::app::AsyncThreadRetirement::activeCount(), originalCount + 1);
        }
        state->resume();
        if (controller) {
            QTRY_VERIFY_WITH_TIMEOUT(controller->isShutdownReady(), 3000);
            QVERIFY(destroyedReenteredShutdown);
            elapsed.restart(); controller.reset(); QVERIFY(elapsed.elapsed() < 100);
        }
        QTRY_VERIFY_WITH_TIMEOUT(guardedThread.isNull(), 3000);
        QVERIFY(state->completed.load());
        QCOMPARE(vision::app::AsyncThreadRetirement::activeCount(), originalCount);
    }
    void asynchronousShutdownCanBeRequestedRepeatedly()
    {
        vision::app::DetectionController controller;
        QSignalSpy ready(&controller, &vision::app::DetectionController::shutdownReady);
        controller.requestShutdown(); controller.requestShutdown();
        QTRY_VERIFY_WITH_TIMEOUT(controller.isShutdownReady(), 3000);
        QCOMPARE(ready.size(), 1);
    }
    void destructionAfterWorkerExitBeforeQueuedFinishedIsDelivered()
    {
        // Reproduce the shutdown race: the worker can already be destroyed
        // while its queued GUI-thread finished notification is still pending.
        const int originalCount = vision::app::AsyncThreadRetirement::activeCount();
        for (int iteration = 0; iteration < 25; ++iteration) {
            auto controller = std::make_unique<vision::app::DetectionController>();
            controller->requestShutdown();
            QThread::msleep(20);
            controller.reset();
            QCoreApplication::processEvents();
        }
        QTRY_COMPARE_WITH_TIMEOUT(vision::app::AsyncThreadRetirement::activeCount(), originalCount, 3000);
    }
    void preparationThenImmediateStopIsPreserved()
    {
        vision::app::DetectionController controller;
        controller.prepare(); controller.requestStop();
        QVERIFY(controller.snapshot().stopping);
        controller.requestShutdown();
        QTRY_VERIFY_WITH_TIMEOUT(controller.isShutdownReady(), 3000);
    }
    void shutdownWaitsForAcceptedSnapshotAndHistoryWithoutBlockingEventLoop()
    {
        QTemporaryDir directory;
        vision::app::HistoryRepository history(directory.filePath("history.json"));
        vision::app::DetectionController controller;
        controller.results()->setHistoryRepository(&history);
        vision::JobRequest request;
        request.files = {"accepted.png"};
        controller.prepareTask(request, 1, 1, "model-at-start", "configuration-at-start",
                               {{"accepted.png", "source-at-start"}});
        controller.beginResultDelivery({});
        vision::InferenceResult result;
        result.source = "accepted.png";
        result.image = QImage(2000, 1500, QImage::Format_RGB32);
        result.image.fill(QColor(30, 70, 120));
        QSignalSpy ready(&controller, &vision::app::DetectionController::shutdownReady);
        QSignalSpy ack(controller.results(), &vision::app::ResultDeliveryService::acknowledged);
        QVERIFY(controller.acceptImageResult(result, 0));
        QCOMPARE(controller.results()->pendingCount(), 1);
        int heartbeats = 0;
        QTimer heartbeat;
        heartbeat.setInterval(0);
        connect(&heartbeat, &QTimer::timeout, this, [&] { ++heartbeats; });
        heartbeat.start();
        QElapsedTimer elapsed;
        elapsed.start();
        controller.requestShutdown();
        controller.requestShutdown();
        QVERIFY(elapsed.elapsed() < 100);
        QVERIFY(!controller.isShutdownReady());
        QTRY_VERIFY_WITH_TIMEOUT(controller.isShutdownReady(), 10000);
        heartbeat.stop();
        QCOMPARE(ready.count(), 1);
        QVERIFY(heartbeats > 0);
        QCOMPARE(controller.results()->pendingCount(), 0);
        QCOMPARE(ack.count(), 1);
        const auto cached = controller.results()->cache().entries().value(result.source);
        QVERIFY(QFileInfo(cached.snapshotPath).isFile());
        QCOMPARE(QImage(cached.snapshotPath).size(), result.image.size());
        QCOMPARE(history.snapshot().size(), 1);
        QFile recordedHistory(history.filePath());
        QVERIFY(recordedHistory.open(QIODevice::ReadOnly));
        QCOMPARE(recordedHistory.readAll(), history.serialize());
    }
    void realVideoFinalization_data()
    {
        QTest::addColumn<QString>("action");
        QTest::newRow("natural-eof") << QStringLiteral("eof");
        QTest::newRow("stop-after-first-frame") << QStringLiteral("stop");
        QTest::newRow("shutdown-after-first-frame") << QStringLiteral("shutdown");
        QTest::newRow("export-failure-before-finished") << QStringLiteral("export-failure");
    }
    void realVideoFinalization()
    {
        QFETCH(QString, action);
        QTemporaryDir directory;
        const QString project = QString::fromUtf8(VISION_PROJECT_DIR);
        const QString model = project + "/models/yolov5n.onnx";
        QVERIFY2(QFileInfo::exists(model), "Real bundled ONNX fixture is required; this test must not silently substitute a mock.");
        const auto frame = cv::imread((project + "/assets/bus.jpg").toStdString());
        QVERIFY(!frame.empty());
        cv::Mat resized;
        cv::resize(frame, resized, cv::Size(640, 480));
        const QString video = directory.filePath("real-stream.avi");
        cv::VideoWriter videoWriter(video.toStdString(), cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                                    30, resized.size());
        QVERIFY2(videoWriter.isOpened(), "OpenCV MJPEG encoder is required for the actual video-path regression.");
        for (int frameIndex = 0; frameIndex < 8; ++frameIndex) {
            auto pixels = resized.clone();
            cv::rectangle(pixels, cv::Rect(5 + frameIndex * 10, 5, 12, 12), cv::Scalar(30, 150, 220), cv::FILLED);
            videoWriter.write(pixels);
        }
        videoWriter.release();
        vision::app::HistoryRepository history(directory.filePath("history.json"));
        vision::app::DetectionController controller;
        controller.results()->setHistoryRepository(&history);
        vision::JobRequest request;
        request.config.modelPath = model;
        request.config.task = vision::ModelTask::YoloV5;
        request.config.device = vision::ComputeDevice::CPU;
        request.sourceKind = vision::SourceKind::Video;
        request.files = {video};
        controller.prepareTask(request, 0, 0, "model-identity-at-start", "configuration-at-start", {});
        const QString output = directory.filePath("output");
        if (action == "export-failure") {
            QFile blocker(output);
            QVERIFY(blocker.open(QIODevice::WriteOnly));
            blocker.write("not a directory");
        }
        controller.beginResultDelivery({true, output});
        vision::InferenceResult latest;
        bool requested = false;
        qint64 requestDuration = -1;
        connect(&controller, &vision::app::DetectionController::resultReady, this,
                [&](const vision::InferenceResult &value) {
            latest = value;
            if (requested || (action != "stop" && action != "shutdown")) return;
            requested = true;
            QElapsedTimer elapsed;
            elapsed.start();
            if (action == "stop") controller.requestStop();
            else controller.requestShutdown();
            requestDuration = elapsed.elapsed();
        });
        QSignalSpy failed(&controller, &vision::app::DetectionController::failed);
        QSignalSpy finished(&controller, &vision::app::DetectionController::finished);
        QSignalSpy ready(&controller, &vision::app::DetectionController::shutdownReady);
        QSignalSpy fatal(controller.results(), &vision::app::ResultDeliveryService::fatalError);
        bool finishObservedCompletePersistence = false;
        connect(&controller, &vision::app::DetectionController::finished, this, [&](bool) {
            QFile savedHistory(history.filePath());
            finishObservedCompletePersistence = controller.results()->pendingCount() == 0 &&
                savedHistory.open(QIODevice::ReadOnly) && savedHistory.readAll() == history.serialize();
        });
        int heartbeats = 0;
        QTimer heartbeat;
        heartbeat.setInterval(1);
        connect(&heartbeat, &QTimer::timeout, this, [&] { ++heartbeats; });
        heartbeat.start();
        controller.run(request);
        QTRY_COMPARE_WITH_TIMEOUT(finished.count(), 1, 20000);
        QVERIFY(finishObservedCompletePersistence);
        QCOMPARE(failed.count(), 0); // Persistence failures retain the original nonmodal error path.
        QVERIFY(!latest.image.isNull());
        QCOMPARE(latest.backend, QStringLiteral("OpenCV DNN / CPU"));
        QCOMPARE(latest.device, vision::ComputeDevice::CPU);
        QCOMPARE(history.snapshot().size(), 1);
        const auto entry = history.snapshot().first().toObject();
        QCOMPARE(entry.value("frame").toInteger(), latest.frameNumber);
        QCOMPARE(entry.value("source").toString(), video);
        if (action == "export-failure") {
            QCOMPARE(fatal.count(), 1);
            QVERIFY(controller.snapshot().failed);
            QVERIFY(controller.snapshot().error.startsWith("自动导出失败："));
        } else {
            QCOMPARE(fatal.count(), 0);
            QVERIFY(!controller.snapshot().failed);
            const auto saved = QDir(output).entryList({"*.json"}, QDir::Files);
            QCOMPARE(saved.size(), 1);
            QFile jsonFile(output + "/" + saved.first());
            QVERIFY(jsonFile.open(QIODevice::ReadOnly));
            const auto json = QJsonDocument::fromJson(jsonFile.readAll()).object();
            QCOMPARE(json.value("frame").toInteger(), latest.frameNumber);
            QCOMPARE(json.value("source").toString(), video);
            QCOMPARE(json.value("actual_device").toString(), QStringLiteral("cpu"));
            QCOMPARE(QDir(output).entryList({"*.png"}, QDir::Files).size(), 1);
            QCOMPARE(QDir(output).entryList({"*.csv"}, QDir::Files).size(), 1);
            if (action == "eof") QCOMPARE(latest.frameNumber, qint64(8));
        }
        if (requested) QVERIFY(requestDuration >= 0 && requestDuration < 100);
        controller.requestShutdown();
        QTRY_VERIFY_WITH_TIMEOUT(controller.isShutdownReady(), 10000);
        heartbeat.stop();
        QVERIFY(heartbeats > 0);
        QCOMPARE(ready.count(), 1);
        QCOMPARE(finished.count(), 1);
        QCOMPARE(controller.results()->pendingCount(), 0);
    }
};
QTEST_MAIN(DetectionControllerTests)
#include "detection_controller_tests.moc"
