#include "core/inferenceworker.h"

#include <QElapsedTimer>
#include <QFileInfo>
#include <QSignalSpy>
#include <QTemporaryDir>
#include <QThread>
#include <QTimer>
#include <QtTest>

namespace
{
vision::JobRequest imageRequest(const QString &directory, int count)
{
    vision::JobRequest request;
    request.config.modelPath = QStringLiteral(VISION_PROJECT_DIR) + QStringLiteral("/models/yolov5n.onnx");
    request.config.task = vision::ModelTask::YoloV5;
    request.config.device = vision::ComputeDevice::CPU;
    request.config.inputSize = 640;
    request.config.confidence = 1.0;
    request.sourceKind = vision::SourceKind::Images;
    for (int row = 0; row < count; ++row)
    {
        QImage image(64 + row, 40 + row, QImage::Format_RGB888);
        image.fill(QColor(30 + row * 10, 60, 90));
        const QString path = directory + QStringLiteral("/image-%1.png").arg(row);
        if (!image.save(path))
            return {};
        request.files.append(path);
    }
    return request;
}

class WorkerHarness
{
  public:
    WorkerHarness()
    {
        worker = new vision::InferenceWorker;
        worker->setImageResultBackpressureEnabled(true);
        worker->moveToThread(&thread);
        QObject::connect(&thread, &QThread::finished, worker, &QObject::deleteLater);
        thread.start();
    }

    ~WorkerHarness()
    {
        worker->requestStop();
        thread.quit();
        if (!thread.wait(30000))
            qFatal("Image-delivery worker did not finish after cancellation.");
    }

    void start(const vision::JobRequest &request)
    {
        worker->prepare();
        QMetaObject::invokeMethod(worker, [worker = worker, request] { worker->run(request); },
                                  Qt::QueuedConnection);
    }

    vision::InferenceWorker *worker = nullptr;
    QThread thread;
};
} // namespace

class InferenceDeliveryTests final : public QObject
{
    Q_OBJECT
  private slots:
    void initTestCase()
    {
        qRegisterMetaType<vision::InferenceResult>();
        qRegisterMetaType<vision::ModelConfig>();
        qRegisterMetaType<vision::JobRequest>();
        QVERIFY(QFileInfo(QStringLiteral(VISION_PROJECT_DIR) +
                           QStringLiteral("/models/yolov5n.onnx")).isFile());
    }

    void slowConsumerBoundsDeliveryAndStopReleasesUnacknowledgedResult()
    {
        QTemporaryDir directory;
        QVERIFY(directory.isValid());
        const auto request = imageRequest(directory.path(), 3);
        QCOMPARE(request.files.size(), 3);
        WorkerHarness runner;
        QSignalSpy images(runner.worker, &vision::InferenceWorker::imageResultReady);
        QSignalSpy legacy(runner.worker, &vision::InferenceWorker::resultReady);
        QSignalSpy progress(runner.worker, &vision::InferenceWorker::progress);
        QSignalSpy finished(runner.worker, &vision::InferenceWorker::finished);
        QSignalSpy errors(runner.worker, &vision::InferenceWorker::failed);
        int heartbeat = 0;
        QTimer timer;
        connect(&timer, &QTimer::timeout, this, [&] { ++heartbeat; });
        timer.start(10);
        runner.start(request);
        QTRY_COMPARE_WITH_TIMEOUT(images.size(), 1, 15000);
        const auto first = qvariant_cast<vision::InferenceResult>(images[0][0]);
        const quint64 firstTicket = images[0][1].toULongLong();
        QVERIFY(firstTicket != 0);
        QCOMPARE(first.source, request.files[0]);
        QCOMPARE(first.frameNumber, qint64(1));
        QVERIFY(first.inferenceMs > 0);
        QCOMPARE(first.device, vision::ComputeDevice::CPU);
        const int beforeHeartbeat = heartbeat;
        QTest::qWait(150);
        QCOMPARE(images.size(), 1);
        QCOMPARE(finished.size(), 0);
        QVERIFY(heartbeat > beforeHeartbeat);
        runner.worker->acknowledgeImageResult(firstTicket + 1);
        QTest::qWait(100);
        QCOMPARE(images.size(), 1); // A wrong completion cannot advance the queue.
        runner.worker->acknowledgeImageResult(firstTicket);
        QTRY_COMPARE_WITH_TIMEOUT(images.size(), 2, 15000);
        const auto second = qvariant_cast<vision::InferenceResult>(images[1][0]);
        QCOMPARE(second.source, request.files[1]);
        QCOMPARE(second.frameNumber, qint64(2));
        QVERIFY(images[1][1].toULongLong() > firstTicket);

        // No expensive forward is in progress: the worker waits only for this
        // already delivered result's cache/export acknowledgement.
        QElapsedTimer stopTime;
        stopTime.start();
        runner.worker->requestStop();
        QTRY_COMPARE_WITH_TIMEOUT(finished.size(), 1, 2000);
        QVERIFY(stopTime.elapsed() < 2000);
        qInfo().noquote() << QStringLiteral("delivery_cancel_evidence: max_unacknowledged=1 stop_elapsed_ms=%1 delivered=2 total=3")
                                .arg(stopTime.elapsed());
        QCOMPARE(finished[0][0].toBool(), true);
        QCOMPARE(images.size(), 2); // The third file must never be delivered.
        QCOMPARE(legacy.size(), 0);
        QCOMPARE(errors.size(), 0);
        QCOMPARE(progress.last()[0].toInt(), 2);
        QCOMPARE(progress.last()[1].toInt(), 3);
    }

    void lateAcknowledgementFromCancelledJobCannotAdvanceNextJob()
    {
        QTemporaryDir directory;
        QVERIFY(directory.isValid());
        const auto request = imageRequest(directory.path(), 2);
        QCOMPARE(request.files.size(), 2);
        WorkerHarness runner;
        QSignalSpy images(runner.worker, &vision::InferenceWorker::imageResultReady);
        QSignalSpy finished(runner.worker, &vision::InferenceWorker::finished);
        QSignalSpy errors(runner.worker, &vision::InferenceWorker::failed);
        runner.start(request);
        QTRY_COMPARE_WITH_TIMEOUT(images.size(), 1, 15000);
        const quint64 oldTicket = images[0][1].toULongLong();
        runner.worker->requestStop();
        QTRY_COMPARE_WITH_TIMEOUT(finished.size(), 1, 2000);

        runner.start(request);
        QTRY_COMPARE_WITH_TIMEOUT(images.size(), 2, 15000);
        const quint64 newTicket = images[1][1].toULongLong();
        QVERIFY(newTicket > oldTicket);
        runner.worker->acknowledgeImageResult(oldTicket);
        QTest::qWait(150);
        QCOMPARE(images.size(), 2);
        QCOMPARE(finished.size(), 1);
        runner.worker->acknowledgeImageResult(newTicket);
        QTRY_COMPARE_WITH_TIMEOUT(images.size(), 3, 15000);
        QCOMPARE(qvariant_cast<vision::InferenceResult>(images[2][0]).source, request.files[1]);
        runner.worker->requestStop();
        QTRY_COMPARE_WITH_TIMEOUT(finished.size(), 2, 2000);
        QCOMPARE(finished[1][0].toBool(), true);
        QCOMPARE(errors.size(), 0);
    }

    void acknowledgedBatchCompletesAllImagesWithoutLegacyDuplicateDelivery()
    {
        QTemporaryDir directory;
        QVERIFY(directory.isValid());
        const auto request = imageRequest(directory.path(), 3);
        QCOMPARE(request.files.size(), 3);
        WorkerHarness runner;
        QSignalSpy images(runner.worker, &vision::InferenceWorker::imageResultReady);
        QSignalSpy legacy(runner.worker, &vision::InferenceWorker::resultReady);
        QSignalSpy progress(runner.worker, &vision::InferenceWorker::progress);
        QSignalSpy finished(runner.worker, &vision::InferenceWorker::finished);
        QSignalSpy errors(runner.worker, &vision::InferenceWorker::failed);
        connect(runner.worker, &vision::InferenceWorker::imageResultReady, this,
                [this, worker = runner.worker](const vision::InferenceResult &, quint64 ticket)
                {
                    QTimer::singleShot(20, this, [worker, ticket] { worker->acknowledgeImageResult(ticket); });
                },
                Qt::QueuedConnection);
        runner.start(request);
        QTRY_COMPARE_WITH_TIMEOUT(finished.size(), 1, 15000);
        QCOMPARE(finished[0][0].toBool(), false);
        QCOMPARE(images.size(), 3);
        for (int index = 0; index < images.size(); ++index)
        {
            const auto result = qvariant_cast<vision::InferenceResult>(images[index][0]);
            QCOMPARE(result.source, request.files[index]);
            QCOMPARE(result.frameNumber, qint64(index + 1));
            QVERIFY(!result.image.isNull());
        }
        QCOMPARE(progress.last()[0].toInt(), 3);
        QCOMPARE(progress.last()[1].toInt(), 3);
        QCOMPARE(legacy.size(), 0);
        QCOMPARE(errors.size(), 0);
    }

    void unboundedStandaloneWorkerRetainsExistingResultSignal()
    {
        QTemporaryDir directory;
        QVERIFY(directory.isValid());
        const auto request = imageRequest(directory.path(), 1);
        QCOMPARE(request.files.size(), 1);
        vision::InferenceWorker worker;
        QSignalSpy legacy(&worker, &vision::InferenceWorker::resultReady);
        QSignalSpy bounded(&worker, &vision::InferenceWorker::imageResultReady);
        QSignalSpy finished(&worker, &vision::InferenceWorker::finished);
        QSignalSpy errors(&worker, &vision::InferenceWorker::failed);
        worker.prepare();
        worker.run(request);
        QCOMPARE(legacy.size(), 1);
        QCOMPARE(bounded.size(), 0);
        QCOMPARE(finished.size(), 1);
        QCOMPARE(finished[0][0].toBool(), false);
        QCOMPARE(errors.size(), 0);
    }
};

QTEST_MAIN(InferenceDeliveryTests)
#include "inference_delivery_tests.moc"
