// SPDX-License-Identifier: AGPL-3.0-only
#include "app/recordingcontroller.h"
#include "app/asyncthreadretirement.h"
#include "thread_retirement_fixtures.h"
#include <QCoreApplication>
#include <QDir>
#include <QEventLoop>
#include <QFile>
#include <QJsonDocument>
#include <QPointer>
#include <QSemaphore>
#include <QTemporaryDir>
#include <QThread>
#include <QTimer>
#include <QtTest>
#include <atomic>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

using vision::SourceKind;
using vision::app::RecordingController;

namespace {
struct WorkerGateState
{
    QSemaphore entered;
    QSemaphore proceed;
    std::atomic<bool> resumed{false};
};

// Pause the event loop before accepted work so shutdown ordering is observable
// without relying on disk speed or creating an excessively large recording.
class WorkerPause
{
    std::shared_ptr<WorkerGateState> state_ = std::make_shared<WorkerGateState>();
public:
    explicit WorkerPause(QThread *thread)
    {
        auto *receiver = new QObject;
        receiver->moveToThread(thread);
        QObject::connect(thread, &QThread::finished, receiver, &QObject::deleteLater);
        QMetaObject::invokeMethod(receiver, [state = state_] {
            state->entered.release();
            state->proceed.acquire();
        }, Qt::QueuedConnection);
    }
    ~WorkerPause() { resume(); }
    bool entered() { return state_->entered.tryAcquire(1, 1000); }
    void resume()
    {
        if (!state_->resumed.exchange(true)) state_->proceed.release();
    }
};

QByteArray bytes(const QString &path)
{
    QFile file(path);
    return file.open(QIODevice::ReadOnly) ? file.readAll() : QByteArray();
}
}

class RecordingControllerTests final : public QObject
{
    Q_OBJECT
    QTemporaryDir temp_;
    QString root_;
    int serial_ = 0;

    QString clip(const QString &name) const { return root_ + "/recordings/" + name + ".avi"; }
    QString sidecar(const QString &path) const
    {
        const QFileInfo info(path);
        return info.absolutePath() + "/" + info.completeBaseName() + ".json";
    }
    void writeClip(const QString &path, const QSize &encodedSize, const QSize &contentSize,
                   int frames = 4, double fps = 25, bool metadata = true)
    {
        QVERIFY(QDir().mkpath(QFileInfo(path).absolutePath()));
        cv::VideoWriter writer(QFile::encodeName(path).constData(),
            cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps,
            cv::Size(encodedSize.width(), encodedSize.height()), true);
        QVERIFY2(writer.isOpened(), "This test requires the installed OpenCV MJPEG AVI encoder");
        for (int index = 0; index < frames; ++index) {
            cv::Mat frame(encodedSize.height(), encodedSize.width(), CV_8UC3,
                          cv::Scalar(20 + index, 80 + index, 180 + index));
            writer.write(frame);
        }
        writer.release();
        QVERIFY(QFileInfo(path).size() > 0);
        if (!metadata) return;
        const QJsonObject object{{"fps", fps}, {"frames", frames},
            {"width", encodedSize.width()}, {"height", encodedSize.height()},
            {"content_width", contentSize.width()}, {"content_height", contentSize.height()},
            {"stereo_view", "left"}, {"color_mode", "grayscale"}};
        QFile file(sidecar(path));
        QVERIFY(file.open(QIODevice::WriteOnly));
        const auto data = QJsonDocument(object).toJson();
        QCOMPARE(file.write(data), qint64(data.size()));
    }
    void modifiedAt(const QString &path, int ageSeconds)
    {
        QFile file(path);
        QVERIFY(file.open(QIODevice::ReadOnly));
        QVERIFY(file.setFileTime(QDateTime::currentDateTimeUtc().addSecs(-ageSeconds), QFileDevice::FileModificationTime));
    }
    QThread *workerThread(RecordingController &controller) const
    {
        // Supports the retained pre-fix reproduction as well as retirement
        // ownership, which deliberately detaches native threads from owners.
        if (auto *legacy = controller.findChild<QThread *>("recordingLibraryWorker")) return legacy;
        for (auto *thread : vision::app::AsyncThreadRetirement::watchedThreads(&controller))
            if (thread->objectName() == "recordingLibraryWorker") return thread;
        return nullptr;
    }
    void flushWorker(RecordingController &controller)
    {
        auto *thread = workerThread(controller);
        QVERIFY(thread);
        auto *marker = new QObject;
        marker->moveToThread(thread);
        QObject::connect(thread, &QThread::finished, marker, &QObject::deleteLater);
        const auto done = std::make_shared<bool>(false);
        QMetaObject::invokeMethod(marker, [this, marker, done] {
            QMetaObject::invokeMethod(this, [done] { *done = true; }, Qt::QueuedConnection);
            marker->deleteLater();
        }, Qt::QueuedConnection);
        QTRY_VERIFY(*done);
    }
    void drain(RecordingController &controller)
    {
        controller.detectionFinished();
        controller.requestShutdown();
        QTRY_VERIFY(controller.isShutdownReady());
    }

private slots:
    void init()
    {
        QVERIFY(temp_.isValid());
        root_ = temp_.path() + "/case-" + QString::number(++serial_);
        QVERIFY(QDir().mkpath(root_));
    }
    void shutdownReadinessWaitsForNativeThreadLocalCleanup()
    {
        auto controller = std::make_unique<RecordingController>(root_);
        auto *thread = workerThread(*controller); QVERIFY(thread);
        QPointer<QThread> guardedThread(thread);
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
        controller->requestShutdown();
        QTRY_VERIFY_WITH_TIMEOUT(state->entered.load(), 3000);
        QVERIFY(thread->isFinished()); QVERIFY(!controller->isShutdownReady());
        const int before = beats; QTest::qWait(30); QVERIFY(beats > before);
        state->resume();
        QTRY_VERIFY_WITH_TIMEOUT(controller->isShutdownReady(), 3000);
        QVERIFY(guardedThread.isNull()); QVERIFY(state->completed.load());
        QElapsedTimer elapsed; elapsed.start(); controller.reset(); QVERIFY(elapsed.elapsed() < 100);
    }
    void recordingIntentCancelsBeforeStartAndRejectsImages()
    {
        RecordingController controller(root_);
        QSignalSpy starts(&controller, &RecordingController::startRecordingRequested);
        QSignalSpy stops(&controller, &RecordingController::stopRecordingRequested);
        controller.setDetectionState(true, SourceKind::Images);
        QVERIFY(!controller.captureState().enabled);
        controller.toggleRecording();
        QCOMPARE(starts.count(), 0);
        controller.setDetectionState(true, SourceKind::Video);
        QVERIFY(controller.captureState().enabled);
        controller.toggleRecording();
        QCOMPARE(starts.count(), 1);
        QVERIFY(controller.captureState().requested);
        QVERIFY(!controller.captureState().active);
        QCOMPARE(controller.captureState().statusText, QString("等待下一帧开始录制…"));
        controller.toggleRecording();
        QCOMPARE(stops.count(), 1);
        QVERIFY(!controller.captureState().requested);
        QVERIFY(!controller.captureState().stopping);
        // A queued start arriving after cancellation must immediately stop again.
        controller.recordingStarted(clip("late"));
        QCOMPARE(stops.count(), 2);
        QVERIFY(controller.captureState().active);
        QVERIFY(controller.captureState().stopping);
        QVERIFY(!controller.captureState().enabled);
        controller.recordingFinished(clip("late"), 1, 25);
        QVERIFY(!controller.captureState().requested);
        QVERIFY(!controller.captureState().active);
        QVERIFY(!controller.captureState().stopping);
        QCOMPARE(controller.captureState().elapsedSeconds, qint64(0));
        drain(controller);
    }
    void activeClockAndStopKeepCaptureActiveUntilFinished()
    {
        RecordingController controller(root_);
        QSignalSpy starts(&controller, &RecordingController::startRecordingRequested);
        QSignalSpy stops(&controller, &RecordingController::stopRecordingRequested);
        controller.setDetectionState(true, SourceKind::Camera);
        controller.toggleRecording();
        controller.recordingStarted(clip("capture"));
        QVERIFY(controller.captureState().active);
        QVERIFY(controller.captureState().requested);
        QCOMPARE(controller.captureState().buttonText, QString("结束录制"));
        QTRY_VERIFY_WITH_TIMEOUT(controller.captureState().elapsedSeconds >= 1, 2500);
        QVERIFY(controller.captureState().statusText.startsWith("● REC  00:"));
        controller.toggleRecording();
        QCOMPARE(starts.count(), 1);
        QCOMPARE(stops.count(), 1);
        QVERIFY(!controller.captureState().requested);
        QVERIFY(controller.captureState().active);
        QVERIFY(controller.captureState().stopping);
        QCOMPARE(controller.captureState().buttonText, QString("正在保存…"));
        controller.toggleRecording();
        QCOMPARE(stops.count(), 1);
        controller.setDetectionState(true, SourceKind::Camera, true);
        QVERIFY(controller.captureState().active);
        QVERIFY(controller.captureState().stopping);
        controller.recordingFinished(clip("capture"), 25, 25);
        QVERIFY(!controller.captureState().active);
        QVERIFY(!controller.captureState().enabled);
        QCOMPARE(controller.captureState().elapsedSeconds, qint64(0));
        controller.setDetectionState(true, SourceKind::Camera, false, true);
        QVERIFY(!controller.captureState().enabled);
        drain(controller);
    }
    void recordingFailureResetsIntentAndReportsError()
    {
        RecordingController controller(root_);
        controller.setDetectionState(true, SourceKind::Video);
        controller.toggleRecording();
        controller.recordingStarted(clip("failed"));
        QSignalSpy notices(&controller, &RecordingController::notice);
        controller.recordingFailed("isolated write failure");
        QVERIFY(!controller.captureState().requested);
        QVERIFY(!controller.captureState().active);
        QVERIFY(!controller.captureState().stopping);
        QVERIFY(controller.captureState().enabled);
        QCOMPARE(notices.count(), 1);
        QVERIFY(notices.front().at(1).toBool());
        QVERIFY(notices.front().at(0).toString().contains("isolated write failure"));
        drain(controller);
    }
    void scansRealRecordingsPreservesSelectionAndCropsPreview()
    {
        const auto older = clip("older"), latest = clip("latest"), legacy = clip("legacy");
        writeClip(older, {48, 32}, {43, 27});
        writeClip(latest, {64, 48}, {53, 37});
        writeClip(legacy, {32, 24}, {}, 2, 25, false);
        modifiedAt(older, 30); modifiedAt(latest, 10); modifiedAt(legacy, 50);
        RecordingController controller(root_);
        QCOMPARE(controller.directory(), root_ + "/recordings");
        QSignalSpy entries(&controller, &RecordingController::recordingsChanged);
        QSignalSpy frames(&controller, &RecordingController::frameReady);
        controller.refresh();
        QTRY_COMPARE(controller.recordings().size(), 3);
        QTRY_VERIFY(!frames.isEmpty());
        QCOMPARE(entries.back().at(1).toString(), latest);
        QCOMPARE(controller.recordings().front().path, latest);
        QCOMPARE(controller.recordings().front().durationText, QString("0.2 s"));
        QCOMPARE(controller.recordings().front().resolutionText, QString("53×37"));
        QCOMPARE(controller.recordings().back().resolutionText, QString("—"));
        QCOMPARE(qvariant_cast<QImage>(frames.back().at(0)).size(), QSize(53, 37));
        QCOMPARE(frames.back().at(1).toString(), latest);
        QVERIFY(controller.playbackState().canPlay);
        QVERIFY(controller.playbackState().canExport);
        QVERIFY(controller.playbackState().details.contains("双目左目 · 灰度 · 25.0 fps"));
        const auto pixel = qvariant_cast<QImage>(frames.back().at(0)).pixelColor(0, 0);
        QVERIFY(pixel.red() > pixel.green() && pixel.green() > pixel.blue());
        frames.clear();
        controller.selectRecording(older);
        QTRY_VERIFY(!frames.isEmpty());
        QCOMPARE(frames.back().at(1).toString(), older);
        QCOMPARE(qvariant_cast<QImage>(frames.back().at(0)).size(), QSize(43, 27));
        controller.refresh();
        QTRY_COMPARE(entries.count(), 2);
        QCOMPARE(entries.back().at(1).toString(), older);
        drain(controller);
    }
    void selectionGenerationDropsQueuedFramesAndEmptySelection()
    {
        const auto first = clip("first"), final = clip("final");
        writeClip(first, {64, 48}, {53, 37});
        writeClip(final, {48, 32}, {43, 27});
        RecordingController controller(root_);
        auto *thread = workerThread(controller);
        QVERIFY(thread);
        QSignalSpy frames(&controller, &RecordingController::frameReady);
        QSignalSpy states(&controller, &RecordingController::playbackChanged);
        {
            WorkerPause pause(thread);
            QVERIFY(pause.entered());
            controller.selectRecording(first);
            controller.selectRecording(final);
            QCOMPARE(controller.playbackState().details, QString("正在载入录像…"));
            QVERIFY(!controller.playbackState().canPlay);
            QVERIFY(!controller.playbackState().canExport);
            QCOMPARE(qvariant_cast<vision::app::RecordingPlaybackState>(states.back().front()).path, final);
            pause.resume();
            flushWorker(controller);
        }
        QCOMPARE(frames.count(), 1);
        QCOMPARE(frames.front().at(1).toString(), final);
        QCOMPARE(controller.playbackState().path, final);
        frames.clear();
        {
            WorkerPause pause(thread);
            QVERIFY(pause.entered());
            controller.selectRecording(first);
            controller.selectRecording({});
            pause.resume();
            flushWorker(controller);
        }
        QCOMPARE(frames.count(), 0);
        QVERIFY(controller.playbackState().path.isEmpty());
        QVERIFY(!controller.playbackState().canPlay);
        QVERIFY(!controller.playbackState().canExport);
        drain(controller);
    }
    void playbackPausesAndReachesEofThenRestarts()
    {
        const auto path = clip("playback");
        writeClip(path, {64, 48}, {53, 37});
        RecordingController controller(root_);
        QSignalSpy frames(&controller, &RecordingController::frameReady);
        controller.selectRecording(path);
        QTRY_COMPARE(frames.count(), 1);
        controller.togglePlayback();
        QVERIFY(controller.playbackState().playing);
        controller.togglePlayback();
        QVERIFY(!controller.playbackState().playing);
        controller.togglePlayback();
        QTRY_VERIFY(!controller.playbackState().playing);
        QCOMPARE(frames.count(), 4);
        controller.togglePlayback();
        QTRY_VERIFY(frames.count() >= 5);
        QTRY_VERIFY(!controller.playbackState().playing);
        QCOMPARE(frames.count(), 8);
        drain(controller);
    }
    void exportsAviAndSidecarExactlyAndRejectsInvalidRequests()
    {
        const auto path = clip("source"), destination = root_ + "/export.AVI";
        writeClip(path, {64, 48}, {53, 37});
        RecordingController controller(root_);
        controller.selectRecording(path);
        QTRY_VERIFY(controller.playbackState().canExport);
        QSignalSpy notices(&controller, &RecordingController::notice);
        QVERIFY(!controller.exportRecording({}));
        QVERIFY(!controller.exportRecording(path));
        QVERIFY(!controller.exportRecording(root_ + "/bad.mp4"));
        QVERIFY(notices.back().at(1).toBool());
        QVERIFY(controller.exportRecording(destination));
        QVERIFY(controller.playbackState().exporting);
        QVERIFY(!controller.exportRecording(root_ + "/duplicate.avi"));
        QTRY_VERIFY(!controller.playbackState().exporting);
        QVERIFY(!notices.back().at(1).toBool());
        QVERIFY(!bytes(destination).isEmpty());
        QCOMPARE(bytes(destination), bytes(path));
        QCOMPARE(bytes(sidecar(destination)), bytes(sidecar(path)));
        drain(controller);
    }
    void shutdownDrainsAcceptedExportAndWaitsForDetectionFinished()
    {
        const auto path = clip("source"), destination = root_ + "/accepted.avi";
        writeClip(path, {64, 48}, {53, 37});
        RecordingController controller(root_);
        auto *thread = workerThread(controller);
        QVERIFY(thread);
        WorkerPause pause(thread);
        QVERIFY(pause.entered());
        controller.setDetectionState(true, SourceKind::Video);
        controller.toggleRecording();
        controller.recordingStarted(path);
        controller.selectRecording(path);
        QVERIFY(controller.exportRecording(destination));
        QSignalSpy ready(&controller, &RecordingController::shutdownReady);
        QSignalSpy starts(&controller, &RecordingController::startRecordingRequested);
        QSignalSpy frames(&controller, &RecordingController::frameReady);
        QElapsedTimer elapsed; elapsed.start();
        controller.requestShutdown(); controller.requestShutdown();
        QVERIFY(elapsed.elapsed() < 100);
        QVERIFY(!controller.isShutdownReady());
        QVERIFY(!controller.captureState().enabled);
        QVERIFY(!QFileInfo::exists(destination));
        QVERIFY(!controller.exportRecording(root_ + "/rejected.avi"));
        controller.toggleRecording(); controller.refresh(); controller.selectRecording({}); controller.togglePlayback();
        QCOMPARE(starts.count(), 0);
        QCOMPARE(controller.playbackState().path, path);
        pause.resume();
        QTRY_VERIFY(vision::app::AsyncThreadRetirement::watchedThreads(&controller).isEmpty());
        QTRY_VERIFY(!controller.playbackState().exporting);
        QCOMPARE(bytes(destination), bytes(path));
        QCOMPARE(bytes(sidecar(destination)), bytes(sidecar(path)));
        QCOMPARE(frames.count(), 0);
        QCOMPARE(ready.count(), 0);
        controller.recordingFinished(path, 4, 25);
        QVERIFY(!controller.isShutdownReady());
        controller.detectionFinished();
        QTRY_VERIFY(controller.isShutdownReady());
        QCOMPARE(ready.count(), 1);
    }
    void closeFromExportingNotificationDrainsTheAcceptedRealCopy()
    {
        const auto path = clip("reentrant-source"), destination = root_ + "/reentrant-accepted.avi";
        writeClip(path, {64, 48}, {53, 37});
        RecordingController controller(root_);
        controller.selectRecording(path);
        QTRY_VERIFY(controller.playbackState().canExport);
        QSignalSpy notices(&controller, &RecordingController::notice);
        bool requested = false, readyInsideNotification = false;
        connect(&controller, &RecordingController::playbackChanged, &controller,
            [&](const vision::app::RecordingPlaybackState &state) {
            if (!state.exporting || requested) return;
            requested = true;
            controller.requestShutdown();
            // The GUI is allowed to keep pumping while a synchronous status
            // notification closes its owner. Readiness must include this copy.
            QEventLoop events;
            connect(&controller, &RecordingController::shutdownReady, &events, &QEventLoop::quit);
            QTimer::singleShot(3000, &events, &QEventLoop::quit);
            if (!controller.isShutdownReady()) events.exec();
            readyInsideNotification = controller.isShutdownReady();
        }, Qt::DirectConnection);
        QVERIFY(controller.exportRecording(destination));
        QVERIFY(requested); QVERIFY(readyInsideNotification); QVERIFY(controller.isShutdownReady());
        QVERIFY(!bytes(destination).isEmpty());
        QCOMPARE(bytes(destination), bytes(path));
        QCOMPARE(bytes(sidecar(destination)), bytes(sidecar(path)));
        QVERIFY(!controller.playbackState().exporting);
        QVERIFY(!notices.isEmpty()); QVERIFY(!notices.back().at(1).toBool());
        QVERIFY(!controller.exportRecording(root_ + "/after-close.avi"));
    }
    void directDestructionIsNonblockingAndReapsThreadAfterAcceptedExport()
    {
        const auto path = clip("source"), destination = root_ + "/accepted-after-delete.avi";
        writeClip(path, {64, 48}, {53, 37});
        auto *controller = new RecordingController(root_);
        QPointer<QThread> thread = workerThread(*controller);
        QVERIFY(thread);
        WorkerPause pause(thread);
        QVERIFY(pause.entered());
        controller->selectRecording(path);
        QVERIFY(controller->exportRecording(destination));
        QElapsedTimer elapsed; elapsed.start();
        delete controller;
        QVERIFY(elapsed.elapsed() < 100);
        QVERIFY(thread);
        QVERIFY(!QFileInfo::exists(destination));
        pause.resume();
        QTRY_VERIFY(thread.isNull());
        QVERIFY(!bytes(destination).isEmpty());
        QCOMPARE(bytes(destination), bytes(path));
        QCOMPARE(bytes(sidecar(destination)), bytes(sidecar(path)));
    }
};

QTEST_GUILESS_MAIN(RecordingControllerTests)
#include "recording_controller_tests.moc"
