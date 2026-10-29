// SPDX-License-Identifier: AGPL-3.0-only
#include "app/operationcoordinator.h"
#include "app/modelsessioncontroller.h"
#include "app/workbenchconfiguration.h"
#include <QTemporaryDir>
#include <QFile>
#include <QDir>
#include <QImage>

#include <QElapsedTimer>
#include <QEventLoop>
#include <QSignalSpy>
#include <QTimer>
#include <QtTest>
#include <memory>

using namespace vision::app;
namespace
{
// Permission contract, independent of the controller's boolean implementation:
// source/model imports exclude every listed activity; start additionally allows
// a modal surface; device edits allow conversion/saving/modal but not preparing.
enum Activity : quint16
{
    Detection = 1 << 0, Input = 1 << 1, Projection = 1 << 2, Evaluation = 1 << 3,
    Conversion = 1 << 4, Preparation = 1 << 5, Saving = 1 << 6, Modal = 1 << 7,
    Closing = 1 << 8
};
ActivitySnapshot stateFor(quint16 flags)
{
    ActivitySnapshot state;
    state.detection = flags & Detection; state.input = flags & Input;
    state.projection = flags & Projection; state.evaluation = flags & Evaluation;
    state.conversion = flags & Conversion; state.gpuPreparation = flags & Preparation;
    state.saving = flags & Saving; state.modal = flags & Modal; state.closing = flags & Closing;
    return state;
}
void runEvents(int milliseconds)
{
    QEventLoop loop;
    QTimer::singleShot(milliseconds, &loop, &QEventLoop::quit);
    loop.exec();
}
}

class OperationCoordinatorTests final : public QObject
{
    Q_OBJECT
  private slots:
    void detectionPlanFreezesSelectedRangeAndConfiguration_data()
    {
        QTest::addColumn<int>("mode"); QTest::addColumn<int>("row");
        QTest::addColumn<bool>("selected"); QTest::addColumn<int>("first"); QTest::addColumn<int>("last");
        QTest::newRow("all-independent-of-current") << 0 << 60 << true << 1 << 64;
        QTest::newRow("all-with-no-selection") << 0 << -1 << false << 1 << 64;
        QTest::newRow("suffix-first") << 1 << 0 << true << 1 << 64;
        QTest::newRow("suffix-middle") << 1 << 12 << true << 13 << 64;
        QTest::newRow("suffix-last") << 1 << 63 << true << 64 << 64;
        QTest::newRow("current-first") << 2 << 0 << true << 1 << 1;
        QTest::newRow("current-last") << 2 << 63 << true << 64 << 64;
        QTest::newRow("suffix-requires-selection") << 1 << 0 << false << 0 << 0;
        QTest::newRow("current-requires-selection") << 2 << -1 << false << 0 << 0;
    }
    void detectionPlanFreezesSelectedRangeAndConfiguration()
    {
        QFETCH(int, mode); QFETCH(int, row); QFETCH(bool, selected); QFETCH(int, first); QFETCH(int, last);
        QTemporaryDir directory; QVERIFY(directory.isValid());
        const QString modelPath = directory.filePath("fixture.onnx");
        QFile modelFile(modelPath); QVERIFY(modelFile.open(QIODevice::WriteOnly)); modelFile.write("identity-fixture"); modelFile.close();
        ModelSessionController model;
        ModelSessionSnapshot models; models.models = {modelPath}; models.activeModel = modelPath; model.initialize(models);
        InputSessionController input(directory.path(), directory.filePath("session.json"));
        connect(&input, &InputSessionController::queueAppend, &input,
            [&input](const QStringList &files, int offset, quint64 generation) { input.acknowledgeQueueProjection(generation, offset + files.size()); });
        QStringList paths;
        QImage image(4, 4, QImage::Format_RGB32); image.fill(Qt::gray);
        for (int i = 0; i < 64; ++i) { const auto path = directory.filePath(QString::number(i) + ".png"); QVERIFY(image.save(path)); paths.append(path); }
        QVERIFY(input.requestImages(paths));
        QTRY_VERIFY_WITH_TIMEOUT(!input.snapshot().importing, 5000);
        input.setSelection(row, selected);
        WorkbenchConfiguration configuration; auto draft = configuration.snapshot();
        draft.confidence = .42; draft.stereoView = vision::StereoView::Right;
        draft.automaticExport = true; draft.exportDirectory = directory.filePath("exports"); configuration.replaceDraft(draft);
        OperationCoordinator coordinator;
        const auto task = coordinator.prepareDetection(input, model, configuration, static_cast<ImageRange>(mode));
        QCOMPARE(task.valid, first != 0);
        if (first) {
            QVERIFY(task.error.isEmpty()); QCOMPARE(task.first, first); QCOMPARE(task.last, last);
            QCOMPARE(task.request.files, paths.mid(first - 1, last - first + 1));
            QCOMPARE(task.sourceIdentities.size(), task.request.files.size());
            QCOMPARE(task.request.stereoView, vision::StereoView::Full);
            QCOMPARE(task.request.config.confidence, float(.42)); QVERIFY(task.output.automaticExport);
            QVERIFY(QDir(task.output.exportDirectory).exists()); QVERIFY(!task.modelIdentity.isEmpty());
            draft.confidence = .9; draft.exportDirectory = directory.filePath("other"); configuration.replaceDraft(draft);
            input.clear();
            QCOMPARE(task.request.files, paths.mid(first - 1, last - first + 1));
            QCOMPARE(task.request.config.confidence, float(.42)); QCOMPARE(task.output.exportDirectory, directory.filePath("exports"));
        } else QVERIFY(!task.error.isEmpty());
        input.requestShutdown(); model.requestShutdown();
        QTRY_VERIFY(input.isShutdownReady()); QTRY_VERIFY(model.isShutdownReady());
    }

    void permissionMatrix_data()
    {
        QTest::addColumn<quint16>("flags");
        for (quint16 flags = 0; flags < (1 << 9); ++flags)
            QTest::newRow(qPrintable(QString("activity-mask-%1").arg(flags, 3, 16, QChar('0')))) << flags;
    }

    void permissionMatrix()
    {
        QFETCH(quint16, flags);
        const auto activity = stateFor(flags);
        const auto available = OperationCoordinator::availability(activity);
        QCOMPARE(activity.workbenchBusy(), bool(flags & (Detection | Input | Projection | Evaluation)));
        QCOMPARE(available.importInput, flags == 0);
        QCOMPARE(available.importModels, flags == 0);
        QCOMPARE(available.start, (flags & ~Modal) == 0);
        QCOMPARE(available.changeDevice, (flags & (Detection | Input | Projection | Evaluation | Preparation | Closing)) == 0);
        QCOMPARE(available.editParameters, (flags & (Detection | Input | Projection | Evaluation | Closing)) == 0);
    }

    void activityProviderIsLiveAndClosingHasOneAuthority()
    {
        OperationCoordinator coordinator;
        ActivitySnapshot provided;
        int reads = 0;
        coordinator.setActivityProvider([&] { ++reads; return provided; });
        QVERIFY(!coordinator.activity().workbenchBusy());
        provided.evaluation = true;
        QVERIFY(coordinator.activity().workbenchBusy());
        provided.evaluation = false; provided.closing = true;
        QVERIFY(!coordinator.activity().closing); // A stale provider cannot start window shutdown.
        QCOMPARE(reads, 3);
        coordinator.requestShutdown();
        QVERIFY(coordinator.activity().closing); QVERIFY(coordinator.isClosing());
        QVERIFY(coordinator.isShutdownReady());
    }

    void probeParticipantDoesNotInventPreparationActivity()
    {
        // A probe can own an outstanding shutdown resource without being an
        // environment installation. Generic "participant busy" must not lock UI.
        OperationCoordinator coordinator;
        QObject gpuProbe;
        bool probeDone = false, preparing = false;
        coordinator.setActivityProvider([&] {
            ActivitySnapshot state; state.gpuPreparation = preparing; return state;
        });
        coordinator.addShutdownParticipant("gpu-probe", &gpuProbe, [&] { probeDone = true; }, [&] { return probeDone; });
        auto available = OperationCoordinator::availability(coordinator.activity());
        QVERIFY(available.importInput); QVERIFY(available.importModels);
        QVERIFY(available.start); QVERIFY(available.changeDevice);
        preparing = true;
        available = OperationCoordinator::availability(coordinator.activity());
        QVERIFY(!available.importInput); QVERIFY(!available.importModels);
        QVERIFY(!available.start); QVERIFY(!available.changeDevice);
        preparing = false;
        coordinator.requestShutdown();
        QVERIFY(coordinator.isShutdownReady());
    }

    void noParticipantsFinishOnceAndRequestsAreIdempotent()
    {
        OperationCoordinator coordinator;
        QSignalSpy started(&coordinator, &OperationCoordinator::shutdownStarted);
        QSignalSpy ready(&coordinator, &OperationCoordinator::shutdownReady);
        coordinator.requestShutdown();
        QCOMPARE(started.count(), 1); QCOMPARE(ready.count(), 1);
        QVERIFY(coordinator.isShutdownReady());
        for (int i = 0; i < 20; ++i) coordinator.requestShutdown();
        runEvents(30);
        QCOMPARE(started.count(), 1); QCOMPARE(ready.count(), 1);
    }

    void repeatedRequestsDoNotCancelAcceptedResourcesTwice()
    {
        OperationCoordinator coordinator;
        QObject owner;
        bool done = false;
        int requests = 0;
        coordinator.addShutdownParticipant("accepted-write", &owner, [&] { ++requests; }, [&] { return done; });
        QSignalSpy started(&coordinator, &OperationCoordinator::shutdownStarted);
        QSignalSpy ready(&coordinator, &OperationCoordinator::shutdownReady);
        for (int i = 0; i < 15; ++i) coordinator.requestShutdown();
        QCOMPARE(requests, 1); QCOMPARE(started.count(), 1); QCOMPARE(ready.count(), 0);
        runEvents(25); QVERIFY(!coordinator.isShutdownReady());
        done = true;
        QTRY_COMPARE_WITH_TIMEOUT(ready.count(), 1, 2000);
        QCOMPARE(requests, 1);
        coordinator.requestShutdown(); runEvents(20); QCOMPARE(ready.count(), 1);
    }

    void asynchronousOwnersMustAllFinish()
    {
        OperationCoordinator coordinator;
        QObject detectionOwner, saveOwner, previewOwner;
        bool detectionDone = false, saveDone = false, previewDone = false;
        QStringList requested, completed;
        coordinator.addShutdownParticipant("detection", &detectionOwner, [&] {
            requested.append("detection");
            QTimer::singleShot(20, &detectionOwner, [&] { detectionDone = true; completed.append("detection"); });
        }, [&] { return detectionDone; });
        coordinator.addShutdownParticipant("saves", &saveOwner, [&] {
            requested.append("saves");
            QTimer::singleShot(75, &saveOwner, [&] { saveDone = true; completed.append("saves"); });
        }, [&] { return saveDone; });
        coordinator.addShutdownParticipant("preview", &previewOwner, [&] {
            requested.append("preview");
            QTimer::singleShot(45, &previewOwner, [&] { previewDone = true; completed.append("preview"); });
        }, [&] { return previewDone; });
        QSignalSpy ready(&coordinator, &OperationCoordinator::shutdownReady);
        bool premature = false;
        connect(&coordinator, &OperationCoordinator::shutdownReady, this, [&] {
            premature = !detectionDone || !saveDone || !previewDone;
        });
        coordinator.requestShutdown();
        QCOMPARE(requested, QStringList({"detection", "saves", "preview"}));
        QVERIFY(!coordinator.isShutdownReady());
        QTRY_COMPARE_WITH_TIMEOUT(ready.count(), 1, 3000);
        QVERIFY(!premature);
        QCOMPARE(completed.size(), 3);
        QVERIFY(coordinator.isShutdownReady());
    }

    void synchronousReadyAndReentrantRequestsCannotSkipLaterOwners()
    {
        OperationCoordinator coordinator;
        QObject first, second;
        bool firstDone = false, secondDone = false, premature = false;
        int requested = 0;
        QSignalSpy ready(&coordinator, &OperationCoordinator::shutdownReady);
        connect(&coordinator, &OperationCoordinator::shutdownStarted, this,
                [&] { coordinator.requestShutdown(); });
        coordinator.addShutdownParticipant("first", &first, [&] {
            ++requested; firstDone = true;
            coordinator.requestShutdown();
            // A nested real event loop exposes an accidental timer-driven
            // readiness check while participant requests are still running.
            runEvents(25);
            premature = premature || coordinator.isShutdownReady();
        }, [&] { return firstDone; });
        coordinator.addShutdownParticipant("second", &second, [&] {
            ++requested; secondDone = true; coordinator.requestShutdown();
            premature = premature || ready.count() != 0;
        }, [&] { return secondDone; });
        coordinator.requestShutdown();
        QCOMPARE(requested, 2); QVERIFY(!premature);
        QCOMPARE(ready.count(), 1); QVERIFY(coordinator.isShutdownReady());
    }

    void shutdownBarrierKeepsEventLoopHeartbeatRunning()
    {
        OperationCoordinator coordinator;
        QObject owner;
        bool done = false;
        int heartbeats = 0, beatsAtReady = 0;
        QTimer heartbeat; heartbeat.setInterval(1);
        connect(&heartbeat, &QTimer::timeout, this, [&] { ++heartbeats; });
        coordinator.addShutdownParticipant("background-save", &owner, [&] {
            QTimer::singleShot(120, &owner, [&] { done = true; });
        }, [&] { return done; });
        connect(&coordinator, &OperationCoordinator::shutdownReady, this, [&] { beatsAtReady = heartbeats; });
        QSignalSpy ready(&coordinator, &OperationCoordinator::shutdownReady);
        heartbeat.start();
        QElapsedTimer command; command.start(); coordinator.requestShutdown();
        QVERIFY2(command.elapsed() < 100, "Shutdown command synchronously waited on a resource.");
        QVERIFY(!done); QVERIFY(!coordinator.isShutdownReady());
        QTRY_COMPARE_WITH_TIMEOUT(ready.count(), 1, 3000);
        heartbeat.stop();
        QVERIFY2(beatsAtReady >= 4, "Event-loop heartbeat did not advance while resources were pending.");
    }

    void destroyedOwnerBeforeRequestCannotBecomeReady()
    {
        OperationCoordinator coordinator;
        auto owner = std::make_unique<QObject>();
        int requests = 0, readinessQueries = 0;
        coordinator.addShutdownParticipant("destroyed-owner", owner.get(), [&] { ++requests; }, [&] {
            ++readinessQueries; return true;
        });
        owner.reset();
        QSignalSpy ready(&coordinator, &OperationCoordinator::shutdownReady);
        coordinator.requestShutdown(); runEvents(35);
        QCOMPARE(requests, 0); QCOMPARE(readinessQueries, 0);
        QCOMPARE(ready.count(), 0); QVERIFY(!coordinator.isShutdownReady());
    }

    void destroyedOwnerDuringShutdownCannotBecomeReady()
    {
        OperationCoordinator coordinator;
        auto owner = std::make_unique<QObject>();
        bool done = false;
        int readinessQueries = 0;
        coordinator.addShutdownParticipant("premature-delete", owner.get(), [] {}, [&] {
            ++readinessQueries; return done;
        });
        QSignalSpy ready(&coordinator, &OperationCoordinator::shutdownReady);
        coordinator.requestShutdown(); QVERIFY(!coordinator.isShutdownReady());
        owner.reset(); const int before = readinessQueries; done = true;
        runEvents(35);
        QCOMPARE(readinessQueries, before);
        QCOMPARE(ready.count(), 0); QVERIFY(!coordinator.isShutdownReady());
    }

    void missingReadinessCallbackCannotSilentlyPass()
    {
        OperationCoordinator coordinator;
        QObject owner;
        coordinator.addShutdownParticipant("missing-readiness", &owner, [] {}, {});
        QSignalSpy ready(&coordinator, &OperationCoordinator::shutdownReady);
        coordinator.requestShutdown(); runEvents(25);
        QCOMPARE(ready.count(), 0); QVERIFY(!coordinator.isShutdownReady());
    }
};
QTEST_GUILESS_MAIN(OperationCoordinatorTests)
#include "operation_coordinator_tests.moc"
