// SPDX-License-Identifier: AGPL-3.0-only
#include "app/inputsessioncontroller.h"
#include "core/datasetsessionstore.h"

#include <QDir>
#include <QColor>
#include <QElapsedTimer>
#include <QEventLoop>
#include <QFile>
#include <QFileInfo>
#include <QImage>
#include <QJsonDocument>
#include <QSignalSpy>
#include <QTemporaryDir>
#include <QTimer>
#include <QtTest>
#include <algorithm>
#include <memory>

using vision::app::ImageRange;
using vision::app::InputPhase;
using vision::app::InputSessionController;

namespace
{
QByteArray contents(const QString &path)
{
    QFile file(path);
    return file.open(QIODevice::ReadOnly) ? file.readAll() : QByteArray();
}
bool writeFile(const QString &path, const QByteArray &data)
{
    QFile file(path);
    return file.open(QIODevice::WriteOnly) && file.write(data) == data.size();
}
}

class InputSessionControllerTests final : public QObject
{
    Q_OBJECT
    QTemporaryDir area_;
    QString dataset_, sessionPath_, project_;
    QStringList images_;
    QByteArray previousPython_;
    bool previousPythonSet_ = false;

    std::unique_ptr<InputSessionController> controller()
    {
        return std::make_unique<InputSessionController>(project_, sessionPath_);
    }
    QMetaObject::Connection acknowledge(InputSessionController &session)
    {
        return connect(&session, &InputSessionController::queueAppend, &session,
            [&session](const QStringList &paths, int firstRow, quint64 generation) {
                session.acknowledgeQueueProjection(generation, firstRow + paths.size());
            });
    }

  private slots:
    void initTestCase()
    {
        QVERIFY(area_.isValid());
        project_ = QStringLiteral(VISION_PROJECT_DIR);
        dataset_ = area_.path() + "/dataset";
        sessionPath_ = area_.path() + "/state/dataset-session.json";
        QVERIFY(QDir().mkpath(dataset_ + "/images/val"));
        QVERIFY(QDir().mkpath(dataset_ + "/labels/val"));
        QImage image(32, 24, QImage::Format_RGB888);
        image.fill(QColor(24, 66, 100));
        for (int i = 0; i < 211; ++i)
        {
            const auto name = QString::number(i).rightJustified(4, '0');
            const auto path = dataset_ + "/images/val/" + name + ".png";
            QVERIFY(image.save(path));
            images_.append(QFileInfo(path).canonicalFilePath());
            QVERIFY(writeFile(dataset_ + "/labels/val/" + name + ".txt",
                i % 2 ? QByteArray() : QByteArray("0 .5 .5 .25 .25\n")));
        }
        QVERIFY(writeFile(dataset_ + "/data.yaml",
            "path: .\nval: images/val\nnames: [football]\n"));
        previousPythonSet_ = qEnvironmentVariableIsSet("VISION_STUDIO_PYTHON");
        previousPython_ = qgetenv("VISION_STUDIO_PYTHON");
        QString python = qEnvironmentVariable("VISION_STUDIO_EVALUATION_PYTHON");
        if (python.isEmpty()) python = "/opt/VisionStudio/runtime/bin/python";
        QVERIFY2(QFileInfo(python).isExecutable(), "Prepared evaluation Python is required.");
        qputenv("VISION_STUDIO_PYTHON", python.toUtf8());
    }

    void init()
    {
        QVERIFY(!QFileInfo::exists(sessionPath_) || QFile::remove(sessionPath_));
    }

    void cleanupTestCase()
    {
        if (previousPythonSet_) qputenv("VISION_STUDIO_PYTHON", previousPython_);
        else qunsetenv("VISION_STUDIO_PYTHON");
    }

    void orderedImagesAreAsynchronousBatchedAndDeduplicated()
    {
        auto input = controller();
        acknowledge(*input);
        QSignalSpy complete(input.get(), &InputSessionController::finished);
        QSignalSpy batches(input.get(), &InputSessionController::queueAppend);
        QStringList ordered = images_;
        std::reverse(ordered.begin(), ordered.end());
        ordered.insert(3, ordered.first());
        QVERIFY(input->requestImages(ordered));
        QCOMPARE(input->snapshot().phase, InputPhase::CheckingImages);
        QVERIFY(input->snapshot().files.isEmpty());
        QTRY_COMPARE_WITH_TIMEOUT(complete.size(), 1, 30000);
        QCOMPARE(input->snapshot().files.size(), 211);
        ordered.removeAt(3);
        QCOMPARE(input->snapshot().files, ordered);
        QVERIFY(batches.size() >= 3);
        int next = 0;
        for (const auto &arguments : batches)
        {
            const auto paths = arguments[0].toStringList();
            QVERIFY(paths.size() <= InputSessionController::MaximumBatchItems);
            QCOMPARE(arguments[1].toInt(), next);
            next += paths.size();
        }
        QCOMPARE(next, 211);
        QCOMPARE(input->snapshot().selectedRow, 0);
        QVERIFY(!QFileInfo::exists(sessionPath_));
    }

    void allTailCurrentAndUnselectedRanges()
    {
        auto input = controller();
        acknowledge(*input);
        QVERIFY(input->requestImages(images_));
        QTRY_VERIFY_WITH_TIMEOUT(!input->snapshot().importing, 30000);
        input->setSelection(52);
        QCOMPARE(input->range(ImageRange::All), images_);
        QCOMPARE(input->range(ImageRange::FromCurrent), images_.mid(52));
        QCOMPARE(input->range(ImageRange::Current), QStringList{images_.at(52)});
        input->setSelection(0);
        QCOMPARE(input->range(ImageRange::FromCurrent), images_);
        input->setSelection(210);
        QCOMPARE(input->range(ImageRange::FromCurrent), QStringList{images_.last()});
        input->setSelection(210, false);
        QString error;
        QVERIFY(input->range(ImageRange::Current, false, &error).isEmpty());
        QVERIFY(!error.isEmpty());
        QCOMPARE(input->range(ImageRange::All), images_);
        input->setSelection(-1);
        QVERIFY(input->range(ImageRange::FromCurrent, false, &error).isEmpty());
        QVERIFY(!error.isEmpty());
    }

    void invalidMixedBatchKeepsOldQueueAndSelection()
    {
        auto input = controller();
        acknowledge(*input);
        QVERIFY(input->requestImages(images_.mid(0, 2)));
        QTRY_VERIFY_WITH_TIMEOUT(!input->snapshot().importing, 30000);
        input->setSelection(1);
        QSignalSpy errors(input.get(), &InputSessionController::failed);
        QSignalSpy resets(input.get(), &InputSessionController::queueReset);
        QVERIFY(input->requestImages({images_.at(5), dataset_ + "/missing.png"}));
        QTRY_COMPARE_WITH_TIMEOUT(errors.size(), 1, 30000);
        QCOMPARE(input->snapshot().files, images_.mid(0, 2));
        QCOMPARE(input->snapshot().selectedRow, 1);
        QCOMPARE(resets.size(), 0);
        QVERIFY(!input->snapshot().importing);
    }

    void datasetPersistsOnlyAfterLastExactProjectionAcknowledgement()
    {
        auto input = controller();
        QSignalSpy ready(input.get(), &InputSessionController::datasetReady);
        QSignalSpy batches(input.get(), &InputSessionController::queueAppend);
        QSignalSpy errors(input.get(), &InputSessionController::failed);
        QSignalSpy complete(input.get(), &InputSessionController::finished);
        QVERIFY(input->loadDataset(dataset_));
        QTRY_VERIFY_WITH_TIMEOUT(!batches.isEmpty() || !errors.isEmpty(), 30000);
        QVERIFY2(errors.isEmpty(), errors.isEmpty() ? "" : qPrintable(errors.first().first().toString()));
        QCOMPARE(ready.size(), 1);
        int acknowledged = 0;
        while (input->snapshot().importing)
        {
            const auto state = input->snapshot();
            QCOMPARE(state.phase, InputPhase::WaitingForProjection);
            QVERIFY(state.files.size() > acknowledged);
            QVERIFY(!QFileInfo::exists(sessionPath_));
            const int before = batches.size();
            input->acknowledgeQueueProjection(state.projectionGeneration + 1, state.files.size());
            input->acknowledgeQueueProjection(state.projectionGeneration, state.files.size() - 1);
            QTest::qWait(5);
            QCOMPARE(batches.size(), before);
            acknowledged = state.files.size();
            input->acknowledgeQueueProjection(state.projectionGeneration, acknowledged);
            if (acknowledged < 211)
                QTRY_VERIFY_WITH_TIMEOUT(batches.size() > before, 10000);
        }
        QCOMPARE(acknowledged, 211);
        QCOMPARE(complete.size(), 1);
        QCOMPARE(input->snapshot().files, images_);
        QCOMPARE(input->snapshot().datasetSamples.size(), 211);
        const auto record = vision::DatasetSessionStore::load(sessionPath_);
        QCOMPARE(record.status, vision::DatasetSessionStatus::Loaded);
        QCOMPARE(record.session.sourcePath, dataset_);
        QCOMPARE(record.session.split, QString("val"));
    }

    void datasetTemporaryImagesAreExcludedFromEvaluationAndCancelRollsBack()
    {
        auto input = controller();
        auto ack = acknowledge(*input);
        QVERIFY(input->loadDataset(dataset_));
        QTRY_VERIFY_WITH_TIMEOUT(!input->snapshot().importing, 30000);
        const auto record = contents(sessionPath_);
        QVERIFY(!record.isEmpty());
        const auto extra = area_.path() + "/temporary.png";
        QVERIFY(QFile::copy(images_.first(), extra));
        QVERIFY(input->requestImages({extra}, true));
        QTRY_VERIFY_WITH_TIMEOUT(!input->snapshot().importing, 30000);
        QCOMPARE(input->snapshot().files.size(), 212);
        QCOMPARE(input->snapshot().datasetSamples.size(), 211);
        QCOMPARE(input->range(ImageRange::All, true), images_);
        QString error;
        QVERIFY(input->range(ImageRange::Current, true, &error).isEmpty());
        QVERIFY(!error.isEmpty());
        input->setSelection(52);
        QCOMPARE(input->range(ImageRange::FromCurrent, true), images_.mid(52));
        QCOMPARE(input->range(ImageRange::Current, true), QStringList{images_.at(52)});
        QCOMPARE(contents(sessionPath_), record);

        disconnect(ack);
        QSignalSpy batches(input.get(), &InputSessionController::queueAppend);
        QSignalSpy resets(input.get(), &InputSessionController::queueReset);
        const auto cancelledExtra = area_.path() + "/cancelled-temporary.png";
        QVERIFY(QFile::copy(images_.first(), cancelledExtra));
        QVERIFY(input->requestImages({cancelledExtra}, true));
        QTRY_VERIFY_WITH_TIMEOUT(!batches.isEmpty(), 30000);
        const auto obsolete = input->snapshot().projectionGeneration;
        QElapsedTimer elapsed;
        elapsed.start();
        input->cancel();
        QVERIFY(elapsed.elapsed() < 500);
        QCOMPARE(input->snapshot().files.size(), 212);
        QCOMPARE(input->snapshot().selectedRow, 52);
        QCOMPARE(input->snapshot().datasetSamples.size(), 211);
        QVERIFY(!input->snapshot().datasetManifest.isEmpty());
        QCOMPARE(resets.last().first().toStringList(), input->snapshot().files);
        input->acknowledgeQueueProjection(obsolete, 96);
        QTest::qWait(10);
        QCOMPARE(input->snapshot().files.size(), 212);
        QCOMPARE(contents(sessionPath_), record);
        QVERIFY(QFile::remove(extra));
        QVERIFY(QFile::remove(cancelledExtra));
    }

    void datasetFailurePreservesLastValidSession()
    {
        auto input = controller();
        acknowledge(*input);
        QVERIFY(input->loadDataset(dataset_));
        QTRY_VERIFY_WITH_TIMEOUT(!input->snapshot().importing, 30000);
        const auto original = contents(sessionPath_);
        const auto manifest = input->snapshot().datasetManifest;
        QSignalSpy errors(input.get(), &InputSessionController::failed);
        QVERIFY(input->loadDataset(area_.path() + "/not-a-dataset"));
        QTRY_COMPARE_WITH_TIMEOUT(errors.size(), 1, 30000);
        QCOMPARE(input->snapshot().files, images_);
        QCOMPARE(input->snapshot().datasetManifest, manifest);
        QCOMPARE(contents(sessionPath_), original);
    }

    void datasetProjectionCancellationKeepsOriginalRecord()
    {
        const vision::DatasetSession saved{dataset_, "val", {}};
        QString error;
        QVERIFY(vision::DatasetSessionStore::save(sessionPath_, saved, &error));
        const auto record = contents(sessionPath_);
        auto input = controller();
        const auto ack = acknowledge(*input);
        QVERIFY(input->requestImages({images_.last()}));
        QTRY_VERIFY_WITH_TIMEOUT(!input->snapshot().importing, 30000);
        disconnect(ack);
        QSignalSpy batches(input.get(), &InputSessionController::queueAppend);
        QVERIFY(input->loadDataset(dataset_));
        QTRY_VERIFY_WITH_TIMEOUT(!batches.isEmpty(), 30000);
        input->cancel();
        QVERIFY(input->snapshot().files.isEmpty());
        QVERIFY(input->snapshot().datasetManifest.isEmpty());
        QCOMPARE(contents(sessionPath_), record);
    }

    void datasetValidationCancellationPreservesPreviousInput()
    {
        auto input = controller();
        acknowledge(*input);
        QVERIFY(input->requestImages({images_.at(4)}));
        QTRY_VERIFY_WITH_TIMEOUT(!input->snapshot().importing, 30000);
        QVERIFY(input->loadDataset(dataset_));
        QCOMPARE(input->snapshot().phase, InputPhase::CheckingDataset);
        input->cancel();
        QCOMPARE(input->snapshot().files, QStringList{images_.at(4)});
        QCOMPARE(input->snapshot().selectedRow, 0);
        QVERIFY(input->snapshot().datasetManifest.isEmpty());
        QTRY_COMPARE_WITH_TIMEOUT(input->activeWorkerCount(), 0, 30000);
        QCOMPARE(input->snapshot().files, QStringList{images_.at(4)});
        QVERIFY(!QFileInfo::exists(sessionPath_));
    }

    void historicalManifestProjectionDoesNotOverwriteStartupSource()
    {
        QString error;
        QVERIFY(vision::DatasetSessionStore::save(sessionPath_, {dataset_, "test", {}}, &error));
        const auto record = contents(sessionPath_);
        const QJsonObject manifest{{"split", "val"}, {"samples", QJsonArray{
            QJsonObject{{"image", images_.first()}, {"gt", QJsonArray{}}}}}};
        auto input = controller();
        acknowledge(*input);
        QVERIFY(input->openDatasetSnapshot(manifest));
        QTRY_VERIFY_WITH_TIMEOUT(!input->snapshot().importing, 10000);
        QCOMPARE(input->snapshot().files, QStringList{images_.first()});
        QCOMPARE(input->snapshot().datasetSamples.size(), 1);
        QCOMPARE(contents(sessionPath_), record);
    }

    void deferredRestoreNeverOverridesManualInputOrDisabledPreference()
    {
        QString error;
        QVERIFY(vision::DatasetSessionStore::save(sessionPath_, {dataset_, "val", {}}, &error));
        auto input = controller();
        acknowledge(*input);
        input->restore();
        QVERIFY(input->requestImages({images_.last()}));
        QTRY_VERIFY_WITH_TIMEOUT(!input->snapshot().importing, 30000);
        QTest::qWait(30);
        QCOMPARE(input->snapshot().files, QStringList{images_.last()});
        QVERIFY(input->snapshot().datasetManifest.isEmpty());

        auto disabled = controller();
        disabled->setRestoreEnabled(false);
        disabled->restore();
        QTest::qWait(20);
        QVERIFY(disabled->snapshot().files.isEmpty());
        QCOMPARE(disabled->activeWorkerCount(), 0);

        auto guarded = controller();
        guarded->setRestoreGuard([] { return false; });
        guarded->restore();
        QTest::qWait(20);
        QCOMPARE(guarded->activeWorkerCount(), 0);
        QVERIFY(guarded->snapshot().files.isEmpty());
    }

    void inFlightRestoreCanBeSupersededAndNormalRestoreIsPersistent()
    {
        QString error;
        QVERIFY(vision::DatasetSessionStore::save(sessionPath_, {dataset_, "val", {}}, &error));
        auto input = controller();
        acknowledge(*input);
        input->restore();
        QTRY_COMPARE_WITH_TIMEOUT(input->snapshot().phase, InputPhase::CheckingDataset, 10000);
        QVERIFY(input->requestImages({images_.at(4)}));
        QTRY_VERIFY_WITH_TIMEOUT(!input->snapshot().importing, 30000);
        QTRY_COMPARE_WITH_TIMEOUT(input->activeWorkerCount(), 0, 30000);
        QCOMPARE(input->snapshot().files, QStringList{images_.at(4)});
        QVERIFY(input->snapshot().datasetManifest.isEmpty());

        auto reopened = controller();
        acknowledge(*reopened);
        reopened->restore();
        QTRY_COMPARE_WITH_TIMEOUT(reopened->snapshot().files.size(), 211, 30000);
        QTRY_VERIFY_WITH_TIMEOUT(!reopened->snapshot().importing, 30000);
        QCOMPARE(reopened->snapshot().datasetSamples.size(), 211);
        QVERIFY(!reopened->snapshot().userInteracted);
        QCOMPARE(vision::DatasetSessionStore::load(sessionPath_).session.sourcePath, dataset_);
    }

    void clearVideoCameraAndShutdownDoNotEraseRememberedDataset()
    {
        QString error;
        QVERIFY(vision::DatasetSessionStore::save(sessionPath_, {dataset_, "val", {}}, &error));
        const auto record = contents(sessionPath_);
        auto input = controller();
        QVERIFY(input->selectVideo(images_.first()));
        QCOMPARE(input->snapshot().sourceKind, vision::SourceKind::Video);
        QVERIFY(input->snapshot().files.isEmpty());
        QCOMPARE(input->snapshot().streamPath, images_.first());
        QVERIFY(input->selectCamera());
        QCOMPARE(input->snapshot().sourceKind, vision::SourceKind::Camera);
        QVERIFY(input->snapshot().streamPath.isEmpty());
        input->clear();
        QCOMPARE(input->snapshot().sourceKind, vision::SourceKind::Images);
        QCOMPARE(contents(sessionPath_), record);
        QVERIFY(input->loadDataset(dataset_));
        QSignalSpy shutdown(input.get(), &InputSessionController::shutdownReady);
        QElapsedTimer elapsed;
        elapsed.start();
        input->requestShutdown();
        QVERIFY(elapsed.elapsed() < 500);
        QVERIFY(!input->requestImages({images_.first()}));
        QTRY_COMPARE_WITH_TIMEOUT(shutdown.size(), 1, 30000);
        QVERIFY(input->isShutdownReady());
        input->requestShutdown();
        QCOMPARE(shutdown.size(), 1);
        QCOMPARE(contents(sessionPath_), record);
    }

    void shutdownDuringStartupNotificationDoesNotLaunchWorker_data()
    {
        QTest::addColumn<bool>("dataset");
        QTest::addColumn<bool>("replaceExisting");
        QTest::newRow("images-checking-state") << false << false;
        QTest::newRow("dataset-checking-state") << true << false;
        QTest::newRow("replacement-cancelling-state") << false << true;
    }

    void shutdownDuringStartupNotificationDoesNotLaunchWorker()
    {
        QFETCH(bool, dataset);
        QFETCH(bool, replaceExisting);
        auto input = controller();
        if (replaceExisting) QVERIFY(input->requestImages({images_.first()}));
        QSignalSpy ready(input.get(), &InputSessionController::shutdownReady);
        bool closed = false, readyWhenClosed = false;
        int workersWhenClosed = -1;
        connect(input.get(), &InputSessionController::stateChanged, input.get(), [&] {
            const auto phase = input->snapshot().phase;
            const auto expected = replaceExisting ? InputPhase::Idle
                : dataset ? InputPhase::CheckingDataset : InputPhase::CheckingImages;
            if (closed || phase != expected) return;
            closed = true;
            workersWhenClosed = input->activeWorkerCount();
            input->requestShutdown();
            readyWhenClosed = input->isShutdownReady();
        }, Qt::DirectConnection);
        const bool accepted = dataset ? input->loadDataset(dataset_)
            : input->requestImages({images_.last()});
        const int workersAfterReturn = input->activeWorkerCount();
        const bool readyAfterReturn = input->isShutdownReady();
        qInfo() << "startup-close" << "accepted" << accepted
                << "workers-at-close" << workersWhenClosed << "workers-after-return" << workersAfterReturn
                << "ready-at-close" << readyWhenClosed << "ready-after-return" << readyAfterReturn;

        // Drain even on the old implementation, so a failing assertion cannot
        // leave its newly launched worker running beyond the fixture lifetime.
        QEventLoop loop;
        QTimer::singleShot(1, &loop, &QEventLoop::quit);
        loop.exec();
        QTRY_COMPARE_WITH_TIMEOUT(input->activeWorkerCount(), 0, 30000);
        QTRY_VERIFY_WITH_TIMEOUT(input->isShutdownReady(), 30000);
        QCOMPARE(ready.count(), 1);
        QVERIFY(closed);
        QVERIFY2(!accepted, "The startup command continued after its state notification closed the controller.");
        QVERIFY2(workersAfterReturn <= workersWhenClosed, "A new worker was registered after shutdown.");
        QVERIFY2(!readyWhenClosed || readyAfterReturn, "Readiness became false again after shutdown had been ready.");
        QCOMPARE(input->snapshot().phase, InputPhase::Idle);
        QVERIFY(!input->requestImages({images_.first()}));
        QVERIFY(!input->loadDataset(dataset_));
        QVERIFY(!QFileInfo::exists(sessionPath_));
    }
};
QTEST_GUILESS_MAIN(InputSessionControllerTests)
#include "input_session_controller_tests.moc"
