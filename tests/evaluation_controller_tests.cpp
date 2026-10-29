#include "app/evaluationcontroller.h"
#include "app/asyncthreadretirement.h"

#include <QCryptographicHash>
#include <QDateTime>
#include <QDir>
#include <QElapsedTimer>
#include <QEventLoop>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QSignalSpy>
#include <QTemporaryDir>
#include <QTimer>
#include <QtTest>
#include <stdexcept>

using namespace vision::app;
namespace
{
void write(const QString &path, const QByteArray &bytes)
{
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly) || file.write(bytes) != bytes.size()) throw std::runtime_error("fixture write failed");
}
QString hash(const QString &path)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) return {};
    return QString::fromLatin1(QCryptographicHash::hash(file.readAll(), QCryptographicHash::Sha256).toHex());
}
QJsonObject box(int id, double score = -1)
{
    QJsonObject value{{"class_id", id}, {"box", QJsonArray{10, 10, 30, 30}}};
    if (score >= 0) { value.insert("confidence", score); value.insert("label", "sports ball"); }
    return value;
}
struct Fixture
{
    QTemporaryDir directory;
    QString modelPath;
    QStringList images, files;
    QHash<QString, QJsonObject> samples;
    QJsonObject manifest, report;
    Fixture()
    {
        QDir().mkpath(directory.filePath("samples"));
        modelPath = directory.filePath("model.pt"); write(modelPath, "trusted fixture model");
        QJsonArray manifestSamples, selected, completed;
        for (int i = 0; i != 4; ++i)
        {
            const QString path = directory.filePath(QString::number(i) + ".png");
            QImage image(64, 64, QImage::Format_RGB888); image.fill(QColor(10 + i, 30, 50));
            if (!image.save(path)) throw std::runtime_error("fixture PNG write failed");
            images.append(path);
            const QJsonArray gt = i == 2 ? QJsonArray{} : QJsonArray{box(0)};
            const QJsonObject sample{{"image", path}, {"width", 64}, {"height", 64}, {"gt", gt},
                {"image_sha256", hash(path)}, {"label_sha256", QString(64, 'a')}};
            samples.insert(path, sample); manifestSamples.append(sample);
            if (i < 3)
            {
                auto cached = sample;
                cached.insert("predictions", i == 1 ? QJsonArray{} : QJsonArray{box(32, .8)});
                cached.insert("backend", "fixture cached predictions");
                const auto bytes = QJsonDocument(cached).toJson();
                const auto base = directory.filePath("samples/" + QString::number(i));
                write(base + ".json", bytes);
                write(base + ".sha256", QCryptographicHash::hash(bytes, QCryptographicHash::Sha256).toHex() + '\n');
                selected.append(path); completed.append(QJsonObject{});
            }
        }
        files = {images[2], images[1], images[0], images[3], directory.filePath("temporary.png")};
        manifest = {{"samples", manifestSamples}, {"names", QJsonArray{"football"}}, {"split", "val"},
                    {"data_yaml", directory.filePath("data.yaml")}, {"fingerprint", "dataset-snapshot"}};
        vision::ModelConfig config;
        config.modelPath = modelPath; config.modelClassCount = 80; config.inputChannels = 3;
        config.inputSize = 64; config.confidence = .55f; config.maxDetections = 100;
        report = {{"schema_version", 1}, {"status", "complete"}, {"full_split", false},
            {"manifest", manifest}, {"per_image", completed},
            {"request", QJsonObject{{"sample_images", selected}, {"model_path", modelPath},
                {"model_sha256", hash(modelPath)}, {"model_class_id", 32}, {"dataset_class_id", 0},
                {"model_identity", "recorded-identity"}, {"confidence", .55}, {"match_iou", .5}}},
            {"config", vision::EvaluationWorker::modelConfigJson(config)},
            {"metrics", QJsonObject{{"gt_boxes", 2}}},
            {"rows", QJsonArray{QJsonObject{{"confidence", .25}}, QJsonObject{{"confidence", .5}}, QJsonObject{{"confidence", .75}}}},
            {"groups", QJsonObject{{"No Ball", QJsonObject{{"available", true}, {"sample_images", QJsonArray{images[2]}}}}}}};
        saveReport();
    }
    void saveReport()
    {
        write(directory.filePath("report.json"), QJsonDocument(report).toJson());
        write(directory.filePath("status.json"), QJsonDocument(QJsonObject{{"status", "complete"}}).toJson());
    }
    void initialize(EvaluationController &controller)
    {
        controller.setReviewContext(files, samples, manifest);
        controller.acceptReport(report, directory.path(), true);
    }
};
struct EvaluationThreadExit
{
    std::atomic_bool cleanupDone{false};
    bool attached = false, finishedObserved = false, cleanupDoneAtFinished = true, busyAtFinished = false, readyAtFinished = true;
};
class SlowEvaluationDeferredCleanup final : public QObject
{
    std::shared_ptr<EvaluationThreadExit> state_;
  public:
    explicit SlowEvaluationDeferredCleanup(std::shared_ptr<EvaluationThreadExit> state) : state_(std::move(state)) {}
    ~SlowEvaluationDeferredCleanup() override { QThread::msleep(1200); state_->cleanupDone.store(true); }
};
}

class EvaluationControllerTests final : public QObject
{
    Q_OBJECT
    void drain(EvaluationController &controller)
    {
        QSignalSpy ready(&controller, &EvaluationController::shutdownReady);
        controller.requestShutdown();
        QTRY_COMPARE_WITH_TIMEOUT(ready.count(), 1, 5000);
        QVERIFY(controller.isShutdownReady());
    }
  private slots:
    void reportLoadAndThresholdUseImmutableSnapshot()
    {
        Fixture fixture;
        const auto loaded = EvaluationController::readReport(fixture.directory.path());
        QVERIFY2(loaded.valid, qPrintable(loaded.error));
        QCOMPARE(loaded.manifest, fixture.manifest);
        EvaluationController controller; fixture.initialize(controller);
        const auto state = controller.snapshot();
        QCOMPARE(state.report, fixture.report);
        QCOMPARE(state.runConfig.modelPath, fixture.modelPath);
        QCOMPARE(state.review.threshold, .5);
        QCOMPARE(state.sampleIndices.value(fixture.images[0]), 0);
        QCOMPARE(state.sampleIndices.value(fixture.images[2]), 2);
        QVERIFY(!state.sampleIndices.contains(fixture.images[3]));
        QVERIFY(state.runFingerprint.isEmpty());
        auto copy = state; copy.report = {};
        QCOMPARE(controller.snapshot().report, fixture.report);
        write(fixture.directory.filePath("status.json"), "{\"status\":\"saving\"}");
        QVERIFY(!EvaluationController::readReport(fixture.directory.path()).valid);
        drain(controller);
    }

    void errorsPreserveQueueOrderAndExcludeUnknownSamples()
    {
        Fixture fixture; EvaluationController controller; fixture.initialize(controller);
        QSignalSpy filtered(&controller, &EvaluationController::filterReady);
        QVERIFY(controller.applyFilter("errors", .5, 2, true));
        QTRY_COMPARE_WITH_TIMEOUT(filtered.count(), 1, 5000);
        const auto outcome = qvariant_cast<EvaluationFilterOutcome>(filtered.first().first());
        QVERIFY2(outcome.valid, qPrintable(outcome.error));
        QVERIFY(outcome.selectFirst);
        QCOMPARE(outcome.visible, QVector<bool>({true, true, false, false, false}));
        QCOMPARE(outcome.selectedRow, 0);
        const auto review = controller.snapshot().review;
        QCOMPARE(review.rows, QVector<int>({0, 1}));
        QCOMPARE(review.errorRows, QVector<int>({0, 1}));
        QCOMPARE(review.evaluated, 3); QCOMPARE(review.falsePositive, 1);
        QCOMPARE(review.falseNegative, 1); QCOMPARE(review.negativeFalsePositive, 1);
        QCOMPARE(controller.navigationTarget(0, 1), 1);
        QCOMPARE(controller.navigationTarget(1, 1), -1);
        QCOMPARE(controller.navigationTarget(0, -1), -1);
        drain(controller);
    }

    void allFilterKeepsTemporarySelectionAndZeroMatchIsValid()
    {
        Fixture fixture; EvaluationController controller; fixture.initialize(controller);
        QSignalSpy filtered(&controller, &EvaluationController::filterReady);
        QVERIFY(controller.applyFilter("all", .5, 4));
        QTRY_COMPARE(filtered.count(), 1);
        auto outcome = qvariant_cast<EvaluationFilterOutcome>(filtered.last().first());
        QVERIFY(outcome.valid); QCOMPARE(outcome.selectedRow, 4);
        QVERIFY(!outcome.selectFirst);
        QCOMPARE(outcome.visible, QVector<bool>(5, true));
        QCOMPARE(controller.snapshot().review.rows, QVector<int>({0, 1, 2, 3}));
        QVERIFY(controller.applyFilter("negative_fp", .95, 0));
        QTRY_COMPARE(filtered.count(), 2);
        outcome = qvariant_cast<EvaluationFilterOutcome>(filtered.last().first());
        QVERIFY(outcome.valid); QCOMPARE(outcome.selectedRow, -1);
        QVERIFY(controller.snapshot().review.rows.isEmpty());
        QCOMPARE(controller.snapshot().review.evaluated, 3);
        QCOMPARE(controller.snapshot().review.negativeFalsePositive, 0);
        drain(controller);
    }

    void sceneFilterAndRapidThresholdChangesDiscardOldCallbacks()
    {
        Fixture fixture; EvaluationController controller; fixture.initialize(controller);
        QSignalSpy filtered(&controller, &EvaluationController::filterReady);
        QVERIFY(controller.applyFilter("errors", .5, 2));
        QVERIFY(controller.applyFilter("scene:No Ball", .5, 2));
        QTRY_COMPARE_WITH_TIMEOUT(filtered.count(), 1, 5000);
        const auto outcome = qvariant_cast<EvaluationFilterOutcome>(filtered.last().first());
        QVERIFY(outcome.valid); QCOMPARE(controller.snapshot().review.rows, QVector<int>{0});
        QCOMPARE(controller.snapshot().review.filter, QString("scene:No Ball"));
        QCOMPARE(controller.snapshot().review.evaluated, 1);
        QString error;
        QVERIFY(!controller.applyFilter("not-a-filter", .5, 0, false, &error)); QVERIFY(!error.isEmpty());
        QVERIFY(!controller.applyFilter("scene:Unknown", .5, 0, false, &error));
        drain(controller);
    }

    void previewKeepsRawClassAndUncompletedGroundTruth()
    {
        Fixture fixture; EvaluationController controller; fixture.initialize(controller);
        QSignalSpy preview(&controller, &EvaluationController::previewReady);
        QVERIFY(controller.requestPreview(2));
        QTRY_COMPARE_WITH_TIMEOUT(preview.count(), 1, 5000);
        auto outcome = qvariant_cast<EvaluationPreviewOutcome>(preview.last().first());
        QVERIFY2(outcome.preview.error.isEmpty(), qPrintable(outcome.preview.error));
        QVERIFY(outcome.preview.completed); QCOMPARE(outcome.preview.result.predictions.size(), 1);
        QCOMPARE(outcome.preview.result.predictions.first().classId, 32);
        QCOMPARE(outcome.preview.result.groundTruth.first().classId, 0);
        QCOMPARE(outcome.config.classFilter, QVector<int>{32});
        QCOMPARE(outcome.config.confidence, .5f);
        QVERIFY(controller.requestPreview(3));
        QTRY_COMPARE_WITH_TIMEOUT(preview.count(), 2, 5000);
        outcome = qvariant_cast<EvaluationPreviewOutcome>(preview.last().first());
        QVERIFY2(outcome.preview.error.isEmpty(), qPrintable(outcome.preview.error));
        QVERIFY(!outcome.preview.completed); QVERIFY(!outcome.preview.result.image.isNull());
        QVERIFY(outcome.preview.result.predictions.isEmpty());
        QVERIFY(!outcome.preview.result.hasEvaluationMatches);
        drain(controller);
    }

    void corruptCacheIsUnavailableRatherThanZeroErrors()
    {
        Fixture fixture; EvaluationController controller; fixture.initialize(controller);
        write(fixture.directory.filePath("samples/0.json"), "{}");
        QSignalSpy filtered(&controller, &EvaluationController::filterReady);
        QVERIFY(controller.applyFilter("errors", .5, 0));
        QTRY_COMPARE(filtered.count(), 1);
        const auto outcome = qvariant_cast<EvaluationFilterOutcome>(filtered.first().first());
        QVERIFY(!outcome.valid); QVERIFY(!outcome.error.isEmpty());
        QVERIFY(!controller.snapshot().review.available);
        QSignalSpy preview(&controller, &EvaluationController::previewReady);
        QVERIFY(controller.requestPreview(2)); QTRY_COMPARE(preview.count(), 1);
        const auto rejected = qvariant_cast<EvaluationPreviewOutcome>(preview.first().first());
        QVERIFY(!rejected.preview.error.isEmpty()); QVERIFY(rejected.preview.result.image.isNull());
        drain(controller);
    }

    void contextReplacementRejectsObsoletePreviewAndFilter()
    {
        Fixture fixture; EvaluationController controller; fixture.initialize(controller);
        QSignalSpy preview(&controller, &EvaluationController::previewReady);
        QSignalSpy filtered(&controller, &EvaluationController::filterReady);
        QVERIFY(controller.requestPreview(2));
        QVERIFY(controller.applyFilter("all", .5, 2));
        controller.setReviewContext({}, {}, {});
        QTRY_COMPARE_WITH_TIMEOUT(controller.activeReviewThreadCount(), 0, 5000);
        QCOMPARE(preview.count(), 0); QCOMPARE(filtered.count(), 0);
        drain(controller);
    }

    void adoptionRequiresExplicitMatchingModelAndHash()
    {
        Fixture fixture; EvaluationController controller; fixture.initialize(controller);
        const auto originalConfig = controller.snapshot().runConfig;
        auto patch = controller.configurationPatch(fixture.modelPath, .75, {{"preserved", true}});
        QVERIFY2(patch.valid, qPrintable(patch.error));
        QCOMPARE(patch.config.classFilter, QVector<int>{32});
        QCOMPARE(patch.config.confidence, .75f); QCOMPARE(patch.profile.value("adopted").toBool(), true);
        QCOMPARE(patch.profile.value("preserved").toBool(), true);
        // Constructing the patch never rewrites the immutable source report/config.
        QCOMPARE(controller.snapshot().runConfig.confidence, originalConfig.confidence);
        QVERIFY(!controller.configurationPatch(fixture.directory.filePath("other.pt"), .5).valid);
        write(fixture.modelPath, "replacement model with different hash");
        QVERIFY(!controller.configurationPatch(fixture.modelPath, .5).valid);
        auto report = fixture.report; report.insert("allow_apply_profile", false);
        controller.acceptReport(report, fixture.directory.path(), true);
        QVERIFY(!controller.configurationPatch(fixture.modelPath, .5).valid);
        drain(controller);
    }

    void historyAndAssociationKeepExistingSchema()
    {
        Fixture fixture; EvaluationController controller; fixture.initialize(controller);
        const auto history = controller.historyEntry("显示名称");
        QCOMPARE(history.value("kind").toString(), QString("evaluation"));
        QCOMPARE(history.value("model").toString(), QString("显示名称"));
        QCOMPARE(history.value("task").toString(), QString("评估 · 子集"));
        QCOMPARE(history.value("split").toString(), QString("val"));
        const QString sha = fixture.report.value("request").toObject().value("model_sha256").toString();
        auto association = controller.reportAssociation({{"model_sha256", sha}, {"adopted", true}});
        QVERIFY(association.value("adopted").toBool());
        association = controller.reportAssociation({{"model_sha256", "other"}, {"adopted", true}});
        QVERIFY(!association.contains("adopted"));
        QCOMPARE(association.value("report_directory").toString(), fixture.directory.path());
        drain(controller);
    }

    void fingerprintIgnoresAnalysisOnlyValues()
    {
        Fixture fixture;
        vision::EvaluationRequest request;
        request.config.modelPath = fixture.modelPath; request.manifest = fixture.manifest;
        const auto baseline = EvaluationController::configurationFingerprint(request);
        request.config.confidence = .9f; request.config.labels = {"renamed"}; request.config.classFilter = {32};
        QCOMPARE(EvaluationController::configurationFingerprint(request), baseline);
        request.config.iou = .7f;
        QVERIFY(EvaluationController::configurationFingerprint(request) != baseline);
    }

    void workerFailureAndShutdownDoNotWaitOnGuiThread()
    {
        EvaluationController controller;
        vision::EvaluationRequest request;
        request.config.task = vision::ModelTask::Classification;
        request.projectRoot = QString(VISION_PROJECT_DIR);
        QSignalSpy failed(&controller, &EvaluationController::failed);
        QSignalSpy finished(&controller, &EvaluationController::finished);
        QSignalSpy shutdown(&controller, &EvaluationController::shutdownReady);
        QVERIFY(controller.start(request, "task-fingerprint"));
        QVERIFY(controller.isBusy()); QVERIFY(!controller.start(request, "another"));
        QElapsedTimer elapsed; elapsed.start(); controller.requestShutdown();
        QVERIFY(elapsed.elapsed() < 100);
        QTRY_COMPARE_WITH_TIMEOUT(shutdown.count(), 1, 5000);
        QCOMPARE(finished.count(), 1); QCOMPARE(failed.count(), 1);
        QVERIFY(!controller.isBusy()); QVERIFY(controller.isShutdownReady());
        QCOMPARE(controller.snapshot().runFingerprint, QString("task-fingerprint"));
    }

    void shutdownDrainsAllReviewThreads()
    {
        Fixture fixture; EvaluationController controller; fixture.initialize(controller);
        QVERIFY(controller.requestPreview(2)); QVERIFY(controller.applyFilter("all", .5, 2));
        QCOMPARE(controller.activeReviewThreadCount(), 2);
        drain(controller);
        QCOMPARE(controller.activeReviewThreadCount(), 0);
        QVERIFY(AsyncThreadRetirement::watchedThreads(&controller).isEmpty());
    }

    void shutdownWaitsForWorkerDeferredCleanupWithoutBlockingGui()
    {
        Fixture fixture;
        EvaluationController controller;
        const auto exit = std::make_shared<EvaluationThreadExit>();
        connect(&controller, &EvaluationController::busyChanged, &controller, [&](bool busy) {
            if (!busy || exit->attached) return;
            const auto watched = AsyncThreadRetirement::watchedThreads(&controller);
            if (watched.size() != 1 || watched.front()->isFinished()) return;
            auto *thread = watched.front();
            auto *cleanup = new SlowEvaluationDeferredCleanup(exit);
            cleanup->moveToThread(thread);
            connect(thread, &QThread::finished, cleanup, &QObject::deleteLater, Qt::DirectConnection);
            connect(thread, &QThread::finished, this, [&, exit] {
                exit->finishedObserved = true;
                exit->cleanupDoneAtFinished = exit->cleanupDone.load();
                exit->busyAtFinished = controller.isBusy();
                exit->readyAtFinished = controller.isShutdownReady();
            }, Qt::QueuedConnection);
            exit->attached = true;
        });
        QTimer heartbeat; heartbeat.setInterval(1);
        QElapsedTimer gap; gap.start(); qint64 maximumGap = 0; int ticks = 0;
        connect(&heartbeat, &QTimer::timeout, this, [&] { maximumGap = qMax(maximumGap, gap.restart()); ++ticks; });
        heartbeat.start();
        QSignalSpy ready(&controller, &EvaluationController::shutdownReady), finished(&controller, &EvaluationController::finished);
        const bool started = controller.importExternal(fixture.directory.path(), {},
            fixture.directory.filePath("cancelled-thread-exit"), QString(VISION_PROJECT_DIR));
        QElapsedTimer shutdownCall; shutdownCall.start(); controller.requestShutdown();
        QVERIFY(shutdownCall.elapsed() < 100);
        QTRY_COMPARE_WITH_TIMEOUT(ready.count(), 1, 5000);
        QTest::qWait(5); heartbeat.stop();
        QVERIFY(started); QVERIFY(exit->attached); QVERIFY(exit->finishedObserved);
        QVERIFY(!exit->cleanupDoneAtFinished); QVERIFY(exit->busyAtFinished); QVERIFY(!exit->readyAtFinished);
        QVERIFY(exit->cleanupDone.load()); QCOMPARE(finished.count(), 1);
        QVERIFY(!controller.isBusy()); QVERIFY(controller.isShutdownReady());
        QVERIFY(AsyncThreadRetirement::watchedThreads(&controller).isEmpty());
        QVERIFY(ticks > 5);
        QVERIFY2(maximumGap < 300, qPrintable(QString("GUI heartbeat blocked for %1 ms").arg(maximumGap)));
        qInfo() << "evaluation_thread_exit_evidence" << "deferred_cleanup_ms" << 1200
                << "heartbeat_ticks" << ticks << "maximum_gap_ms" << maximumGap;
    }

    void shutdownDuringStartupNotificationDoesNotLaunchWorker_data()
    {
        QTest::addColumn<QString>("operation");
        QTest::newRow("evaluation") << QString("evaluation");
        QTest::newRow("reanalysis") << QString("reanalysis");
        QTest::newRow("external-import") << QString("external-import");
    }

    void shutdownDuringStartupNotificationDoesNotLaunchWorker()
    {
        QFETCH(QString, operation);
        Fixture fixture;
        EvaluationController controller;
        fixture.initialize(controller);
        QSignalSpy ready(&controller, &EvaluationController::shutdownReady);
        QSignalSpy busy(&controller, &EvaluationController::busyChanged);
        bool closed = false, readyWhenClosed = false;
        connect(&controller, &EvaluationController::reviewChanged, &controller, [&] {
            if (closed) return;
            closed = true;
            controller.requestShutdown();
            readyWhenClosed = controller.isShutdownReady();
        }, Qt::DirectConnection);
        vision::EvaluationRequest request;
        request.config.task = vision::ModelTask::Classification;
        request.projectRoot = QString(VISION_PROJECT_DIR);
        const auto output = fixture.directory.filePath("new-results");
        QString error;
        const bool accepted = operation == "evaluation" ? controller.start(request, "closing-task", &error)
            : operation == "reanalysis" ? controller.reanalyze({}, output, request.projectRoot, &error)
            : controller.importExternal(fixture.directory.path(), {}, output, request.projectRoot, &error);
        const bool busyAfterReturn = controller.isBusy();
        const bool readyAfterReturn = controller.isShutdownReady();
        int launches = 0;
        for (const auto &event : busy) if (event.first().toBool()) ++launches;
        qInfo() << "startup-close" << operation << "accepted" << accepted << "launches" << launches
                << "ready-at-close" << readyWhenClosed << "ready-after-return" << readyAfterReturn;

        // Explicitly stop and drain a worker accidentally launched by the old
        // implementation before asserting, keeping the red test self-contained.
        controller.requestStop();
        QEventLoop loop;
        QTimer::singleShot(1, &loop, &QEventLoop::quit);
        loop.exec();
        QTRY_VERIFY_WITH_TIMEOUT(!controller.isBusy(), 5000);
        QTRY_COMPARE_WITH_TIMEOUT(ready.count(), 1, 5000);
        QVERIFY(controller.isShutdownReady());
        QVERIFY(closed);
        QVERIFY2(!accepted, "The startup command continued after its review notification closed the controller.");
        QVERIFY2(!busyAfterReturn && launches == 0, "A new evaluation worker was launched after shutdown.");
        QVERIFY2(!readyWhenClosed || readyAfterReturn, "Readiness became false again after shutdown had been ready.");
        QVERIFY(!controller.start(request, "after-close", &error));
        QVERIFY(!controller.reanalyze({}, output, request.projectRoot, &error));
        QVERIFY(!controller.importExternal(fixture.directory.path(), {}, output, request.projectRoot, &error));
        QVERIFY(!QFileInfo::exists(output));
    }
};
QTEST_GUILESS_MAIN(EvaluationControllerTests)
#include "evaluation_controller_tests.moc"
