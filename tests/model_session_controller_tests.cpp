#include "app/modelsessioncontroller.h"

#include <QElapsedTimer>
#include <QFile>
#include <QJsonArray>
#include <QSemaphore>
#include <QSignalSpy>
#include <QTemporaryDir>
#include <QThread>
#include <QtTest>
#include <stdexcept>

using namespace vision::app;
namespace
{
QString model(const QTemporaryDir &directory, const QString &name)
{
    const auto path = directory.filePath(name);
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly) || file.write("model fixture") < 0) return {};
    return path;
}
vision::ModelConfig inspected(vision::ModelConfig config, int count)
{
    config.modelClassCount = count;
    config.inputChannels = 3;
    config.labels = {"native"};
    return config;
}
QJsonObject convertedMetadata(const QString &path)
{
    return {{"format", "onnx"}, {"output", path}, {"task", "detect"}, {"layout", "v8"},
            {"input_channels", 1}, {"ch", 1}, {"names", QJsonObject{{"0", "football"}}},
            {"precision", "fp32"}, {"nms", false}, {"dynamic", false},
            {"shape", QJsonArray{1, 1, 640, 640}}, {"bytes", double(QFileInfo(path).size())}};
}
}

class ModelSessionControllerTests final : public QObject
{
    Q_OBJECT
  private slots:
    void canonicalOrderedImportAndSingleSelection()
    {
        QTemporaryDir directory;
        const auto first = model(directory, "first.onnx");
        const auto second = model(directory, "second.pt");
        const auto alias = directory.filePath("alias.onnx");
        QVERIFY(QFile::link(first, alias));
        ModelSessionController controller;
        QSignalSpy selected(&controller, &ModelSessionController::selectionChanged);
        int added = -1;
        QString error;
        QVERIFY2(controller.importOrdered({first, alias, second}, &added, &error), qPrintable(error));
        QCOMPARE(added, 2);
        QCOMPARE(controller.snapshot().models, QStringList({first, second}));
        QCOMPARE(controller.snapshot().activeModel, second);
        QCOMPARE(selected.count(), 1);
        QVERIFY(controller.rename(first, "实验一"));
        QVERIFY(controller.setNote(first, "备注"));
        QVERIFY(controller.importOrdered({alias, second}, &added));
        QCOMPARE(added, 0);
        QCOMPARE(selected.count(), 1); // Reimporting the active path never reloads its graph.
        QCOMPARE(controller.displayName(first), QString("实验一"));
        QCOMPARE(controller.snapshot().notes.value(first).toString(), QString("备注"));
        auto copy = controller.snapshot(); copy.models.clear();
        QCOMPARE(controller.snapshot().models.size(), 2);
    }

    void invalidBatchHasNoPartialChanges()
    {
        QTemporaryDir directory;
        const auto first = model(directory, "first.onnx");
        const auto second = model(directory, "second.pt");
        const auto invalid = model(directory, "not-a-model.txt");
        ModelSessionController controller;
        QVERIFY(controller.importOrdered({first}));
        const auto before = controller.snapshot();
        QString error;
        QVERIFY(!controller.importOrdered({second, invalid}, nullptr, &error));
        QVERIFY(!error.isEmpty());
        QCOMPARE(controller.snapshot().models, before.models);
        QCOMPARE(controller.snapshot().activeModel, before.activeModel);
        QVERIFY(!controller.importOrdered({}, nullptr, &error));
        QVERIFY(!controller.select(directory.filePath("missing.onnx"), false, &error));
        QCOMPARE(controller.snapshot().activeModel, first);
    }

    void libraryEditsPreservePathsAndProfiles()
    {
        QTemporaryDir directory;
        const auto first = model(directory, "first.onnx");
        const auto second = model(directory, "second.pt");
        ModelSessionController controller;
        QVERIFY(controller.importOrdered({first, second}));
        QVERIFY(controller.reorder({second, first}));
        QVERIFY(!controller.reorder({first}));
        QVERIFY(controller.rename(first, "  名称    一  "));
        QCOMPARE(controller.displayName(first), QString("名称 一"));
        QVERIFY(controller.setNote(first, "  说明\n第二行  "));
        QCOMPARE(controller.snapshot().notes.value(first).toString(), QString("说明\n第二行"));
        controller.setEvaluationProfile(first, {{"report_directory", "report"}});
        QVERIFY(controller.select(first));
        QVERIFY(controller.remove(first));
        QCOMPARE(controller.snapshot().activeModel, second);
        QCOMPARE(controller.snapshot().evaluationProfiles.value(first).toMap().value("report_directory").toString(), QString("report"));
        QCOMPARE(controller.snapshot().notes.value(first).toString(), QString("说明\n第二行"));
        QVERIFY(QFileInfo::exists(first));
        QVERIFY(controller.importOrdered({first}));
        QCOMPARE(controller.displayName(first), QString("名称 一"));
        QVERIFY(controller.rename(first, " "));
        QCOMPARE(controller.displayName(first), QString("first.onnx"));
        QVERIFY(controller.remove(first));
        QVERIFY(controller.remove(second));
        QVERIFY(controller.snapshot().activeModel.isEmpty());
    }

    void convertedProfileUsesExactFileIdentity()
    {
        QTemporaryDir directory;
        const auto path = model(directory, "converted.onnx");
        ModelSessionController controller;
        auto metadata = convertedMetadata(path);
        QString error;
        QVERIFY2(controller.rememberConverted(path, metadata, &error), qPrintable(error));
        QVERIFY(controller.select(path));
        QCOMPARE(controller.snapshot().inputChannels, 1);
        QCOMPARE(controller.snapshot().nativeLabels, QStringList{"football"});
        QCOMPARE(controller.convertedProfile(path), metadata);
        auto broken = metadata; broken.insert("nms", true);
        QVERIFY(!controller.rememberConverted(path, broken, &error));
        QCOMPARE(controller.convertedProfile(path), metadata);
        QFile file(path); QVERIFY(file.open(QIODevice::Append)); file.write("replacement"); file.close();
        QVERIFY(controller.convertedProfile(path).isEmpty());
        QVERIFY(controller.select(path)); // Same path with stale profile resets metadata.
        QCOMPARE(controller.snapshot().inputChannels, 0);
    }

    void oldInspectionCannotOverwriteNewSelection()
    {
        QTemporaryDir directory;
        const auto first = model(directory, "first.onnx");
        const auto second = model(directory, "second.onnx");
        const auto firstGate = std::make_shared<QSemaphore>();
        ModelSessionController controller([first, firstGate](const vision::ModelConfig &config, std::function<bool()>)
        {
            if (config.modelPath == first) firstGate->tryAcquire(1, 3000);
            return inspected(config, config.modelPath == first ? 80 : 2);
        });
        QVERIFY(controller.select(first)); controller.inspect({});
        QVERIFY(controller.select(second)); controller.inspect({});
        QTRY_COMPARE_WITH_TIMEOUT(controller.snapshot().classCount, 2, 3000);
        QCOMPARE(controller.activeInspectionCount(), 1);
        firstGate->release();
        QTRY_COMPARE_WITH_TIMEOUT(controller.activeInspectionCount(), 0, 3000);
        QCOMPARE(controller.snapshot().classCount, 2);
        QCOMPARE(controller.snapshot().activeModel, second);
        QSignalSpy shutdown(&controller, &ModelSessionController::shutdownReady);
        controller.requestShutdown(); QTRY_COMPARE(shutdown.count(), 1);
    }

    void fileReplacementRejectsInspectionAndRequestsFreshConfig()
    {
        QTemporaryDir directory;
        const auto path = model(directory, "changed.onnx");
        const auto gate = std::make_shared<QSemaphore>();
        ModelSessionController controller([gate](const vision::ModelConfig &config, std::function<bool()>)
        { gate->tryAcquire(1, 3000); return inspected(config, 80); });
        QSignalSpy retry(&controller, &ModelSessionController::inspectionRetryRequested);
        QVERIFY(controller.select(path)); controller.inspect({});
        QFile file(path); QVERIFY(file.open(QIODevice::Append)); file.write("changed"); file.close();
        gate->release();
        QTRY_COMPARE_WITH_TIMEOUT(retry.count(), 1, 3000);
        QCOMPARE(controller.snapshot().classCount, 0);
        QVERIFY(controller.snapshot().inspectedIdentity.isEmpty());
        QVERIFY(!controller.snapshot().inspectionPending);
        QSignalSpy shutdown(&controller, &ModelSessionController::shutdownReady);
        controller.requestShutdown(); QTRY_COMPARE(shutdown.count(), 1);
    }

    void replacementDetectionDoesNotRetainClassificationSuggestion_data()
    {
        QTest::addColumn<int>("requestedTask");
        QTest::newRow("explicit-yolov5") << int(vision::ModelTask::YoloV5);
        QTest::newRow("explicit-yolov8") << int(vision::ModelTask::YoloV8);
    }

    void replacementDetectionDoesNotRetainClassificationSuggestion()
    {
        QFETCH(int, requestedTask);
        QTemporaryDir directory;
        const auto path = model(directory, "same-path.onnx");
        QVERIFY(!path.isEmpty());
        const auto calls = std::make_shared<std::atomic_int>(0);
        const auto inspectedTask = std::make_shared<std::atomic_int>(-1);
        ModelSessionController controller([calls, inspectedTask](const vision::ModelConfig &requested, std::function<bool()>) {
            inspectedTask->store(int(requested.task));
            auto result = inspected(requested, 2);
            if (calls->fetch_add(1) == 0) result.task = vision::ModelTask::Classification;
            return result;
        });
        QVERIFY(controller.select(path));
        controller.inspect({});
        QTRY_VERIFY_WITH_TIMEOUT(!controller.snapshot().inspectionPending, 3000);
        QVERIFY(controller.snapshot().inspectionError.isEmpty());
        QVERIFY(controller.snapshot().hasSuggestedTask);
        QCOMPARE(controller.snapshot().suggestedTask, vision::ModelTask::Classification);
        const auto classifiedIdentity = controller.snapshot().inspectedIdentity;

        // The file is replaced in place, so selecting the same library entry
        // does not reset its metadata. A task edit requests fresh inspection.
        QFile file(path); QVERIFY(file.open(QIODevice::Append));
        QCOMPARE(file.write(" replacement detection model"), 28LL); file.close();
        const auto replacementIdentity = ModelSessionController::fileIdentity(path);
        QVERIFY(replacementIdentity != classifiedIdentity);
        QVERIFY(controller.select(path));
        vision::ModelConfig requested;
        requested.task = static_cast<vision::ModelTask>(requestedTask);
        controller.inspect(requested);
        QTRY_VERIFY_WITH_TIMEOUT(!controller.snapshot().inspectionPending, 3000);
        QVERIFY(controller.snapshot().inspectionError.isEmpty());
        QCOMPARE(calls->load(), 2);
        QCOMPARE(controller.snapshot().inspectedIdentity, replacementIdentity);
        QCOMPARE(controller.snapshot().classCount, 2);
        // Detection ONNX inspection must not force a task format: retain the
        // caller's explicit YOLO choice, without a stale classification hint.
        QVERIFY(!controller.snapshot().hasSuggestedTask);
        QCOMPARE(inspectedTask->load(), requestedTask);
        QSignalSpy shutdown(&controller, &ModelSessionController::shutdownReady);
        controller.requestShutdown(); QTRY_COMPARE(shutdown.count(), 1);
    }

    void shutdownCancelsAllOutstandingInspectionsWithoutWaiting()
    {
        QTemporaryDir directory;
        const auto first = model(directory, "first.onnx");
        const auto second = model(directory, "second.onnx");
        ModelSessionController controller([](const vision::ModelConfig &config, std::function<bool()> cancelled)
        {
            while (!cancelled()) QThread::msleep(5);
            return inspected(config, 80);
        });
        QVERIFY(controller.select(first)); controller.inspect({});
        QVERIFY(controller.select(second)); controller.inspect({});
        QCOMPARE(controller.activeInspectionCount(), 2);
        QSignalSpy shutdown(&controller, &ModelSessionController::shutdownReady);
        QElapsedTimer elapsed; elapsed.start(); controller.requestShutdown();
        QVERIFY(elapsed.elapsed() < 100);
        QVERIFY(!controller.select(first));
        QTRY_COMPARE_WITH_TIMEOUT(shutdown.count(), 1, 3000);
        QCOMPARE(controller.activeInspectionCount(), 0);
        QVERIFY(controller.isShutdownReady());
        controller.requestShutdown(); QCoreApplication::processEvents();
        QCOMPARE(shutdown.count(), 1);
    }

    void loadedMetadataPreservesCapturedIdentity()
    {
        QTemporaryDir directory;
        const auto path = model(directory, "loaded.pt");
        ModelSessionController controller;
        QVERIFY(controller.select(path));
        vision::ModelConfig config;
        config.modelPath = path; config.modelClassCount = 1; config.inputChannels = 1;
        config.labels = {"football"};
        QVERIFY(controller.acceptLoadedMetadata(config, "captured-before-replacement"));
        QCOMPARE(controller.snapshot().inspectedIdentity, QString("captured-before-replacement"));
        QCOMPARE(controller.snapshot().nativeLabels, QStringList{"football"});
        QCOMPARE(controller.snapshot().classCount, 1);
        config.modelPath = directory.filePath("unselected.pt");
        QVERIFY(!controller.acceptLoadedMetadata(config));
        QCOMPARE(controller.snapshot().classCount, 1);
    }

    void inspectionErrorDoesNotBecomeEmptySuccessfulModel()
    {
        QTemporaryDir directory;
        const auto path = model(directory, "broken.onnx");
        ModelSessionController controller([](const vision::ModelConfig &, std::function<bool()>) -> vision::ModelConfig
        { throw std::runtime_error("declared inspection failure"); });
        QVERIFY(controller.select(path)); controller.inspect({});
        QTRY_VERIFY_WITH_TIMEOUT(!controller.snapshot().inspectionPending, 3000);
        QCOMPARE(controller.snapshot().inspectionError, QString("declared inspection failure"));
        QCOMPARE(controller.snapshot().classCount, 0);
        QVERIFY(controller.snapshot().inspectedIdentity.isEmpty());
        QSignalSpy shutdown(&controller, &ModelSessionController::shutdownReady);
        controller.requestShutdown(); QTRY_COMPARE(shutdown.count(), 1);
    }

    void realOnnxDeclaredMetadata()
    {
        const QString path = QString(VISION_PROJECT_DIR) + "/models/yolov5n.onnx";
        QVERIFY(QFileInfo::exists(path));
        ModelSessionController controller;
        QVERIFY(controller.select(path));
        vision::ModelConfig config; config.task = vision::ModelTask::YoloV5;
        controller.inspect(config);
        QTRY_VERIFY_WITH_TIMEOUT(!controller.snapshot().inspectionPending, 15000);
        QVERIFY2(controller.snapshot().inspectionError.isEmpty(), qPrintable(controller.snapshot().inspectionError));
        QCOMPARE(controller.snapshot().classCount, 80);
        QCOMPARE(controller.snapshot().inputChannels, 3);
        QCOMPARE(controller.snapshot().inspectedIdentity, ModelSessionController::fileIdentity(path));
        QSignalSpy shutdown(&controller, &ModelSessionController::shutdownReady);
        controller.requestShutdown(); QTRY_COMPARE(shutdown.count(), 1);
    }
};
QTEST_GUILESS_MAIN(ModelSessionControllerTests)
#include "model_session_controller_tests.moc"
