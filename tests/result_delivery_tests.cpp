// SPDX-License-Identifier: AGPL-3.0-only
#include "app/resultdeliveryservice.h"
#include "app/historyrepository.h"

#include <QDir>
#include <QFile>
#include <QJsonDocument>
#include <QSignalSpy>
#include <QTemporaryDir>
#include <QTimer>
#include <QtTest>

using namespace vision;
using namespace vision::app;

namespace
{
InferenceResult result(const QString &source)
{
    InferenceResult value;
    value.source = source;
    value.modelName = "model.pt";
    value.image = QImage(128, 80, QImage::Format_RGB32);
    value.image.fill(QColor(24, 39, 52));
    value.originalImage = value.image;
    value.predictions = {{32, "sports ball", .9f, QRectF(12, 20, 40, 30)}};
    value.classFilter = {32};
    value.modelClassCount = 80;
    return value;
}

ModelConfig configuration()
{
    ModelConfig value;
    value.modelPath = "model.pt";
    value.confidence = .55f;
    value.classFilter = {32};
    value.modelClassCount = 80;
    value.inputSize = 960;
    return value;
}

QByteArray read(const QString &path)
{
    QFile file(path);
    return file.open(QIODevice::ReadOnly) ? file.readAll() : QByteArray();
}

QStringList exports(const QString &directory)
{
    return QDir(directory).entryList({"*.json"}, QDir::Files, QDir::Name);
}

bool blockDirectory(const QString &path)
{
    QFile file(path);
    return file.open(QIODevice::WriteOnly) && file.write("not a directory") > 0;
}
}

class ResultDeliveryServiceTests final : public QObject
{
    Q_OBJECT
  private slots:
    void serialDeliveryFreezesConfigurationAndOutputAndAcknowledgesEveryTicket()
    {
        QTemporaryDir directory;
        QVERIFY(directory.isValid());
        HistoryRepository history(directory.filePath("history.json"));
        ResultDeliveryService service;
        service.setHistoryRepository(&history);
        auto config = configuration();
        ResultDeliveryOptions options{true, directory.filePath("accepted-output")};
        const QString acceptedOutput = options.exportDirectory;
        QVERIFY(service.beginTask(config, "frozen-config", {{"first.png", "first-v1"}, {"second.png", "second-v1"}}, options));
        config.confidence = .01f;
        config.classFilter = {0};
        options.exportDirectory = directory.filePath("unaccepted-output");
        options.automaticExport = false;
        QSignalSpy ack(&service, &ResultDeliveryService::acknowledged);
        QSignalSpy fatal(&service, &ResultDeliveryService::fatalError);
        QSignalSpy notice(&service, &ResultDeliveryService::notice);
        QSignalSpy changed(&service, &ResultDeliveryService::historyChanged);
        QSignalSpy ready(&service, &ResultDeliveryService::shutdownReady);
        service.acceptImage(result("first.png"), 11);
        service.acceptImage(result("second.png"), 12);
        QCOMPARE(history.snapshot().size(), 2);
        QCOMPARE(history.snapshot().first().toObject().value("source").toString(), QStringLiteral("second.png"));
        QCOMPARE(changed.count(), 2);
        QCOMPARE(service.pendingCount(), 2);
        QCOMPARE(ack.count(), 0);
        service.requestShutdown();
        QTRY_COMPARE_WITH_TIMEOUT(ready.count(), 1, 10000);
        QCOMPARE(service.pendingCount(), 0);
        QCOMPARE(ack.count(), 2);
        QCOMPARE(ack.at(0).first().toULongLong(), quint64(11));
        QCOMPARE(ack.at(1).first().toULongLong(), quint64(12));
        QCOMPARE(fatal.count(), 0);
        QCOMPARE(notice.count(), 0);
        QCOMPARE(read(history.filePath()), history.serialize());
        const auto files = exports(acceptedOutput);
        QCOMPARE(files.size(), 2);
        QVERIFY(!QDir(options.exportDirectory).exists());
        for (const auto &file : files) {
            const auto json = QJsonDocument::fromJson(read(acceptedOutput + "/" + file)).object();
            const auto savedConfig = json.value("config").toObject();
            QCOMPARE(savedConfig.value("confidence_threshold").toDouble(), double(.55f));
            QCOMPARE(savedConfig.value("class_filter").toArray(), QJsonArray{32});
            QCOMPARE(savedConfig.value("input_size").toInt(), 960);
            QCOMPARE(json.value("predictions").toArray().first().toObject().value("class_id").toInt(), 32);
            const auto source = json.value("source").toString();
            const auto entry = service.cache().entries().value(source);
            QCOMPARE(entry.config.classFilter, QVector<int>{32});
            QCOMPARE(entry.config.confidence, .55f);
            QVERIFY(entry.result.image.isNull());
            QVERIFY(QFileInfo(entry.snapshotPath).isFile());
        }
    }

    void loadedBackendMetadataKeepsFrozenSourceAndModelFingerprint()
    {
        QTemporaryDir directory;
        HistoryRepository history(directory.filePath("history.json"));
        ResultDeliveryService service;
        service.setHistoryRepository(&history);
        auto config = configuration();
        config.inputChannels = 0;
        service.beginTask(config, "model-before-load", {{"source.png", "source-at-start"}}, {});
        config.inputChannels = 1;
        config.resolvedDevice = ComputeDevice::CUDA;
        config.deviceName = "actual worker device";
        service.updateLoadedConfiguration(config);
        service.setConfigurationFingerprint("frozen-model-loaded-metadata");
        auto value = result("source.png");
        value.device = ComputeDevice::CUDA;
        value.deviceName = config.deviceName;
        QSignalSpy ready(&service, &ResultDeliveryService::shutdownReady);
        service.acceptImage(value, 20);
        const auto *accepted = service.findCached("source.png", "source-at-start");
        QVERIFY(accepted);
        QCOMPARE(accepted->config.inputChannels, 1);
        QCOMPARE(accepted->config.resolvedDevice, ComputeDevice::CUDA);
        QCOMPARE(accepted->sourceIdentity, QStringLiteral("source-at-start"));
        QCOMPARE(service.cache().fingerprint(), QStringLiteral("frozen-model-loaded-metadata"));
        service.requestShutdown();
        QTRY_COMPARE_WITH_TIMEOUT(ready.count(), 1, 10000);
        QVERIFY(service.findCached("source.png", "source-at-start"));
        QVERIFY(!service.findCached("source.png", "externally-replaced-source"));
    }

    void zeroMatchesStayEmptyInCacheHistoryAndExport()
    {
        QTemporaryDir directory;
        HistoryRepository history(directory.filePath("history.json"));
        ResultDeliveryService service;
        service.setHistoryRepository(&history);
        const QString output = directory.filePath("results");
        service.beginTask(configuration(), "configuration", {{"no-match.png", "source-v1"}}, {true, output});
        auto value = result("no-match.png");
        value.predictions.clear();
        QSignalSpy ready(&service, &ResultDeliveryService::shutdownReady);
        QSignalSpy ack(&service, &ResultDeliveryService::acknowledged);
        service.acceptImage(value, 32);
        service.requestShutdown();
        QTRY_COMPARE_WITH_TIMEOUT(ready.count(), 1, 10000);
        QCOMPARE(ack.count(), 1);
        QVERIFY(service.cache().entries().value(value.source).result.predictions.isEmpty());
        QVERIFY(history.snapshot().first().toObject().contains("objects"));
        QCOMPARE(history.snapshot().first().toObject().value("objects").toInt(), 0);
        QCOMPARE(exports(output).size(), 1);
        const auto json = QJsonDocument::fromJson(read(output + "/" + exports(output).first())).object();
        QVERIFY(json.value("predictions").toArray().isEmpty());
        QCOMPARE(json.value("config").toObject().value("class_filter").toArray(), QJsonArray{32});
    }

    void exportFailureReportsFatalAndStillAcknowledgesAfterHistorySave()
    {
        QTemporaryDir directory;
        HistoryRepository history(directory.filePath("history.json"));
        ResultDeliveryService service;
        service.setHistoryRepository(&history);
        const QString blocked = directory.filePath("output-is-a-regular-file");
        QVERIFY(blockDirectory(blocked));
        service.beginTask(configuration(), "configuration", {{"source.png", "source-v1"}}, {true, blocked});
        QSignalSpy fatal(&service, &ResultDeliveryService::fatalError);
        QSignalSpy ack(&service, &ResultDeliveryService::acknowledged);
        QSignalSpy notice(&service, &ResultDeliveryService::notice);
        QSignalSpy ready(&service, &ResultDeliveryService::shutdownReady);
        QStringList order;
        connect(&service, &ResultDeliveryService::fatalError, this, [&] { order << "fatal"; });
        connect(&service, &ResultDeliveryService::acknowledged, this, [&] { order << "ack"; });
        service.acceptImage(result("source.png"), 51);
        service.requestShutdown();
        QTRY_COMPARE_WITH_TIMEOUT(ready.count(), 1, 10000);
        QCOMPARE(fatal.count(), 1);
        QVERIFY(fatal.first().first().toString().startsWith("自动导出失败："));
        QCOMPARE(ack.count(), 1);
        QCOMPARE(ack.first().first().toULongLong(), quint64(51));
        QCOMPARE(order, QStringList({"fatal", "ack"}));
        QCOMPARE(notice.count(), 1);
        QCOMPARE(read(history.filePath()), history.serialize());
        const auto entry = service.cache().entries().value("source.png");
        QVERIFY(QFileInfo(entry.snapshotPath).isFile());
        QVERIFY(entry.result.image.isNull());
    }

    void historyWriteFailureRetainsMemoryAndDoesNotPreventAcknowledgement()
    {
        QTemporaryDir directory;
        const QString blocker = directory.filePath("not-a-directory");
        QVERIFY(blockDirectory(blocker));
        HistoryRepository history(blocker + "/history.json");
        ResultDeliveryService service;
        service.setHistoryRepository(&history);
        service.beginTask(configuration(), "configuration", {{"source.png", "source-v1"}}, {});
        QSignalSpy notice(&service, &ResultDeliveryService::notice);
        QSignalSpy fatal(&service, &ResultDeliveryService::fatalError);
        QSignalSpy ack(&service, &ResultDeliveryService::acknowledged);
        QSignalSpy ready(&service, &ResultDeliveryService::shutdownReady);
        service.acceptImage(result("source.png"), 60);
        service.requestShutdown();
        QTRY_COMPARE_WITH_TIMEOUT(ready.count(), 1, 10000);
        QCOMPARE(notice.count(), 1);
        QVERIFY(notice.first().first().toString().startsWith("运行记录保存失败："));
        QCOMPARE(fatal.count(), 0);
        QCOMPARE(ack.count(), 1);
        QCOMPARE(history.snapshot().size(), 1);
        QVERIFY(QFileInfo(service.cache().entries().value("source.png").snapshotPath).isFile());
    }

    void staleCompletionAcknowledgesWithoutFailingNewConfiguration()
    {
        QTemporaryDir directory;
        HistoryRepository history(directory.filePath("history.json"));
        ResultDeliveryService service;
        service.setHistoryRepository(&history);
        const QString blocked = directory.filePath("blocked-output");
        QVERIFY(blockDirectory(blocked));
        service.beginTask(configuration(), "old-configuration", {{"source.png", "source-v1"}}, {true, blocked});
        QSignalSpy ack(&service, &ResultDeliveryService::acknowledged);
        QSignalSpy fatal(&service, &ResultDeliveryService::fatalError);
        QSignalSpy notice(&service, &ResultDeliveryService::notice);
        QSignalSpy ready(&service, &ResultDeliveryService::shutdownReady);
        service.acceptImage(result("source.png"), 71);
        QVERIFY(service.invalidateConfiguration("new-configuration"));
        service.requestShutdown();
        QTRY_COMPARE_WITH_TIMEOUT(ready.count(), 1, 10000);
        QCOMPARE(ack.count(), 1);
        QCOMPARE(ack.first().first().toULongLong(), quint64(71));
        QCOMPARE(fatal.count(), 0);
        QCOMPARE(notice.count(), 0);
        QVERIFY(service.cache().entries().isEmpty());
        // The old accepted write still persists its captured history.
        QCOMPARE(read(history.filePath()), history.serialize());
    }

    void shutdownIsAsynchronousIdempotentAndDrainsAcceptedResults()
    {
        QTemporaryDir directory;
        HistoryRepository history(directory.filePath("history.json"));
        ResultDeliveryService service;
        service.setHistoryRepository(&history);
        service.beginTask(configuration(), "configuration", {{"source.png", "source-v1"}}, {});
        QSignalSpy ack(&service, &ResultDeliveryService::acknowledged);
        QSignalSpy ready(&service, &ResultDeliveryService::shutdownReady);
        int ticks = 0;
        QTimer heartbeat;
        heartbeat.setInterval(0);
        connect(&heartbeat, &QTimer::timeout, this, [&] { ++ticks; });
        heartbeat.start();
        auto value = result("source.png");
        value.image = QImage(1600, 1200, QImage::Format_RGB32);
        value.image.fill(QColor(40, 80, 125));
        value.originalImage = value.image;
        service.acceptImage(value, 90);
        QElapsedTimer elapsed;
        elapsed.start();
        service.requestShutdown();
        service.requestShutdown();
        QVERIFY(elapsed.elapsed() < 100);
        QVERIFY(!service.isShutdownReady());
        QTRY_COMPARE_WITH_TIMEOUT(ready.count(), 1, 10000);
        heartbeat.stop();
        QVERIFY(ticks > 0);
        QVERIFY(service.isShutdownReady());
        QCOMPARE(service.pendingCount(), 0);
        QCOMPARE(ack.count(), 1);
        QCOMPARE(ack.first().first().toULongLong(), quint64(90));
        QCOMPARE(read(history.filePath()), history.serialize());
        const auto cached = service.cache().entries().value("source.png");
        QVERIFY(QFileInfo(cached.snapshotPath).isFile());
        QCOMPARE(QImage(cached.snapshotPath).size(), QSize(1600, 1200));
        service.requestShutdown();
        QCoreApplication::processEvents();
        QCOMPARE(ready.count(), 1);
        service.acceptImage(result("rejected.png"), 91);
        QCOMPARE(ack.count(), 2);
        QCOMPARE(ack.last().first().toULongLong(), quint64(91));
        QCOMPARE(history.snapshot().size(), 1);
        QVERIFY(!service.cache().entries().contains("rejected.png"));
    }

    void sameConfigurationRerunPreservesResultsOutsideNewRange()
    {
        QTemporaryDir directory;
        HistoryRepository history(directory.filePath("history.json"));
        ResultDeliveryService service;
        service.setHistoryRepository(&history);
        service.beginTask(configuration(), "configuration", {{"first.png", "first-v1"}, {"last.png", "last-v1"}}, {});
        service.acceptImage(result("first.png"), 101);
        service.acceptImage(result("last.png"), 102);
        QTRY_COMPARE_WITH_TIMEOUT(service.pendingCount(), 0, 10000);
        const auto generation = service.cache().generation();
        QVERIFY(!service.beginTask(configuration(), "configuration", {{"last.png", "last-v1"}}, {}));
        QCOMPARE(service.cache().generation(), generation);
        QVERIFY(service.findCached("first.png", "first-v1"));
        service.acceptImage(result("last.png"), 103);
        QSignalSpy ready(&service, &ResultDeliveryService::shutdownReady);
        service.requestShutdown();
        QTRY_COMPARE_WITH_TIMEOUT(ready.count(), 1, 10000);
        QCOMPARE(service.cache().entries().size(), 2);
        QVERIFY(service.findCached("first.png", "first-v1"));
        QCOMPARE(history.snapshot().size(), 3);
    }

    void streamFinalizationPreservesHistoryAndSignalsOnlyAfterSerialWrite()
    {
        QTemporaryDir directory;
        HistoryRepository history(directory.filePath("history.json"));
        ResultDeliveryService service;
        service.setHistoryRepository(&history);
        service.beginTask(configuration(), "configuration", {}, {});
        QSignalSpy changed(&service, &ResultDeliveryService::historyChanged);
        QSignalSpy finalized(&service, &ResultDeliveryService::streamFinalized);
        QSignalSpy ack(&service, &ResultDeliveryService::acknowledged);
        service.finalizeStream(result("video.mp4"));
        QCOMPARE(changed.count(), 1);
        QCOMPARE(history.snapshot().size(), 1);
        QCOMPARE(finalized.count(), 0);
        QVERIFY(service.cache().entries().isEmpty());
        QSignalSpy ready(&service, &ResultDeliveryService::shutdownReady);
        service.requestShutdown();
        QTRY_COMPARE_WITH_TIMEOUT(ready.count(), 1, 10000);
        QCOMPARE(read(history.filePath()), history.serialize());
        QCOMPARE(finalized.count(), 1);
        QCOMPARE(ack.count(), 0);
    }
};

QTEST_MAIN(ResultDeliveryServiceTests)
#include "result_delivery_tests.moc"
