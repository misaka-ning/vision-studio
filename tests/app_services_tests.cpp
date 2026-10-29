#include "app/annotationrenderer.h"
#include "app/historyrepository.h"
#include "app/preferencesrepository.h"
#include "app/resultwriter.h"

#include <QFile>
#include <QJsonDocument>
#include <QSettings>
#include <QSignalSpy>
#include <QTemporaryDir>
#include <QtTest>

using namespace vision::app;
namespace
{
vision::InferenceResult sample(const QString &source = QStringLiteral("sample image.jpg"))
{
    vision::InferenceResult result;
    result.source = source;
    result.modelName = "test.pt";
    result.image = QImage(192, 128, QImage::Format_RGB888);
    result.image.fill(QColor(25, 40, 65));
    result.predictions = {{32, "=ball,\"quoted\"", .8f, QRectF(20, 35, 65, 45)}};
    return result;
}
QByteArray read(const QString &path)
{
    QFile file(path);
    return file.open(QIODevice::ReadOnly) ? file.readAll() : QByteArray();
}
}

class AppServicesTests final : public QObject
{
    Q_OBJECT
  private slots:
    void preferencesPreserveLegacyEncoding()
    {
        QTemporaryDir directory;
        const QString path = directory.filePath("preferences.ini");
        const QVariantMap notes{{"/tmp/足球.pt", "保持备注\n第二行"}};
        {
            QSettings legacy(path, QSettings::IniFormat);
            legacy.setValue("models", QStringList{"/tmp/a.onnx", "/tmp/足球.pt"});
            legacy.setValue("modelNotes", notes);
            legacy.setValue("restoreDatasetOnStartup", false);
            legacy.setValue("confidence", .55);
            legacy.sync();
        }
        PreferencesRepository preferences(path);
        QCOMPARE(preferences.value("modelNotes").toMap(), notes);
        QCOMPARE(preferences.value("models").toStringList().size(), 2);
        QVERIFY(!preferences.value("restoreDatasetOnStartup", true).toBool());
        QCOMPARE(preferences.value("missing", 640).toInt(), 640);
        preferences.setValue("classFilter", "0,2,32");
        preferences.sync();
        QCOMPARE(preferences.status(), QSettings::NoError);
        QSettings legacy(path, QSettings::IniFormat);
        QCOMPARE(legacy.value("classFilter").toString(), QString("0,2,32"));
        QCOMPARE(legacy.value("confidence").toDouble(), .55);
        QCOMPARE(preferences.snapshot().value("modelNotes").toMap(), notes);
    }

    void exportContractAndCsvEscaping()
    {
        QTemporaryDir directory;
        auto result = sample();
        vision::ModelConfig config;
        config.modelPath = "/tmp/test.pt";
        config.classFilter = {32};
        config.modelClassCount = 80;
        const auto exported = ResultWriter::prepare(result, directory.path(), config);
        const auto json = QJsonDocument::fromJson(exported.json).object();
        QCOMPARE(json.value("task").toString(), QString("YOLOv8 / 11"));
        QCOMPARE(json.value("source").toString(), result.source);
        QCOMPARE(json.value("config").toObject().value("class_filter").toArray(), QJsonArray{32});
        QCOMPARE(json.value("config").toObject().value("preprocess").toObject().value("mode").toString(),
                 QString("model_native"));
        QVERIFY(exported.csv.startsWith("class_id,label,confidence,x,y,width,height\n"));
        QVERIFY(exported.csv.contains("32,\"'=ball,\"\"quoted\"\"\",0.800000,20.00,35.00,65.00,45.00\n"));
        QString error;
        QVERIFY2(ResultWriter::writePrepared(exported, &error), qPrintable(error));
        QCOMPARE(read(exported.base + ".json"), exported.json);
        QCOMPARE(read(exported.base + ".csv"), exported.csv);
        QCOMPARE(QImage(exported.base + ".png").convertToFormat(QImage::Format_ARGB32_Premultiplied),
                 AnnotationRenderer::render(result));
    }

    void historicalMetadataRemainsUnknown()
    {
        auto result = sample();
        result.hasCompleteInferenceConfig = false;
        result.hasVerifiedDevice = false;
        result.hasInferenceTiming = false;
        result.sourceRuntimeDevice = "historical-backend";
        vision::ModelConfig config;
        config.confidence = .4f;
        const QJsonObject original{{"original_field", 17}, {"confidence_threshold", .2}};
        const auto exported = ResultWriter::prepare(result, "/tmp", config, original);
        const auto json = QJsonDocument::fromJson(exported.json).object();
        QVERIFY(json.value("actual_device").isNull());
        QVERIFY(json.value("inference_ms").isNull());
        QCOMPARE(json.value("source_runtime_device").toString(), QString("historical-backend"));
        const auto recorded = json.value("config").toObject();
        QCOMPARE(recorded.value("original_field").toInt(), 17);
        QCOMPARE(recorded.value("metadata_complete").toBool(), false);
        QVERIFY(recorded.value("input_channels").isNull());
        QCOMPARE(recorded.value("confidence_threshold").toDouble(), double(config.confidence));
    }

    void groundTruthAndMatchingKeepOriginalIds()
    {
        auto result = sample();
        result.hasGroundTruth = true;
        result.hasEvaluationMatches = true;
        result.groundTruth = {{32, "sports ball", 1, QRectF(20, 35, 65, 45)}};
        result.matchedPredictions = {0};
        result.matchedGroundTruth = {0};
        const auto exported = ResultWriter::prepare(result, "/tmp", {});
        const auto json = QJsonDocument::fromJson(exported.json).object();
        QCOMPARE(json.value("ground_truth").toArray().first().toObject().value("class_id").toInt(), 32);
        QCOMPARE(json.value("predictions").toArray().first().toObject().value("match_status").toString(), QString("TP"));
        QVERIFY(exported.csv.contains("match_status\n"));
        QVERIFY(exported.csv.endsWith(",TP\n"));
    }

    void demonstrationCannotBeExported()
    {
        auto result = sample();
        result.demonstration = true;
        QVERIFY(ResultWriter::prepare(result, "/tmp", {}).base.isEmpty());
        QString error;
        QVERIFY(!ResultWriter::writePrepared({}, &error));
        QVERIFY(!error.isEmpty());
    }

    void historyPreservesCapacityAndUnknownDevice()
    {
        QTemporaryDir directory;
        HistoryRepository repository(directory.filePath("history.json"));
        QVERIFY(repository.load().isEmpty());
        QJsonArray entries;
        for (int i = 0; i != 205; ++i)
            HistoryRepository::prepend(entries, sample(QString::number(i)));
        QCOMPARE(entries.size(), 200);
        QCOMPARE(entries.first().toObject().value("source").toString(), QString("204"));
        QCOMPARE(entries.last().toObject().value("source").toString(), QString("5"));
        QVERIFY(repository.save(entries));
        QCOMPARE(read(repository.filePath()), QJsonDocument(entries).toJson());
        QCOMPARE(repository.load(), entries);
        auto result = sample();
        result.hasVerifiedDevice = false;
        result.hasInferenceTiming = false;
        result.hasCompleteInferenceConfig = false;
        const auto entry = HistoryRepository::entry(result);
        QVERIFY(entry.value("actual_device").isNull());
        QVERIFY(entry.value("inference_ms").isNull());
        QCOMPARE(entry.value("task").toString(), QString("目标检测 · 历史预测"));
    }

    void serialWritesDrainBeforeShutdown()
    {
        QTemporaryDir directory;
        auto cache = std::make_shared<QTemporaryDir>();
        QVERIFY(cache->isValid());
        ResultWriter writer;
        QSignalSpy completion(&writer, &ResultWriter::completed);
        QSignalSpy shutdown(&writer, &ResultWriter::shutdownReady);
        const QString history = directory.filePath("history.json");
        for (int i = 0; i != 8; ++i)
        {
            ResultWriteRequest request;
            request.result = sample(QString::number(i));
            request.result.originalImage = QImage(192, 128, QImage::Format_RGB888);
            request.result.originalImage.fill(Qt::red);
            request.ticket = i + 1;
            request.generation = 42;
            request.directory = cache;
            request.snapshotPath = cache->filePath(QString::number(i) + ".png");
            request.historyPath = history;
            request.history = QByteArray::number(i);
            QVERIFY(writer.enqueue(std::move(request)));
        }
        QCOMPARE(writer.pendingCount(), 8);
        writer.requestShutdown();
        QVERIFY(!writer.enqueue({}));
        QTRY_COMPARE_WITH_TIMEOUT(shutdown.count(), 1, 10000);
        QCOMPARE(completion.count(), 8);
        QCOMPARE(writer.pendingCount(), 0);
        QCOMPARE(read(history), QByteArray("7"));
        for (int i = 0; i != completion.count(); ++i)
        {
            const auto outcome = qvariant_cast<ResultWriteOutcome>(completion.at(i).at(0));
            QCOMPARE(outcome.ticket, quint64(i + 1));
            QCOMPARE(outcome.generation, quint64(42));
            QCOMPARE(outcome.source, QString::number(i));
            QVERIFY(outcome.snapshotSaved);
            QVERIFY(outcome.historySaved);
            QCOMPARE(QImage(outcome.snapshotPath).pixelColor(0, 0), QColor(Qt::red));
        }
        writer.requestShutdown();
        QCoreApplication::processEvents();
        QCOMPARE(shutdown.count(), 1);
    }

    void failedExportStillCompletesTicketAndHistory()
    {
        QTemporaryDir directory;
        ResultWriter writer;
        QSignalSpy completion(&writer, &ResultWriter::completed);
        QSignalSpy shutdown(&writer, &ResultWriter::shutdownReady);
        ResultWriteRequest request;
        request.result = sample();
        request.ticket = 19;
        request.automaticExport = true;
        request.historyPath = directory.filePath("history.json");
        request.history = "[]";
        request.snapshotPath = directory.filePath("missing/failed.png");
        QVERIFY(writer.enqueue(request));
        writer.requestShutdown();
        QTRY_COMPARE_WITH_TIMEOUT(shutdown.count(), 1, 10000);
        QCOMPARE(completion.count(), 1);
        const auto outcome = qvariant_cast<ResultWriteOutcome>(completion.first().first());
        QCOMPARE(outcome.ticket, quint64(19));
        QVERIFY(!outcome.exportSaved);
        QVERIFY(!outcome.exportError.isEmpty());
        QVERIFY(!outcome.snapshotSaved);
        QVERIFY(!outcome.snapshotError.isEmpty());
        QVERIFY(outcome.historySaved);
        QCOMPARE(read(request.historyPath), QByteArray("[]"));
    }
};
QTEST_MAIN(AppServicesTests)
#include "app_services_tests.moc"
