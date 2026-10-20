#include "core/evaluationworker.h"
#include "core/evaluationreview.h"

#include <QCryptographicHash>
#include <QDir>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QSignalSpy>
#include <QTemporaryDir>
#include <QtTest>
#include <algorithm>
#include <cstring>
#include <functional>
#include <stdexcept>

namespace
{
QByteArray number(quint64 value)
{
    QByteArray out;
    do
    {
        quint8 byte = value & 127;
        value >>= 7;
        out.append(char(byte | (value ? 128 : 0)));
    } while (value);
    return out;
}
QByteArray integer(int field, quint64 value) { return number(quint64(field) << 3) + number(value); }
QByteArray bytes(int field, const QByteArray &value)
{
    return number((quint64(field) << 3) | 2) + number(value.size()) + value;
}
QByteArray text(int field, const char *value) { return bytes(field, QByteArray(value)); }
QByteArray info(const char *name, const QVector<qint64> &dimensions)
{
    QByteArray shape;
    for (auto dimension : dimensions)
        shape += bytes(1, integer(1, dimension));
    return text(1, name) + bytes(2, bytes(1, integer(1, 1) + bytes(2, shape)));
}
template <class T> QByteArray tensor(const char *name, const QVector<qint64> &dimensions,
                                    int type, const QVector<T> &values)
{
    QByteArray data, result;
    for (const auto &value : values)
    {
        quint8 encoded[sizeof(T)];
        std::memcpy(encoded, &value, sizeof(T));
#if Q_BYTE_ORDER == Q_BIG_ENDIAN
        std::reverse(encoded, encoded + sizeof(T));
#endif
        data.append(reinterpret_cast<const char *>(encoded), sizeof(T));
    }
    for (auto dimension : dimensions)
        result += integer(1, dimension);
    return result + integer(2, type) + text(8, name) + bytes(9, data);
}
QByteArray node(const char *op, const QVector<const char *> &inputs, const char *output,
                const QByteArray &attributes = {})
{
    QByteArray result;
    for (const auto input : inputs)
        result += text(1, input);
    return result + text(2, output) + text(4, op) + attributes;
}
void writeFile(const QString &path, const QByteArray &value)
{
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly) || file.write(value) != value.size())
        throw std::runtime_error("Cannot write evaluation fixture");
}
QString realOnnx(const QString &root)
{
    // Actual Conv -> GlobalAveragePool -> Reshape ONNX, evaluated by OpenCV.
    // Its one class has one 0.8-score box [22,22,42,42] for a 64-pixel image.
    QVector<float> values(5 * 8, 0);
    values[0] = values[8] = 32;
    values[16] = values[24] = 20;
    values[32] = .8f;
    QByteArray graph;
    const auto kernel = bytes(5, text(1, "kernel_shape") + integer(8, 1) + integer(8, 1) + integer(20, 7));
    graph += bytes(1, node("Conv", {"images", "weights", "bias"}, "conv", kernel));
    graph += bytes(1, node("GlobalAveragePool", {"conv"}, "average"));
    graph += bytes(1, node("Reshape", {"average", "shape"}, "output"));
    graph += text(2, "vision_studio_evaluation_fixture");
    graph += bytes(5, tensor("weights", {40, 3, 1, 1}, 1, QVector<float>(120, 0)));
    graph += bytes(5, tensor("bias", {40}, 1, values));
    graph += bytes(5, tensor("shape", {3}, 7, QVector<qint64>{1, 5, 8}));
    graph += bytes(11, info("images", {1, 3, 64, 64}));
    graph += bytes(12, info("output", {1, 5, 8}));
    const auto path = root + "/known.onnx";
    writeFile(path, integer(1, 7) + text(2, "Vision Studio evaluation tests") + bytes(7, graph) +
                    bytes(8, integer(2, 11)));
    return path;
}
QJsonObject readJson(const QString &path)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly))
        return {};
    return QJsonDocument::fromJson(file.readAll()).object();
}
} // namespace

class EvaluationWorkerTests final : public QObject
{
    Q_OBJECT
  private:
    QTemporaryDir root_;
    QString project_, python_, model_;
    QByteArray previousPython_;
    bool previousPythonSet_ = false;
    int counter_ = 0;

    QJsonObject dataset(int count, bool missingLabel = false)
    {
        const auto directory = root_.path() + "/dataset-" + QString::number(++counter_);
        QDir().mkpath(directory + "/images/val");
        QDir().mkpath(directory + "/labels/val");
        QImage image(64, 64, QImage::Format_RGB888);
        image.fill(QColor(20, 40, 80));
        for (int index = 0; index < count; ++index)
        {
            const auto name = QString::number(index).rightJustified(3, '0');
            if (!image.save(directory + "/images/val/" + name + ".png"))
                throw std::runtime_error("Cannot save evaluation image");
            if (!missingLabel || index != count - 1)
                writeFile(directory + "/labels/val/" + name + ".txt",
                          index == count - 1 ? QByteArray() : QByteArray("0 0.5 0.5 0.3125 0.3125\n"));
        }
        writeFile(directory + "/dataset.yaml", ("path: " + directory +
            "\nval: images/val\nnames: [football]\n").toUtf8());
        vision::EvaluationWorker worker;
        QSignalSpy ready(&worker, &vision::EvaluationWorker::datasetReady);
        QSignalSpy failed(&worker, &vision::EvaluationWorker::failed);
        worker.configureDataset(directory + "/dataset.yaml", "val", {}, project_);
        worker.start();
        if (!worker.wait(30000))
        {
            worker.requestStop();
            worker.wait();
            throw std::runtime_error("Dataset preparation timed out");
        }
        QCoreApplication::processEvents();
        if (ready.size() != 1)
            throw std::runtime_error(failed.isEmpty() ? "Dataset prepare emitted no manifest" :
                                     failed.first().first().toString().toUtf8().constData());
        return ready.first().first().toJsonObject();
    }

    vision::EvaluationRequest request(const QJsonObject &manifest) const
    {
        vision::EvaluationRequest value;
        value.config.modelPath = model_;
        value.config.task = vision::ModelTask::YoloV8;
        value.config.inputSize = 64;
        value.config.device = vision::ComputeDevice::CPU;
        value.config.modelClassCount = 1;
        value.config.labels = {"football"};
        value.manifest = manifest;
        for (const auto &sample : manifest.value("samples").toArray())
            value.files.append(sample.toObject().value("image").toString());
        value.outputRoot = root_.path() + "/results";
        value.projectRoot = project_;
        value.scanStart = .1;
        value.scanEnd = .9;
        value.scanStep = .2;
        return value;
    }

    QJsonObject evaluate(const vision::EvaluationRequest &request, QString *directory = nullptr,
                         const std::function<void(vision::EvaluationWorker &)> &setup = {})
    {
        vision::EvaluationWorker worker;
        QSignalSpy ready(&worker, &vision::EvaluationWorker::reportReady);
        QSignalSpy failed(&worker, &vision::EvaluationWorker::failed);
        if (setup)
            setup(worker);
        worker.configureEvaluation(request);
        worker.start();
        if (!worker.wait(60000))
        {
            worker.requestStop();
            worker.wait();
            throw std::runtime_error("Evaluation timed out");
        }
        QCoreApplication::processEvents();
        if (ready.size() != 1)
            throw std::runtime_error(failed.isEmpty() ? "Evaluation emitted no report" :
                                     failed.first().first().toString().toUtf8().constData());
        if (directory)
            *directory = ready.first().at(1).toString();
        return ready.first().first().toJsonObject();
    }

  private slots:
    void initTestCase()
    {
        QVERIFY(root_.isValid());
        project_ = QStringLiteral(VISION_PROJECT_DIR);
        python_ = qEnvironmentVariable("VISION_STUDIO_EVALUATION_PYTHON");
        if (python_.isEmpty())
        {
            const QStringList candidates{project_ + "/runtime/bin/python", "/opt/VisionStudio/runtime/bin/python",
                                          QDir::homePath() + "/VisionStudio/runtime/bin/python"};
            for (const auto &candidate : candidates)
                if (QFileInfo(candidate).isExecutable()) { python_ = candidate; break; }
        }
        QVERIFY2(QFileInfo(python_).isExecutable(), "A prepared CPU evaluation runtime is required.");
        QVERIFY(QFileInfo(project_ + "/scripts/evaluation_worker.py").isFile());
        previousPythonSet_ = qEnvironmentVariableIsSet("VISION_STUDIO_PYTHON");
        previousPython_ = qgetenv("VISION_STUDIO_PYTHON");
        qputenv("VISION_STUDIO_PYTHON", python_.toUtf8());
        model_ = realOnnx(root_.path());
    }

    void cleanupTestCase()
    {
        if (previousPythonSet_)
            qputenv("VISION_STUDIO_PYTHON", previousPython_);
        else
            qunsetenv("VISION_STUDIO_PYTHON");
    }

    void prepareUsesRealDimensionsAndFlagsMissingLabels()
    {
        const auto complete = dataset(3);
        QCOMPARE(complete.value("samples").toArray().size(), 3);
        QVERIFY(!complete.value("requires_attention").toBool());
        const auto first = complete.value("samples").toArray().first().toObject();
        QCOMPARE(first.value("width").toInt(), 64);
        QCOMPARE(first.value("height").toInt(), 64);
        QCOMPARE(first.value("image_sha256").toString().size(), 64);
        QCOMPARE(first.value("gt").toArray().first().toObject().value("box").toArray(),
                 QJsonArray({22., 22., 42., 42.}));
        const auto incomplete = dataset(3, true);
        QVERIFY(incomplete.value("requires_attention").toBool());
    }

    void actualOnnxAllAndSubsets_data()
    {
        QTest::addColumn<int>("count");
        QTest::addColumn<int>("start");
        QTest::addColumn<int>("length");
        QTest::addColumn<bool>("full");
        QTest::newRow("all-three") << 3 << 0 << 3 << true;
        QTest::newRow("from-second") << 3 << 1 << 2 << false;
        QTest::newRow("only-last") << 3 << 2 << 1 << false;
        QTest::newRow("more-than-fifty-all") << 61 << 0 << 61 << true;
        QTest::newRow("more-than-fifty-tail") << 61 << 50 << 11 << false;
    }

    void actualOnnxAllAndSubsets()
    {
        QFETCH(int, count);
        QFETCH(int, start);
        QFETCH(int, length);
        QFETCH(bool, full);
        auto value = request(dataset(count));
        value.files = value.files.mid(start, length);
        QString directory;
        const auto report = evaluate(value, &directory);
        QCOMPARE(report.value("status").toString(), QString("complete"));
        QCOMPARE(report.value("full_split").toBool(), full);
        QCOMPARE(report.value("request").toObject().value("sample_images").toArray().size(), length);
        QCOMPARE(QDir(directory + "/samples").entryList({"*.json"}, QDir::Files).size(), length);
        for (int index = 0; index < length; ++index)
        {
            const auto sample = vision::EvaluationWorker::readSample(directory, index);
            QCOMPARE(sample.value("image").toString(), value.files.at(index));
            QCOMPARE(sample.value("predictions").toArray().size(), 1);
            QCOMPARE(sample.value("predictions").toArray().first().toObject().value("class_id").toInt(), 0);
            QCOMPARE(sample.value("backend").toString(), QString("OpenCV DNN / CPU"));
        }
        QCOMPARE(readJson(directory + "/status.json").value("completed").toInt(), length);
        QVERIFY(QFileInfo(directory + "/helper.log").isFile());
        const auto metrics = report.value("metrics").toObject();
        QCOMPARE(metrics.value("TP").toInt(), length - 1);
        QCOMPARE(metrics.value("FP").toInt(), 1);
        QCOMPARE(metrics.value("FN").toInt(), 0);
        QCOMPARE(metrics.value("no_ball_false_positive_rate").toDouble(), 1.);
        if (full)
        {
            QVERIFY(!metrics.value("mAP50").isNull());
            QVERIFY(metrics.value("mAP50").toDouble() > .9);
        }
        else
            QVERIFY(metrics.value("mAP50").isNull());
    }

    void stopRetainsOnlyCompletedPrefix()
    {
        auto value = request(dataset(61));
        QString directory;
        const auto report = evaluate(value, &directory, [](vision::EvaluationWorker &worker)
        {
            QObject::connect(&worker, &vision::EvaluationWorker::progress, &worker,
                [&worker](int completed, int, const QString &)
                { if (completed == 1) worker.requestStop(); }, Qt::DirectConnection);
        });
        QCOMPARE(report.value("status").toString(), QString("cancelled"));
        QCOMPARE(readJson(directory + "/status.json").value("completed").toInt(), 1);
        QCOMPARE(QDir(directory + "/samples").entryList({"*.json"}, QDir::Files).size(), 1);
        QCOMPARE(vision::EvaluationWorker::readSample(directory, 0).value("image").toString(), value.files.first());
        QString error;
        QVERIFY(vision::EvaluationWorker::readSample(directory, 1, &error).isEmpty());
        QVERIFY(!error.isEmpty());
        QVERIFY(report.value("metrics").toObject().value("mAP50").isNull());
        QVERIFY(!report.value("full_split").toBool());
    }

    void stopBeforeLoadProducesEmptyCancelledReport()
    {
        auto value = request(dataset(3));
        QString directory;
        const auto report = evaluate(value, &directory, [](vision::EvaluationWorker &worker)
        {
            QObject::connect(&worker, &vision::EvaluationWorker::progress, &worker,
                [&worker](int completed, int, const QString &)
                { if (!completed) worker.requestStop(); }, Qt::DirectConnection);
        });
        QCOMPARE(report.value("status").toString(), QString("cancelled"));
        QCOMPARE(report.value("completed_images").toInt(), 0);
        QCOMPARE(QDir(directory + "/samples").entryList({"*.json"}, QDir::Files).size(), 0);
        QVERIFY(report.value("metrics").toObject().value("mAP50").isNull());
    }

    void staleDatasetRejectedBeforePrediction()
    {
        auto value = request(dataset(3));
        QImage changed(64, 64, QImage::Format_RGB888);
        changed.fill(Qt::red);
        QVERIFY(changed.save(value.files.first()));
        vision::EvaluationWorker worker;
        QSignalSpy ready(&worker, &vision::EvaluationWorker::reportReady);
        QSignalSpy failed(&worker, &vision::EvaluationWorker::failed);
        worker.configureEvaluation(value);
        worker.start();
        QVERIFY(worker.wait(30000));
        QCoreApplication::processEvents();
        QCOMPARE(ready.size(), 0);
        QCOMPARE(failed.size(), 1);
        QVERIFY(failed.first().first().toString().contains(QStringLiteral("发生变化")));
    }

    void duplicateAndForeignRangesRejected()
    {
        const auto manifest = dataset(2);
        for (bool foreign : {false, true})
        {
            auto value = request(manifest);
            value.files = {value.files.first(), foreign ? root_.path() + "/foreign.png" : value.files.first()};
            vision::EvaluationWorker worker;
            QSignalSpy ready(&worker, &vision::EvaluationWorker::reportReady);
            QSignalSpy failed(&worker, &vision::EvaluationWorker::failed);
            worker.configureEvaluation(value);
            worker.start();
            QVERIFY(worker.wait(5000));
            QCoreApplication::processEvents();
            QCOMPARE(ready.size(), 0);
            QCOMPARE(failed.size(), 1);
        }
    }

    void invalidMappingAndScanPreventInference()
    {
        const auto manifest = dataset(2);
        for (bool mapping : {false, true})
        {
            auto value = request(manifest);
            if (mapping) value.datasetClassId = 1;
            else value.scanStart = 0;
            vision::EvaluationWorker worker;
            QSignalSpy failed(&worker, &vision::EvaluationWorker::failed);
            worker.configureEvaluation(value);
            worker.start();
            QVERIFY(worker.wait(5000));
            QCoreApplication::processEvents();
            QCOMPARE(failed.size(), 1);
        }
    }

    void backgroundReviewKeepsSnapshotAndGrayscale()
    {
        const auto manifest = dataset(2);
        QString directory;
        const auto report = evaluate(request(manifest), &directory);
        Q_UNUSED(report);
        vision::EvaluationPreviewRequest value;
        value.directory = directory;
        value.index = 0;
        value.sample = manifest.value("samples").toArray()[0].toObject();
        value.config = request(manifest).config;
        value.config.colorMode = vision::InputColorMode::Grayscale;
        value.datasetClassName = "football snapshot";
        value.confidence = .5;
        const auto preview = vision::loadEvaluationPreview(value);
        QVERIFY2(preview.error.isEmpty(), qPrintable(preview.error));
        QVERIFY(preview.completed);
        QVERIFY(preview.result.hasGroundTruth);
        QVERIFY(preview.result.hasEvaluationMatches);
        QCOMPARE(preview.result.predictions.size(), 1);
        QCOMPARE(preview.result.groundTruth.size(), 1);
        QCOMPARE(preview.result.matchedPredictions.size(), 1);
        QCOMPARE(preview.result.groundTruth.first().label, QString("football snapshot"));
        QCOMPARE(preview.result.classFilter, QVector<int>({0}));
        const auto gray = preview.result.image.pixelColor(10, 10);
        QCOMPARE(gray.red(), gray.green());
        QCOMPARE(gray.green(), gray.blue());
        QCOMPARE(preview.result.evaluationReport, directory);
    }

    void unprocessedPreviewCannotBecomeCompletedResult()
    {
        const auto manifest = dataset(2);
        vision::EvaluationPreviewRequest value;
        value.sample = manifest.value("samples").toArray()[0].toObject();
        // Even unexpected prediction fields in a supplied manifest must never
        // make an unprocessed image look like a completed evaluation result.
        value.sample.insert("predictions", QJsonArray{QJsonObject{{"class_id", 0}, {"confidence", .99},
            {"box", QJsonArray{22., 22., 42., 42.}}}});
        value.config = request(manifest).config;
        const auto preview = vision::loadEvaluationPreview(value);
        QVERIFY2(preview.error.isEmpty(), qPrintable(preview.error));
        QVERIFY(!preview.completed);
        QVERIFY(preview.result.hasGroundTruth);
        QVERIFY(!preview.result.hasEvaluationMatches);
        QVERIFY(preview.result.predictions.isEmpty());
        QCOMPARE(preview.result.groundTruth.size(), 1);
    }

    void reviewRejectsWrongRowAndChangedPixels()
    {
        const auto manifest = dataset(2);
        QString directory;
        evaluate(request(manifest), &directory);
        vision::EvaluationPreviewRequest value;
        value.directory = directory;
        value.index = 0;
        value.config = request(manifest).config;
        value.sample = manifest.value("samples").toArray()[1].toObject();
        auto preview = vision::loadEvaluationPreview(value);
        QVERIFY(!preview.error.isEmpty());
        QVERIFY(preview.result.image.isNull());
        QVERIFY(!preview.completed);
        value.sample = manifest.value("samples").toArray()[0].toObject();
        QImage changed(64, 64, QImage::Format_RGB888);
        changed.fill(Qt::green);
        QVERIFY(changed.save(value.sample.value("image").toString()));
        preview = vision::loadEvaluationPreview(value);
        QVERIFY(preview.error.contains(QStringLiteral("已改变")));
        QVERIFY(preview.result.predictions.isEmpty());
        QVERIFY(preview.result.image.isNull());
        QVERIFY(!preview.completed);
    }

    void modifiedPredictionCacheCannotDisagreeWithSavedMetrics()
    {
        const auto manifest = dataset(2);
        auto value = request(manifest);
        QString directory;
        const auto report = evaluate(value, &directory);
        const QString base = directory + "/samples/0";
        QFile source(base + ".json"); QVERIFY(source.open(QIODevice::ReadOnly));
        const auto originalBytes = source.readAll(); source.close();
        QFile checksum(base + ".sha256"); QVERIFY(checksum.open(QIODevice::ReadOnly));
        const auto originalChecksum = checksum.readAll(); checksum.close();
        QCOMPARE(originalChecksum.size(), 65);
        QCOMPARE(originalChecksum, QCryptographicHash::hash(originalBytes, QCryptographicHash::Sha256).toHex() + '\n');
        QString error;
        const auto original = vision::EvaluationWorker::readSample(directory, 0, &error);
        QVERIFY2(!original.isEmpty(), qPrintable(error));
        auto changed = original;
        auto predictions = changed.value("predictions").toArray(); QVERIFY(!predictions.isEmpty());
        auto prediction = predictions[0].toObject(); prediction.insert("confidence", .01);
        prediction.insert("box", QJsonArray{0., 0., 10., 10.}); predictions[0] = prediction;
        changed.insert("predictions", predictions);
        writeFile(base + ".json", QJsonDocument(changed).toJson());
        QVERIFY(vision::EvaluationWorker::readSample(directory, 0, &error).isEmpty());
        QVERIFY(error.contains(QStringLiteral("校验失败")));
        vision::EvaluationPreviewRequest preview;
        preview.directory = directory; preview.index = 0;
        preview.sample = manifest.value("samples").toArray()[0].toObject(); preview.config = value.config;
        const auto rejected = vision::loadEvaluationPreview(preview);
        QVERIFY(!rejected.error.isEmpty()); QVERIFY(rejected.result.predictions.isEmpty());
        QVERIFY(rejected.result.image.isNull()); QVERIFY(!rejected.completed);
        QHash<QString, int> indices; indices.insert(value.files[0], 0); indices.insert(value.files[1], 1);
        QVERIFY(vision::filterEvaluationSamples(value.files, indices, directory, 0, 0, .5, .5, "fp", {}, &error).isEmpty());
        QVERIFY(error.contains(QStringLiteral("校验失败")));
        QCOMPARE(readJson(directory + "/report.json").value("metrics"), report.value("metrics"));
        writeFile(base + ".json", originalBytes);
        QVERIFY2(!vision::EvaluationWorker::readSample(directory, 0, &error).isEmpty(), qPrintable(error));
        writeFile(base + ".sha256", QByteArray(65, '0'));
        QVERIFY(vision::EvaluationWorker::readSample(directory, 0, &error).isEmpty());
        QVERIFY(error.contains(QStringLiteral("格式损坏")));
        QVERIFY(QFile::remove(base + ".sha256"));
        QVERIFY(vision::EvaluationWorker::readSample(directory, 0, &error).isEmpty());
        QVERIFY(error.contains(QStringLiteral("缺少校验文件")));
    }

    void filtersReadCompactCacheWithoutOpeningImages()
    {
        const auto manifest = dataset(3);
        auto value = request(manifest);
        QString directory;
        evaluate(value, &directory);
        QHash<QString, int> indices;
        for (int i = 0; i < value.files.size(); ++i)
        {
            indices.insert(value.files[i], i);
            QVERIFY(QFile::remove(value.files[i]));
        }
        QString error;
        QCOMPARE(vision::filterEvaluationSamples(value.files, indices, directory, 0, 0, .5, .5, "fp", {}, &error),
                 QVector<bool>({false, false, true}));
        QVERIFY2(error.isEmpty(), qPrintable(error));
        QCOMPARE(vision::filterEvaluationSamples(value.files, indices, directory, 0, 0, .5, .5, "negative_fp"),
                 QVector<bool>({false, false, true}));
        QCOMPARE(vision::filterEvaluationSamples(value.files, indices, directory, 0, 0, .9, .5, "fn"),
                 QVector<bool>({true, true, false}));
        QCOMPARE(vision::filterEvaluationSamples(value.files, {}, "missing", 0, 0, .5, .5, "all"),
                 QVector<bool>({true, true, true}));
        QVERIFY(vision::filterEvaluationSamples(value.files, indices, directory, 0, 0, .5, .5, "fp",
                    [] { return true; }, &error).isEmpty());
        QVERIFY(!error.isEmpty());
        auto cancelled = vision::loadEvaluationPreview({}, [] { return true; });
        QVERIFY(!cancelled.error.isEmpty());
        QVERIFY(cancelled.result.image.isNull());
    }
};

QTEST_GUILESS_MAIN(EvaluationWorkerTests)
#include "evaluation_worker_tests.moc"
