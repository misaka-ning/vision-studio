// SPDX-License-Identifier: AGPL-3.0-only
#include "app/imageresultstore.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QSignalSpy>
#include <QTemporaryDir>
#include <QtTest>

using namespace vision;
using namespace vision::app;

namespace
{
InferenceResult sample(const QString &source = QStringLiteral("frame.png"))
{
    InferenceResult result;
    result.source = source;
    result.modelName = QStringLiteral("selected-model.pt");
    result.image = QImage(80, 48, QImage::Format_RGB32);
    result.image.fill(QColor(35, 35, 35));
    result.originalImage = QImage(80, 48, QImage::Format_RGB32);
    result.originalImage.fill(QColor(16, 125, 220));
    result.predictions = {{32, QStringLiteral("sports ball"), .91f, QRectF(5, 7, 20, 24)}};
    result.inferenceMs = 7.125;
    result.frameNumber = 13;
    result.classFilter = {32};
    result.modelClassCount = 80;
    return result;
}

ModelConfig config()
{
    ModelConfig value;
    value.modelPath = QStringLiteral("models/selected-model.pt");
    value.classFilter = {32};
    value.modelClassCount = 80;
    value.colorMode = InputColorMode::Grayscale;
    value.confidence = .55f;
    value.inputSize = 960;
    return value;
}

ResultWriteRequest snapshotRequest(ImageResultStore &store, const InferenceResult &result)
{
    ResultWriteRequest request;
    request.result = result;
    request.generation = store.generation();
    request.ticket = 123;
    request.directory = store.snapshotDirectory();
    request.snapshotPath = request.directory->filePath(QStringLiteral("snapshot.png"));
    return request;
}

QImage normalized(const QImage &image)
{
    return image.convertToFormat(QImage::Format_ARGB32_Premultiplied);
}
}

class ImageResultStoreTests final : public QObject
{
    Q_OBJECT
  private slots:
    void nullAndDemonstrationResultsCannotReplaceRealResult()
    {
        ImageResultStore store;
        const auto real = sample();
        store.accept(real, config(), "frozen-config", "source-v1");
        auto empty = real;
        empty.image = {};
        store.accept(empty, {}, "wrong-config", "wrong-source");
        auto demonstration = real;
        demonstration.demonstration = true;
        store.accept(demonstration, {}, "wrong-config", "wrong-source");
        QCOMPARE(store.entries().size(), 1);
        QCOMPARE(store.fingerprint(), QStringLiteral("frozen-config"));
        const auto *cached = store.findValid(real.source, "source-v1");
        QVERIFY(cached);
        QCOMPARE(cached->sourceIdentity, QStringLiteral("source-v1"));
        QCOMPARE(cached->result.image, real.image);
        QCOMPARE(cached->config.inputSize, 960);
    }

    void acceptedResultKeepsImmutableConfigurationAndIdentity()
    {
        ImageResultStore store;
        auto result = sample();
        auto parameters = config();
        store.accept(result, parameters, "model-and-parameters-v1", "source-v1");
        parameters.classFilter.clear();
        parameters.confidence = .1f;
        parameters.modelPath = "another-model.onnx";
        result.image.fill(Qt::red);
        result.predictions.clear();
        const auto *cached = store.findValid(result.source, "source-v1");
        QVERIFY(cached);
        QCOMPARE(cached->config.modelPath, QStringLiteral("models/selected-model.pt"));
        QCOMPARE(cached->config.classFilter, QVector<int>{32});
        QCOMPARE(cached->config.confidence, .55f);
        QCOMPARE(cached->config.colorMode, InputColorMode::Grayscale);
        QCOMPARE(cached->result.predictions.size(), 1);
        QCOMPARE(cached->result.image.pixelColor(0, 0), QColor(35, 35, 35));
        QCOMPARE(cached->sourceIdentity, QStringLiteral("source-v1"));
    }

    void unchangedFingerprintPreservesPreviouslyCompletedResults()
    {
        ImageResultStore store;
        store.accept(sample("first.png"), config(), "same-model-and-parameters", "first-v1");
        store.accept(sample("outside-current-range.png"), config(), "same-model-and-parameters", "outside-v1");
        const auto generation = store.generation();
        const auto directory = store.snapshotDirectory();
        QVERIFY(!store.invalidateConfiguration("same-model-and-parameters"));
        QCOMPARE(store.entries().size(), 2);
        QCOMPARE(store.generation(), generation);
        QCOMPARE(store.snapshotDirectory(), directory);
        QVERIFY(store.findValid("outside-current-range.png", "outside-v1"));
    }

    void changedModelOrParameterFingerprintInvalidatesWholeGeneration_data()
    {
        QTest::addColumn<QString>("replacementFingerprint");
        QTest::newRow("model-identity-change") << QStringLiteral("model-v2_parameters-v1");
        QTest::newRow("inference-parameters-change") << QStringLiteral("model-v1_parameters-v2");
    }

    void changedModelOrParameterFingerprintInvalidatesWholeGeneration()
    {
        QFETCH(QString, replacementFingerprint);
        ImageResultStore store;
        store.accept(sample("first.png"), config(), "model-v1_parameters-v1", "first-v1");
        store.accept(sample("second.png"), config(), "model-v1_parameters-v1", "second-v1");
        const auto oldGeneration = store.generation();
        QVERIFY(store.invalidateConfiguration(replacementFingerprint));
        QVERIFY(store.entries().isEmpty());
        QVERIFY(store.fingerprint().isEmpty());
        QCOMPARE(store.generation(), oldGeneration + 1);
        QVERIFY(!store.findValid("first.png", "first-v1"));
        // The task owner sets the accepted fingerprint when starting its new task.
        store.setFingerprint(replacementFingerprint);
        QCOMPARE(store.fingerprint(), replacementFingerprint);
        QVERIFY(!store.invalidateConfiguration("yet-another-empty-generation"));
        QCOMPARE(store.generation(), oldGeneration + 1);
    }

    void replacedSourceInvalidatesOnlyItsOwnEntry()
    {
        ImageResultStore store;
        store.accept(sample("changed.png"), config(), "configuration", "changed-v1");
        store.accept(sample("unchanged.png"), config(), "configuration", "unchanged-v1");
        const auto generation = store.generation();
        QVERIFY(store.findValid("changed.png", "changed-v1"));
        // A temporarily inaccessible source has no usable identity; retain its result.
        QVERIFY(store.findValid("changed.png", {}));
        QVERIFY(!store.findValid("changed.png", "changed-v2"));
        QCOMPARE(store.entries().size(), 1);
        QVERIFY(store.findValid("unchanged.png", "unchanged-v1"));
        QCOMPARE(store.generation(), generation);
        QCOMPARE(store.fingerprint(), QStringLiteral("configuration"));
        QVERIFY(!store.findValid("not-detected.png", "new-file"));
        auto replacement = sample("changed.png");
        replacement.predictions.clear();
        store.accept(replacement, config(), "configuration", "changed-v2");
        const auto *cached = store.findValid("changed.png", "changed-v2");
        QVERIFY(cached);
        QVERIFY(cached->result.predictions.isEmpty());
        QCOMPARE(cached->sourceIdentity, QStringLiteral("changed-v2"));
    }

    void successfulRealSnapshotCommitReleasesPixelsAndPreservesResult()
    {
        ImageResultStore store;
        const auto result = sample();
        store.accept(result, config(), "configuration", "source-v1");
        const auto request = snapshotRequest(store, result);
        ResultWriter writer;
        QSignalSpy complete(&writer, &ResultWriter::completed);
        QSignalSpy ready(&writer, &ResultWriter::shutdownReady);
        QVERIFY(writer.enqueue(request));
        writer.requestShutdown();
        QTRY_COMPARE_WITH_TIMEOUT(ready.count(), 1, 10000);
        QCOMPARE(complete.count(), 1);
        const auto outcome = qvariant_cast<ResultWriteOutcome>(complete.first().first());
        QVERIFY2(outcome.snapshotSaved, qPrintable(outcome.snapshotError));
        QCOMPARE(outcome.ticket, request.ticket);
        QCOMPARE(normalized(QImage(outcome.snapshotPath)), normalized(result.originalImage));
        const auto accepted = store.commit(outcome);
        QVERIFY(accepted.current);
        QVERIFY(!accepted.warnMemoryFallback);
        const auto *cached = store.findValid(result.source, "source-v1");
        QVERIFY(cached);
        QVERIFY(cached->result.image.isNull());
        QVERIFY(cached->result.originalImage.isNull());
        QCOMPARE(cached->snapshotPath, request.snapshotPath);
        QCOMPARE(cached->result.predictions.size(), 1);
        QCOMPARE(cached->result.predictions.first().classId, 32);
        QCOMPARE(cached->result.inferenceMs, result.inferenceMs);
        QCOMPARE(cached->result.frameNumber, result.frameNumber);
        QCOMPARE(cached->config.classFilter, QVector<int>{32});
        QCOMPARE(cached->sourceIdentity, QStringLiteral("source-v1"));
    }

    void snapshotWithoutOriginalImageUsesPreviewPixels()
    {
        ImageResultStore store;
        auto result = sample();
        result.originalImage = {};
        store.accept(result, config(), "configuration", "source-v1");
        const auto request = snapshotRequest(store, result);
        ResultWriter writer;
        QSignalSpy complete(&writer, &ResultWriter::completed);
        QSignalSpy ready(&writer, &ResultWriter::shutdownReady);
        QVERIFY(writer.enqueue(request));
        writer.requestShutdown();
        QTRY_COMPARE_WITH_TIMEOUT(ready.count(), 1, 10000);
        QCOMPARE(complete.count(), 1);
        const auto outcome = qvariant_cast<ResultWriteOutcome>(complete.first().first());
        QVERIFY2(outcome.snapshotSaved, qPrintable(outcome.snapshotError));
        QCOMPARE(normalized(QImage(outcome.snapshotPath)), normalized(result.image));
        QVERIFY(store.commit(outcome).current);
    }

    void failedRealSnapshotRetainsMemoryAndWarnsOncePerGeneration()
    {
        ImageResultStore store;
        const auto result = sample();
        store.accept(result, config(), "configuration", "source-v1");
        auto request = snapshotRequest(store, result);
        // A regular file as parent fails deterministically, including under root.
        QFile blocker(request.directory->filePath("regular-file"));
        QVERIFY(blocker.open(QIODevice::WriteOnly));
        blocker.close();
        request.snapshotPath = blocker.fileName() + "/snapshot.png";
        ResultWriter writer;
        QSignalSpy complete(&writer, &ResultWriter::completed);
        QSignalSpy ready(&writer, &ResultWriter::shutdownReady);
        QVERIFY(writer.enqueue(request));
        writer.requestShutdown();
        QTRY_COMPARE_WITH_TIMEOUT(ready.count(), 1, 10000);
        QCOMPARE(complete.count(), 1);
        const auto outcome = qvariant_cast<ResultWriteOutcome>(complete.first().first());
        QVERIFY(!outcome.snapshotSaved);
        QVERIFY(!outcome.snapshotError.isEmpty());
        const auto accepted = store.commit(outcome);
        QVERIFY(accepted.current);
        QVERIFY(accepted.warnMemoryFallback);
        const auto *cached = store.findValid(result.source, "source-v1");
        QVERIFY(cached);
        QCOMPARE(cached->result.image, result.image);
        QCOMPARE(cached->result.originalImage, result.originalImage);
        QVERIFY(cached->snapshotPath.isEmpty());
        QVERIFY(!store.commit(outcome).warnMemoryFallback);
        store.clear();
        store.accept(result, config(), "configuration", "source-v1");
        auto nextFailure = outcome;
        nextFailure.generation = store.generation();
        QVERIFY(store.commit(nextFailure).warnMemoryFallback);
    }

    void staleSuccessfulAndFailedCompletionsCannotAffectNewGeneration()
    {
        ImageResultStore store;
        const auto oldResult = sample();
        store.accept(oldResult, config(), "old-configuration", "source-v1");
        auto oldWrite = snapshotRequest(store, oldResult);
        ResultWriter writer;
        QSignalSpy complete(&writer, &ResultWriter::completed);
        QSignalSpy ready(&writer, &ResultWriter::shutdownReady);
        QVERIFY(writer.enqueue(oldWrite));
        QVERIFY(store.invalidateConfiguration("new-configuration"));
        auto newResult = sample();
        newResult.image.fill(Qt::green);
        newResult.predictions.clear();
        auto newConfig = config();
        newConfig.confidence = .8f;
        store.accept(newResult, newConfig, "new-configuration", "source-v2");
        writer.requestShutdown();
        QTRY_COMPARE_WITH_TIMEOUT(ready.count(), 1, 10000);
        QCOMPARE(complete.count(), 1);
        auto outcome = qvariant_cast<ResultWriteOutcome>(complete.first().first());
        QVERIFY2(outcome.snapshotSaved, qPrintable(outcome.snapshotError));
        auto rejected = store.commit(outcome);
        QVERIFY(!rejected.current);
        QVERIFY(!rejected.warnMemoryFallback);
        outcome.snapshotSaved = false;
        rejected = store.commit(outcome);
        QVERIFY(!rejected.current);
        QVERIFY(!rejected.warnMemoryFallback);
        const auto *cached = store.findValid(newResult.source, "source-v2");
        QVERIFY(cached);
        QCOMPARE(cached->result.image, newResult.image);
        QVERIFY(cached->result.predictions.isEmpty());
        QVERIFY(cached->snapshotPath.isEmpty());
        QCOMPARE(cached->config.confidence, .8f);
        // A rejected old failure must not consume this generation's one warning.
        outcome.generation = store.generation();
        QVERIFY(store.commit(outcome).warnMemoryFallback);
    }

    void completionForRemovedSourceDoesNotResurrectItOrConsumeWarning()
    {
        ImageResultStore store;
        auto result = sample();
        store.accept(result, config(), "configuration", "source-v1");
        ResultWriteOutcome outcome;
        outcome.source = result.source;
        outcome.generation = store.generation();
        store.remove(result.source);
        QVERIFY(store.commit(outcome).current);
        QVERIFY(!store.commit(outcome).warnMemoryFallback);
        QVERIFY(store.entries().isEmpty());
        store.accept(result, config(), "configuration", "source-v2");
        QVERIFY(store.commit(outcome).warnMemoryFallback);
    }

    void temporaryDirectoriesFollowAcceptedWriteLifetimeAcrossInvalidation()
    {
        QString oldPath;
        QString nextPath;
        {
            ImageResultStore store;
            auto result = sample();
            store.accept(result, config(), "configuration-v1", "source-v1");
            auto request = snapshotRequest(store, result);
            oldPath = request.directory->path();
            QVERIFY(request.directory->isValid());
            QCOMPARE(store.snapshotDirectory(), request.directory);
            ResultWriter writer;
            QSignalSpy complete(&writer, &ResultWriter::completed);
            QSignalSpy ready(&writer, &ResultWriter::shutdownReady);
            QVERIFY(writer.enqueue(request));
            request.directory.reset();
            QVERIFY(store.invalidateConfiguration("configuration-v2"));
            // The accepted disk write and its queued completion still own the old tree.
            QVERIFY(QDir(oldPath).exists());
            nextPath = store.snapshotDirectory()->path();
            QVERIFY(nextPath != oldPath);
            QVERIFY(QDir(nextPath).exists());
            writer.requestShutdown();
            QTRY_COMPARE_WITH_TIMEOUT(ready.count(), 1, 10000);
            QCOMPARE(complete.count(), 1);
            auto outcome = qvariant_cast<ResultWriteOutcome>(complete.first().first());
            QVERIFY2(outcome.snapshotSaved, qPrintable(outcome.snapshotError));
            QVERIFY(QFileInfo(outcome.snapshotPath).isFile());
            QVERIFY(!store.commit(outcome).current);
            complete.clear();
            QVERIFY(QDir(oldPath).exists());
            outcome.directory.reset();
            QTRY_VERIFY_WITH_TIMEOUT(!QDir(oldPath).exists(), 10000);
            QVERIFY(QDir(nextPath).exists());
        }
        QVERIFY(!QDir(oldPath).exists());
        QVERIFY(!QDir(nextPath).exists());
    }

    void destroyingStoreReleasesItsUnsharedTemporaryDirectory()
    {
        QString path;
        {
            ImageResultStore store;
            const auto directory = store.snapshotDirectory();
            QVERIFY(directory->isValid());
            path = directory->path();
            QVERIFY(QDir(path).exists());
        }
        QVERIFY(!QDir(path).exists());
    }
};

QTEST_GUILESS_MAIN(ImageResultStoreTests)
#include "image_result_store_tests.moc"
