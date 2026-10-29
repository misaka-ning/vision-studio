// SPDX-License-Identifier: AGPL-3.0-only
#include "app/workbenchconfiguration.h"

#include <QFile>
#include <QSignalSpy>
#include <QTemporaryDir>
#include <QtTest>
#include <cmath>
#include <functional>

using namespace vision;
using namespace vision::app;

namespace
{
WorkbenchDraft populatedDraft()
{
    WorkbenchDraft draft;
    draft.confidence = .123456789012345; draft.iou = .234567890123456;
    auto &p = draft.parameters;
    p.modelPath = "/authority/model.onnx"; p.labels = {"first", "second"}; p.classFilter = {1};
    p.modelClassCount = 2; p.task = ModelTask::YoloV5; p.inputSize = 416;
    p.confidence = .1f; p.iou = .2f; p.maxDetections = 789;
    p.cpuThreads = 4; p.resolvedCpuThreads = 2; p.swapRB = false;
    p.colorMode = InputColorMode::Color; p.inputChannels = 3;
    p.scale = .00392156862745098; p.meanR = 7.125; p.meanG = 8.25; p.meanB = 9.5;
    p.device = ComputeDevice::CUDA; p.deviceIndex = 3; p.resolvedDevice = ComputeDevice::CPU;
    p.deviceName = "loaded CPU"; p.deviceNotice = "verified notice";
    draft.classFilterText = "1，0"; draft.labelsPath = "/authority/classes.names";
    draft.exportDirectory = "/authority/results"; draft.cameraIndex = 5;
    draft.stereoView = StereoView::Right; draft.automaticExport = true;
    return draft;
}

QVariantMap draftFields(const WorkbenchDraft &draft)
{
    const auto &p = draft.parameters;
    QVariantList filter;
    for (int id : p.classFilter) filter.append(id);
    return {{"confidence", draft.confidence}, {"iou", draft.iou},
        {"modelPath", p.modelPath}, {"labels", p.labels}, {"classFilter", filter},
        {"modelClassCount", p.modelClassCount}, {"task", int(p.task)}, {"inputSize", p.inputSize},
        {"backendConfidence", p.confidence}, {"backendIou", p.iou},
        {"maxDetections", p.maxDetections}, {"cpuThreads", p.cpuThreads}, {"resolvedCpuThreads", p.resolvedCpuThreads},
        {"swapRB", p.swapRB}, {"colorMode", int(p.colorMode)}, {"inputChannels", p.inputChannels},
        {"scale", p.scale}, {"meanR", p.meanR}, {"meanG", p.meanG}, {"meanB", p.meanB},
        {"device", int(p.device)}, {"deviceIndex", p.deviceIndex}, {"resolvedDevice", int(p.resolvedDevice)},
        {"deviceName", p.deviceName}, {"deviceNotice", p.deviceNotice},
        {"classFilterText", draft.classFilterText}, {"labelsPath", draft.labelsPath},
        {"exportDirectory", draft.exportDirectory}, {"cameraIndex", draft.cameraIndex},
        {"stereoView", int(draft.stereoView)}, {"automaticExport", draft.automaticExport}};
}

struct FieldCommand
{
    const char *name;
    std::function<void(WorkbenchConfiguration &)> apply;
    QVariantMap expectedChanges;
};
const QVector<FieldCommand> &fieldCommands()
{
    static const QVector<FieldCommand> commands{
        {"task", [](auto &c) { c.setTask(ModelTask::Classification); }, {{"task", int(ModelTask::Classification)}}},
        {"input-size", [](auto &c) { c.setInputSize(736); }, {{"inputSize", 736}}},
        {"confidence", [](auto &c) { c.setConfidence(.7000000000000001); }, {{"confidence", .7000000000000001}}},
        {"iou", [](auto &c) { c.setIou(.6100000000000001); }, {{"iou", .6100000000000001}}},
        {"scale", [](auto &c) { c.setScale(.0123456789012345); }, {{"scale", .0123456789012345}}},
        {"mean-r", [](auto &c) { c.setMeanR(.123456789012345); }, {{"meanR", .123456789012345}}},
        {"mean-g", [](auto &c) { c.setMeanG(.234567890123456); }, {{"meanG", .234567890123456}}},
        {"mean-b", [](auto &c) { c.setMeanB(.345678901234567); }, {{"meanB", .345678901234567}}},
        {"input-color-both", [](auto &c) { c.setInputColor(InputColorMode::Grayscale, true); },
            {{"colorMode", int(InputColorMode::Grayscale)}, {"swapRB", true}}},
        {"input-color-mode", [](auto &c) { c.setInputColor(InputColorMode::Grayscale, false); },
            {{"colorMode", int(InputColorMode::Grayscale)}}},
        {"input-color-swap", [](auto &c) { c.setInputColor(InputColorMode::Color, true); }, {{"swapRB", true}}},
        {"compute-device", [](auto &c) { c.setComputeDevice(ComputeDevice::Auto); }, {{"device", int(ComputeDevice::Auto)}}},
        {"device-index", [](auto &c) { c.setDeviceIndex(9); }, {{"deviceIndex", 9}}},
        {"class-filter", [](auto &c) { c.setClassFilterText("32，0 32"); }, {{"classFilterText", "32，0 32"}}},
        {"camera-index", [](auto &c) { c.setCameraIndex(12); }, {{"cameraIndex", 12}}},
        {"stereo-view", [](auto &c) { c.setStereoView(StereoView::Left); }, {{"stereoView", int(StereoView::Left)}}},
        {"automatic-export", [](auto &c) { c.setAutomaticExport(false); }, {{"automaticExport", false}}}};
    return commands;
}
}

class WorkbenchConfigurationTests final : public QObject
{
    Q_OBJECT
  private slots:
    void fieldCommandsPreserveUnrelatedAuthority_data()
    {
        QTest::addColumn<int>("commandIndex");
        const auto &commands = fieldCommands();
        for (int index = 0; index < commands.size(); ++index)
            QTest::newRow(commands[index].name) << index;
    }

    void fieldCommandsPreserveUnrelatedAuthority()
    {
        QFETCH(int, commandIndex);
        WorkbenchConfiguration configuration;
        configuration.replaceDraft(populatedDraft());
        const auto &command = fieldCommands()[commandIndex];
        auto expected = draftFields(configuration.snapshot());
        for (auto it = command.expectedChanges.cbegin(); it != command.expectedChanges.cend(); ++it)
            expected.insert(it.key(), it.value());
        QSignalSpy changed(&configuration, &WorkbenchConfiguration::changed);
        QVector<QVariantMap> published;
        connect(&configuration, &WorkbenchConfiguration::changed, this, [&] {
            published.append(draftFields(configuration.snapshot()));
        });
        command.apply(configuration);
        QCOMPARE(changed.count(), 1);
        QCOMPARE(draftFields(configuration.snapshot()), expected);
        QCOMPARE(published.size(), 1);
        QCOMPARE(published.first(), expected);
        command.apply(configuration);
        QCOMPARE(changed.count(), 1);
        QCOMPARE(published.size(), 1);
        QCOMPARE(draftFields(configuration.snapshot()), expected);
    }

    void currentLegalValuesAreNoOps()
    {
        WorkbenchConfiguration configuration;
        const auto draft = populatedDraft();
        configuration.replaceDraft(draft);
        QSignalSpy changed(&configuration, &WorkbenchConfiguration::changed);
        const auto &p = draft.parameters;
        configuration.setTask(p.task); configuration.setInputSize(p.inputSize);
        configuration.setConfidence(draft.confidence); configuration.setIou(draft.iou);
        configuration.setScale(p.scale);
        configuration.setMeanR(p.meanR); configuration.setMeanG(p.meanG); configuration.setMeanB(p.meanB);
        configuration.setInputColor(p.colorMode, p.swapRB);
        configuration.setComputeDevice(p.device); configuration.setDeviceIndex(p.deviceIndex);
        configuration.setClassFilterText(draft.classFilterText); configuration.setCameraIndex(draft.cameraIndex);
        configuration.setStereoView(draft.stereoView); configuration.setAutomaticExport(draft.automaticExport);
        QCOMPARE(changed.count(), 0);
        QCOMPARE(draftFields(configuration.snapshot()), draftFields(draft));
    }

    void doubleCommandsRetainAdjacentRepresentableValues()
    {
        WorkbenchConfiguration configuration;
        const auto before = populatedDraft();
        configuration.replaceDraft(before);
        const double confidence = std::nextafter(before.confidence, 1.0);
        const double iou = std::nextafter(before.iou, 1.0);
        const double scale = std::nextafter(before.parameters.scale, 1.0);
        const double meanR = std::nextafter(before.parameters.meanR, 10.0);
        const double meanG = std::nextafter(before.parameters.meanG, 10.0);
        const double meanB = std::nextafter(before.parameters.meanB, 10.0);
        QSignalSpy changed(&configuration, &WorkbenchConfiguration::changed);
        configuration.setConfidence(confidence); configuration.setIou(iou); configuration.setScale(scale);
        configuration.setMeanR(meanR); configuration.setMeanG(meanG); configuration.setMeanB(meanB);
        const auto after = configuration.snapshot();
        QVERIFY(after.confidence == confidence); QVERIFY(after.iou == iou);
        QVERIFY(after.parameters.scale == scale); QVERIFY(after.parameters.meanR == meanR);
        QVERIFY(after.parameters.meanG == meanG); QVERIFY(after.parameters.meanB == meanB);
        QCOMPARE(after.parameters.confidence, before.parameters.confidence);
        QCOMPARE(after.parameters.iou, before.parameters.iou);
        QCOMPARE(changed.count(), 6);
    }

    void labelImportPreservesExistingMappingOnFailureAndKeepsUtf8Names()
    {
        QTemporaryDir directory; QVERIFY(directory.isValid());
        WorkbenchConfiguration configuration; configuration.setLabels({"original"}, "/original.names");
        QString error;
        QVERIFY(!configuration.importLabels(directory.filePath("missing"), &error)); QVERIFY(!error.isEmpty());
        QCOMPARE(configuration.snapshot().parameters.labels, QStringList{"original"});
        QFile empty(directory.filePath("empty.txt")); QVERIFY(empty.open(QIODevice::WriteOnly)); empty.write(" \r\n"); empty.close();
        QVERIFY(!configuration.importLabels(empty.fileName(), &error)); QCOMPARE(error, QString("标签文件为空。"));
        QCOMPARE(configuration.snapshot().labelsPath, QString("/original.names"));
        QFile valid(directory.filePath("labels.names")); QVERIFY(valid.open(QIODevice::WriteOnly));
        valid.write(QString(QChar(0xfeff)).toUtf8() + QString(" 足球 \r\n\n 球门 \n").toUtf8()); valid.close();
        QVERIFY(configuration.importLabels(valid.fileName(), &error));
        QCOMPARE(configuration.snapshot().parameters.labels, QStringList({"足球", "球门"}));
        QCOMPARE(configuration.snapshot().labelsPath, valid.fileName());
    }

    void compositionPreservesDraftAndOriginalClassIds()
    {
        WorkbenchConfiguration configuration;
        WorkbenchDraft draft;
        draft.parameters.task = ModelTask::YoloV5;
        draft.parameters.inputSize = 512;
        draft.confidence = .42;
        draft.iou = .6;
        draft.parameters.colorMode = InputColorMode::Grayscale;
        draft.parameters.meanR = 7; draft.parameters.meanG = 8; draft.parameters.meanB = 9;
        draft.classFilterText = "32， 0,32";
        draft.cameraIndex = 3; draft.stereoView = StereoView::Right; draft.automaticExport = true;
        configuration.replaceDraft(draft);
        ModelSessionSnapshot model;
        model.activeModel = "/a/model.onnx";
        model.classCount = 80; model.inputChannels = 1;
        const auto composed = configuration.compose(model, false);
        QCOMPARE(composed.modelPath, model.activeModel);
        QCOMPARE(composed.inputSize, 512);
        QCOMPARE(composed.confidence, .42f);
        QCOMPARE(composed.iou, .6f);
        QCOMPARE(composed.classFilter, QVector<int>({0, 32}));
        QCOMPARE(composed.modelClassCount, 80);
        QCOMPARE(composed.inputChannels, 1);
        QCOMPARE(composed.meanR, 7.0); QCOMPARE(composed.meanG, 7.0); QCOMPARE(composed.meanB, 7.0);
        QCOMPARE(configuration.snapshot().parameters.meanG, 8.0);
        QCOMPARE(configuration.snapshot().parameters.meanB, 9.0);
        QCOMPARE(configuration.snapshot().classFilterText, draft.classFilterText);
        QCOMPARE(configuration.snapshot().cameraIndex, 3);
        QCOMPARE(configuration.snapshot().stereoView, StereoView::Right);
        QVERIFY(configuration.snapshot().automaticExport);
    }

    void editorThresholdsRetainDoublePrecisionUntilBackendComposition()
    {
        WorkbenchConfiguration configuration;
        WorkbenchDraft draft;
        draft.confidence = .7; draft.iou = .61;
        // The editable double is authoritative even if an old ModelConfig
        // value was included in a draft supplied by a caller.
        draft.parameters.confidence = .1f; draft.parameters.iou = .2f;
        configuration.replaceDraft(draft);
        QCOMPARE(configuration.snapshot().confidence, .7);
        QCOMPARE(configuration.snapshot().iou, .61);
        QCOMPARE(configuration.compose({}, false).confidence, .7f);
        QCOMPARE(configuration.compose({}, false).iou, .61f);
        QVERIFY(double(configuration.compose({}, false).confidence) != configuration.snapshot().confidence);
    }

    void labelsOnlyFallBackForKnownConvertedOnnx()
    {
        WorkbenchConfiguration configuration;
        ModelSessionSnapshot model;
        model.activeModel = "/model.onnx"; model.nativeLabels = {"native"};
        QVERIFY(configuration.compose(model, false).labels.isEmpty());
        QCOMPARE(configuration.compose(model, true).labels, QStringList{"native"});
        QSignalSpy changed(&configuration, &WorkbenchConfiguration::changed);
        configuration.setLabels({"custom"}, "/labels.txt");
        QCOMPARE(changed.size(), 1);
        QCOMPARE(configuration.compose(model, true).labels, QStringList{"custom"});
        QCOMPARE(configuration.snapshot().labelsPath, QString("/labels.txt"));
        configuration.setLabels({}, {});
        model.activeModel = "/model.pt";
        QVERIFY(configuration.compose(model, true).labels.isEmpty());
    }

    void pytorchUsesNativeNormalizationWithoutMutatingEditableValues()
    {
        WorkbenchConfiguration configuration;
        WorkbenchDraft draft;
        draft.parameters.swapRB = false; draft.parameters.scale = .5;
        draft.parameters.meanR = 3; draft.parameters.meanG = 4; draft.parameters.meanB = 5;
        draft.parameters.labels = {"a", "b"}; draft.classFilterText = "1";
        configuration.replaceDraft(draft);
        ModelSessionSnapshot model;
        model.activeModel = "/weights.torchscript"; model.classCount = 2;
        const auto composed = configuration.compose(model, false);
        QVERIFY(composed.swapRB);
        QCOMPARE(composed.scale, 1.0 / 255.0);
        QCOMPARE(composed.meanR, 0.0); QCOMPARE(composed.meanG, 0.0); QCOMPARE(composed.meanB, 0.0);
        QCOMPARE(composed.labels, draft.parameters.labels);
        QCOMPARE(composed.classFilter, QVector<int>{1});
        QCOMPARE(configuration.snapshot().parameters.scale, .5);
        QVERIFY(!configuration.snapshot().parameters.swapRB);
        QCOMPARE(configuration.snapshot().parameters.meanR, 3.0);
    }

    void classificationDoesNotApplyDetectionFilter()
    {
        WorkbenchConfiguration configuration;
        WorkbenchDraft draft; draft.parameters.task = ModelTask::Classification;
        draft.classFilterText = "0,32";
        configuration.replaceDraft(draft);
        QVERIFY(configuration.compose({}, false).classFilter.isEmpty());
        QCOMPARE(configuration.snapshot().classFilterText, QString("0,32"));
    }

    void adoptedBackendOptionsRequireExactCurrentFileIdentity()
    {
        QTemporaryDir directory;
        QVERIFY(directory.isValid());
        const QString path = directory.filePath("model.onnx");
        QFile file(path); QVERIFY(file.open(QIODevice::WriteOnly)); QCOMPARE(file.write("model"), 5LL); file.close();
        ModelSessionSnapshot model; model.activeModel = path;
        QVariantMap profile{{"adopted", true}, {"model_identity", ModelSessionController::fileIdentity(path)},
            {"config", QVariantMap{{"max_detections", 1234}, {"cpu_threads", 5}}}};
        model.evaluationProfiles.insert(path, profile);
        WorkbenchConfiguration configuration;
        QCOMPARE(configuration.compose(model, false).maxDetections, 1234);
        QCOMPARE(configuration.compose(model, false).cpuThreads, 5);
        profile.insert("model_identity", "stale"); model.evaluationProfiles.insert(path, profile);
        QCOMPARE(configuration.compose(model, false).maxDetections, 300);
        QCOMPARE(configuration.compose(model, false).cpuThreads, 0);
    }

    void exportDirectoryHasOneEditableAuthorityAndSurvivesViewDraftUpdates()
    {
        WorkbenchConfiguration configuration;
        QSignalSpy changed(&configuration, &WorkbenchConfiguration::changed);
        configuration.setExportDirectory("results/选择的目录");
        QCOMPARE(changed.count(), 1);
        auto draft = configuration.snapshot();
        draft.confidence = .7;
        configuration.replaceDraft(draft);
        QCOMPARE(configuration.snapshot().exportDirectory, QStringLiteral("results/选择的目录"));
        auto detached = configuration.snapshot();
        detached.exportDirectory = "unaccepted-directory";
        QCOMPARE(configuration.snapshot().exportDirectory, QStringLiteral("results/选择的目录"));
    }

    void fingerprintPreservesLegacyBytesAndCanonicalClassOrder()
    {
        ModelConfig config;
        config.labels = {"person", "sports ball"};
        config.classFilter = {32, 0};
        // Captured from the pre-extraction MainWindow QDataStream field sequence.
        const QString legacy = "f23ab25e6d97c51d270885b0d47e76e5c608b73dbda0fb727c5c504bfdcbe3f0";
        QCOMPARE(WorkbenchConfiguration::configurationFingerprint(config, "frozen-model-identity"), legacy);
        QCOMPARE(config.classFilter, QVector<int>({32, 0}));
        config.classFilter = {0, 32};
        QCOMPARE(WorkbenchConfiguration::configurationFingerprint(config, "frozen-model-identity"), legacy);
        config.modelClassCount = 80;
        config.inputChannels = 3;
        config.resolvedDevice = ComputeDevice::CUDA;
        config.deviceName = "verified runtime information";
        // Inspection/runtime metadata was not part of the legacy editable-configuration identity.
        QCOMPARE(WorkbenchConfiguration::configurationFingerprint(config, "frozen-model-identity"), legacy);
        QVERIFY(WorkbenchConfiguration::configurationFingerprint(config, "replacement-model-identity") != legacy);
        const QVector<std::function<void(ModelConfig &)>> changes{
            [](ModelConfig &c) { c.task = ModelTask::YoloV5; },
            [](ModelConfig &c) { c.inputSize = 512; },
            [](ModelConfig &c) { c.confidence = .5f; },
            [](ModelConfig &c) { c.iou = .6f; },
            [](ModelConfig &c) { c.colorMode = InputColorMode::Grayscale; },
            [](ModelConfig &c) { c.swapRB = false; },
            [](ModelConfig &c) { c.scale = 1; },
            [](ModelConfig &c) { c.meanR = 1; },
            [](ModelConfig &c) { c.meanG = 1; },
            [](ModelConfig &c) { c.meanB = 1; },
            [](ModelConfig &c) { c.device = ComputeDevice::CUDA; },
            [](ModelConfig &c) { c.deviceIndex = 2; },
            [](ModelConfig &c) { c.labels = {"custom name"}; },
            [](ModelConfig &c) { c.classFilter = {32}; },
            [](ModelConfig &c) { c.maxDetections = 500; },
            [](ModelConfig &c) { c.cpuThreads = 2; }};
        for (qsizetype index = 0; index < changes.size(); ++index) {
            auto changedConfig = config;
            changes.at(index)(changedConfig);
            QVERIFY2(WorkbenchConfiguration::configurationFingerprint(changedConfig, "frozen-model-identity") != legacy,
                     qPrintable(QString("configuration dependency %1 did not invalidate the fingerprint").arg(index)));
        }
    }

    void filterValidationUsesCurrentModelMetadata_data()
    {
        QTest::addColumn<QString>("text");
        QTest::addColumn<int>("classCount");
        QTest::addColumn<QString>("inspectedIdentity");
        QTest::addColumn<bool>("pending");
        QTest::addColumn<bool>("classification");
        QTest::addColumn<bool>("valid");
        QTest::newRow("all-without-inspection") << QString() << 0 << QString() << false << false << true;
        QTest::newRow("raw-coco-class-32") << QString("32") << 80 << QString("current-model") << false << false << true;
        QTest::newRow("normalized-multiple-ids") << QString("32， 0,32") << 80 << QString("current-model") << false << false << true;
        QTest::newRow("negative-id") << QString("-1") << 80 << QString("current-model") << false << false << false;
        QTest::newRow("non-integer") << QString("1.2") << 80 << QString("current-model") << false << false << false;
        QTest::newRow("past-model-range") << QString("32") << 1 << QString("current-model") << false << false << false;
        QTest::newRow("unknown-class-count") << QString("0") << 0 << QString("current-model") << false << false << false;
        QTest::newRow("stale-model-inspection") << QString("32") << 80 << QString("previous-model") << false << false << false;
        QTest::newRow("inspection-pending") << QString("32") << 80 << QString("current-model") << true << false << false;
        QTest::newRow("classification-ignores-detection-filter") << QString("bad-filter") << 80 << QString() << false << true << true;
    }

    void filterValidationUsesCurrentModelMetadata()
    {
        QFETCH(QString, text);
        QFETCH(int, classCount);
        QFETCH(QString, inspectedIdentity);
        QFETCH(bool, pending);
        QFETCH(bool, classification);
        QFETCH(bool, valid);
        WorkbenchConfiguration configuration;
        WorkbenchDraft draft;
        draft.classFilterText = text;
        if (classification) draft.parameters.task = ModelTask::Classification;
        configuration.replaceDraft(draft);
        ModelSessionSnapshot model;
        model.classCount = classCount;
        model.inspectedIdentity = inspectedIdentity;
        model.inspectionPending = pending;
        const auto status = configuration.classFilterStatus(model, "current-model");
        QCOMPARE(status.error.isEmpty(), valid);
        QVERIFY(!status.hint.isEmpty());
        if (!valid) QCOMPARE(status.hint, status.error);
        if (classification) {
            QVERIFY(status.ids.isEmpty());
            QVERIFY(status.hint.contains("图像分类"));
        } else if (valid && !text.isEmpty()) {
            QVERIFY(status.ids.contains(32));
            QVERIFY(!status.hint.contains("ID 0 ·"));
            QVERIFY(status.hint.contains("sports ball"));
            QVERIFY(status.hint.contains("并非只识别足球"));
            if (text.contains("，")) QCOMPARE(status.ids, QVector<int>({0, 32}));
            else QCOMPARE(status.ids, QVector<int>{32});
        }
        QCOMPARE(configuration.snapshot().classFilterText, text);
    }
};
QTEST_GUILESS_MAIN(WorkbenchConfigurationTests)
#include "workbench_configuration_tests.moc"
