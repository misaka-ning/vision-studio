// SPDX-License-Identifier: AGPL-3.0-only
#include "app/workbenchconfiguration.h"

#include <QFile>
#include <QSignalSpy>
#include <QTemporaryDir>
#include <QtTest>

using namespace vision;
using namespace vision::app;

class WorkbenchConfigurationTests final : public QObject
{
    Q_OBJECT
  private slots:
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
};
QTEST_GUILESS_MAIN(WorkbenchConfigurationTests)
#include "workbench_configuration_tests.moc"
