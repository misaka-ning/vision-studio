// SPDX-License-Identifier: AGPL-3.0-only
#include "ui/evaluationpanel.h"
#include <QApplication>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QFile>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QJsonArray>
#include <QJsonDocument>
#include <QLabel>
#include <QPushButton>
#include <QSignalSpy>
#include <QSpinBox>
#include <QTableWidget>
#include <QToolButton>
#include <QTemporaryDir>
#include <QWheelEvent>
#include <QtTest>

namespace {
QJsonObject dataset()
{
    return {{"data_yaml", "/dataset/data.yaml"}, {"split", "val"}, {"names", QJsonArray{"football"}},
        {"samples", QJsonArray{QJsonObject{{"image", "/dataset/positive.png"}}, QJsonObject{{"image", "/dataset/negative.png"}}, QJsonObject{{"image", "/dataset/second.png"}}}},
        {"validation", QJsonObject{{"positive_images", 2}, {"negative_images", 1}, {"missing_labels", 0}, {"excluded", 0}}},
        {"requires_attention", false}, {"fingerprint", "sample-dataset"}};
}
vision::ModelConfig model(int count = 80)
{
    vision::ModelConfig config;
    config.modelPath = "/models/yolo.pt"; config.modelClassCount = count;
    for (int i = 0; i < count; ++i) config.labels.append(i == 32 ? "sports ball" : QString("class_%1").arg(i));
    return config;
}
QJsonObject operating(double threshold, double precision, double recall, double f1, QJsonValue fpr)
{
    return {{"confidence", threshold}, {"precision", precision}, {"recall", recall}, {"F1", f1},
        {"no_ball_false_positive_rate", fpr}, {"TP", 2}, {"FP", 1}, {"FN", 0}, {"images", 3}, {"gt_boxes", 2},
        {"no_ball_images", 1}, {"no_ball_false_positive_images", 1}};
}
QJsonObject report()
{
    auto metrics = operating(.25, .7, .9, .79, .2);
    metrics.insert("mAP50", .85); metrics.insert("mAP50_95", .65);
    return {{"status", "complete"}, {"full_split", true}, {"metrics", metrics}, {"manifest", dataset()},
        {"request", QJsonObject{{"model_path", "/models/yolo.pt"}, {"model_class_id", 32}, {"dataset_class_id", 0}, {"match_iou", .5}, {"analysis_threshold", .25}}},
        {"rows", QJsonArray{operating(.1, .5, 1, .66, .5), operating(.25, .7, .9, .79, .2), operating(.5, .95, .8, .87, 0), operating(.9, 1, .3, .46, 0)}}};
}
template<class T> T *control(EvaluationPanel &panel, const char *name)
{
    auto *found = panel.findChild<T *>(name);
    return found ? found : panel.controlsWidget()->findChild<T *>(name);
}
}

class EvaluationPanelTests final : public QObject
{
    Q_OBJECT
private slots:
    void modeRequiresValidDatasetAndDetection()
    {
        EvaluationPanel panel;
        QVERIFY(!panel.canEvaluate()); panel.setMode(true); QVERIFY(!panel.evaluationMode());
        panel.setModel(model()); panel.setDataset(dataset());
        QVERIFY(panel.canEvaluate()); panel.setMode(true); QVERIFY(panel.evaluationMode());
        auto classification = model(); classification.task = vision::ModelTask::Classification;
        panel.setModel(classification);
        QVERIFY(!panel.canEvaluate()); QVERIFY(panel.validationMessage().contains("分类"));
        panel.setMode(false); panel.setMode(true); QVERIFY(!panel.evaluationMode());
    }
    void missingLabelsAndNegativeSamples()
    {
        EvaluationPanel panel; panel.setModel(model());
        auto manifest = dataset(); manifest.insert("requires_attention", true);
        panel.setDataset(manifest); QVERIFY(!panel.canEvaluate());
        QVERIFY(panel.validationMessage().contains("缺失"));
        manifest = dataset(); manifest.insert("validation", QJsonObject{{"positive_images", 0}, {"negative_images", 3}});
        panel.setDataset(manifest); QVERIFY(panel.canEvaluate());
        manifest.insert("samples", QJsonArray{}); panel.setDataset(manifest); QVERIFY(!panel.canEvaluate());
    }
    void mappingPreservesOriginalIdsAndValidatesNewModel()
    {
        EvaluationPanel panel; panel.setDataset(dataset()); panel.setModel(model());
        auto *modelId = control<QSpinBox>(panel, "evaluationModelClassIdSpin"); QVERIFY(modelId);
        modelId->setValue(32); QCOMPARE(panel.modelClassId(), 32);
        QVERIFY(control<QLabel>(panel, "evaluationMappingHint")->toolTip().contains("不代表模型只识别足球"));
        panel.setModel(model(1)); QCOMPARE(panel.modelClassId(), 32);
        QVERIFY(!panel.canEvaluate()); QVERIFY(panel.validationMessage().contains("超出模型"));
        modelId->setValue(0); QVERIFY(panel.canEvaluate());
        auto unknown = model(); unknown.modelClassCount = 0; panel.setModel(unknown);
        QVERIFY(!panel.canEvaluate()); QVERIFY(panel.validationMessage().contains("实际类别"));
    }
    void refreshDoesNotEmitRecursiveConfigurationSignals()
    {
        EvaluationPanel panel; QSignalSpy config(&panel, &EvaluationPanel::configChanged);
        panel.setModel(model()); QCOMPARE(config.count(), 0);
        panel.setDataset(dataset()); QCOMPARE(config.count(), 1);
        panel.setDataset(dataset()); panel.setModel(model()); QCOMPARE(config.count(), 1);
    }
    void sameManifestRestoresFailedDatasetSelectionWithoutRecursiveSignals()
    {
        EvaluationPanel panel; panel.setModel(model()); panel.setDataset(dataset());
        auto *split = control<QComboBox>(panel, "evaluationSplitCombo");
        QSignalSpy changes(&panel, &EvaluationPanel::configChanged);
        QSignalSpy requested(&panel, &EvaluationPanel::datasetRequested);
        split->setCurrentIndex(split->findData("test")); QCOMPARE(requested.count(), 1);
        panel.setDataset(dataset()); QCOMPARE(panel.datasetSplit(), "val");
        QCOMPARE(changes.count(), 0); QCOMPARE(requested.count(), 1);
        panel.clearDataset(); QCOMPARE(panel.datasetSplit(), "val"); QVERIFY(panel.datasetYaml().isEmpty());
    }
    void busyLocksConfigurationAndMapping()
    {
        EvaluationPanel panel; panel.setDataset(dataset()); panel.setModel(model()); panel.setMode(true); panel.setBusy(true);
        for (const char *name : {"evaluationModeCombo", "evaluationSplitCombo", "evaluationModelClassIdSpin", "evaluationDatasetClassIdSpin", "evaluationMatchIouSpin", "evaluationScanStartSpin"})
            QVERIFY2(!control<QWidget>(panel, name)->isEnabled(), name);
        panel.setMode(false); QVERIFY(panel.evaluationMode());
        panel.setBusy(false); QVERIFY(control<QWidget>(panel, "evaluationModeCombo")->isEnabled());
    }
    void spinBoxesIgnoreWheel()
    {
        EvaluationPanel panel; panel.setDataset(dataset()); panel.setModel(model()); panel.setReport(report());
        for (const char *name : {"evaluationMatchIouSpin", "evaluationScanStartSpin", "evaluationScanEndSpin", "evaluationScanStepSpin", "evaluationAnalysisThresholdSpin"}) {
            auto *spin = control<QDoubleSpinBox>(panel, name); QVERIFY(spin); const double old = spin->value();
            QWheelEvent event(QPointF(8, 8), QPointF(8, 8), QPoint(), QPoint(0, 120), Qt::NoButton, Qt::NoModifier, Qt::NoScrollPhase, false);
            QApplication::sendEvent(spin, &event); QCOMPARE(spin->value(), old);
        }
    }
    void operatingPointSelectionNeverAutoApplies()
    {
        EvaluationPanel panel; panel.setReport(report());
        QSignalSpy selected(&panel, &EvaluationPanel::thresholdChanged);
        QSignalSpy applied(&panel, &EvaluationPanel::applyRequested);
        panel.setAnalysisThreshold(.44);
        QCOMPARE(panel.analysisThreshold(), .5); QCOMPARE(selected.count(), 1);
        QCOMPARE(selected.takeFirst().at(0).toDouble(), .5);
        QCOMPARE(control<QLabel>(panel, "evaluationMetric2")->text(), "95.0%");
        QCOMPARE(applied.count(), 0);
        control<QPushButton>(panel, "evaluationApplyThresholdButton")->click();
        QCOMPARE(applied.count(), 1); QCOMPARE(applied.at(0).at(0).toDouble(), .5);
        panel.setBusy(true); control<QPushButton>(panel, "evaluationApplyThresholdButton")->click(); QCOMPARE(applied.count(), 1);
    }
    void arrowsSelectAdjacentComputedThresholds()
    {
        EvaluationPanel panel; panel.setReport(report());
        auto *spin = control<QDoubleSpinBox>(panel, "evaluationAnalysisThresholdSpin");
        QCOMPARE(spin->value(), .25); spin->stepUp(); QCOMPARE(spin->value(), .5);
        spin->stepUp(); QCOMPARE(spin->value(), .9); spin->stepUp(); QCOMPARE(spin->value(), .9);
        spin->stepDown(); QCOMPARE(spin->value(), .5);
    }
    void tableAndCurveSelectActualRows()
    {
        EvaluationPanel panel; panel.resize(850, 900); panel.setReport(report()); panel.show();
        auto *table = control<QTableWidget>(panel, "evaluationThresholdTable"); QVERIFY(table); QCOMPARE(table->rowCount(), 4);
        const QRect row = table->visualItemRect(table->item(3, 0));
        QTest::mouseClick(table->viewport(), Qt::LeftButton, Qt::NoModifier, row.center());
        QCOMPARE(panel.analysisThreshold(), .9);
        auto *curve = control<QWidget>(panel, "evaluationThresholdCurve"); QVERIFY(curve);
        const int x = 48 + qRound((curve->width() - 72) * .5);
        QTest::mouseClick(curve, Qt::LeftButton, Qt::NoModifier, QPoint(x, curve->height() / 2));
        QCOMPARE(panel.analysisThreshold(), .5);
    }
    void unavailableAndPartialMetricsRemainNA()
    {
        EvaluationPanel panel; auto data = report();
        auto metrics = data.value("metrics").toObject(); metrics.insert("mAP50", QJsonValue::Null); metrics.insert("mAP50_95", QJsonValue::Null); data.insert("metrics", metrics);
        data.insert("rows", QJsonArray{operating(.25, .7, .9, .79, QJsonValue::Null)});
        panel.setReport(data);
        QCOMPARE(control<QLabel>(panel, "evaluationMetric0")->text(), "N/A");
        QCOMPARE(control<QLabel>(panel, "evaluationMetric5")->text(), "N/A");
        data = report(); data.insert("status", "cancelled"); panel.setReport(data);
        QCOMPARE(control<QLabel>(panel, "evaluationMetric0")->text(), "N/A");
        QVERIFY(control<QLabel>(panel, "evaluationReportStatus")->text().contains("未完成"));
        data = report(); data.insert("full_split", false); panel.setReport(data);
        QCOMPARE(control<QLabel>(panel, "evaluationMetric0")->text(), "N/A");
        QVERIFY(control<QLabel>(panel, "evaluationReportStatus")->text().contains("子集"));
    }
    void constrainedRecommendationAndTestPolicy()
    {
        EvaluationPanel panel; panel.setReport(report());
        auto *use = control<QPushButton>(panel, "evaluationUseRecommendationButton"); QVERIFY(use->isEnabled());
        use->click(); QCOMPARE(panel.analysisThreshold(), .5);
        auto *strategy = control<QComboBox>(panel, "evaluationRecommendationStrategyCombo");
        auto *constraint = control<QDoubleSpinBox>(panel, "evaluationRecommendationConstraintSpin");
        strategy->setCurrentIndex(1); constraint->setValue(1); use->click(); QCOMPARE(panel.analysisThreshold(), .9);
        auto data = report(); auto manifest = dataset(); manifest.insert("split", "test"); data.insert("manifest", manifest); panel.setReport(data);
        QVERIFY(!use->isEnabled()); QVERIFY(control<QLabel>(panel, "evaluationRecommendationHint")->text().contains("Test"));
        manifest.insert("split", "train"); data.insert("manifest", manifest); panel.setReport(data); QVERIFY(!use->isEnabled());
        data = report(); data.insert("rows", QJsonArray{operating(.5, .5, .8, .6, QJsonValue::Null)}); panel.setReport(data);
        strategy->setCurrentIndex(2); QVERIFY(!use->isEnabled());
        QVERIFY(control<QLabel>(panel, "evaluationRecommendationHint")->text().contains("无满足条件"));
    }
    void recommendationPlateauUsesHighestConfidence()
    {
        EvaluationPanel panel; auto data = report();
        data.insert("rows", QJsonArray{operating(.3, .95, .8, .87, 0), operating(.7, .95, .8, .87, 0)});
        panel.setReport(data);
        auto *use = control<QPushButton>(panel, "evaluationUseRecommendationButton");
        auto *strategy = control<QComboBox>(panel, "evaluationRecommendationStrategyCombo");
        for (int mode : {0, 1, 2}) {
            strategy->setCurrentIndex(mode); QVERIFY(use->isEnabled()); use->click(); QCOMPARE(panel.analysisThreshold(), .7);
        }
    }
    void scanValidationAndImageFilter()
    {
        EvaluationPanel panel; panel.setDataset(dataset()); panel.setModel(model());
        auto *begin = control<QDoubleSpinBox>(panel, "evaluationScanStartSpin");
        auto *end = control<QDoubleSpinBox>(panel, "evaluationScanEndSpin");
        begin->setValue(.9); end->setValue(.1); QVERIFY(!panel.canEvaluate());
        QVERIFY(panel.validationMessage().contains("起点"));
        begin->setValue(.05); end->setValue(.95); QVERIFY(panel.canEvaluate());
        panel.setReport(report()); QSignalSpy filters(&panel, &EvaluationPanel::imageFilterChanged);
        auto *combo = control<QComboBox>(panel, "evaluationImageFilterCombo"); combo->setCurrentIndex(3);
        QCOMPARE(filters.count(), 1); QCOMPARE(filters.at(0).at(0).toString(), "negative_fp");
    }
    void exportedCopyKeepsOriginalAndAddsAnalysisSnapshot()
    {
        QTemporaryDir area; QVERIFY(area.isValid());
        const QString source = area.path() + "/source", destination = area.path() + "/exports";
        QVERIFY(QDir().mkdir(source)); QVERIFY(QDir().mkdir(destination));
        QFile original(source + "/report.json"); QVERIFY(original.open(QIODevice::WriteOnly));
        const QByteArray bytes = QJsonDocument(report()).toJson(); QCOMPARE(original.write(bytes), bytes.size()); original.close();
        QFile markdown(source + "/REPORT.md"); QVERIFY(markdown.open(QIODevice::WriteOnly)); markdown.write("# Report\n"); markdown.close();
        EvaluationPanel panel; panel.setReport(report(), source); panel.setAnalysisThreshold(.5);
        QVERIFY(control<QPushButton>(panel, "evaluationOpenReportButton")->isEnabled());
        panel.exportReportTo(destination);
        auto *exportButton = control<QPushButton>(panel, "evaluationExportReportButton");
        QTRY_VERIFY_WITH_TIMEOUT(exportButton->isEnabled(), 10000);
        const auto entries = QDir(destination).entryList(QDir::Dirs | QDir::NoDotAndDotDot);
        QCOMPARE(entries.size(), 1);
        QFile copy(destination + "/" + entries.first() + "/report.json"); QVERIFY(copy.open(QIODevice::ReadOnly)); QCOMPARE(copy.readAll(), bytes);
        QFile selection(destination + "/" + entries.first() + "/analysis-selection.json"); QVERIFY(selection.open(QIODevice::ReadOnly));
        QCOMPARE(QJsonDocument::fromJson(selection.readAll()).object().value("analysis_threshold").toDouble(), .5);
        QVERIFY(!QFileInfo::exists(source + "/analysis-selection.json"));
        original.open(QIODevice::ReadOnly); QCOMPARE(original.readAll(), bytes); original.close();
        panel.exportReportTo(source); QTRY_VERIFY_WITH_TIMEOUT(exportButton->isEnabled(), 10000);
        QCOMPARE(QDir(source).entryList(QDir::Dirs | QDir::NoDotAndDotDot).size(), 0);
        QVERIFY(control<QLabel>(panel, "evaluationRecommendationHint")->text().contains("不能位于原报告内部"));
    }
    void compactToolbarFitsMinimumWorkbenchWidth()
    {
        QWidget window; window.setObjectName("workspace"); auto *layout = new QVBoxLayout(&window);
        auto *panel = new EvaluationPanel; layout->addWidget(panel->controlsWidget());
        QFile theme(QStringLiteral(VISION_PROJECT_DIR) + "/assets/fluent-dark.qss"); QVERIFY(theme.open(QIODevice::ReadOnly));
        window.setStyleSheet(QString::fromUtf8(theme.readAll()));
        window.resize(970, 500); panel->setModel(model()); window.show(); QTest::qWait(10);
        QVERIFY(panel->controlsWidget()->sizeHint().height() <= 65);
        panel->setDataset(dataset()); QTest::qWait(10);
        QVERIFY(panel->controlsWidget()->sizeHint().height() <= 120);
        control<QToolButton>(*panel, "evaluationAdvancedButton")->click(); QTest::qWait(10);
        for (const char *name : {"evaluationModeCombo", "evaluationSplitCombo", "evaluationModelClassIdSpin", "evaluationDatasetClassIdSpin", "evaluationMetadataEdit", "evaluationMatchIouSpin", "evaluationScanStartSpin", "evaluationScanEndSpin", "evaluationScanStepSpin"}) {
            auto *widget = control<QWidget>(*panel, name); QVERIFY2(widget->isVisible(), name);
            const QRect bounds(widget->mapTo(panel->controlsWidget(), QPoint()), widget->size());
            QVERIFY2(panel->controlsWidget()->rect().contains(bounds), name);
            QVERIFY2(widget->height() >= 30, name);
        }
        // The detached analysis view is not inserted into this toolbar-only test.
        delete panel;
    }
    void captureFluentLayout()
    {
        const QString output = qEnvironmentVariable("VISION_EVALUATION_PANEL_SCREENSHOT");
        if (output.isEmpty()) return;
        QWidget window; window.setObjectName("workspace"); auto *layout = new QVBoxLayout(&window);
        auto *panel = new EvaluationPanel;
        panel->setDataset(dataset()); panel->setModel(model());
        control<QSpinBox>(*panel, "evaluationModelClassIdSpin")->setValue(32); panel->setMode(true); panel->setReport(report());
        layout->addWidget(panel->controlsWidget()); layout->addWidget(panel, 1);
        QFile theme(QStringLiteral(VISION_PROJECT_DIR) + "/assets/fluent-dark.qss"); QVERIFY(theme.open(QIODevice::ReadOnly));
        window.setStyleSheet(QString::fromUtf8(theme.readAll())); window.resize(1260, 960); window.show(); QTest::qWait(60);
        QVERIFY(window.grab().save(output));
    }
};
QTEST_MAIN(EvaluationPanelTests)
#include "evaluation_panel_tests.moc"
