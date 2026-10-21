// SPDX-License-Identifier: AGPL-3.0-only
#include "ui/evaluationpanel.h"
#include <QApplication>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QElapsedTimer>
#include <QFile>
#include <QFileDialog>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QJsonArray>
#include <QJsonDocument>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QScreen>
#include <QScrollArea>
#include <QScrollBar>
#include <QSignalSpy>
#include <QSpinBox>
#include <QStandardItemModel>
#include <QTableWidget>
#include <QToolButton>
#include <QTemporaryDir>
#include <QTimer>
#include <QWheelEvent>
#include <QtTest>
#include <functional>
#include <memory>

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
    bool choosePath(const std::function<void()> &open, const QString &path, bool directory, bool cancel = false)
    {
        bool seen = false, correctMode = false;
        QElapsedTimer elapsed; elapsed.start(); QTimer timer;
        connect(&timer, &QTimer::timeout, this, [&] {
            for (auto *widget : QApplication::topLevelWidgets()) {
                auto *dialog = qobject_cast<QFileDialog *>(widget);
                if (!dialog || !dialog->isVisible()) continue;
                correctMode = dialog->objectName() == "yoloDatasetSourceDialog" && dialog->fileMode() == QFileDialog::AnyFile;
                if (cancel || elapsed.elapsed() > 15000) { dialog->reject(); return; }
                if (!seen) { seen = true; dialog->setDirectory(QFileInfo(path).absolutePath()); }
                if (auto *filename = dialog->findChild<QLineEdit *>("fileNameEdit")) {
                    if (filename->text() != path) { filename->setFocus(); filename->selectAll(); QTest::keyClicks(filename, path); }
                    if (filename->text() == path) QMetaObject::invokeMethod(dialog, "accept", Qt::QueuedConnection);
                }
            }
        });
        timer.start(10); open(); timer.stop();
        return correctMode && (cancel || (seen && QFileInfo(path).isDir() == directory)) && elapsed.elapsed() < 15000;
    }
private slots:
    void initTestCase()
    {
        QApplication::setAttribute(Qt::AA_DontUseNativeDialogs);
    }
    void sameDatasetPickerAcceptsDirectoryAndYaml()
    {
        QTemporaryDir area; QVERIFY(area.isValid());
        const QString yaml = area.path() + "/dataset.yaml";
        QFile file(yaml); QVERIFY(file.open(QIODevice::WriteOnly)); file.write("names: [ball]\nval: images/val\n"); file.close();
        EvaluationPanel panel; QSignalSpy requested(&panel, &EvaluationPanel::datasetRequested);
        QVERIFY(choosePath([&] { panel.chooseDataset(); }, area.path(), true));
        QCOMPARE(requested.count(), 1); QCOMPARE(requested.takeFirst().at(0).toString(), area.path());
        QCOMPARE(panel.datasetYaml(), area.path());
        QVERIFY(choosePath([&] { panel.chooseDataset(); }, yaml, false));
        QCOMPARE(requested.count(), 1); QCOMPARE(requested.takeFirst().at(0).toString(), yaml);
        QCOMPARE(panel.datasetYaml(), yaml);
    }
    void cancellingOrBusyDoesNotRequestDataset()
    {
        EvaluationPanel panel; panel.setDataset(dataset());
        QSignalSpy requested(&panel, &EvaluationPanel::datasetRequested);
        QVERIFY(choosePath([&] { panel.chooseDataset(); }, {}, true, true));
        QCOMPARE(requested.count(), 0); QCOMPARE(panel.datasetYaml(), QString("/dataset/data.yaml"));
        QVERIFY(choosePath([&] { panel.chooseDataset(); }, {}, false, true));
        QCOMPARE(requested.count(), 0); QCOMPARE(panel.datasetYaml(), QString("/dataset/data.yaml"));
        panel.setBusy(true); panel.chooseDataset(); panel.chooseDataset();
        QCOMPARE(requested.count(), 0); QCOMPARE(panel.datasetYaml(), QString("/dataset/data.yaml"));
    }
    void unifiedPickerRejectsInvalidSourceInline()
    {
        QTemporaryDir area; QVERIFY(area.isValid()); const QString invalid = area.path() + "/weights.pt";
        QFile file(invalid); QVERIFY(file.open(QIODevice::WriteOnly)); file.write("not a dataset configuration"); file.close();
        EvaluationPanel panel; QSignalSpy requested(&panel, &EvaluationPanel::datasetRequested);
        bool seen = false, rejectedInline = false, attempted = false; QElapsedTimer elapsed; elapsed.start(); QTimer timer;
        connect(&timer, &QTimer::timeout, this, [&] {
            auto *dialog = qobject_cast<QFileDialog *>(QApplication::activeModalWidget());
            if (!dialog || dialog->objectName() != "yoloDatasetSourceDialog") return;
            seen = true;
            if (elapsed.elapsed() > 10000) { dialog->reject(); return; }
            if (!attempted) {
                dialog->setDirectory(area.path());
                if (auto *editor = dialog->findChild<QLineEdit *>("fileNameEdit")) {
                    editor->setFocus(); editor->selectAll(); QTest::keyClicks(editor, invalid);
                }
                attempted = true; QMetaObject::invokeMethod(dialog, "accept", Qt::QueuedConnection); return;
            }
            auto *hint = dialog->findChild<QLabel *>("yoloDatasetSourceHint");
            if (hint && hint->isVisible() && hint->text().startsWith("请选择")) {
                rejectedInline = dialog->isVisible() && requested.isEmpty(); dialog->reject();
            }
        });
        timer.start(10); panel.chooseDataset(); timer.stop();
        QVERIFY(seen); QVERIFY(rejectedInline); QCOMPARE(requested.count(), 0); QVERIFY(panel.datasetYaml().isEmpty());
    }
    void failedFirstImportKeepsSourceAndAllowsExplicitSplitRetry()
    {
        EvaluationPanel panel; panel.setModel(model());
        QSignalSpy requested(&panel, &EvaluationPanel::datasetRequested);
        QSignalSpy changes(&panel, &EvaluationPanel::configChanged);
        panel.setDatasetSource("/dataset/train-only", "val", "/dataset/audit.csv");
        QCOMPARE(requested.count(), 0); QCOMPARE(changes.count(), 0);
        panel.showDatasetImportError("该数据集未配置 Val，请选择 Train 后重试。");
        QCOMPARE(panel.datasetYaml(), QString("/dataset/train-only"));
        QCOMPARE(panel.metadataPath(), QString("/dataset/audit.csv"));
        QVERIFY(panel.dataset().isEmpty()); QVERIFY(!panel.canEvaluate());
        auto *split = control<QComboBox>(panel, "evaluationSplitCombo"); QVERIFY(split);
        auto *reload = control<QPushButton>(panel, "evaluationReloadDatasetButton"); QVERIFY(reload);
        QVERIFY(!split->isHidden()); QVERIFY(!reload->isHidden()); QVERIFY(reload->isEnabled());
        QVERIFY(control<QLabel>(panel, "evaluationDatasetHint")->toolTip().contains("未配置 Val"));
        split->setCurrentIndex(split->findData("train"));
        QCOMPARE(requested.count(), 1); const auto request = requested.takeFirst();
        QCOMPARE(request.at(0).toString(), QString("/dataset/train-only"));
        QCOMPARE(request.at(1).toString(), QString("train"));
        QCOMPARE(request.at(2).toString(), QString("/dataset/audit.csv"));
        panel.clearDataset(); QVERIFY(panel.datasetYaml().isEmpty()); QVERIFY(split->isHidden()); QVERIFY(reload->isHidden());
    }
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
    void settingsPopupKeepsToolbarGeometryAndDismissesWithoutConfigurationChanges()
    {
        QWidget window; window.setObjectName("workspace"); auto *layout = new QVBoxLayout(&window);
        auto panelOwner = std::make_unique<EvaluationPanel>(); auto *panel = panelOwner.get(); layout->addWidget(panel->controlsWidget()); layout->addStretch();
        QFile theme(QStringLiteral(VISION_PROJECT_DIR) + "/assets/fluent-dark.qss"); QVERIFY(theme.open(QIODevice::ReadOnly));
        window.setStyleSheet(QString::fromUtf8(theme.readAll()));
        window.resize(970, 600); panel->setModel(model()); panel->setDataset(dataset()); window.show(); QTest::qWait(20);
        auto *settings = control<QWidget>(*panel, "evaluationDatasetSettings");
        auto *settingsButton = control<QToolButton>(*panel, "evaluationDatasetSettingsButton");
        auto *advanced = control<QWidget>(*panel, "evaluationAdvancedControls");
        QVERIFY(settings && settingsButton && advanced); QCOMPARE(settings->windowType(), Qt::Popup);
        QVERIFY(!settings->isVisible()); QVERIFY(!advanced->isVisible());
        QVERIFY(panel->controlsWidget()->sizeHint().height() <= 48);
        const QRect toolbar = panel->controlsWidget()->geometry();
        const bool mode = panel->evaluationMode(); const int modelId = panel->modelClassId(), datasetId = panel->datasetClassId();
        const double scanStart = panel->scanStart(), scanEnd = panel->scanEnd(), scanStep = panel->scanStep();
        QTest::mouseClick(settingsButton, Qt::LeftButton); QTRY_VERIFY(settings->isVisible());
        QCOMPARE(panel->controlsWidget()->geometry(), toolbar);
        QVERIFY(control<QWidget>(*panel, "evaluationModelClassIdSpin")->isVisible());
        QTest::mouseClick(control<QToolButton>(*panel, "evaluationAdvancedButton"), Qt::LeftButton);
        QTRY_VERIFY(advanced->isVisible()); QCOMPARE(panel->controlsWidget()->geometry(), toolbar);
        for (const char *name : {"evaluationModelClassIdSpin", "evaluationDatasetClassIdSpin", "evaluationMetadataEdit", "evaluationMatchIouSpin", "evaluationScanStartSpin", "evaluationScanEndSpin", "evaluationScanStepSpin"}) {
            auto *widget = control<QWidget>(*panel, name); QVERIFY2(widget && widget->isVisible(), name);
            QVERIFY2(settings->rect().contains(QRect(widget->mapTo(settings, QPoint()), widget->size())), name);
            QVERIFY2(widget->height() >= 30, name);
        }
        const QString capture = qEnvironmentVariable("VISION_STUDIO_UI_CAPTURE_DIR");
        if (!capture.isEmpty()) { QVERIFY(QDir().mkpath(capture)); QVERIFY(settings->grab().save(capture + "/panel-dataset-settings.png")); }
        const QSize popupSize = settings->size();
        QTest::keyClick(settings, Qt::Key_Escape); QTRY_VERIFY(!settings->isVisible());
        QVERIFY(!settingsButton->isChecked()); QCOMPARE(panel->controlsWidget()->geometry(), toolbar);
        QTest::mouseClick(settingsButton, Qt::LeftButton); QTRY_VERIFY(settings->isVisible());
        QTest::mouseClick(&window, Qt::LeftButton, Qt::NoModifier, QPoint(2, window.height() - 2));
        QTRY_VERIFY(!settings->isVisible()); QVERIFY(!settingsButton->isChecked());
        QCOMPARE(panel->controlsWidget()->geometry(), toolbar);
        QCOMPARE(panel->evaluationMode(), mode); QCOMPARE(panel->modelClassId(), modelId); QCOMPARE(panel->datasetClassId(), datasetId);
        QCOMPARE(panel->scanStart(), scanStart); QCOMPARE(panel->scanEnd(), scanEnd); QCOMPARE(panel->scanStep(), scanStep);
        QTest::mouseClick(settingsButton, Qt::LeftButton); QTRY_VERIFY(settings->isVisible());
        panel->setBusy(true); QTRY_VERIFY(!settings->isVisible()); QVERIFY(!settingsButton->isEnabled());
        panel->setBusy(false); QVERIFY(settingsButton->isEnabled());
        qInfo().noquote() << "dataset_popup_evidence" << "toolbar" << toolbar << "popup" << popupSize
                         << "outside_click_closed" << !settings->isVisible() << "configuration_unchanged" << (panel->modelClassId() == modelId);
        panelOwner.reset();
    }
    void longImportErrorsAreReadableInPopupWithoutGrowingToolbar()
    {
        QWidget window; window.setObjectName("workspace"); auto *layout = new QVBoxLayout(&window);
        auto panelOwner = std::make_unique<EvaluationPanel>(); auto *panel = panelOwner.get(); layout->addWidget(panel->controlsWidget()); layout->addStretch();
        window.resize(970, 600); panel->setModel(model()); window.show();
        panel->setDatasetSource("/dataset/invalid", "val", {});
        const QString error = "ValueError: " + QString(8192, 'x') + "\nTraceback (most recent call last):\n" + QString(8192, 'y');
        panel->showDatasetImportError(error); QTest::qWait(10);
        auto *status = control<QToolButton>(*panel, "evaluationEligibilityStatusButton");
        auto *settings = control<QWidget>(*panel, "evaluationDatasetSettings");
        auto *validation = control<QLabel>(*panel, "evaluationValidationHint");
        auto *scroll = control<QScrollArea>(*panel, "evaluationDatasetSettingsScroll");
        QVERIFY(status && settings && validation && scroll); QCOMPARE(status->text(), QString("导入失败"));
        QVERIFY(!settings->isVisible()); QVERIFY(panel->controlsWidget()->sizeHint().height() <= 48);
        const QRect toolbar = panel->controlsWidget()->geometry();
        QTest::mouseClick(status, Qt::LeftButton); QTRY_VERIFY(settings->isVisible());
        QCOMPARE(panel->controlsWidget()->geometry(), toolbar);
        QCOMPARE(validation->text(), error); QCOMPARE(validation->toolTip(), error); QCOMPARE(validation->accessibleDescription(), error);
        QVERIFY(status->toolTip().contains(error));
        QTRY_VERIFY(scroll->verticalScrollBar()->maximum() > 0);
        scroll->verticalScrollBar()->setValue(scroll->verticalScrollBar()->maximum());
        QCOMPARE(scroll->verticalScrollBar()->value(), scroll->verticalScrollBar()->maximum());
        QVERIFY(!panel->canEvaluate());
        QTest::keyClick(settings, Qt::Key_Escape); QTRY_VERIFY(!settings->isVisible());
        QCOMPARE(panel->controlsWidget()->geometry(), toolbar);
        panelOwner.reset();
    }
    void collapsedStatusExplainsEligibilityAndOpensDetailsWithoutChangingMode()
    {
        QWidget window; auto *layout = new QVBoxLayout(&window); auto panelOwner = std::make_unique<EvaluationPanel>(); auto *panel = panelOwner.get();
        layout->addWidget(panel->controlsWidget()); layout->addStretch(); window.resize(970, 600); window.show();
        panel->setModel(model()); panel->setDataset(dataset());
        auto *status = control<QToolButton>(*panel, "evaluationEligibilityStatusButton");
        auto *settings = control<QWidget>(*panel, "evaluationDatasetSettings");
        auto *settingsButton = control<QToolButton>(*panel, "evaluationDatasetSettingsButton");
        auto *mode = control<QComboBox>(*panel, "evaluationModeCombo");
        QVERIFY(status && settings && settingsButton && mode);
        auto classification = model(); classification.task = vision::ModelTask::Classification;
        panel->setModel(classification);
        QVERIFY(!settingsButton->isChecked()); QVERIFY(settings->isHidden());
        QVERIFY(status->text().contains("仅预览")); QVERIFY(status->toolTip().contains("分类"));
        QVERIFY(!panel->canEvaluate()); QVERIFY(!panel->evaluationMode());
        auto *items = qobject_cast<QStandardItemModel *>(mode->model()); QVERIFY(items);
        QVERIFY(!items->item(1)->isEnabled());
        QTest::mouseClick(status, Qt::LeftButton); QTRY_VERIFY(settings->isVisible());
        QVERIFY(settingsButton->isChecked()); QVERIFY(control<QLabel>(*panel, "evaluationValidationHint")->text().contains("分类"));
        QVERIFY(!panel->evaluationMode());
        QTest::keyClick(settings, Qt::Key_Escape); QTRY_VERIFY(settings->isHidden()); panel->setModel(model());
        QVERIFY(status->text().contains("可评估")); QVERIFY(items->item(1)->isEnabled());
        panelOwner.reset();
    }
    void captureFluentLayout()
    {
        const QString output = qEnvironmentVariable("VISION_EVALUATION_PANEL_SCREENSHOT");
        if (output.isEmpty()) return;
        QWidget window; window.setObjectName("workspace"); auto *layout = new QVBoxLayout(&window);
        auto panelOwner = std::make_unique<EvaluationPanel>(); auto *panel = panelOwner.get();
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
