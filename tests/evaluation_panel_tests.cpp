// SPDX-License-Identifier: AGPL-3.0-only
#include "ui/evaluationpanel.h"
#include <QApplication>
#include <QComboBox>
#include <QCheckBox>
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
#include <QSplitter>
#include <QStandardItemModel>
#include <QTableWidget>
#include <QTabWidget>
#include <QToolButton>
#include <QToolTip>
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
bool fullyVisibleWithin(const QWidget *widget, const QWidget *window)
{
    if (!widget || !widget->isVisibleTo(window) || widget->size().isEmpty()) return false;
    for (auto *parent = widget->parentWidget(); parent; parent = parent->parentWidget()) {
        if (!parent->rect().contains(QRect(widget->mapTo(parent, QPoint()), widget->size()))) return false;
        if (parent == window) return true;
    }
    return false;
}
bool labelTextFits(const QLabel *label)
{
    if (!label || label->contentsRect().width() <= 0) return false;
    const int textHeight = label->fontMetrics().boundingRect(QRect(0, 0, label->contentsRect().width(), 100000),
        Qt::AlignLeft | Qt::AlignTop | Qt::TextWordWrap, label->text()).height();
    return label->contentsRect().height() >= textHeight && label->height() >= label->heightForWidth(label->width());
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
        control<QTabWidget>(panel, "evaluationAnalysisTabs")->setCurrentIndex(0); QTest::qWait(10);
        auto *table = control<QTableWidget>(panel, "evaluationThresholdTable"); QVERIFY(table); QCOMPARE(table->rowCount(), 4);
        const QRect row = table->visualItemRect(table->item(3, 0));
        QTest::mouseClick(table->viewport(), Qt::LeftButton, Qt::NoModifier, row.center());
        QCOMPARE(panel.analysisThreshold(), .9);
        auto *curve = control<QWidget>(panel, "evaluationThresholdCurve"); QVERIFY(curve);
        const int x = 48 + qRound((curve->width() - 72) * .5);
        QTest::mouseClick(curve, Qt::LeftButton, Qt::NoModifier, QPoint(x, curve->height() / 2));
        QCOMPARE(panel.analysisThreshold(), .5);
    }
    void decisionSidebarSharesCurveTableAndIndependentApply()
    {
        EvaluationPanel panel; panel.resize(1100, 720); panel.setReport(report()); panel.show(); QTest::qWait(15);
        auto *tabs = control<QTabWidget>(panel, "evaluationAnalysisTabs"); QCOMPARE(tabs->currentIndex(), 0);
        auto *page = control<QWidget>(panel, "evaluationThresholdPage");
        auto *curve = control<QWidget>(panel, "evaluationThresholdCurve");
        auto *sidebar = control<QWidget>(panel, "evaluationDecisionSidebar");
        auto *table = control<QTableWidget>(panel, "evaluationThresholdTable");
        auto *threshold = control<QDoubleSpinBox>(panel, "evaluationAnalysisThresholdSpin");
        for (auto *widget : QList<QWidget *>{curve, sidebar, table, threshold, control<QLabel>(panel, "evaluationMetric2"), control<QPushButton>(panel, "evaluationApplyThresholdButton")})
            QVERIFY(page->isAncestorOf(widget));
        QVERIFY(sidebar->mapTo(page, QPoint()).x() > curve->mapTo(page, QPoint()).x());
        QVERIFY(table->mapTo(page, QPoint()).y() > curve->mapTo(page, QPoint()).y());
        QCOMPARE(curve->property("analysisThreshold").toDouble(), .25);
        QCOMPARE(curve->property("recommendedThreshold").toDouble(), .5);
        QSignalSpy applied(&panel, &EvaluationPanel::applyRequested), changes(&panel, &EvaluationPanel::configChanged);
        control<QPushButton>(panel, "evaluationUseRecommendationButton")->click();
        QCOMPARE(panel.analysisThreshold(), .5); QCOMPARE(table->currentRow(), 2);
        QCOMPARE(curve->property("recommendedThreshold").toDouble(), .5); QCOMPARE(applied.count(), 0); QCOMPARE(changes.count(), 0);
        table->setFocus(); QTest::keyClick(table, Qt::Key_Down);
        QCOMPARE(panel.analysisThreshold(), .9); QCOMPARE(curve->property("analysisThreshold").toDouble(), .9);
        QCOMPARE(curve->property("recommendedThreshold").toDouble(), .5); QCOMPARE(applied.count(), 0);
        control<QPushButton>(panel, "evaluationApplyThresholdButton")->click(); QCOMPARE(applied.count(), 1); QCOMPARE(applied.at(0).at(0).toDouble(), .9);
        auto *splitter = control<QSplitter>(panel, "evaluationDecisionSplit"); splitter->setSizes({1, 10000}); QTest::qWait(5);
        const auto beforeHover = curve->grab().toImage();
        const int hoverX = 48 + qRound((curve->width() - 72) * .25);
        QTest::mouseMove(curve, QPoint(hoverX, curve->height() / 2));
        QTRY_VERIFY(curve->property("hoverDetails").toString().contains("阈值 0.250"));
        QTRY_VERIFY(QToolTip::isVisible()); QVERIFY(QToolTip::text().contains("无目标图片误报率 20.0%"));
        const auto afterHover = curve->grab().toImage();
        const int plotTop = curve->width() < 560 ? 54 : 34;
        const QRect recommendationMarker(48 + qRound((curve->width() - 72) * .5), plotTop + 23, 100, 19);
        QCOMPARE(afterHover.copy(recommendationMarker), beforeHover.copy(recommendationMarker));
        QCOMPARE(panel.analysisThreshold(), .9); QCOMPARE(applied.count(), 1); QToolTip::hideText();
        const auto source = panel.report(); const auto sizes = splitter->sizes();
        panel.setImageFilter("errors", false); tabs->setCurrentIndex(2); panel.hide(); panel.show(); tabs->setCurrentIndex(0);
        QCOMPARE(panel.analysisThreshold(), .9); QCOMPARE(panel.imageFilter(), QString("errors")); QCOMPARE(panel.report(), source);
        QCOMPARE(control<QSplitter>(panel, "evaluationDecisionSplit")->sizes(), sizes); QCOMPARE(changes.count(), 0);
    }
    void recommendationScrollsLongThresholdTableAndReviewIsExplicit()
    {
        EvaluationPanel panel; panel.resize(1000, 650); auto data = report(); QJsonArray rows;
        for (int i = 0; i < 91; ++i) rows.append(operating(.05 + .01 * i, .8, .8, i == 70 ? .99 : .5, .1));
        data.insert("rows", rows); panel.setReport(data); panel.show(); QTest::qWait(15);
        auto *table = control<QTableWidget>(panel, "evaluationThresholdTable");
        auto *split = control<QSplitter>(panel, "evaluationDecisionSplit"); split->setSizes({320, 135}); QTest::qWait(5);
        QSignalSpy applied(&panel, &EvaluationPanel::applyRequested), review(&panel, &EvaluationPanel::errorReviewRequested);
        control<QPushButton>(panel, "evaluationUseRecommendationButton")->click(); QTest::qWait(5);
        QCOMPARE(table->rowCount(), 91); QCOMPARE(table->currentRow(), 70); QCOMPARE(panel.analysisThreshold(), .75);
        QVERIFY(table->verticalScrollBar()->value() > 0);
        QVERIFY(table->viewport()->rect().contains(table->visualItemRect(table->item(70, 0))));
        panel.resize(1000, 1000); QTest::qWait(5); panel.resize(1000, 600); QTest::qWait(5);
        QCOMPARE(panel.analysisThreshold(), .75); QCOMPARE(table->currentRow(), 70);
        QTRY_VERIFY(table->viewport()->rect().contains(table->visualItemRect(table->item(70, 0))));
        auto *strategy = control<QComboBox>(panel, "evaluationRecommendationStrategyCombo");
        for (int mode : {1, 2, 0}) {
            strategy->setCurrentIndex(mode); QTest::qWait(5);
            QCOMPARE(panel.analysisThreshold(), .75); QCOMPARE(table->currentRow(), 70);
            QTRY_VERIFY(table->viewport()->rect().contains(table->visualItemRect(table->item(70, 0))));
        }
        split->setSizes({10000, 1}); QTest::qWait(5);
        QCOMPARE(panel.analysisThreshold(), .75); QCOMPARE(table->currentRow(), 70);
        QTRY_VERIFY(table->viewport()->rect().contains(table->visualItemRect(table->item(70, 0))));
        panel.setAnalysisThreshold(.95); QTest::qWait(5); QCOMPARE(table->currentRow(), 90);
        QVERIFY(table->viewport()->rect().contains(table->visualItemRect(table->item(90, 0))));
        panel.setAnalysisThreshold(.05); QTest::qWait(5); QCOMPARE(table->currentRow(), 0);
        QVERIFY(table->viewport()->rect().contains(table->visualItemRect(table->item(0, 0))));
        panel.setErrorSummary(3, 2, 1, 2, 1, 3);
        QVERIFY(control<QLabel>(panel, "evaluationErrorSummaryLabel")->text().contains("错图 2"));
        auto *button = control<QPushButton>(panel, "evaluationReviewErrorsButton"); button->click(); QCOMPARE(review.count(), 1);
        QCOMPARE(applied.count(), 0); QCOMPARE(panel.analysisThreshold(), .05);
        panel.setBusy(true); QVERIFY(!button->isEnabled()); button->click(); QCOMPARE(review.count(), 1);
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
        const auto originalReport = panel.report();
        const QString recommendation = control<QLabel>(panel, "evaluationRecommendationHint")->text();
        auto *feedback = control<QLabel>(panel, "evaluationExportStatusLabel"); QVERIFY(feedback);
        QVERIFY(control<QWidget>(panel, "evaluationReportPage")->isAncestorOf(feedback));
        QVERIFY(control<QPushButton>(panel, "evaluationOpenReportButton")->isEnabled());
        panel.exportReportTo(destination);
        auto *exportButton = control<QPushButton>(panel, "evaluationExportReportButton");
        QTRY_VERIFY_WITH_TIMEOUT(exportButton->isEnabled(), 10000);
        QVERIFY(feedback->text().contains("已导出")); QCOMPARE(feedback->toolTip(), feedback->text());
        QCOMPARE(control<QLabel>(panel, "evaluationRecommendationHint")->text(), recommendation);
        QCOMPARE(panel.analysisThreshold(), .5); QCOMPARE(panel.report(), originalReport);
        const auto entries = QDir(destination).entryList(QDir::Dirs | QDir::NoDotAndDotDot);
        QCOMPARE(entries.size(), 1);
        QFile copy(destination + "/" + entries.first() + "/report.json"); QVERIFY(copy.open(QIODevice::ReadOnly)); QCOMPARE(copy.readAll(), bytes);
        QFile selection(destination + "/" + entries.first() + "/analysis-selection.json"); QVERIFY(selection.open(QIODevice::ReadOnly));
        QCOMPARE(QJsonDocument::fromJson(selection.readAll()).object().value("analysis_threshold").toDouble(), .5);
        QVERIFY(!QFileInfo::exists(source + "/analysis-selection.json"));
        original.open(QIODevice::ReadOnly); QCOMPARE(original.readAll(), bytes); original.close();
        panel.exportReportTo(source); QTRY_VERIFY_WITH_TIMEOUT(exportButton->isEnabled(), 10000);
        QCOMPARE(QDir(source).entryList(QDir::Dirs | QDir::NoDotAndDotDot).size(), 0);
        QVERIFY(feedback->text().contains("不能位于原报告内部")); QCOMPARE(feedback->toolTip(), feedback->text());
        QCOMPARE(control<QLabel>(panel, "evaluationRecommendationHint")->text(), recommendation);
        QCOMPARE(panel.analysisThreshold(), .5); QCOMPARE(panel.report(), originalReport);
        panel.setReport(report(), source);
        QVERIFY(feedback->text().isEmpty()); QVERIFY(feedback->toolTip().isEmpty()); QVERIFY(feedback->isHidden());
        original.open(QIODevice::ReadOnly); QCOMPARE(original.readAll(), bytes); original.close();
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
    void advancedInferenceOptionsValidateAndLock()
    {
        EvaluationPanel panel; panel.setDataset(dataset()); panel.setModel(model());
        QCOMPARE(panel.batchSize(), 1); QCOMPARE(panel.cpuThreads(), 0); QCOMPARE(panel.maxDetections(), 300); QCOMPARE(panel.scoreFloor(), .001);
        control<QSpinBox>(panel, "evaluationBatchSizeSpin")->setValue(8);
        control<QSpinBox>(panel, "evaluationCpuThreadsSpin")->setValue(4);
        control<QSpinBox>(panel, "evaluationMaxDetectionsSpin")->setValue(600);
        QCOMPARE(panel.batchSize(), 8); QCOMPARE(panel.cpuThreads(), 4); QCOMPARE(panel.maxDetections(), 600); QVERIFY(panel.canEvaluate());
        auto onnx = model(); onnx.modelPath = "/models/yolo.onnx"; panel.setModel(onnx);
        QVERIFY(!panel.canEvaluate()); QVERIFY(panel.validationMessage().contains("ONNX"));
        control<QSpinBox>(panel, "evaluationCpuThreadsSpin")->setValue(0); QVERIFY(panel.canEvaluate());
        control<QDoubleSpinBox>(panel, "evaluationScoreFloorSpin")->setValue(.1);
        QVERIFY(!panel.canEvaluate()); QVERIFY(panel.validationMessage().contains("扫描起点"));
        control<QDoubleSpinBox>(panel, "evaluationScoreFloorSpin")->setValue(.001); QVERIFY(panel.canEvaluate());
        panel.setBusy(true);
        for (const char *name : {"evaluationBatchSizeSpin", "evaluationCpuThreadsSpin", "evaluationMaxDetectionsSpin", "evaluationScoreFloorSpin"})
            QVERIFY2(!control<QWidget>(panel, name)->isEnabled(), name);
    }
    void diagnosticMetricsAndMatricesRetainCachedFields()
    {
        EvaluationPanel panel; auto data = report(); auto row = operating(.25, .7, .9, .79, .2);
        row.insert("no_ball_fpr_wilson95", QJsonArray{.02, .62}); row.insert("fp_boxes_per_image", .333);
        row.insert("no_ball_fp_boxes_per_image", 1.5); row.insert("positive_image_miss_rate", .25);
        row.insert("no_ball_false_positive_boxes", 3);
        row.insert("presence_confusion", QJsonObject{{"TP", 2}, {"FP", 1}, {"FN", 0}, {"TN", 4}});
        auto metrics = row; metrics.insert("mAP50", .85); metrics.insert("mAP75", .75); metrics.insert("mAP50_95", .65);
        data.insert("metrics", metrics); data.insert("rows", QJsonArray{row}); panel.setReport(data);
        auto *details = control<QTableWidget>(panel, "evaluationExtendedMetricsTable"); QVERIFY(details);
        QCOMPARE(details->item(0, 1)->text(), QString("75.0%")); QCOMPARE(details->item(0, 3)->text(), QString("0.333"));
        QCOMPARE(details->item(2, 1)->text(), QString("2.0% ～ 62.0%")); QCOMPARE(details->item(2, 3)->text(), QString("3"));
        auto *presence = control<QTableWidget>(panel, "evaluationPresenceMatrix");
        auto *localization = control<QTableWidget>(panel, "evaluationLocalizationMatrix");
        QCOMPARE(presence->item(1, 1)->text(), QString("TN  4")); QCOMPARE(localization->item(1, 1)->text(), QString("TN  不定义"));
        panel.setOperatingMetrics(operating(.25, .7, .9, .79, .2));
        QCOMPARE(details->item(0, 3)->text(), QString("0.333")); QCOMPARE(presence->item(1, 1)->text(), QString("TN  4"));
        data.insert("full_split", false); panel.setReport(data); QCOMPARE(details->item(0, 1)->text(), QString("N/A"));
        panel.setReport(report()); QCOMPARE(details->item(0, 3)->text(), QString("N/A")); QCOMPARE(presence->item(1, 1)->text(), QString("TN  N/A"));
    }
    void reportTabsAndPlotsDoNotChangeAnalysisOrRunConfiguration()
    {
        EvaluationPanel panel; auto data = report(); auto metrics = data.value("metrics").toObject();
        metrics.insert("ap_per_iou", QJsonArray{.9, .88, .85, .83, .8, .75, .7, .6, .5, .4}); data.insert("metrics", metrics);
        data.insert("pr_curve", QJsonObject{{"recall", QJsonArray{0., .5, 1.}}, {"precision_envelope", QJsonArray{1., .9, .2}}});
        panel.setDataset(dataset()); panel.setModel(model()); panel.setReport(data); panel.resize(1100, 900); panel.show();
        QSignalSpy config(&panel, &EvaluationPanel::configChanged), threshold(&panel, &EvaluationPanel::thresholdChanged), apply(&panel, &EvaluationPanel::applyRequested);
        auto *tabs = control<QTabWidget>(panel, "evaluationAnalysisTabs"); QCOMPARE(tabs->count(), 4);
        QCOMPARE(tabs->tabText(0), QString("阈值调优")); QCOMPARE(tabs->tabText(1), QString("能力诊断"));
        QCOMPARE(tabs->tabText(2), QString("场景分析")); QCOMPARE(tabs->tabText(3), QString("报告管理"));
        for (int i = 0; i < tabs->count(); ++i) tabs->setCurrentIndex(i);
        tabs->setCurrentIndex(1); auto *kind = control<QComboBox>(panel, "evaluationDiagnosticPlotCombo");
        for (int i = 0; i < kind->count(); ++i) {
            kind->setCurrentIndex(i); QTest::qWait(1);
            const auto image = control<QWidget>(panel, "evaluationDiagnosticPlot")->grab().toImage(); QVERIFY(!image.isNull());
        }
        QCOMPARE(panel.analysisThreshold(), .25); QCOMPARE(config.count(), 0); QCOMPARE(threshold.count(), 0); QCOMPARE(apply.count(), 0);
    }
    void reanalysisOptionsAreIndependentAndSourceReportStaysImmutable()
    {
        QTemporaryDir area; QVERIFY(area.isValid()); QFile original(area.path() + "/report.json"); QVERIFY(original.open(QIODevice::WriteOnly));
        const QByteArray bytes = QJsonDocument(report()).toJson(); original.write(bytes); original.close();
        EvaluationPanel panel; panel.setReport(report(), area.path());
        QSignalSpy config(&panel, &EvaluationPanel::configChanged), requested(&panel, &EvaluationPanel::reanalysisRequested), apply(&panel, &EvaluationPanel::applyRequested);
        auto *run = control<QPushButton>(panel, "evaluationReanalyzeButton"); QVERIFY(run->isEnabled());
        control<QDoubleSpinBox>(panel, "evaluationReanalysisIouSpin")->setValue(.65);
        control<QDoubleSpinBox>(panel, "evaluationReanalysisConfidenceSpin")->setValue(.4);
        control<QDoubleSpinBox>(panel, "evaluationReanalysisStartSpin")->setValue(.1);
        control<QDoubleSpinBox>(panel, "evaluationReanalysisEndSpin")->setValue(.8);
        control<QDoubleSpinBox>(panel, "evaluationReanalysisStepSpin")->setValue(.02);
        control<QCheckBox>(panel, "evaluationFigurePdfCheck")->setChecked(true); control<QSpinBox>(panel, "evaluationFigureDpiSpin")->setValue(600);
        run->click(); QCOMPARE(requested.count(), 1); const auto options = requested.takeFirst().at(0).toJsonObject();
        QCOMPARE(options.value("match_iou").toDouble(), .65); QCOMPARE(options.value("confidence").toDouble(), .4);
        QCOMPARE(options.value("scan_start").toDouble(), .1); QCOMPARE(options.value("scan_step").toDouble(), .02);
        QCOMPARE(options.value("figure_dpi").toInt(), 600); QCOMPARE(options.value("figure_formats").toArray(), (QJsonArray{"png", "svg", "pdf"}));
        QCOMPARE(panel.matchIou(), .5); QCOMPARE(panel.scanStart(), .05); QCOMPARE(panel.analysisThreshold(), .25);
        QCOMPARE(config.count(), 0); QCOMPARE(apply.count(), 0); QCOMPARE(panel.report(), report());
        control<QDoubleSpinBox>(panel, "evaluationReanalysisStartSpin")->setValue(.9); QVERIFY(!run->isEnabled());
        QVERIFY(control<QLabel>(panel, "evaluationReanalysisValidationLabel")->text().contains("起点"));
        control<QDoubleSpinBox>(panel, "evaluationReanalysisStartSpin")->setValue(.1); QVERIFY(run->isEnabled());
        control<QDoubleSpinBox>(panel, "evaluationReanalysisConfidenceSpin")->setValue(0); QVERIFY(!run->isEnabled());
        QVERIFY(control<QLabel>(panel, "evaluationReanalysisValidationLabel")->text().contains("采集下限"));
        control<QDoubleSpinBox>(panel, "evaluationReanalysisConfidenceSpin")->setValue(.4); QVERIFY(run->isEnabled());
        for (const char *name : {"evaluationFigurePngCheck", "evaluationFigureSvgCheck", "evaluationFigurePdfCheck"}) control<QCheckBox>(panel, name)->setChecked(false);
        QVERIFY(!run->isEnabled()); QVERIFY(control<QLabel>(panel, "evaluationReanalysisValidationLabel")->text().contains("格式"));
        control<QCheckBox>(panel, "evaluationFigurePdfCheck")->setChecked(true); panel.setBusy(true); QVERIFY(!run->isEnabled());
        QVERIFY(!control<QDoubleSpinBox>(panel, "evaluationReanalysisIouSpin")->isEnabled());
        QVERIFY(original.open(QIODevice::ReadOnly)); QCOMPARE(original.readAll(), bytes);
    }
    void sceneTableDistinguishesUnknownAndKnownAndRequestsOriginalGroup()
    {
        QTemporaryDir area; QVERIFY(area.isValid()); EvaluationPanel panel; auto data = report();
        auto all = data.value("metrics").toObject(); all.insert("available", true); all.insert("review_status", "not_applicable");
        all.insert("sample_images", QJsonArray{"/dataset/positive.png", "/dataset/negative.png", "/dataset/second.png"});
        auto small = all; small.insert("review_status", "reviewed"); small.insert("images", 2); small.insert("mAP75", .55);
        data.insert("groups", QJsonObject{{"All", all}, {"Small", small}, {"Blurred", QJsonObject{{"available", false}, {"review_status", "unreviewed"}}}});
        data.insert("scene_review", QJsonObject{{"reviewed_images", 2}, {"unreviewed_images", 1}}); panel.setReport(data, area.path());
        auto *table = control<QTableWidget>(panel, "evaluationSceneMetricsTable"); QCOMPARE(table->rowCount(), 6);
        QCOMPARE(table->item(1, 4)->text(), QString("55.0%")); QCOMPARE(table->item(3, 1)->text(), QString("未审核")); QCOMPARE(table->item(3, 2)->text(), QString("N/A"));
        auto *review = control<QPushButton>(panel, "evaluationReviewSceneButton"); QSignalSpy requested(&panel, &EvaluationPanel::sceneReviewRequested);
        table->selectRow(1); QVERIFY(review->isEnabled()); review->click(); QCOMPARE(requested.count(), 1); QCOMPARE(requested.at(0).at(0).toString(), QString("Small"));
        table->selectRow(3); QVERIFY(!review->isEnabled()); review->click(); QCOMPARE(requested.count(), 1);
        panel.setAnalysisThreshold(.5); QVERIFY(control<QLabel>(panel, "evaluationSceneHint")->text().contains("0.250"));
        QVERIFY(control<QLabel>(panel, "evaluationSceneHint")->text().contains("0.500"));
        QSignalSpy reanalysis(&panel, &EvaluationPanel::reanalysisRequested); const auto source = panel.report();
        control<QPushButton>(panel, "evaluationSceneReanalysisButton")->click();
        QCOMPARE(control<QTabWidget>(panel, "evaluationAnalysisTabs")->currentIndex(), 3);
        QCOMPARE(control<QDoubleSpinBox>(panel, "evaluationReanalysisConfidenceSpin")->value(), .5);
        QCOMPARE(reanalysis.count(), 0); QCOMPARE(panel.report(), source);
        auto legacy = report(); legacy.insert("groups", QJsonObject{{"Small", operating(.25, .7, .9, .79, .2)}});
        panel.setReport(legacy, area.path()); table->selectRow(1); QVERIFY(!review->isEnabled());
        QVERIFY(review->toolTip().contains("重新分析")); QCOMPARE(table->item(1, 6)->text(), QString("70.0%"));
    }
    void responsiveDiagnosticLayouts()
    {
        QTemporaryDir area; QVERIFY(area.isValid());
        QWidget window; window.setObjectName("workspace"); auto *layout = new QVBoxLayout(&window);
        auto *fixture = new QLabel("界面布局测试 · 合成指标（非模型实测）"); fixture->setStyleSheet("color:#E8C688; font-size:11px;"); layout->addWidget(fixture);
        auto *panel = new EvaluationPanel; layout->addWidget(panel->controlsWidget()); layout->addWidget(panel, 1);
        panel->setDataset(dataset()); panel->setModel(model()); auto data = report();
        auto metrics = data.value("metrics").toObject(); metrics.insert("mAP75", .75);
        metrics.insert("ap_per_iou", QJsonArray{.9, .88, .85, .83, .8, .75, .7, .6, .5, .4}); data.insert("metrics", metrics);
        data.insert("pr_curve", QJsonObject{{"recall", QJsonArray{0., .5, 1.}}, {"precision_envelope", QJsonArray{1., .9, .2}}});
        auto group = metrics; group.insert("available", true); group.insert("review_status", "reviewed");
        group.insert("sample_images", QJsonArray{"/dataset/positive.png", "/dataset/negative.png", "/dataset/second.png"});
        data.insert("groups", QJsonObject{{"All", group}, {"Small", group}, {"Occluded", group}, {"Blurred", group}, {"No Ball", group}, {"No difficulty", group}});
        data.insert("scene_review", QJsonObject{{"reviewed_images", 3}, {"unreviewed_images", 0}}); panel->setReport(data, area.path());
        QFile theme(QStringLiteral(VISION_PROJECT_DIR) + "/assets/fluent-dark.qss"); QVERIFY(theme.open(QIODevice::ReadOnly)); window.setStyleSheet(QString::fromUtf8(theme.readAll()));
        const QString capture = qEnvironmentVariable("VISION_EVALUATION_PANEL_FIXTURE_DIR");
        if (!capture.isEmpty()) QVERIFY(QDir().mkpath(capture));
        auto *tabs = control<QTabWidget>(*panel, "evaluationAnalysisTabs");
        for (const QSize size : {QSize(1100, 640), QSize(1100, 720), QSize(1260, 820), QSize(1366, 768), QSize(1366, 900), QSize(1600, 1000)}) {
            window.resize(size); window.show(); QTest::qWait(15);
            QCOMPARE(window.size(), size);
            for (int i = 0; i < tabs->count(); ++i) {
                tabs->setCurrentIndex(i); QTest::qWait(5);
                const QRect tabsGeometry(tabs->mapTo(&window, QPoint()), tabs->size());
                if (!capture.isEmpty()) QVERIFY(window.grab().save(QString("%1/%2x%3-tab-%4.png").arg(capture).arg(size.width()).arg(size.height()).arg(i)));
                qInfo() << "layout_geometry" << size << "tab" << i << tabsGeometry << "window" << window.rect();
                QVERIFY2(window.rect().contains(tabsGeometry), qPrintable(QString("tab %1 outside %2×%3").arg(i).arg(size.width()).arg(size.height())));
                if (i == 0) {
                    for (const char *debug : {"evaluationThresholdPage", "evaluationDecisionSplit", "evaluationChartDecisionSplit", "evaluationDecisionSidebar", "evaluationThresholdTable", "evaluationApplyThresholdButton"}) {
                        auto *w = control<QWidget>(*panel, debug); qInfo() << "decision_geometry" << debug << w->geometry() << w->minimumSizeHint() << w->minimumSize() << w->sizeHint();
                    }
                    auto *strategy = control<QComboBox>(*panel, "evaluationRecommendationStrategyCombo");
                    for (int mode = 0; mode < strategy->count(); ++mode) {
                        strategy->setCurrentIndex(mode); QTest::qWait(1);
                        auto *constraint = control<QDoubleSpinBox>(*panel, "evaluationRecommendationConstraintSpin");
                        if (mode) QVERIFY(fullyVisibleWithin(constraint, &window));
                        for (const char *name : {"evaluationThresholdCurve", "evaluationThresholdTable", "evaluationUseRecommendationButton"})
                            QVERIFY2(fullyVisibleWithin(control<QWidget>(*panel, name), &window), name);
                        auto *hint = control<QLabel>(*panel, "evaluationRecommendationHint");
                        QTRY_VERIFY(labelTextFits(hint)); QVERIFY(fullyVisibleWithin(hint, &window));
                        QVERIFY(!hint->geometry().intersects(control<QPushButton>(*panel, "evaluationUseRecommendationButton")->geometry()));
                        auto *strategySplit = control<QSplitter>(*panel, "evaluationDecisionSplit"); const auto strategySizes = strategySplit->sizes();
                        for (const QList<int> requested : {QList<int>{1, 10000}, QList<int>{10000, 1}}) {
                            strategySplit->setSizes(requested); QTest::qWait(1); QTRY_VERIFY(labelTextFits(hint));
                            QVERIFY(fullyVisibleWithin(hint, &window));
                            QVERIFY(!hint->geometry().intersects(control<QPushButton>(*panel, "evaluationUseRecommendationButton")->geometry()));
                        }
                        strategySplit->setSizes(strategySizes); QTest::qWait(1);
                        auto *apply = control<QPushButton>(*panel, "evaluationApplyThresholdButton");
                        QVERIFY(fullyVisibleWithin(apply, &window));
                    }
                    strategy->setCurrentIndex(0); QTest::qWait(1);
                    for (const char *name : {"evaluationThresholdCurve", "evaluationThresholdTable", "evaluationDecisionSidebar"})
                        QVERIFY2(fullyVisibleWithin(control<QWidget>(*panel, name), &window), name);
                    auto *splitter = control<QSplitter>(*panel, "evaluationDecisionSplit"); const auto originalSizes = splitter->sizes();
                    for (const QList<int> requested : {QList<int>{1, 10000}, QList<int>{10000, 1}}) {
                        splitter->setSizes(requested); QTest::qWait(1);
                        for (const char *name : {"evaluationThresholdCurve", "evaluationThresholdTable", "evaluationUseRecommendationButton", "evaluationApplyThresholdButton"})
                            QVERIFY2(fullyVisibleWithin(control<QWidget>(*panel, name), &window), name);
                    }
                    splitter->setSizes(originalSizes); QTest::qWait(1);
                    for (const char *name : {"evaluationApplyThresholdButton", "evaluationReviewErrorsButton", "evaluationUseRecommendationButton"}) {
                        const auto *action = control<QPushButton>(*panel, name); QVERIFY(action->isVisible());
                        QVERIFY2(fullyVisibleWithin(action, &window), name);
                    }
                }
                auto *pageScroll = qobject_cast<QScrollArea *>(tabs->widget(i)); QVERIFY(pageScroll);
                const auto pageImage = pageScroll->viewport()->grab().toImage(); QVERIFY(!pageImage.isNull());
                const auto background = pageImage.pixelColor(1, 1);
                QVERIFY2(background.red() < 100 && background.green() < 100 && background.blue() < 100, "tab page background must remain dark");
                if (i == 2) {
                    auto *sceneTable = control<QTableWidget>(*panel, "evaluationSceneMetricsTable");
                    const auto tableImage = sceneTable->viewport()->grab().toImage();
                    const auto tableBackground = tableImage.pixelColor(tableImage.width() - 4, tableImage.height() - 4);
                    QVERIFY2(tableBackground.red() < 100 && tableBackground.green() < 100 && tableBackground.blue() < 100, "scene table background must remain dark");
                }
                pageScroll->verticalScrollBar()->setValue(pageScroll->verticalScrollBar()->maximum()); QTest::qWait(1);
                const char *actionNames[]{"evaluationApplyThresholdButton", "evaluationPresenceMatrix", "evaluationReviewSceneButton", "evaluationImportExternalReportButton"};
                auto *bottomAction = control<QWidget>(*panel, actionNames[i]); QVERIFY(bottomAction);
                const QRect actionGeometry(bottomAction->mapTo(pageScroll->viewport(), QPoint()), bottomAction->size());
                QVERIFY2(pageScroll->viewport()->rect().contains(actionGeometry), "page bottom action must be reachable within its scroller");
                pageScroll->verticalScrollBar()->setValue(0);
            }
        }
        qInfo() << "diagnostic_layout_evidence" << "synthetic_fixture_only" << "sizes 1100x640 1100x720 1260x820 1366x768 1366x900 1600x1000; panel fixtures, not application minimum" << "tabs" << tabs->count();
    }
    void externalSnapshotImportNeedsNoModelAndCannotApplyUnknownProfile()
    {
        QTemporaryDir area; QVERIFY(area.isValid()); EvaluationPanel panel;
        auto *import = control<QPushButton>(panel, "evaluationImportExternalReportButton"); QVERIFY(import); QVERIFY(import->isEnabled());
        QVERIFY(!control<QPushButton>(panel, "evaluationReanalyzeButton")->isEnabled());
        QSignalSpy requested(&panel, &EvaluationPanel::externalReportImportRequested);
        auto choose = [&](bool cancel) {
            bool seen = false; QElapsedTimer elapsed; elapsed.start(); QTimer timer;
            connect(&timer, &QTimer::timeout, this, [&] {
                auto *dialog = qobject_cast<QFileDialog *>(QApplication::activeModalWidget()); if (!dialog) return;
                if (dialog->windowTitle() != "导入脚本评估报告目录" || elapsed.elapsed() > 5000) { dialog->reject(); return; }
                seen = true;
                if (cancel) dialog->reject(); else { dialog->setDirectory(area.path()); QMetaObject::invokeMethod(dialog, "accept", Qt::QueuedConnection); }
            });
            timer.start(10); import->click(); timer.stop(); return seen;
        };
        QVERIFY(choose(false)); QCOMPARE(requested.count(), 1); const auto event = requested.takeFirst();
        QCOMPARE(event[0].toString(), area.path()); QCOMPARE(event[1].toJsonObject(), panel.analysisOptions());
        QVERIFY(choose(true)); QCOMPARE(requested.count(), 0);
        panel.setBusy(true); QVERIFY(!import->isEnabled()); import->click(); QCOMPARE(requested.count(), 0); panel.setBusy(false); QVERIFY(import->isEnabled());
        auto external = report(); external.insert("allow_apply_profile", false); panel.setReport(external, area.path());
        QVERIFY(control<QLabel>(panel, "evaluationReportIdentity")->text().contains("颜色未记录"));
        auto *apply = control<QPushButton>(panel, "evaluationApplyThresholdButton"); QVERIFY(!apply->isEnabled()); QVERIFY(apply->toolTip().contains("配置"));
        QSignalSpy applied(&panel, &EvaluationPanel::applyRequested); apply->click(); QCOMPARE(applied.count(), 0);
        QVERIFY(QMetaObject::invokeMethod(apply, "clicked", Qt::DirectConnection)); QCOMPARE(applied.count(), 0);
        panel.setReport(report(), area.path()); QVERIFY(apply->isEnabled()); apply->click(); QCOMPARE(applied.count(), 1);
        auto native = report(); auto nativeRequest = native.value("request").toObject();
        nativeRequest.insert("config", QJsonObject{{"color_mode", "color"}}); native.insert("request", nativeRequest); panel.setReport(native, area.path());
        QVERIFY(control<QLabel>(panel, "evaluationReportIdentity")->text().contains("彩色"));
        nativeRequest.insert("config", QJsonObject{{"color_mode", "grayscale"}}); native.insert("request", nativeRequest); panel.setReport(native, area.path());
        QVERIFY(control<QLabel>(panel, "evaluationReportIdentity")->text().contains("灰度"));
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
