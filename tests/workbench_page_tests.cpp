#include "ui/pages/workbenchpage.h"
#include "ui/evaluationpanel.h"
#include "ui/imagecanvas.h"

#include <QCheckBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QFileInfo>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QPushButton>
#include <QSignalSpy>
#include <QSpinBox>
#include <QTabWidget>
#include <QTableWidget>
#include <QTemporaryDir>
#include <QTimer>
#include <QWheelEvent>
#include <QtTest>

class WorkbenchPageTests final : public QObject
{
    Q_OBJECT
  private slots:
    void widgetIdentityAndSubpagesRemainStable()
    {
        WorkbenchPage page("/not-created/results");
        const auto &c = page.controls();
        QCOMPARE(page.objectName(), QString("workbenchPage"));
        QCOMPARE(c.workbenchTabs->count(), 2);
        QCOMPARE(c.workbenchTabs->tabText(0), QString("图像预览"));
        QCOMPARE(c.workbenchTabs->tabText(1), QString("评估与阈值"));
        QCOMPARE(c.workbenchTabs->widget(0)->objectName(), QString("imagePreviewPage"));
        QCOMPARE(c.workbenchTabs->widget(1)->objectName(), QString("evaluationWorkspacePage"));
        QCOMPARE(c.inputSize->objectName(), QString("inputSize"));
        QCOMPARE(c.confidence->objectName(), QString("confidence"));
        QCOMPARE(c.iou->objectName(), QString("nmsIou"));
        QCOMPARE(c.queue->objectName(), QString("inputQueue"));
        QCOMPARE(c.evaluation->parentWidget(), c.workbenchTabs->widget(1));
        QCOMPARE(c.inputSize->value(), 640); QCOMPARE(c.confidence->value(), .25); QCOMPARE(c.iou->value(), .45);
        QCOMPARE(c.detectionRange->currentIndex(), 0); QCOMPARE(c.detectionRange->count(), 3);
        QVERIFY(c.lockedControls.contains(c.classFilter));
        QVERIFY(c.lockedControls.contains(c.inputSize));
        QVERIFY(c.lockedControls.contains(c.datasetButton));
        QCOMPARE(c.recordingClock->interval(), 250); QVERIFY(!c.recordingClock->isActive());
    }

    void rangeChangesRemainSynchronizedWithoutDuplicateIntents()
    {
        WorkbenchPage page({}); const auto &c = page.controls();
        QSignalSpy ranges(&page, &WorkbenchPage::rangeChanged);
        c.evaluationRange->setCurrentIndex(2);
        QCOMPARE(c.detectionRange->currentIndex(), 2); QCOMPARE(ranges.count(), 1);
        c.detectionRange->setCurrentIndex(1);
        QCOMPARE(c.evaluationRange->currentIndex(), 1); QCOMPARE(ranges.count(), 2);
    }

    void spinWheelCannotChangeInferenceParameters()
    {
        WorkbenchPage page({}); const auto &c = page.controls();
        QSignalSpy parameters(&page, &WorkbenchPage::parametersChanged);
        for (QWidget *control : QList<QWidget *>{c.inputSize, c.confidence, c.iou})
        {
            QWheelEvent wheel(QPointF(5, 5), QPointF(control->mapToGlobal(QPoint(5, 5))),
                QPoint(), QPoint(0, 120), Qt::NoButton, Qt::NoModifier, Qt::NoScrollPhase, false);
            QApplication::sendEvent(control, &wheel);
        }
        QCOMPARE(c.inputSize->value(), 640); QCOMPARE(c.confidence->value(), .25); QCOMPARE(c.iou->value(), .45);
        QCOMPARE(parameters.count(), 0);
        c.confidence->setValue(.6); QCOMPARE(parameters.count(), 1);
    }

    void actionsAreIntentsAndNeverCreateFiles()
    {
        QTemporaryDir directory; const auto output = directory.filePath("not-created");
        WorkbenchPage page(output); const auto &c = page.controls();
        QSignalSpy starts(&page, &WorkbenchPage::startRequested);
        QSignalSpy stops(&page, &WorkbenchPage::stopRequested);
        QSignalSpy open(&page, &WorkbenchPage::openExportDirectoryRequested);
        QSignalSpy clear(&page, &WorkbenchPage::clearQueueRequested);
        QSignalSpy video(&page, &WorkbenchPage::videoRequested);
        QSignalSpy camera(&page, &WorkbenchPage::cameraRequested);
        c.runButton->click(); c.evaluationStartButton->click();
        c.stopButton->click(); c.evaluationStopButton->click();
        c.openExportDirectory->click(); c.clearQueue->click(); c.videoButton->click(); c.cameraButton->click();
        QCOMPARE(starts.count(), 2); QCOMPARE(stops.count(), 2); QCOMPARE(open.count(), 1);
        QCOMPARE(clear.count(), 1); QCOMPARE(video.count(), 1); QCOMPARE(camera.count(), 1);
        QVERIFY(!QFileInfo::exists(output));
        QCOMPARE(c.openExportDirectory->toolTip(), output);
        page.setExportDirectory(directory.path());
        QCOMPARE(c.openExportDirectory->toolTip(), directory.path());
    }

    void classAndQueueSignalsCarryUserValues()
    {
        WorkbenchPage page({}); const auto &c = page.controls();
        QSignalSpy classes(&page, &WorkbenchPage::classFilterChanged);
        QSignalSpy selection(&page, &WorkbenchPage::queueSelectionChanged);
        QSignalSpy row(&page, &WorkbenchPage::queueRowChanged);
        c.classFilter->setText("0,2,32");
        QCOMPARE(classes.count(), 1); QCOMPARE(classes.first().first().toString(), QString("0,2,32"));
        c.queue->addItems({"a", "b"}); c.queue->setCurrentRow(1);
        QVERIFY(!selection.isEmpty()); QVERIFY(!row.isEmpty());
        QCOMPARE(selection.last()[0].toInt(), 1); QVERIFY(selection.last()[1].toBool());
        QCOMPARE(row.last()[0].toInt(), 1);
    }

    void reviewControlsAndNavigationRemainLocal()
    {
        WorkbenchPage page({}); const auto &c = page.controls();
        QSignalSpy filters(&page, &WorkbenchPage::reviewFilterRequested);
        QSignalSpy navigate(&page, &WorkbenchPage::reviewNavigateRequested);
        QSignalSpy tabs(&page, &WorkbenchPage::workbenchTabChanged);
        c.evaluationPreviewFilter->setCurrentIndex(c.evaluationPreviewFilter->findData("errors"));
        QCOMPARE(filters.count(), 1); QCOMPARE(filters.first().first().toString(), QString("errors"));
        c.evaluationReviewNextButton->click(); c.evaluationReviewPreviousButton->click();
        QCOMPARE(navigate.count(), 2); QCOMPARE(navigate[0][0].toInt(), 1); QCOMPARE(navigate[1][0].toInt(), -1);
        page.findChild<QPushButton *>("evaluationReviewAnalysisButton")->click();
        QCOMPARE(c.workbenchTabs->currentIndex(), 1);
        page.findChild<QPushButton *>("evaluationConfigureButton")->click();
        QCOMPARE(c.workbenchTabs->currentIndex(), 0); QCOMPARE(tabs.count(), 2);
    }

    void canvasVisibilityAndTableSelectionStayWithinPage()
    {
        WorkbenchPage page({}); const auto &c = page.controls();
        vision::InferenceResult result;
        result.image = QImage(100, 100, QImage::Format_RGB888); result.image.fill(Qt::black);
        result.predictions = {{32, "sports ball", .8f, QRectF(20, 30, 30, 30)}};
        c.canvas->setResult(result);
        const auto annotated = c.canvas->annotatedImage();
        QVERIFY(annotated != result.image.convertToFormat(QImage::Format_ARGB32_Premultiplied));
        c.showBoxes->setChecked(false); c.showLabels->setChecked(false);
        QCOMPARE(c.canvas->annotatedImage(), result.image.convertToFormat(QImage::Format_ARGB32_Premultiplied));
        c.predictionTable->setRowCount(2);
        c.predictionTable->setItem(0, 0, new QTableWidgetItem("first"));
        c.predictionTable->setItem(1, 0, new QTableWidgetItem("second"));
        c.canvas->predictionSelected(1); QCOMPARE(c.predictionTable->currentRow(), 1);
        c.canvas->predictionSelected(-1); QVERIFY(c.predictionTable->selectedItems().isEmpty());
    }
};
QTEST_MAIN(WorkbenchPageTests)
#include "workbench_page_tests.moc"
