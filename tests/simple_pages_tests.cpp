// SPDX-License-Identifier: AGPL-3.0-only
#include "ui/pages/morepage.h"
#include "ui/pages/guidepage.h"
#include "ui/pages/historypage.h"
#include <QCheckBox>
#include <QJsonArray>
#include <QJsonObject>
#include <QLabel>
#include <QPlainTextEdit>
#include <QProgressBar>
#include <QPushButton>
#include <QScrollArea>
#include <QTableWidget>
#include <QTextBrowser>
#include <QtTest>

class SimplePagesTests final : public QObject
{
    Q_OBJECT
private slots:
    void moreForwardsActionsAndMirrorsPreferenceWithoutWritingBack()
    {
        MorePage page(true, false);
        const auto &controls = page.controls();
        QSignalSpy restore(&page, &MorePage::restoreDatasetOnStartupChanged);
        QSignalSpy check(&page, &MorePage::gpuCheckRequested);
        QSignalSpy prepare(&page, &MorePage::gpuPrepareRequested);
        QSignalSpy cancel(&page, &MorePage::gpuCancelRequested);
        QSignalSpy exports(&page, &MorePage::exportRequested);
        QVERIFY(controls.restoreDataset->isChecked());
        page.setRestoreDatasetOnStartup(false);
        QVERIFY(!controls.restoreDataset->isChecked());
        QCOMPARE(restore.count(), 0);
        controls.restoreDataset->click();
        QCOMPARE(restore.count(), 1);
        QCOMPARE(restore.front().front().toBool(), true);
        controls.gpuCheckButton->click();
        controls.gpuPrepareButton->click();
        controls.gpuCancelButton->click();
        QCOMPARE(check.count(), 1);
        QCOMPARE(prepare.count(), 1);
        QCOMPARE(cancel.count(), 1);
        controls.exportButton->click();
        QCOMPARE(exports.count(), 0);
        controls.exportButton->setEnabled(true);
        controls.exportButton->click();
        QCOMPARE(exports.count(), 1);
    }
    void moreRetainsInitialVisibilityAndStableWidgetNames()
    {
        MorePage page(false, false);
        const auto &controls = page.controls();
        QCOMPARE(page.objectName(), QString("morePage"));
        QVERIFY(page.findChild<QScrollArea *>("gpuMoreScroll"));
        QCOMPARE(controls.restoreDataset->objectName(), QString("restoreDatasetOnStartup"));
        QCOMPARE(controls.gpuEnvironmentStatus->objectName(), QString("gpuEnvironmentStatus"));
        QCOMPARE(controls.gpuEnvironmentStatus->text(), QString("正在检查 GPU 环境…"));
        QVERIFY(controls.gpuCancelButton->isHidden());
        QVERIFY(controls.gpuSetupProgress->isHidden());
        QVERIFY(controls.gpuLog->isHidden());
        QVERIFY(controls.gpuLog->isReadOnly());
        QCOMPARE(controls.gpuLog->maximumBlockCount(), 600);
        QCOMPARE(controls.exportButton->objectName(), QString("exportResultButton"));
        QVERIFY(!controls.exportButton->isEnabled());
#ifndef Q_OS_WIN
        QVERIFY(!controls.cpuPrepareButton);
#else
        QVERIFY(controls.cpuPrepareButton);
        QVERIFY(controls.cpuPrepareButton->isHidden());
#endif
    }
    void historyRendersOldAndCurrentRecordsAndForwardsActions()
    {
        HistoryPage page("/isolated/history.json");
        const auto &controls = page.controls();
        QCOMPARE(controls.hint->toolTip(), QString("/isolated/history.json"));
        QJsonObject detection{{"time", "2026-10-29T12:34:56.000+08:00"}, {"source", "/images/frame.png"},
            {"model", "model.pt"}, {"stereo_view", "left"}, {"objects", 3}, {"inference_ms", 12.36},
            {"task", "YOLOv8 / 11"}, {"actual_device", "cuda"}, {"backend", "PT"},
            {"device_name", "Isolated GPU"}, {"device_notice", "notice"}};
        const QJsonObject old{{"source", "/images/old.png"}, {"model", "old.onnx"}, {"stereo_view", "right"}};
        const QJsonArray records{detection, old};
        page.setRecords(records);
        QCOMPARE(controls.table->rowCount(), 2);
        QCOMPARE(controls.table->columnCount(), 6);
        QCOMPARE(controls.table->item(0, 0)->text(), QString("10-29  12:34:56"));
        QCOMPARE(controls.table->item(0, 1)->text(), QString("frame.png · 双目左目"));
        QCOMPARE(controls.table->item(0, 1)->toolTip(), QString("/images/frame.png"));
        QCOMPARE(controls.table->item(0, 2)->text(), QString("model.pt · GPU"));
        QCOMPARE(controls.table->item(0, 2)->toolTip(), QString("model.pt · GPU\nPT\nIsolated GPU\nnotice"));
        QCOMPARE(controls.table->item(0, 3)->text(), QString("3"));
        QCOMPARE(controls.table->item(0, 4)->text(), QString("12.4 ms"));
        QCOMPARE(controls.table->item(0, 0)->data(HistoryPage::RowData).toJsonObject(), detection);
        QCOMPARE(controls.table->item(1, 1)->text(), QString("old.png · 双目右目"));
        QCOMPARE(controls.table->item(1, 2)->text(), QString("old.onnx"));
        QSignalSpy exports(&page, &HistoryPage::exportRequested);
        QSignalSpy activated(&page, &HistoryPage::activated);
        controls.exportButton->click();
        controls.table->cellDoubleClicked(1, 2);
        QCOMPARE(exports.count(), 1);
        QCOMPARE(activated.count(), 1);
        QCOMPARE(activated.front().front().toInt(), 1);
        page.setRecords({old});
        QCOMPARE(controls.table->rowCount(), 1);
        QCOMPARE(controls.table->item(0, 0)->data(HistoryPage::RowData).toJsonObject(), old);
        page.setRecords({});
        QCOMPARE(controls.table->rowCount(), 0);
    }
    void guideLoadsBundledResourceWithExternalLinks()
    {
        GuidePage page;
        QCOMPARE(page.objectName(), QString("guidePage"));
        QCOMPARE(page.controls().browser->objectName(), QString("guideBrowser"));
        QVERIFY(page.controls().browser->openExternalLinks());
        QVERIFY(page.controls().browser->toPlainText().contains("Vision Studio"));
        QVERIFY(!page.controls().browser->toPlainText().contains("使用指南资源无法读取"));
    }
};

QTEST_MAIN(SimplePagesTests)
#include "simple_pages_tests.moc"
