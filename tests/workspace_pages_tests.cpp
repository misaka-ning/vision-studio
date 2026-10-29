// SPDX-License-Identifier: AGPL-3.0-only
#include "ui/pages/modelslibrarypage.h"
#include "ui/pages/modeldisplaypage.h"
#include "ui/pages/recordingspage.h"
#include "ui/modelviewer.h"
#include "ui/imagecanvas.h"
#include <QContextMenuEvent>
#include <QLabel>
#include <QProcess>
#include <QPushButton>
#include <QSignalSpy>
#include <QTableWidget>
#include <QtTest>

class WorkspacePagesTests final : public QObject
{
    Q_OBJECT
  private slots:
    void modelLibraryOwnsItsPresentationAndEmitsActions()
    {
        ModelsLibraryPage page; page.resize(1100, 800); page.show();
        const auto &c = page.controls();
        QCOMPARE(page.objectName(), QString("modelLibraryPage"));
        QCOMPARE(c.list->objectName(), QString("globalModelList"));
        QCOMPARE(c.list->dragDropMode(), QAbstractItemView::InternalMove);
        QSignalSpy selection(&page, &ModelsLibraryPage::selectionRequested);
        QSignalSpy order(&page, &ModelsLibraryPage::orderChanged);
        QSignalSpy imports(&page, &ModelsLibraryPage::importRequested);
        QSignalSpy remove(&page, &ModelsLibraryPage::removeRequested);
        const QVector<ModelLibraryRow> rows = {
            {"/models/a.onnx", "模型A", "1.2 MB  ·  ONNX", "a note", "评估 · val · 完整集合", "original A", 143},
            {"/models/b.pt", "模型B", "2.0 MB  ·  PT", {}, {}, "original B", 94},
            {"/models/c.pt", "模型C", "文件已移动或不存在", {}, {}, "original C", 94}};
        page.setRows(rows, "/models/b.pt");
        QCOMPARE(selection.size(), 0); QCOMPARE(c.list->count(), 3);
        QVERIFY(c.list->item(0)->text().contains("备注 · a note"));
        QVERIFY(c.list->item(0)->text().contains("评估 · val · 完整集合"));
        QCOMPARE(c.list->currentItem()->data(Qt::UserRole).toString(), QString("/models/b.pt"));
        QVERIFY(c.list->model()->moveRows({}, 0, 1, {}, 3));
        QCOMPARE(order.size(), 1);
        QCOMPARE(order[0][0].toStringList(), QStringList({"/models/b.pt", "/models/c.pt", "/models/a.onnx"}));
        QCOMPARE(selection.size(), 0);
        QCOMPARE(c.list->currentItem()->data(Qt::UserRole).toString(), QString("/models/b.pt"));
        QTest::mouseClick(c.import, Qt::LeftButton); QCOMPARE(imports.size(), 1);
        QTest::mouseClick(c.remove, Qt::LeftButton); QCOMPARE(remove.size(), 1);
        QCOMPARE(remove[0][0].toString(), QString("/models/b.pt"));
        page.setDropHint("invalid mixed batch", true);
        QCOMPARE(c.dropHint->text(), QString("invalid mixed batch")); QVERIFY(c.dropHint->styleSheet().contains("FFB4BB"));
    }

    void modelLibraryBusyAndContextClickDoNotChangeGlobalSelection()
    {
        ModelsLibraryPage page; page.resize(1100, 800); page.show();
        page.setRows({{"a", "A", "ONNX", {}, {}, {}, 94}, {"b", "B", "PT", {}, {}, {}, 94}}, "a");
        const auto &c = page.controls(); QSignalSpy selection(&page, &ModelsLibraryPage::selectionRequested);
        QSignalSpy context(&page, &ModelsLibraryPage::contextMenuRequested);
        const QPoint second = c.list->visualItemRect(c.list->item(1)).center();
        QTest::mouseClick(c.list->viewport(), Qt::RightButton, Qt::NoModifier, second);
        QCOMPARE(c.list->currentRow(), 0); QCOMPARE(selection.size(), 0);
        QContextMenuEvent menu(QContextMenuEvent::Mouse, second, c.list->viewport()->mapToGlobal(second));
        QApplication::sendEvent(c.list->viewport(), &menu); QCOMPARE(context.size(), 1);
        page.setBusy(true); QVERIFY(!c.list->dragEnabled()); QVERIFY(!c.list->acceptDrops());
        QVERIFY(!c.import->isEnabled()); QVERIFY(!c.activate->isEnabled()); QVERIFY(!c.remove->isEnabled());
        QTest::mouseClick(c.list->viewport(), Qt::LeftButton, Qt::NoModifier, second);
        QTest::keyClick(c.list, Qt::Key_Down); QCOMPARE(c.list->currentRow(), 0); QCOMPARE(selection.size(), 0);
        page.setBusy(false); QTest::mouseClick(c.list->viewport(), Qt::LeftButton, Qt::NoModifier, second);
        QCOMPARE(selection.size(), 1); QCOMPARE(selection[0][0].toString(), QString("b"));
    }

    void modelDisplayHeaderDoesNotOwnModelSelectionOrStartParsing()
    {
        ModelDisplayPage page; page.show(); const auto &c = page.controls();
        page.setSelectedModelHeader("/fixtures/model.onnx", "全局名称");
        QCOMPARE(c.modelName->text(), QString("全局名称")); QVERIFY(c.hint->isHidden());
        QVERIFY(c.viewer->findChildren<QProcess *>().isEmpty());
        page.setSelectedModelHeader("/fixtures/model.pt", "PT 名称", true);
        QVERIFY(!c.hint->isHidden()); QVERIFY(c.metadata->text().contains("首次查看"));
        QVERIFY(c.viewer->findChildren<QProcess *>().isEmpty());
        QSignalSpy failed(&page, &ModelDisplayPage::loadFailed);
        c.viewer->openModel("/path/that/does/not/exist.pt");
        QCOMPARE(failed.size(), 1); QCOMPARE(c.status->text(), QString("无法显示"));
        QSignalSpy shutdown(c.viewer, &ModelViewer::shutdownReady); c.viewer->requestShutdown();
        QTRY_COMPARE_WITH_TIMEOUT(shutdown.size(), 1, 1000);
    }

    void recordingPageRendersValuesWithoutTouchingFiles()
    {
        RecordingsPage page("/no/real/recordings"); page.resize(1100, 800); page.show();
        const auto &c = page.controls(); QSignalSpy selected(&page, &RecordingsPage::selectionRequested);
        QVector<vision::app::RecordingEntry> rows{
            {"/recordings/a.avi", "time\na.avi", "1.0 s", "64×48", "0.1 MB", {}},
            {"/recordings/b.avi", "time\nb.avi", "2.0 s", "80×60", "0.2 MB", {}}};
        page.setRecords(rows, rows[1].path);
        QCOMPARE(c.table->currentRow(), 1); QCOMPARE(selected.size(), 0);
        QCOMPARE(c.table->item(1, 0)->data(Qt::UserRole).toString(), rows[1].path);
        QCOMPARE(c.count->text(), QString("本地录像 · 2 个文件"));
        c.table->setCurrentCell(0, 0); QCOMPARE(selected.size(), 1); QCOMPARE(selected[0][0].toString(), rows[0].path);
        vision::app::RecordingPlaybackState state; state.path = rows[0].path;
        state.canPlay = state.canExport = state.playing = true; state.details = "左目 · 灰度 · 25 fps";
        page.setPlayback(state); QVERIFY(c.play->isEnabled()); QCOMPARE(c.play->text(), QString("暂停播放"));
        QVERIFY(c.exportVideo->isEnabled());
        QImage image(64, 48, QImage::Format_RGB888); image.fill(Qt::red); page.showFrame(image, state.path);
        QCOMPARE(c.canvas->result().image, image); QCOMPARE(c.canvas->result().source, state.path);
        state.exporting = true; page.setPlayback(state); QVERIFY(!c.exportVideo->isEnabled());
        state.canPlay = false; page.setPlayback(state); QVERIFY(c.canvas->result().image.isNull());
    }
};
QTEST_MAIN(WorkspacePagesTests)
#include "workspace_pages_tests.moc"
