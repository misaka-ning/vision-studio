// SPDX-License-Identifier: AGPL-3.0-only
#include "ui/pages/modelslibrarypage.h"
#include "ui/pages/modeldisplaypage.h"
#include "ui/pages/recordingspage.h"
#include "ui/modelviewer.h"
#include "ui/imagecanvas.h"
#include "app/modelsessioncontroller.h"
#include "app/operationcoordinator.h"
#include <QAction>
#include <QContextMenuEvent>
#include <QDialogButtonBox>
#include <QInputDialog>
#include <QLabel>
#include <QMenu>
#include <QPointer>
#include <QProcess>
#include <QPushButton>
#include <QSignalSpy>
#include <QTableWidget>
#include <QTimer>
#include <QtTest>
#include <functional>

class WorkspacePagesTests final : public QObject
{
    Q_OBJECT
    const QString selectedPath_ = QStringLiteral("/models/selected.onnx");
    const QString targetPath_ = QStringLiteral("/models/pointed.pt");

    void initializeModelContext(ModelsLibraryPage &page, vision::app::ModelSessionController &models,
                                vision::app::OperationCoordinator &operations,
                                vision::app::ActivitySnapshot &activity)
    {
        vision::app::ModelSessionSnapshot snapshot;
        snapshot.models = {selectedPath_, targetPath_}; snapshot.activeModel = selectedPath_;
        snapshot.displayNames.insert(targetPath_, "Stored target name");
        snapshot.notes.insert(targetPath_, "Stored note\nfrom authority");
        models.initialize(snapshot);
        operations.setActivityProvider([&activity] { return activity; });
        page.setApplicationContext(&models, &operations);
        // Deliberately stale row text must never become an editor's default.
        page.setRows({{selectedPath_, "Selected row", "ONNX", {}, {}, {}, 94},
                      {targetPath_, "Stale row name", "PT", "Stale row note", {}, {}, 94}}, selectedPath_);
        page.resize(1100, 800); page.show();
    }

    QMenu *openModelContextMenu(ModelsLibraryPage &page)
    {
        auto *list = page.controls().list;
        const QPoint point = list->visualItemRect(list->item(1)).center();
        QTest::mouseClick(list->viewport(), Qt::RightButton, Qt::NoModifier, point);
        QContextMenuEvent event(QContextMenuEvent::Mouse, point, list->viewport()->mapToGlobal(point));
        QApplication::sendEvent(list->viewport(), &event);
        const auto menus = page.findChildren<QMenu *>("modelContextMenu");
        for (auto *menu : menus) if (menu->isVisible()) return menu;
        return nullptr;
    }

    struct DialogObservation
    {
        bool shown = false, buttonAvailable = false;
        QString objectName, initialText, label;
    };
    DialogObservation editThroughMenu(ModelsLibraryPage &page, QMenu *menu, QAction *action,
                                     const QString &value, const QString &completion,
                                     std::function<void()> nestedChange = {})
    {
        DialogObservation observed;
        const QPointer<QMenu> menuGuard(menu);
        QTimer editor;
        editor.setInterval(1);
        connect(&editor, &QTimer::timeout, &page, [&] {
            auto *dialog = qobject_cast<QInputDialog *>(QApplication::activeModalWidget());
            if (!dialog) return;
            editor.stop();
            observed.shown = true; observed.objectName = dialog->objectName();
            observed.initialText = dialog->textValue(); observed.label = dialog->labelText();
            dialog->setTextValue(value);
            if (nestedChange) nestedChange();
            if (completion == "escape") { QTest::keyClick(dialog, Qt::Key_Escape); return; }
            auto *buttons = dialog->findChild<QDialogButtonBox *>();
            auto *button = buttons ? buttons->button(completion == "cancel" ? QDialogButtonBox::Cancel : QDialogButtonBox::Ok) : nullptr;
            observed.buttonAvailable = button != nullptr;
            if (button) QTest::mouseClick(button, Qt::LeftButton);
            else dialog->reject();
        });
        QTimer watchdog;
        watchdog.setSingleShot(true);
        connect(&watchdog, &QTimer::timeout, &page, [&] {
            editor.stop();
            if (auto *dialog = qobject_cast<QDialog *>(QApplication::activeModalWidget())) dialog->reject();
            else if (menuGuard) menuGuard->close();
        });
        editor.start(); watchdog.start(2000);
        QTest::mouseClick(menu, Qt::LeftButton, Qt::NoModifier, menu->actionGeometry(action).center());
        editor.stop(); watchdog.stop();
        return observed;
    }

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

    void modelContextMenuTargetsPointerWithoutChangingGlobalSelection()
    {
        vision::app::ActivitySnapshot activity;
        vision::app::ModelSessionController models;
        vision::app::OperationCoordinator operations;
        ModelsLibraryPage page;
        initializeModelContext(page, models, operations, activity);
        auto *selectedItem = page.controls().list->currentItem();
        QSignalSpy selection(&page, &ModelsLibraryPage::selectionRequested);
        QSignalSpy remove(&page, &ModelsLibraryPage::removeRequested);
        QSignalSpy changed(&models, &vision::app::ModelSessionController::changed);
        auto *menu = openModelContextMenu(page); QVERIFY(menu);
        const QStringList names{"modelOpenLocationAction", "modelNoteAction", "modelRenameAction", "modelRemoveAction"};
        QCOMPARE(menu->actions().size(), names.size());
        for (int index = 0; index < names.size(); ++index) {
            QCOMPARE(menu->actions()[index]->objectName(), names[index]);
            QCOMPARE(menu->actions()[index]->data().toString(), targetPath_);
            QVERIFY(menu->actions()[index]->isEnabled());
        }
        auto *action = menu->findChild<QAction *>("modelRemoveAction"); QVERIFY(action);
        QTest::mouseClick(menu, Qt::LeftButton, Qt::NoModifier, menu->actionGeometry(action).center());
        QCOMPARE(remove.size(), 1); QCOMPARE(remove.front().front().toString(), targetPath_);
        QCOMPARE(page.controls().list->currentItem(), selectedItem);
        QCOMPARE(models.snapshot().activeModel, selectedPath_);
        QVERIFY(models.snapshot().models.contains(targetPath_));
        QCOMPARE(selection.size(), 0); QCOMPARE(changed.size(), 0);
    }

    void modelEditorsUseAuthorityDefaultsAndOnlyEmitValues_data()
    {
        QTest::addColumn<QString>("kind");
        QTest::newRow("rename") << QString("rename");
        QTest::newRow("note") << QString("note");
    }
    void modelEditorsUseAuthorityDefaultsAndOnlyEmitValues()
    {
        QFETCH(QString, kind);
        vision::app::ActivitySnapshot activity;
        vision::app::ModelSessionController models;
        vision::app::OperationCoordinator operations;
        ModelsLibraryPage page;
        initializeModelContext(page, models, operations, activity);
        const auto before = models.snapshot();
        auto *selectedItem = page.controls().list->currentItem();
        QSignalSpy rename(&page, &ModelsLibraryPage::renameRequested), note(&page, &ModelsLibraryPage::noteRequested);
        QSignalSpy selection(&page, &ModelsLibraryPage::selectionRequested);
        QSignalSpy changed(&models, &vision::app::ModelSessionController::changed);
        auto *menu = openModelContextMenu(page); QVERIFY(menu);
        const bool renaming = kind == "rename";
        auto *action = menu->findChild<QAction *>(renaming ? "modelRenameAction" : "modelNoteAction"); QVERIFY(action && action->isEnabled());
        const auto observed = editThroughMenu(page, menu, action,
            renaming ? QString("  New   display  name  ") : QString(" \n Line 1\nLine 2 \n "), "accept");
        QVERIFY(observed.shown); QVERIFY(observed.buttonAvailable);
        QCOMPARE(observed.objectName, renaming ? QString("modelRenameDialog") : QString("modelNoteDialog"));
        QCOMPARE(observed.initialText, renaming ? before.displayNames.value(targetPath_).toString() : before.notes.value(targetPath_).toString());
        if (!renaming) QVERIFY(observed.label.startsWith("Stored target name\n"));
        auto &intent = renaming ? rename : note;
        QCOMPARE(intent.size(), 1); QCOMPARE(intent.front().at(0).toString(), targetPath_);
        QCOMPARE(intent.front().at(1).toString(), renaming ? QString("New display name") : QString("Line 1\nLine 2"));
        QCOMPARE(renaming ? note.size() : rename.size(), 0);
        QCOMPARE(models.snapshot().displayNames, before.displayNames);
        QCOMPARE(models.snapshot().notes, before.notes);
        QCOMPARE(models.snapshot().models, before.models);
        QCOMPARE(models.snapshot().activeModel, selectedPath_);
        QCOMPARE(page.controls().list->currentItem(), selectedItem);
        QCOMPARE(selection.size(), 0); QCOMPARE(changed.size(), 0);
    }

    void modelEditorsRecheckAuthorityAfterNestedDialog_data()
    {
        QTest::addColumn<QString>("kind"); QTest::addColumn<bool>("removeTarget");
        for (const auto &kind : {QString("rename"), QString("note")}) {
            QTest::newRow(qPrintable(kind + "-target-removed")) << kind << true;
            QTest::newRow(qPrintable(kind + "-operation-busy")) << kind << false;
        }
    }
    void modelEditorsRecheckAuthorityAfterNestedDialog()
    {
        QFETCH(QString, kind); QFETCH(bool, removeTarget);
        vision::app::ActivitySnapshot activity;
        vision::app::ModelSessionController models;
        vision::app::OperationCoordinator operations;
        ModelsLibraryPage page;
        initializeModelContext(page, models, operations, activity);
        const auto before = models.snapshot();
        QSignalSpy rename(&page, &ModelsLibraryPage::renameRequested), note(&page, &ModelsLibraryPage::noteRequested);
        QSignalSpy selection(&page, &ModelsLibraryPage::selectionRequested);
        QSignalSpy changed(&models, &vision::app::ModelSessionController::changed);
        auto *menu = openModelContextMenu(page); QVERIFY(menu);
        auto *action = menu->findChild<QAction *>(kind == "rename" ? "modelRenameAction" : "modelNoteAction"); QVERIFY(action && action->isEnabled());
        bool nestedMutationApplied = false;
        const auto observed = editThroughMenu(page, menu, action, "Must not be submitted", "accept", [&] {
            if (removeTarget) nestedMutationApplied = models.remove(targetPath_);
            else {
                activity.saving = true;
                nestedMutationApplied = !vision::app::OperationCoordinator::availability(operations.activity()).importModels;
            }
        });
        QVERIFY(observed.shown); QVERIFY(observed.buttonAvailable); QVERIFY(nestedMutationApplied);
        QCOMPARE(rename.size(), 0); QCOMPARE(note.size(), 0);
        QCOMPARE(models.snapshot().displayNames, before.displayNames);
        QCOMPARE(models.snapshot().notes, before.notes);
        QCOMPARE(models.snapshot().models.contains(targetPath_), !removeTarget);
        QCOMPARE(models.snapshot().activeModel, selectedPath_);
        QCOMPARE(page.controls().list->currentItem()->data(Qt::UserRole).toString(), selectedPath_);
        QCOMPARE(selection.size(), 0); QCOMPARE(changed.size(), removeTarget ? 1 : 0);
    }

    void modelEditorEscapeAndCancelDiscardEdits_data()
    {
        QTest::addColumn<QString>("kind"); QTest::addColumn<QString>("completion");
        for (const auto &kind : {QString("rename"), QString("note")}) {
            QTest::newRow(qPrintable(kind + "-escape")) << kind << QString("escape");
            QTest::newRow(qPrintable(kind + "-cancel")) << kind << QString("cancel");
        }
    }
    void modelEditorEscapeAndCancelDiscardEdits()
    {
        QFETCH(QString, kind); QFETCH(QString, completion);
        vision::app::ActivitySnapshot activity;
        vision::app::ModelSessionController models;
        vision::app::OperationCoordinator operations;
        ModelsLibraryPage page;
        initializeModelContext(page, models, operations, activity);
        const auto before = models.snapshot();
        QSignalSpy rename(&page, &ModelsLibraryPage::renameRequested), note(&page, &ModelsLibraryPage::noteRequested);
        QSignalSpy changed(&models, &vision::app::ModelSessionController::changed);
        auto *menu = openModelContextMenu(page); QVERIFY(menu);
        auto *action = menu->findChild<QAction *>(kind == "rename" ? "modelRenameAction" : "modelNoteAction"); QVERIFY(action && action->isEnabled());
        const auto observed = editThroughMenu(page, menu, action, "Discard this edit", completion);
        QVERIFY(observed.shown);
        if (completion == "cancel") QVERIFY(observed.buttonAvailable);
        QCOMPARE(rename.size(), 0); QCOMPARE(note.size(), 0); QCOMPARE(changed.size(), 0);
        QCOMPARE(models.snapshot().displayNames, before.displayNames);
        QCOMPARE(models.snapshot().notes, before.notes);
        QCOMPARE(models.snapshot().models, before.models);
        QCOMPARE(models.snapshot().activeModel, selectedPath_);
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

    void modelDisplayDestructionDisconnectsViewerBeforeHeaderChildren()
    {
        // An invalid path reaches Error without creating a helper. The viewer
        // destructor still calls stop(), which changes Error to Empty after
        // the earlier-created header widgets have already been deleted.
        // Exercise that actual destruction order, not a prior stop() that
        // would leave Empty and hide a stale page callback.
        for (int iteration = 0; iteration < 32; ++iteration) {
            QObject observer;
            auto *page = new ModelDisplayPage;
            const QPointer<QLabel> status(page->controls().status);
            const QPointer<ModelViewer> viewer(page->controls().viewer);
            bool headerDestroyed = false;
            int finalStateAfterHeader = 0;
            connect(status.data(), &QObject::destroyed, &observer, [&] { headerDestroyed = true; });
            connect(viewer.data(), &ModelViewer::stateChanged, &observer, [&](ModelViewer::State state) {
                if (headerDestroyed && state == ModelViewer::State::Empty) ++finalStateAfterHeader;
            });
            viewer->openModel("/path/that/does/not/exist.pt");
            QCOMPARE(viewer->state(), ModelViewer::State::Error);
            QVERIFY(viewer->findChildren<QProcess *>().isEmpty());
            delete page;
            QVERIFY(headerDestroyed); QVERIFY(status.isNull()); QVERIFY(viewer.isNull());
            QCOMPARE(finalStateAfterHeader, 1);
        }
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
