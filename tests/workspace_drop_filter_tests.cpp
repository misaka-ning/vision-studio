// SPDX-License-Identifier: AGPL-3.0-only
#include "ui/pages/workspacedropfilter.h"
#include "ui/imagecanvas.h"
#include "app/modelsessioncontroller.h"
#include "app/operationcoordinator.h"
#include <QApplication>
#include <QDir>
#include <QDragEnterEvent>
#include <QDragLeaveEvent>
#include <QDragMoveEvent>
#include <QDropEvent>
#include <QFile>
#include <QImage>
#include <QLabel>
#include <QListWidget>
#include <QMimeData>
#include <QSignalSpy>
#include <QTemporaryDir>
#include <QTest>
#include <QUrl>

namespace
{
bool writeFile(const QString &path)
{
    QFile file(path);
    return file.open(QIODevice::WriteOnly) && file.write("fixture\n") == 8;
}
QList<QUrl> localUrls(const QStringList &paths)
{
    QList<QUrl> result;
    for (const auto &path : paths) result.append(QUrl::fromLocalFile(path));
    return result;
}
bool enter(QWidget *target, const QMimeData &mime, Qt::DropActions actions = Qt::CopyAction)
{
    QDragEnterEvent event(QPoint(4, 4), actions, &mime, Qt::LeftButton, Qt::NoModifier);
    QApplication::sendEvent(target, &event);
    return event.isAccepted();
}
bool drop(QWidget *target, const QMimeData &mime, Qt::DropActions actions = Qt::CopyAction)
{
    if (!enter(target, mime, actions)) return false;
    QDragMoveEvent move(QPoint(4, 4), actions, &mime, Qt::LeftButton, Qt::NoModifier);
    QApplication::sendEvent(target, &move);
    if (!move.isAccepted()) return false;
    QDropEvent event(QPointF(4, 4), actions, &mime, Qt::LeftButton, Qt::NoModifier);
    QApplication::sendEvent(target, &event);
    return event.isAccepted() && event.dropAction() == Qt::CopyAction;
}
}

class WorkspaceDropFilterTests : public QObject
{
    Q_OBJECT
  private slots:
    void orderedCanonicalModelKeysUseCopyAndSynchronousAcknowledgement()
    {
        QTemporaryDir files; QVERIFY(files.isValid());
        const QString first = files.filePath("first.onnx"), second = files.filePath("second.pt");
        QVERIFY(writeFile(first)); QVERIFY(writeFile(second)); QVERIFY(QDir().mkpath(files.filePath("alias")));
        const QString oldKey = files.filePath("alias/../first.onnx");
        vision::app::ModelSessionSnapshot initial;
        initial.models = {oldKey}; initial.activeModel = oldKey;
        initial.displayNames.insert(oldKey, "original name"); initial.notes.insert(oldKey, "original note");
        vision::app::ModelSessionController models; models.initialize(initial);
        vision::app::OperationCoordinator operations;
        QWidget window, surface(&window); QLabel descendant(&surface);
        surface.setAcceptDrops(true); descendant.setAcceptDrops(true);
        WorkspaceDropFilter filter(WorkspaceDropFilter::Surface::ModelLibrary, &surface, nullptr,
            nullptr, &operations, &models, &surface);
        QStringList received; int requests = 0;
        connect(&filter, &WorkspaceDropFilter::modelDropRequested, &filter,
            [&](quint64 token, const QStringList &paths) { ++requests; received = paths; filter.resolveDrop(token, true); }, Qt::DirectConnection);
        QMimeData mime; mime.setUrls(localUrls({first, second, first}));
        QVERIFY(drop(&descendant, mime, Qt::CopyAction | Qt::MoveAction));
        QCOMPARE(requests, 1); QCOMPARE(received, QStringList({oldKey, second}));
        QCOMPARE(models.snapshot().models, initial.models);
        QCOMPARE(models.snapshot().displayNames, initial.displayNames);
        QCOMPARE(models.snapshot().notes, initial.notes);
    }

    void rejectedMixedRemoteMoveAndLinkBatchesEmitNoImport()
    {
        QTemporaryDir files; QVERIFY(files.isValid());
        const QString model = files.filePath("model.onnx"), text = files.filePath("invalid.txt");
        QVERIFY(writeFile(model)); QVERIFY(writeFile(text));
        vision::app::ModelSessionController models; vision::app::OperationCoordinator operations;
        QWidget window, surface(&window); surface.setAcceptDrops(true);
        WorkspaceDropFilter filter(WorkspaceDropFilter::Surface::ModelLibrary, &surface, nullptr,
            nullptr, &operations, &models, &surface);
        QSignalSpy intents(&filter, &WorkspaceDropFilter::modelDropRequested);
        QSignalSpy notices(&filter, &WorkspaceDropFilter::noticeRequested);
        QMimeData mixed; mixed.setUrls(localUrls({model, text}));
        QVERIFY(!drop(&surface, mixed));
        QMimeData remote; remote.setUrls({QUrl::fromLocalFile(model), QUrl("https://example.invalid/model.pt")});
        QVERIFY(!drop(&surface, remote));
        QMimeData local; local.setUrls(localUrls({model}));
        QVERIFY(!drop(&surface, local, Qt::MoveAction)); QVERIFY(!drop(&surface, local, Qt::LinkAction));
        QCOMPARE(intents.count(), 0); QCOMPARE(notices.count(), 4);
        QVERIFY(notices.last().at(1).toBool());
        QVERIFY(models.snapshot().models.isEmpty());
    }

    void dragLeaveRestoresOnlyActiveHoverAndPreservesRealError()
    {
        QTemporaryDir files; QVERIFY(files.isValid());
        const QString model = files.filePath("model.onnx"); QVERIFY(writeFile(model));
        vision::app::ModelSessionController models; vision::app::OperationCoordinator operations;
        QWidget window, surface(&window); surface.setAcceptDrops(true);
        WorkspaceDropFilter filter(WorkspaceDropFilter::Surface::ModelLibrary, &surface, nullptr,
            nullptr, &operations, &models, &surface);
        QString hint; bool error = false;
        connect(&filter, &WorkspaceDropFilter::modelHintChanged, &filter,
            [&](const QString &text, bool failed) { hint = text; error = failed; });
        QMimeData valid; valid.setUrls(localUrls({model}));
        QVERIFY(enter(&surface, valid)); QVERIFY(hint.startsWith("松开以导入")); QVERIFY(!error);
        QDragLeaveEvent leave; QApplication::sendEvent(&surface, &leave);
        QCOMPARE(hint, WorkspaceDropFilter::modelPrompt());
        QMimeData invalid; invalid.setUrls(localUrls({files.filePath("missing.pt")}));
        QVERIFY(!enter(&surface, invalid)); QVERIFY(error); const QString invalidHint = hint;
        QDragLeaveEvent failedLeave; QApplication::sendEvent(&surface, &failedLeave);
        QCOMPARE(hint, invalidHint); QVERIFY(error); QVERIFY(models.snapshot().models.isEmpty());
    }

    void permissionSnapshotBlocksEveryDomainActivityWithoutMutation()
    {
        QTemporaryDir files; QVERIFY(files.isValid());
        const QString model = files.filePath("model.onnx"); QVERIFY(writeFile(model));
        vision::app::ModelSessionController models; vision::app::OperationCoordinator operations;
        vision::app::ActivitySnapshot state;
        operations.setActivityProvider([&] { return state; });
        QWidget window, surface(&window); surface.setAcceptDrops(true);
        WorkspaceDropFilter filter(WorkspaceDropFilter::Surface::ModelLibrary, &surface, nullptr,
            nullptr, &operations, &models, &surface);
        QSignalSpy intents(&filter, &WorkspaceDropFilter::modelDropRequested);
        QMimeData mime; mime.setUrls(localUrls({model}));
        const QList<bool vision::app::ActivitySnapshot::*> fields{
            &vision::app::ActivitySnapshot::detection, &vision::app::ActivitySnapshot::input,
            &vision::app::ActivitySnapshot::projection, &vision::app::ActivitySnapshot::evaluation,
            &vision::app::ActivitySnapshot::conversion, &vision::app::ActivitySnapshot::gpuPreparation,
            &vision::app::ActivitySnapshot::saving, &vision::app::ActivitySnapshot::modal};
        for (auto field : fields) { state = {}; state.*field = true; QVERIFY(!drop(&surface, mime)); }
        state = {}; operations.requestShutdown(); QVERIFY(!drop(&surface, mime));
        QCOMPARE(intents.count(), 0); QVERIFY(models.snapshot().models.isEmpty());
    }

    void intentsRequireTheCurrentTokenAndPreserveCancelledDrop()
    {
        QTemporaryDir files; QVERIFY(files.isValid());
        const QString model = files.filePath("model.onnx"); QVERIFY(writeFile(model));
        vision::app::ModelSessionController models; vision::app::OperationCoordinator operations;
        QWidget window, surface(&window); surface.setAcceptDrops(true);
        WorkspaceDropFilter filter(WorkspaceDropFilter::Surface::ModelLibrary, &surface, nullptr,
            nullptr, &operations, &models, &surface);
        int requests = 0; quint64 previous = 0;
        connect(&filter, &WorkspaceDropFilter::modelDropRequested, &filter,
            [&](quint64 token, const QStringList &) {
            ++requests;
            filter.resolveDrop(previous, true); // a cancelled previous drop cannot acknowledge the new one
            if (requests == 2) filter.resolveDrop(token, true);
            previous = token;
        }, Qt::DirectConnection);
        QMimeData mime; mime.setUrls(localUrls({model}));
        QVERIFY(!drop(&surface, mime)); QVERIFY(drop(&surface, mime)); QVERIFY(!drop(&surface, mime));
        QCOMPARE(requests, 3); QVERIFY(models.snapshot().models.isEmpty());
    }

    void previewScopeHandlesRealTypesAndIgnoresOtherSurfacesAndWindows()
    {
        QTemporaryDir files; QVERIFY(files.isValid());
        const QString image = files.filePath("image.png"), video = files.filePath("video.mp4");
        QImage pixels(4, 3, QImage::Format_RGB32); pixels.fill(Qt::cyan); QVERIFY(pixels.save(image));
        QVERIFY(writeFile(video));
        vision::app::ModelSessionController models; vision::app::OperationCoordinator operations;
        QWidget window, surface(&window), unrelated(&window), secondWindow;
        ImageCanvas canvas(&surface);
        surface.setAcceptDrops(true); unrelated.setAcceptDrops(true); secondWindow.setAcceptDrops(true);
        WorkspaceDropFilter filter(WorkspaceDropFilter::Surface::Preview, &surface, &canvas,
            nullptr, &operations, &models, &surface);
        QStringList received; int requests = 0;
        connect(&filter, &WorkspaceDropFilter::previewDropRequested, &filter,
            [&](quint64 token, const QStringList &paths) { ++requests; received = paths; filter.resolveDrop(token, true); }, Qt::DirectConnection);
        QMimeData mime; mime.setUrls(localUrls({image}));
        QVERIFY(!drop(&unrelated, mime)); QVERIFY(!drop(&secondWindow, mime)); QCOMPARE(requests, 0);
        QVERIFY(drop(&canvas, mime)); QCOMPARE(received, QStringList({image}));
        mime.setUrls(localUrls({image, video})); QVERIFY(!drop(&canvas, mime)); QCOMPARE(requests, 1);
        mime.setUrls(localUrls({files.path()})); QVERIFY(drop(&surface, mime)); QCOMPARE(received, QStringList({files.path()}));
        mime.setUrls(localUrls({video})); QVERIFY(drop(&surface, mime)); QCOMPARE(received, QStringList({video}));
        QCOMPARE(requests, 3);
    }

    void itemModelMimeStaysWithTheNativeList()
    {
        vision::app::ModelSessionController models; vision::app::OperationCoordinator operations;
        QWidget window, surface(&window); QListWidget list(&surface);
        list.setAcceptDrops(true); list.setDragDropMode(QAbstractItemView::InternalMove);
        WorkspaceDropFilter filter(WorkspaceDropFilter::Surface::ModelLibrary, &surface, nullptr,
            &list, &operations, &models, &surface);
        QSignalSpy imports(&filter, &WorkspaceDropFilter::modelDropRequested);
        QSignalSpy hints(&filter, &WorkspaceDropFilter::modelHintChanged);
        QMimeData internal; internal.setData("application/x-qabstractitemmodeldatalist", "fixture");
        enter(list.viewport(), internal, Qt::MoveAction);
        QCOMPARE(imports.count(), 0); QCOMPARE(hints.count(), 0);
    }
};
QTEST_MAIN(WorkspaceDropFilterTests)
#include "workspace_drop_filter_tests.moc"
