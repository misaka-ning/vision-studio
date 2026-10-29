// SPDX-License-Identifier: AGPL-3.0-only
#include "workspacedropfilter.h"
#include "app/operationcoordinator.h"
#include "app/modelsessioncontroller.h"
#include "core/previewdrop.h"
#include "ui/imagecanvas.h"
#include <QApplication>
#include <QDropEvent>
#include <QListWidget>
#include <QMimeData>
#include <QUrl>
#include <QWidget>

WorkspaceDropFilter::WorkspaceDropFilter(Surface kind, QWidget *surface, ImageCanvas *canvas,
    QListWidget *modelList, const vision::app::OperationCoordinator *operations,
    const vision::app::ModelSessionController *models, QObject *owner)
    : QObject(owner), kind_(kind), surface_(surface), canvas_(canvas), modelList_(modelList),
      operations_(operations), models_(models)
{
    if (qApp) qApp->installEventFilter(this);
}

QString WorkspaceDropFilter::modelPrompt()
{
    return QStringLiteral("将 ONNX / PT / TorchScript 文件拖到此处，或点击导入；支持多选，原文件保留。");
}
void WorkspaceDropFilter::showModelHint(const QString &message, bool error)
{
    externalModelDragActive_ = false;
    emit modelHintChanged(message, error);
    emit noticeRequested(message, error);
}
bool WorkspaceDropFilter::locked() const
{
    if (!operations_) return true;
    const auto available = vision::app::OperationCoordinator::availability(operations_->activity());
    return kind_ == Surface::ModelLibrary ? !available.importModels : !available.importInput;
}
void WorkspaceDropFilter::resolveDrop(quint64 requestId, bool accepted)
{
    if (requestId == pendingRequest_) pendingAccepted_ = accepted;
}
bool WorkspaceDropFilter::dispatch(const QStringList &paths)
{
    const quint64 request = ++nextRequest_;
    pendingRequest_ = request; pendingAccepted_ = false;
    if (kind_ == Surface::ModelLibrary) emit modelDropRequested(request, paths);
    else emit previewDropRequested(request, paths);
    const bool accepted = pendingRequest_ == request && pendingAccepted_;
    if (pendingRequest_ == request) pendingRequest_ = 0;
    return accepted;
}
bool WorkspaceDropFilter::eventFilter(QObject *watched, QEvent *event)
{
    if (event->type() != QEvent::DragEnter && event->type() != QEvent::DragMove &&
        event->type() != QEvent::Drop && event->type() != QEvent::DragLeave) return false;
    auto *widget = qobject_cast<QWidget *>(watched);
    if (!surface_ || !widget || widget->window() != surface_->window() ||
        (widget != surface_ && !surface_->isAncestorOf(widget))) return false;
    if (event->type() == QEvent::DragLeave) {
        if (kind_ == Surface::ModelLibrary) {
            if (externalModelDragActive_) showModelHint(modelPrompt());
        } else if (canvas_) canvas_->setDropActive(false);
        return false;
    }
    auto *drop = static_cast<QDropEvent *>(event);
    return kind_ == Surface::ModelLibrary ? filterModels(drop, event) : filterPreview(drop, event);
}
bool WorkspaceDropFilter::filterModels(QDropEvent *drop, QEvent *event)
{
    // Qt's InternalMove remains owned by the list. Non-URL model MIME must
    // never enter the external import path.
    if ((modelList_ && drop->source() == modelList_) || !drop->mimeData()->hasUrls()) return false;
    drop->ignore();
    if (locked()) {
        showModelHint("请先结束当前任务，再导入模型；当前模型与任务保持不变。", true); return true;
    }
    if (!(drop->possibleActions() & Qt::CopyAction)) {
        showModelHint("模型库仅引用本地文件，不支持移动或链接操作；请直接拖入文件。", true); return true;
    }
    QStringList input;
    for (const auto &url : drop->mimeData()->urls()) {
        if (!url.isLocalFile() || (!url.host().isEmpty() && url.host() != "localhost") ||
            !url.query().isEmpty() || !url.fragment().isEmpty()) {
            showModelHint("只支持本地模型文件，不能混入网络地址；本次未导入任何模型。", true); return true;
        }
        input.append(url.toLocalFile());
    }
    QStringList paths; QString error;
    if (!models_ || !models_->validateImport(input, &paths, &error)) {
        showModelHint(error, true); return true;
    }
    if (event->type() == QEvent::Drop) {
        if (externalModelDragActive_) showModelHint(modelPrompt());
        if (!dispatch(paths)) return true;
    } else {
        showModelHint(QString("松开以导入 %1 个模型；同一路径自动去重，原文件保留。").arg(paths.size()));
        externalModelDragActive_ = true;
    }
    drop->setDropAction(Qt::CopyAction); drop->accept();
    return true;
}
bool WorkspaceDropFilter::filterPreview(QDropEvent *drop, QEvent *event)
{
    if (!drop->mimeData()->hasUrls()) return false;
    drop->ignore();
    if (canvas_) canvas_->setDropActive(false);
    if (locked()) { emit noticeRequested("请先停止当前任务，再拖入新的输入。", true); return true; }
    QStringList input;
    for (const auto &url : drop->mimeData()->urls()) {
        if (!url.isLocalFile()) {
            emit noticeRequested("只支持拖入本地文件，不能混入网络地址。", true); return true;
        }
        input.append(url.toLocalFile());
    }
    QStringList paths; QString error; vision::PreviewDropKind kind;
    if (!vision::classifyPreviewDrop(input, &paths, &kind, &error)) {
        emit noticeRequested(error, true); return true;
    }
    if (event->type() == QEvent::Drop && !dispatch(paths)) return true;
    if (event->type() != QEvent::Drop && canvas_) canvas_->setDropActive(true);
    drop->setDropAction(Qt::CopyAction); drop->accept();
    return true;
}
