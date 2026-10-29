// SPDX-License-Identifier: AGPL-3.0-only
#pragma once
#include <QObject>
#include <QPointer>
#include <QStringList>
class QWidget;
class QListWidget;
class QDropEvent;
class ImageCanvas;
namespace vision::app { class OperationCoordinator; class ModelSessionController; }

// Presentation-owned application event filter. It handles only descendants of
// its own surface and never owns model/input business state or an import job.
class WorkspaceDropFilter final : public QObject
{
    Q_OBJECT
  public:
    enum class Surface { Preview, ModelLibrary };
    WorkspaceDropFilter(Surface kind, QWidget *surface, ImageCanvas *canvas,
                        QListWidget *modelList,
                        const vision::app::OperationCoordinator *operations,
                        const vision::app::ModelSessionController *models,
                        QObject *owner);
    static QString modelPrompt();
    void showModelHint(const QString &message, bool error = false);
    // Import intents are resolved synchronously by a DirectConnection. A token
    // prevents a stale or nested callback from acknowledging a different event.
    void resolveDrop(quint64 requestId, bool accepted);
  signals:
    void previewDropRequested(quint64 requestId, const QStringList &paths);
    void modelDropRequested(quint64 requestId, const QStringList &paths);
    void noticeRequested(const QString &message, bool error);
    void modelHintChanged(const QString &message, bool error);
  protected:
    bool eventFilter(QObject *watched, QEvent *event) override;
  private:
    bool locked() const;
    bool dispatch(const QStringList &paths);
    bool filterModels(QDropEvent *drop, QEvent *event);
    bool filterPreview(QDropEvent *drop, QEvent *event);
    Surface kind_;
    QPointer<QWidget> surface_;
    QPointer<ImageCanvas> canvas_;
    QPointer<QListWidget> modelList_;
    QPointer<const vision::app::OperationCoordinator> operations_;
    QPointer<const vision::app::ModelSessionController> models_;
    bool externalModelDragActive_ = false;
    quint64 nextRequest_ = 0, pendingRequest_ = 0;
    bool pendingAccepted_ = false;
};
