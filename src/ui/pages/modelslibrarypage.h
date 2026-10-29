// SPDX-License-Identifier: AGPL-3.0-only
#pragma once
#include <QListWidget>
#include <QPointer>
#include <QWidget>

class QLabel;
class QPushButton;
namespace vision::app { class ModelSessionController; class OperationCoordinator; }

struct ModelLibraryRow
{
    QString path, displayName, detail, note, evaluationSummary, tooltip;
    int height = 94;
};

class ModelsLibraryPage final : public QWidget
{
    Q_OBJECT
  public:
    struct Controls
    {
        QLabel *count = nullptr, *dropHint = nullptr;
        QListWidget *list = nullptr;
        QPushButton *import = nullptr, *activate = nullptr, *structure = nullptr,
                    *remove = nullptr, *evaluationReport = nullptr;
    };
    explicit ModelsLibraryPage(QWidget *parent = nullptr);
    const Controls &controls() const { return controls_; }
    void setBusy(bool busy);
    void setRows(const QVector<ModelLibraryRow> &rows, const QString &selectedPath);
    void setDropHint(const QString &message, bool error = false);
    void setApplicationContext(const vision::app::ModelSessionController *models,
                               const vision::app::OperationCoordinator *operations);

  signals:
    void importRequested();
    void selectionRequested(const QString &path);
    void orderChanged(const QStringList &paths);
    void contextMenuRequested(const QPoint &position);
    void detectRequested();
    void structureRequested();
    void removeRequested(const QString &path);
    void renameRequested(const QString &path, const QString &name);
    void noteRequested(const QString &path, const QString &note);
    void noticeRequested(const QString &message, bool error = false);
    void evaluationRequested();

  private:
    bool canMutate(const QString &path) const;
    void showContextMenu(const QPoint &position);
    void openModelLocation(const QString &path);
    void editModelNote(const QString &path);
    void renameModel(const QString &path);
    Controls controls_;
    QPointer<const vision::app::ModelSessionController> models_;
    QPointer<const vision::app::OperationCoordinator> operations_;
    bool busy_ = false, reordering_ = false;
};
