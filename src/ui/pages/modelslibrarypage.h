// SPDX-License-Identifier: AGPL-3.0-only
#pragma once
#include <QListWidget>
#include <QWidget>

class QLabel;
class QPushButton;

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

  signals:
    void importRequested();
    void selectionRequested(const QString &path);
    void orderChanged(const QStringList &paths);
    void contextMenuRequested(const QPoint &position);
    void detectRequested();
    void structureRequested();
    void removeRequested(const QString &path);
    void evaluationRequested();

  private:
    Controls controls_;
    bool busy_ = false, reordering_ = false;
};
