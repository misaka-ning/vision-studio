// SPDX-License-Identifier: AGPL-3.0-only
#pragma once
#include <QWidget>

class QJsonArray;
class QLabel;
class QPushButton;
class QTableWidget;

class HistoryPage final : public QWidget
{
    Q_OBJECT
public:
    struct Controls
    {
        QTableWidget *table = nullptr;
        QPushButton *exportButton = nullptr;
        QLabel *hint = nullptr;
    };
    static constexpr int RowData = Qt::UserRole;
    explicit HistoryPage(const QString &historyFilePath, QWidget *parent = nullptr);
    const Controls &controls() const { return controls_; }
    void setRecords(const QJsonArray &records);
    void setHistoryFilePath(const QString &path);

signals:
    void exportRequested();
    void activated(int row);

private:
    Controls controls_;
};
