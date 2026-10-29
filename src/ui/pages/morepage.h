// SPDX-License-Identifier: AGPL-3.0-only
#pragma once
#include <QWidget>

class QCheckBox;
class QLabel;
class QPlainTextEdit;
class QProgressBar;
class QPushButton;

class MorePage final : public QWidget
{
    Q_OBJECT
public:
    struct Controls
    {
        QCheckBox *restoreDataset = nullptr;
        QLabel *gpuEnvironmentStatus = nullptr;
        QPushButton *gpuCheckButton = nullptr;
        QPushButton *gpuPrepareButton = nullptr;
        QPushButton *gpuCancelButton = nullptr;
        QProgressBar *gpuSetupProgress = nullptr;
        QPlainTextEdit *gpuLog = nullptr;
        QPushButton *cpuPrepareButton = nullptr;
        QPushButton *exportButton = nullptr;
    };

    explicit MorePage(bool restoreDatasetOnStartup, bool cpuDownloadNeeded, QWidget *parent = nullptr);
    const Controls &controls() const { return controls_; }
    void setRestoreDatasetOnStartup(bool enabled);

signals:
    void restoreDatasetOnStartupChanged(bool enabled);
    void gpuCheckRequested();
    void gpuPrepareRequested();
    void gpuCancelRequested();
    void cpuPrepareRequested();
    void exportRequested();

private:
    Controls controls_;
};
