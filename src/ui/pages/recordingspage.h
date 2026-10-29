// SPDX-License-Identifier: AGPL-3.0-only
#pragma once
#include "app/recordingcontroller.h"
#include <QWidget>
class QLabel;
class QPushButton;
class QTableWidget;
class ImageCanvas;

class RecordingsPage final : public QWidget
{
    Q_OBJECT
  public:
    struct Controls
    {
        QLabel *count = nullptr, *details = nullptr;
        QTableWidget *table = nullptr;
        ImageCanvas *canvas = nullptr;
        QPushButton *refresh = nullptr, *folder = nullptr, *play = nullptr, *exportVideo = nullptr;
    };
    explicit RecordingsPage(const QString &recordingsDirectory, QWidget *parent = nullptr);
    const Controls &controls() const { return controls_; }
    void setRecords(const QVector<vision::app::RecordingEntry> &entries, const QString &selectedPath);
    void setPlayback(const vision::app::RecordingPlaybackState &state);
    void showFrame(const QImage &image, const QString &path);
  signals:
    void refreshRequested();
    void openFolderRequested();
    void playRequested();
    void exportRequested();
    void selectionRequested(const QString &path);
  private:
    Controls controls_;
};
