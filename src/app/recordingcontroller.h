// SPDX-License-Identifier: AGPL-3.0-only
#pragma once
#include "core/visiontypes.h"
#include <QElapsedTimer>
#include <QJsonObject>
#include <QObject>
#include <QPointer>

class QThread;
class QTimer;
namespace vision::app
{
struct RecordingEntry
{
    QString path, caption, durationText, resolutionText, sizeText;
    QJsonObject metadata;
};
struct RecordingCaptureState
{
    bool requested = false, active = false, stopping = false, enabled = false;
    qint64 elapsedSeconds = 0;
    QString buttonText = QStringLiteral("开始录制");
    QString statusText = QStringLiteral("视频 / 摄像头检测时可录制");
};
struct RecordingPlaybackState
{
    QString path;
    QString details = QStringLiteral("还没有录像。在工作台检测视频或摄像头时，点击「开始录制」。");
    bool canPlay = false, canExport = false, playing = false, exporting = false;
};
class RecordingWorker;

// Presentation-free recording intent, library and playback. Encoding stays in
// InferenceWorker. Disk enumeration, playback decode and accepted export copies
// execute on a dedicated event loop, including during asynchronous shutdown.
class RecordingController final : public QObject
{
    Q_OBJECT
  public:
    explicit RecordingController(const QString &dataRoot, QObject *parent = nullptr);
    ~RecordingController() override;
    QString directory() const { return directory_; }
    const QVector<RecordingEntry> &recordings() const { return recordings_; }
    const RecordingCaptureState &captureState() const { return capture_; }
    const RecordingPlaybackState &playbackState() const { return playback_; }
    void setDetectionState(bool running, SourceKind source, bool stopping = false, bool failed = false);
    void toggleRecording();
    void recordingStarted(const QString &path);
    void recordingFinished(const QString &path, qint64 frames, double fps);
    void recordingFailed(const QString &error);
    void detectionFinished();
    void refresh();
    void selectRecording(const QString &path);
    void togglePlayback();
    void stopPlayback();
    bool exportRecording(const QString &destination);
    void requestShutdown();
    bool isShutdownReady() const;

  signals:
    void startRecordingRequested();
    void stopRecordingRequested();
    void captureStateChanged(const vision::app::RecordingCaptureState &state);
    void recordingsChanged(const QVector<vision::app::RecordingEntry> &entries, const QString &selectedPath);
    void playbackChanged(const vision::app::RecordingPlaybackState &state);
    void frameReady(const QImage &image, const QString &path);
    void notice(const QString &message, bool error);
    void shutdownReady();

  private:
    void publishCapture();
    void scheduleFrame();
    void openPlayback(bool autoplay);
    void maybeFinishShutdown();
    QString directory_;
    QVector<RecordingEntry> recordings_;
    RecordingCaptureState capture_;
    RecordingPlaybackState playback_;
    SourceKind source_ = SourceKind::Images;
    bool detectionRunning_ = false, detectionStopping_ = false, detectionFailed_ = false;
    bool playbackOpened_ = false, readPending_ = false, autoplay_ = false;
    bool shuttingDown_ = false, threadFinished_ = false, shutdownSignalled_ = false;
    double playbackFps_ = 25;
    quint64 playbackGeneration_ = 0, scanGeneration_ = 0;
    QElapsedTimer elapsed_;
    QTimer *captureClock_ = nullptr, *playbackClock_ = nullptr;
    QThread *thread_ = nullptr;
    QPointer<RecordingWorker> worker_;
};
} // namespace vision::app
Q_DECLARE_METATYPE(vision::app::RecordingEntry)
Q_DECLARE_METATYPE(QVector<vision::app::RecordingEntry>)
Q_DECLARE_METATYPE(vision::app::RecordingCaptureState)
Q_DECLARE_METATYPE(vision::app::RecordingPlaybackState)
