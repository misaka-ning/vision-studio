// SPDX-License-Identifier: AGPL-3.0-only
#include "recordingcontroller.h"
#include "asyncthreadretirement.h"
#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QSaveFile>
#include <QThread>
#include <QTimer>
#include <algorithm>
#include <cmath>
#include <memory>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

namespace vision::app
{
namespace
{
QJsonObject metadataFor(const QString &path)
{
    const QFileInfo info(path);
    QFile file(info.absolutePath() + "/" + info.completeBaseName() + ".json");
    return file.open(QIODevice::ReadOnly) ? QJsonDocument::fromJson(file.readAll()).object() : QJsonObject{};
}
QString stereoName(const QString &mode)
{
    return mode == "left" ? QStringLiteral("双目左目") : mode == "right" ? QStringLiteral("双目右目") : QStringLiteral("完整画面");
}
}

class RecordingWorker final : public QObject
{
    Q_OBJECT
  public:
    void scan(const QString &directory, quint64 generation)
    {
        QVector<RecordingEntry> entries;
        const auto files = QDir(directory).entryInfoList({"*.avi"}, QDir::Files, QDir::Time);
        for (const auto &file : files) {
            const auto metadata = metadataFor(file.absoluteFilePath());
            const double fps = metadata.value("fps").toDouble();
            const double duration = metadata.value("duration_seconds").toDouble(fps > 0 ? metadata.value("frames").toDouble() / fps : 0);
            const int width = metadata.value("content_width").toInt(metadata.value("width").toInt());
            const int height = metadata.value("content_height").toInt(metadata.value("height").toInt());
            entries.append({file.absoluteFilePath(), file.lastModified().toString("yyyy-MM-dd HH:mm:ss") + "\n" + file.fileName(),
                duration > 0 ? QString::number(duration, 'f', 1) + " s" : QStringLiteral("—"),
                width > 0 && height > 0 ? QString("%1×%2").arg(width).arg(height) : QStringLiteral("—"),
                QString::number(file.size() / 1048576.0, 'f', 1) + " MB", metadata});
        }
        emit scanned(generation, entries);
    }
    void open(const QString &path, quint64 generation)
    {
        release(); path_ = path; generation_ = generation;
        const auto metadata = metadataFor(path);
        contentSize_ = QSize(metadata.value("content_width").toInt(metadata.value("width").toInt()),
                             metadata.value("content_height").toInt(metadata.value("height").toInt()));
        RecordingPlaybackState state; state.path = path;
        try { capture_ = std::make_unique<cv::VideoCapture>(QFile::encodeName(path).constData()); }
        catch (const cv::Exception &) {
            state.details = "无法解码此录像，请检查文件是否完整。";
            state.canExport = QFileInfo(path).isFile();
            emit opened(generation, state, 25); return;
        }
        state.canPlay = capture_->isOpened(); state.canExport = QFileInfo(path).isFile();
        double fps = 25;
        if (!state.canPlay) state.details = "无法解码此录像，请检查文件是否完整。";
        else {
            const double rawFps = capture_->get(cv::CAP_PROP_FPS);
            fps = std::isfinite(rawFps) && rawFps >= .001 ? rawFps : 25;
            state.details = QString("%1 · %2 · %3 fps\n%4").arg(stereoName(metadata.value("stereo_view").toString()))
                .arg(metadata.value("color_mode").toString() == "grayscale" ? "灰度" : "彩色")
                .arg(rawFps, 0, 'f', 1).arg(QFileInfo(path).fileName());
        }
        emit opened(generation, state, fps);
        if (state.canPlay) read(generation);
    }
    void read(quint64 generation)
    {
        if (generation != generation_ || !capture_ || !capture_->isOpened()) return;
        try {
            cv::Mat pixels;
            if (!capture_->read(pixels) || pixels.empty()) {
                capture_->set(cv::CAP_PROP_POS_FRAMES, 0); emit ended(generation); return;
            }
            cv::Mat rgb; cv::cvtColor(pixels, rgb, cv::COLOR_BGR2RGB);
            QImage image = QImage(rgb.data, rgb.cols, rgb.rows, qsizetype(rgb.step), QImage::Format_RGB888).copy();
            if (contentSize_.isValid() && contentSize_.width() <= image.width() && contentSize_.height() <= image.height())
                image = image.copy(QRect(QPoint(), contentSize_));
            emit frame(generation, image, path_);
        } catch (const cv::Exception &error) {
            release(); emit readFailed(generation, QString::fromUtf8(error.what()));
        }
    }
    void release()
    {
        if (capture_) capture_->release();
        capture_.reset();
    }
    void exportFile(const QString &sourcePath, const QString &destination)
    {
        QFile source(sourcePath); QSaveFile target(destination);
        if (!source.open(QIODevice::ReadOnly) || !target.open(QIODevice::WriteOnly)) {
            emit exported("无法导出录像，请检查保存位置。", true); return;
        }
        while (!source.atEnd()) {
            const auto block = source.read(1024 * 1024);
            if (block.isEmpty() || target.write(block) != block.size()) {
                emit exported("录像写入失败，文件未提交。", true); return;
            }
        }
        if (!target.commit()) { emit exported("录像保存失败：" + target.errorString(), true); return; }
        const QFileInfo info(sourcePath), output(destination);
        QFile metadata(info.absolutePath() + "/" + info.completeBaseName() + ".json");
        if (metadata.open(QIODevice::ReadOnly)) {
            const auto bytes = metadata.readAll();
            QSaveFile result(output.absolutePath() + "/" + output.completeBaseName() + ".json");
            if (!result.open(QIODevice::WriteOnly) || result.write(bytes) != bytes.size() || !result.commit()) {
                emit exported("录像已导出，元数据未保存：" + result.errorString(), true); return;
            }
        }
        emit exported("录像已导出：" + destination, false);
    }
  signals:
    void scanned(quint64 generation, const QVector<vision::app::RecordingEntry> &entries);
    void opened(quint64 generation, const vision::app::RecordingPlaybackState &state, double fps);
    void frame(quint64 generation, const QImage &image, const QString &path);
    void ended(quint64 generation);
    void readFailed(quint64 generation, const QString &error);
    void exported(const QString &message, bool error);
  private:
    std::unique_ptr<cv::VideoCapture> capture_;
    QString path_;
    QSize contentSize_;
    quint64 generation_ = 0;
};

RecordingController::RecordingController(const QString &dataRoot, QObject *parent)
    : QObject(parent), directory_(dataRoot + "/recordings")
{
    qRegisterMetaType<QVector<RecordingEntry>>(); qRegisterMetaType<RecordingCaptureState>(); qRegisterMetaType<RecordingPlaybackState>();
    captureClock_ = new QTimer(this); captureClock_->setInterval(250);
    connect(captureClock_, &QTimer::timeout, this, &RecordingController::publishCapture);
    playbackClock_ = new QTimer(this);
    connect(playbackClock_, &QTimer::timeout, this, &RecordingController::scheduleFrame);
    thread_ = new QThread(this); thread_->setObjectName("recordingLibraryWorker");
    worker_ = new RecordingWorker; worker_->moveToThread(thread_);
    connect(thread_, &QThread::finished, worker_, &QObject::deleteLater);
    AsyncThreadRetirement::watch(thread_, this, [this] {
        thread_ = nullptr; threadFinished_ = true; maybeFinishShutdown();
    });
    connect(worker_, &RecordingWorker::scanned, this, [this](quint64 generation, const QVector<RecordingEntry> &entries) {
        if (shuttingDown_ || generation != scanGeneration_) return;
        recordings_ = entries; QString selected;
        for (const auto &entry : entries) if (entry.path == playback_.path) selected = entry.path;
        if (selected.isEmpty() && !entries.isEmpty()) selected = entries.first().path;
        emit recordingsChanged(recordings_, selected); selectRecording(selected);
    });
    connect(worker_, &RecordingWorker::opened, this, [this](quint64 generation, const RecordingPlaybackState &state, double fps) {
        if (shuttingDown_ || generation != playbackGeneration_) return;
        const bool exporting = playback_.exporting;
        playback_ = state; playback_.exporting = exporting; playbackOpened_ = state.canPlay; playbackFps_ = fps;
        if (autoplay_ && state.canPlay) {
            playback_.playing = true;
            playbackClock_->start(std::max(1, int(std::round(1000 / std::min(fps, 1000.0)))));
        }
        autoplay_ = false; if (!state.canPlay) readPending_ = false;
        emit playbackChanged(playback_);
    });
    connect(worker_, &RecordingWorker::frame, this, [this](quint64 generation, const QImage &image, const QString &path) {
        if (shuttingDown_ || generation != playbackGeneration_) return;
        readPending_ = false; emit frameReady(image, path);
    });
    connect(worker_, &RecordingWorker::ended, this, [this](quint64 generation) {
        if (shuttingDown_ || generation != playbackGeneration_) return;
        readPending_ = false; playbackClock_->stop(); playback_.playing = false; emit playbackChanged(playback_);
    });
    connect(worker_, &RecordingWorker::readFailed, this, [this](quint64 generation, const QString &error) {
        if (shuttingDown_ || generation != playbackGeneration_) return;
        readPending_ = playbackOpened_ = false; playbackClock_->stop();
        playback_.playing = playback_.canPlay = false;
        playback_.details = "无法解码此录像，请检查文件是否完整。";
        emit playbackChanged(playback_); emit notice("录像播放失败：" + error, true);
    });
    connect(worker_, &RecordingWorker::exported, this, [this](const QString &message, bool error) {
        playback_.exporting = false; emit playbackChanged(playback_); emit notice(message, error);
    });
    thread_->start();
}

RecordingController::~RecordingController()
{
    captureClock_->stop(); playbackClock_->stop();
    if (!threadFinished_) {
        thread_->disconnect(this);
        if (worker_)
            QMetaObject::invokeMethod(worker_, [worker = worker_] { if (worker) worker->release(); QThread::currentThread()->quit(); }, Qt::QueuedConnection);
    }
}

void RecordingController::setDetectionState(bool running, SourceKind source, bool stopping, bool failed)
{
    detectionRunning_ = running; source_ = source; detectionStopping_ = stopping; detectionFailed_ = failed;
    if (!running) { capture_.requested = capture_.active = capture_.stopping = false; captureClock_->stop(); }
    else if (stopping) { capture_.requested = false; capture_.stopping = capture_.active; }
    publishCapture(); maybeFinishShutdown();
}

void RecordingController::publishCapture()
{
    capture_.enabled = !shuttingDown_ && detectionRunning_ && source_ != SourceKind::Images &&
                       !capture_.stopping && !detectionStopping_ && !detectionFailed_;
    const bool recording = capture_.requested || capture_.active;
    capture_.buttonText = capture_.stopping ? "正在保存…" : recording ? "结束录制" : "开始录制";
    capture_.elapsedSeconds = capture_.active ? elapsed_.elapsed() / 1000 : 0;
    if (capture_.stopping) capture_.statusText = "正在完成录像文件…";
    else if (capture_.active) capture_.statusText = QString("● REC  %1:%2")
        .arg(capture_.elapsedSeconds / 60, 2, 10, QChar('0')).arg(capture_.elapsedSeconds % 60, 2, 10, QChar('0'));
    else if (capture_.requested) capture_.statusText = "等待下一帧开始录制…";
    else capture_.statusText = detectionRunning_ && source_ != SourceKind::Images ? "可录制当前检测画面" : "视频 / 摄像头检测时可录制";
    emit captureStateChanged(capture_);
}

void RecordingController::toggleRecording()
{
    if (!capture_.enabled) return;
    if (capture_.requested || capture_.active) {
        capture_.requested = false; capture_.stopping = capture_.active; emit stopRecordingRequested();
    } else { capture_.requested = true; emit startRecordingRequested(); }
    publishCapture();
}
void RecordingController::recordingStarted(const QString &)
{
    capture_.active = true; elapsed_.restart();
    if (!shuttingDown_) captureClock_->start();
    if (!capture_.requested && !shuttingDown_) { capture_.stopping = true; emit stopRecordingRequested(); }
    publishCapture();
}
void RecordingController::recordingFinished(const QString &path, qint64 frames, double)
{
    capture_.requested = capture_.active = capture_.stopping = false; captureClock_->stop();
    if (!shuttingDown_) refresh();
    publishCapture(); emit notice(QString("录制已保存 · %1 帧 · %2").arg(frames).arg(QFileInfo(path).fileName()), false);
    maybeFinishShutdown();
}
void RecordingController::recordingFailed(const QString &error)
{
    capture_.requested = capture_.active = capture_.stopping = false; captureClock_->stop();
    publishCapture(); emit notice("录制未保存：" + error, true); maybeFinishShutdown();
}
void RecordingController::detectionFinished() { setDetectionState(false, source_); }

void RecordingController::refresh()
{
    if (shuttingDown_) return;
    const auto generation = ++scanGeneration_;
    QMetaObject::invokeMethod(worker_, [worker = worker_, directory = directory_, generation] { worker->scan(directory, generation); }, Qt::QueuedConnection);
}
void RecordingController::selectRecording(const QString &path)
{
    if (shuttingDown_) return;
    stopPlayback(); playback_.path = path; playback_.canPlay = playback_.canExport = false;
    if (path.isEmpty()) { playback_.details = "还没有录像。在工作台检测视频或摄像头时，点击「开始录制」。"; emit playbackChanged(playback_); return; }
    playback_.details = "正在载入录像…"; emit playbackChanged(playback_);
    openPlayback(false);
}
void RecordingController::openPlayback(bool autoplay)
{
    const auto generation = ++playbackGeneration_;
    readPending_ = true; autoplay_ = autoplay;
    QMetaObject::invokeMethod(worker_, [worker = worker_, path = playback_.path, generation] { worker->open(path, generation); }, Qt::QueuedConnection);
}
void RecordingController::togglePlayback()
{
    if (shuttingDown_ || playback_.path.isEmpty()) return;
    if (playback_.playing) { playbackClock_->stop(); playback_.playing = false; emit playbackChanged(playback_); return; }
    if (!playbackOpened_) { openPlayback(true); return; }
    playback_.playing = true;
    playbackClock_->start(std::max(1, int(std::round(1000 / std::min(playbackFps_, 1000.0)))));
    emit playbackChanged(playback_);
}
void RecordingController::scheduleFrame()
{
    if (shuttingDown_ || readPending_ || !playback_.playing || !playbackOpened_) return;
    readPending_ = true; const auto generation = playbackGeneration_;
    QMetaObject::invokeMethod(worker_, [worker = worker_, generation] { worker->read(generation); }, Qt::QueuedConnection);
}
void RecordingController::stopPlayback()
{
    playbackClock_->stop(); playback_.playing = false; playbackOpened_ = false; readPending_ = false;
    ++playbackGeneration_;
    if (!threadFinished_ && !shuttingDown_)
        QMetaObject::invokeMethod(worker_, [worker = worker_] { worker->release(); }, Qt::QueuedConnection);
    emit playbackChanged(playback_);
}
bool RecordingController::exportRecording(const QString &destination)
{
    if (shuttingDown_ || playback_.exporting || !QFileInfo(playback_.path).isFile() || destination.isEmpty() ||
        QFileInfo(destination).canonicalFilePath() == QFileInfo(playback_.path).canonicalFilePath()) return false;
    if (QFileInfo(destination).suffix().compare("avi", Qt::CaseInsensitive) != 0) { emit notice("请将录像保存为 .avi 文件。", true); return false; }
    playback_.exporting = true;
    // A status observer may synchronously close the window and pump events.
    // Accept the copy before publishing that status so shutdown queues its
    // release/quit after the accepted export, never before it.
    if (!QMetaObject::invokeMethod(worker_, [worker = worker_, source = playback_.path, destination] {
            worker->exportFile(source, destination);
        }, Qt::QueuedConnection)) {
        playback_.exporting = false;
        return false;
    }
    emit playbackChanged(playback_);
    return true;
}
void RecordingController::requestShutdown()
{
    if (shuttingDown_) return;
    shuttingDown_ = true; captureClock_->stop(); playbackClock_->stop(); ++playbackGeneration_; ++scanGeneration_;
    publishCapture();
    QMetaObject::invokeMethod(worker_, [worker = worker_] { worker->release(); QThread::currentThread()->quit(); }, Qt::QueuedConnection);
    QTimer::singleShot(0, this, &RecordingController::maybeFinishShutdown);
}
bool RecordingController::isShutdownReady() const
{
    return shuttingDown_ && threadFinished_ && !detectionRunning_ && !capture_.active && !capture_.requested && !capture_.stopping;
}
void RecordingController::maybeFinishShutdown()
{
    if (isShutdownReady() && !shutdownSignalled_) { shutdownSignalled_ = true; emit shutdownReady(); }
}
} // namespace vision::app
#include "recordingcontroller.moc"
