#pragma once

#include "visiontypes.h"
#include <QDateTime>
#include <QElapsedTimer>
#include <QSize>
#include <QString>
#include <QVector>
#include <opencv2/videoio.hpp>

namespace vision
{

struct RecordingSummary
{
    QString path;
    QString metadataPath;
    qint64 frames = 0;
    double fps = 0;
    double durationSeconds = 0;
    QSize contentSize;
    QSize encodedSize;
};

// Used only by the inference thread. Commands for it are marshalled by InferenceWorker.
// A final AVI is published only after its writer is released and its index is validated.
class VideoRecorder final
{
  public:
    ~VideoRecorder();
    bool start(const QString &directory, const InferenceResult &firstFrame, double sourceFps, bool realtime,
               const ModelConfig &config, QString *error = nullptr);
    bool append(const InferenceResult &frame, QString *error = nullptr);
    bool finish(RecordingSummary *summary = nullptr, QString *error = nullptr);
    void discard() noexcept;
    bool active() const noexcept
    {
        return m_active;
    }
    QString path() const
    {
        return m_path;
    }
    qint64 frames() const noexcept
    {
        return m_frames;
    }
    static QImage annotatedFrame(const InferenceResult &frame);

  private:
    cv::VideoWriter m_writer;
    QString m_path, m_temporaryPath;
    QSize m_contentSize, m_encodedSize;
    InferenceResult m_firstFrame;
    ModelConfig m_config;
    QDateTime m_created;
    QElapsedTimer m_timer;
    QVector<qint64> m_frameTimes;
    qint64 m_frames = 0, m_lastSourceFrame = 0;
    double m_sourceFps = 30;
    bool m_active = false, m_realtime = false;
};

} // namespace vision
