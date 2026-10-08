#pragma once

#include "visiontypes.h"
#include <QMutex>
#include <QObject>
#include <QWaitCondition>
#include <atomic>

namespace vision
{

// A side-by-side stream is split before inference; coordinates are relative to the selected eye.
QImage selectStereoView(const QImage &image, StereoView view);

class InferenceWorker final : public QObject
{
    Q_OBJECT
  public:
    explicit InferenceWorker(QObject *parent = nullptr) : QObject(parent) {}
    // Called by the GUI before queueing a job, so a concurrent stop is never reset by run().
    void prepare() noexcept;
    void requestStop() noexcept;
    // Configure only while idle. The GUI acknowledges a ticket after its bounded
    // cache/export work completes, so fast inference cannot flood its event queue.
    void setImageResultBackpressureEnabled(bool enabled) noexcept;
    void acknowledgeImageResult(quint64 ticket) noexcept;
    // Direct GUI calls: only small command state is touched, never the writer itself.
    void prepareRecording(const QString &directory);
    void requestStartRecording() noexcept
    {
        m_recordRequested.store(true, std::memory_order_release);
    }
    void requestStopRecording() noexcept
    {
        m_recordRequested.store(false, std::memory_order_release);
    }

  public slots:
    void run(vision::JobRequest request);

  signals:
    void modelReady(QString backend, vision::ModelConfig config);
    void resultReady(vision::InferenceResult result);
    void imageResultReady(vision::InferenceResult result, quint64 ticket);
    void progress(int completed, int total);
    void status(QString message);
    void failed(QString message);
    void finished(bool cancelled);
    void recordingStarted(QString path);
    void recordingFinished(QString path, qint64 frames, double fps);
    void recordingFailed(QString error);

  private:
    bool stopping() const noexcept
    {
        return m_stop.load(std::memory_order_acquire);
    }
    bool deliverImageResult(vision::InferenceResult result, int completed, int total);
    std::atomic_bool m_stop{false};
    std::atomic_bool m_recordRequested{false};
    QMutex m_recordMutex;
    QString m_recordDirectory;
    std::atomic_bool m_imageResultBackpressure{false};
    QMutex m_deliveryMutex;
    QWaitCondition m_deliveryAcknowledged;
    quint64 m_pendingImageTicket = 0;
    quint64 m_nextImageTicket = 0;
};

} // namespace vision
