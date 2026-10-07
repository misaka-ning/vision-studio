#pragma once

#include "visiontypes.h"
#include <QObject>
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
    void prepare() noexcept
    {
        m_stop.store(false, std::memory_order_release);
    }
    void requestStop() noexcept
    {
        m_stop.store(true, std::memory_order_release);
    }

  public slots:
    void run(vision::JobRequest request);

  signals:
    void modelReady(QString backend, vision::ModelConfig config);
    void resultReady(vision::InferenceResult result);
    void progress(int completed, int total);
    void status(QString message);
    void failed(QString message);
    void finished(bool cancelled);

  private:
    bool stopping() const noexcept
    {
        return m_stop.load(std::memory_order_acquire);
    }
    std::atomic_bool m_stop{false};
};

} // namespace vision
