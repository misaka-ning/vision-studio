// SPDX-License-Identifier: AGPL-3.0-only
#pragma once
#include "core/visiontypes.h"
#include "resultdeliveryservice.h"
#include <QObject>
#include <QSet>
#include <QHash>
#include <QPointer>

class QThread;
namespace vision { class InferenceWorker; }
namespace vision::app
{
struct DetectionTaskSnapshot
{
    JobRequest request;
    ModelConfig loadedConfig;
    bool running = false, stopping = false, failed = false;
    int completed = 0, total = 0;
    QString error;
    QString modelIdentity, configurationFingerprint;
    QHash<QString, QString> sourceIdentities;
    int rangeFirst = 0, rangeLast = 0;
};

// Owns execution and ticket acknowledgement. All commands except the worker's
// explicitly thread-safe controls are dispatched on this object's event loop.
class DetectionController final : public QObject
{
    Q_OBJECT
  public:
    explicit DetectionController(QObject *parent = nullptr);
    ~DetectionController() override;
    const DetectionTaskSnapshot &snapshot() const { return state_; }
    void prepare();
    void prepareTask(const JobRequest &request, int first, int last,
                     const QString &modelIdentity, const QString &fingerprint,
                     const QHash<QString, QString> &sourceIdentities);
    void setConfigurationFingerprint(const QString &fingerprint);
    ResultDeliveryService *results() const { return results_; }
    bool beginResultDelivery(const ResultDeliveryOptions &options);
    bool acceptImageResult(const InferenceResult &result, quint64 ticket);
    void reportFailure(const QString &error);
    void prepareRecording(const QString &directory);
    void requestStop();
    void acknowledgeImageResult(quint64 ticket);
    void requestStartRecording();
    void requestStopRecording();
    void requestShutdown();
    bool isShutdownReady() const;
  public slots:
    void run(vision::JobRequest request);
  signals:
    void dispatch(vision::JobRequest request);
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
    void shutdownReady();
  private:
    QThread *thread_;
    QPointer<vision::InferenceWorker> worker_;
    DetectionTaskSnapshot state_;
    QSet<quint64> tickets_;
    ResultDeliveryService *results_;
    bool shuttingDown_ = false, threadFinished_ = false;
    bool shutdownReadyEmitted_ = false;
    InferenceResult lastStreamResult_;
    bool finishing_ = false, finishCancelled_ = false;
    void beginFinish(bool cancelled);
    void completeFinish();
    void checkShutdownReady();
};
}
