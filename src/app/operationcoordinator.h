// SPDX-License-Identifier: AGPL-3.0-only
#pragma once
#include <QObject>
#include <QPointer>
#include <QVector>
#include <functional>
#include "core/visiontypes.h"
#include "resultdeliveryservice.h"
#include "inputsessioncontroller.h"

class QTimer;
namespace vision::app
{
class ModelSessionController;
class WorkbenchConfiguration;
class DetectionController;
class EvaluationController;
class RecordingController;
struct PreparedDetectionTask
{
    bool valid = false;
    QString error, modelIdentity, fingerprint;
    JobRequest request;
    QHash<QString, QString> sourceIdentities;
    ResultDeliveryOptions output;
    int first = 0, last = 0;
};
enum class StopTarget { None, Input, Evaluation, Detection };
struct DetectionActivation { bool started = false, cacheInvalidated = false; };
enum class VisibleTask { None, Detection, Evaluation, Input };
struct ActivitySnapshot
{
    bool detection = false, input = false, projection = false, evaluation = false;
    bool conversion = false, gpuPreparation = false, saving = false, modal = false, closing = false;
    bool workbenchBusy() const { return detection || input || projection || evaluation; }
};
struct OperationAvailability
{
    bool importInput = false, importModels = false, start = false, changeDevice = false, editParameters = false;
};

class OperationCoordinator final : public QObject
{
    Q_OBJECT
  public:
    explicit OperationCoordinator(QObject *parent = nullptr);
    static OperationAvailability availability(const ActivitySnapshot &activity);
    void setActivityProvider(std::function<ActivitySnapshot()> provider) { activityProvider_ = std::move(provider); }
    ActivitySnapshot activity() const;
    PreparedDetectionTask prepareDetection(const InputSessionController &input,
        const ModelSessionController &model, const WorkbenchConfiguration &configuration,
        ImageRange range) const;
    DetectionActivation beginDetection(const PreparedDetectionTask &task, DetectionController &detection);
    StopTarget stopWorkbench(InputSessionController &input, EvaluationController &evaluation,
        DetectionController &detection, RecordingController &recordings);
    // Owners remain alive until the barrier is ready. Readiness is queried,
    // never copied into a separately mutable domain-status cache.
    void addShutdownParticipant(const QString &name, QObject *owner,
                                std::function<void()> request, std::function<bool()> ready);
    void requestShutdown();
    bool isClosing() const { return closing_; }
    bool isShutdownReady() const { return ready_; }
    void setVisibleTask(VisibleTask task) { visibleTask_ = task; }
    VisibleTask visibleTask() const { return visibleTask_; }
  signals:
    void shutdownStarted();
    void shutdownReady();
  private:
    void checkShutdown();
    struct Participant { QString name; QPointer<QObject> owner; std::function<void()> request; std::function<bool()> ready; };
    QVector<Participant> participants_;
    std::function<ActivitySnapshot()> activityProvider_;
    QTimer *timer_;
    bool closing_ = false, ready_ = false, requesting_ = false;
    VisibleTask visibleTask_ = VisibleTask::None;
};
}
