// SPDX-License-Identifier: AGPL-3.0-only
#include "operationcoordinator.h"
#include "modelsessioncontroller.h"
#include "workbenchconfiguration.h"
#include "detectioncontroller.h"
#include "evaluationcontroller.h"
#include "recordingcontroller.h"
#include <QDir>
#include <QTimer>

namespace vision::app
{
OperationCoordinator::OperationCoordinator(QObject *parent) : QObject(parent), timer_(new QTimer(this))
{
    timer_->setInterval(10);
    connect(timer_, &QTimer::timeout, this, &OperationCoordinator::checkShutdown);
}
OperationAvailability OperationCoordinator::availability(const ActivitySnapshot &state)
{
    OperationAvailability result;
    result.importInput = result.importModels = !state.workbenchBusy() && !state.conversion &&
        !state.gpuPreparation && !state.saving && !state.modal && !state.closing;
    result.start = !state.workbenchBusy() && !state.conversion && !state.gpuPreparation && !state.saving && !state.closing;
    result.changeDevice = !state.workbenchBusy() && !state.gpuPreparation && !state.closing;
    result.editParameters = !state.workbenchBusy() && !state.closing;
    return result;
}
ActivitySnapshot OperationCoordinator::activity() const
{
    auto result = activityProvider_ ? activityProvider_() : ActivitySnapshot{};
    result.closing = closing_;
    return result;
}
PreparedDetectionTask OperationCoordinator::prepareDetection(const InputSessionController &input,
    const ModelSessionController &model, const WorkbenchConfiguration &configuration, ImageRange range) const
{
    PreparedDetectionTask task;
    if (!availability(activity()).start) return task;
    const auto source = input.snapshot();
    const auto selected = model.snapshot();
    const auto draft = configuration.snapshot();
    task.modelIdentity = ModelSessionController::fileIdentity(selected.activeModel);
    const auto filter = configuration.classFilterStatus(selected, task.modelIdentity);
    if (!filter.error.isEmpty()) { task.error = filter.error; return task; }
    const auto files = source.sourceKind == SourceKind::Images ? input.range(range, false, &task.error) : QStringList{};
    if (!task.error.isEmpty()) return task;
    if (selected.activeModel.isEmpty()) { task.error = "请先导入 ONNX 或 PT 模型。"; return task; }
    if (source.sourceKind == SourceKind::Images && source.files.isEmpty()) {
        task.error = "请先导入 YOLO 数据集或选择视频／摄像头。"; return task;
    }
    if (draft.automaticExport && !QDir().mkpath(draft.exportDirectory)) {
        task.error = "无法创建自动导出目录：" + draft.exportDirectory; return task;
    }
    task.request.config = configuration.compose(selected, !model.convertedProfile(selected.activeModel).isEmpty());
    task.request.sourceKind = source.sourceKind;
    task.request.stereoView = source.sourceKind == SourceKind::Images ? StereoView::Full : draft.stereoView;
    task.request.cameraIndex = draft.cameraIndex;
    task.request.files = source.sourceKind == SourceKind::Video ? QStringList{source.streamPath} : files;
    task.first = files.isEmpty() ? 0 : source.files.indexOf(files.first()) + 1;
    task.last = files.isEmpty() ? 0 : source.files.indexOf(files.last()) + 1;
    for (const auto &path : files) task.sourceIdentities.insert(path, ModelSessionController::fileIdentity(path));
    task.fingerprint = WorkbenchConfiguration::configurationFingerprint(task.request.config, task.modelIdentity);
    task.output = {draft.automaticExport, draft.exportDirectory};
    task.valid = true;
    return task;
}
DetectionActivation OperationCoordinator::beginDetection(const PreparedDetectionTask &task, DetectionController &detection)
{
    if (!task.valid || detection.snapshot().running || !availability(activity()).start) return {};
    visibleTask_ = VisibleTask::Detection;
    detection.prepareTask(task.request, task.first, task.last, task.modelIdentity, task.fingerprint, task.sourceIdentities);
    return {true, detection.beginResultDelivery(task.output)};
}
StopTarget OperationCoordinator::stopWorkbench(InputSessionController &input, EvaluationController &evaluation,
    DetectionController &detection, RecordingController &recordings)
{
    input.noteUserInteraction();
    if (!activity().workbenchBusy()) return StopTarget::None;
    if (input.snapshot().importing) { input.cancel(); return StopTarget::Input; }
    if (evaluation.isBusy()) { evaluation.requestStop(); return StopTarget::Evaluation; }
    recordings.setDetectionState(detection.snapshot().running, input.snapshot().sourceKind, true, detection.snapshot().failed);
    detection.requestStop();
    return StopTarget::Detection;
}
void OperationCoordinator::addShutdownParticipant(const QString &name, QObject *owner,
    std::function<void()> request, std::function<bool()> ready)
{
    Q_ASSERT(!closing_);
    participants_.append(Participant{name, owner, std::move(request), std::move(ready)});
}
void OperationCoordinator::requestShutdown()
{
    if (closing_) return;
    closing_ = true; requesting_ = true; emit shutdownStarted();
    for (const auto &participant : participants_)
        if (participant.owner && participant.request) participant.request();
    requesting_ = false; timer_->start(); checkShutdown();
}
void OperationCoordinator::checkShutdown()
{
    if (!closing_ || requesting_ || ready_) return;
    for (const auto &participant : participants_)
        if (!participant.owner || !participant.ready || !participant.ready()) return;
    ready_ = true; timer_->stop(); emit shutdownReady();
}
}
