// SPDX-License-Identifier: AGPL-3.0-only
#include "gpuenvironmentservice.h"

#include "core/cpuruntime.h"
#include "core/gpuruntime.h"
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QProcess>
#include <QProcessEnvironment>
#include <QStandardPaths>
#include <QTimer>
#include <utility>

namespace vision::app
{
namespace
{
void terminateAsync(QProcess *process, int killDelayMs)
{
    const QPointer<QProcess> target(process);
    process->terminate();
    QTimer::singleShot(killDelayMs, process, [target] {
        if (target && target->state() != QProcess::NotRunning)
            target->kill();
    });
}
} // namespace

GpuEnvironmentService::GpuEnvironmentService(const QString &projectRoot, QObject *parent)
    : GpuEnvironmentService(Options{}, parent)
{
    options_.projectRoot = projectRoot;
}

GpuEnvironmentService::GpuEnvironmentService(Options options, QObject *parent)
    : QObject(parent), options_(std::move(options))
{
    options_.probeTimeoutMs = qMax(1, options_.probeTimeoutMs);
    options_.preparationKillDelayMs = qMax(1, options_.preparationKillDelayMs);
    options_.shutdownKillDelayMs = qMax(1, options_.shutdownKillDelayMs);
}

GpuEnvironmentService::~GpuEnvironmentService()
{
    // Normal destruction follows shutdownReady. If an owner disappears early,
    // do not make QProcess's destructor wait on the GUI thread: detached helpers
    // retain their own terminate/kill/reap callbacks until they have exited.
    for (QProcess *process : std::as_const(processes_)) {
        process->disconnect(this);
        process->setParent(nullptr);
        connect(process, qOverload<int, QProcess::ExitStatus>(&QProcess::finished), process,
                &QObject::deleteLater);
        connect(process, &QProcess::errorOccurred, process, [process](QProcess::ProcessError error) {
            if (error == QProcess::FailedToStart) process->deleteLater();
        });
        if (process->state() == QProcess::NotRunning) process->deleteLater();
        else terminateAsync(process, options_.shutdownKillDelayMs);
    }
}

QString GpuEnvironmentService::runtimeDirectory() const
{
    return options_.runtimeDirectory.isEmpty() ? vision::gpuRuntimeDirectory() : options_.runtimeDirectory;
}

bool GpuEnvironmentService::isChecking() const { return !probeProcess_.isNull(); }
bool GpuEnvironmentService::isPreparing() const { return !setupProcess_.isNull(); }
bool GpuEnvironmentService::isBusy() const { return isChecking() || isPreparing(); }

QString GpuEnvironmentService::launcherPython() const
{
    if (!options_.launcherPython.isEmpty()) return options_.launcherPython;
#ifdef Q_OS_WIN
    return vision::cpuPython();
#else
    const QStringList candidates{qEnvironmentVariable("VISION_STUDIO_PYTHON"),
                                 options_.projectRoot + "/runtime/bin/python", "/usr/bin/python3.10",
                                 QStandardPaths::findExecutable("python3")};
    for (const auto &candidate : candidates)
        if (!candidate.isEmpty() && QFileInfo(candidate).isExecutable()) return candidate;
    return {};
#endif
}

QString GpuEnvironmentService::scriptPath(bool setup) const
{
    const QString configured = setup ? options_.setupScript : options_.probeScript;
    return configured.isEmpty() ? vision::gpuScriptPath(setup ? "gpu_setup.py" : "gpu_probe.py") : configured;
}

QProcess *GpuEnvironmentService::createProcess(const QString &name)
{
    auto *process = new QProcess(this);
    process->setObjectName(name);
    processes_.insert(process);
    connect(process, &QObject::destroyed, this, [this, process] {
        processes_.remove(process);
        maybeFinishShutdown();
    });
    return process;
}

void GpuEnvironmentService::setStatus(const QString &status)
{
    status_ = status;
    emit statusChanged(status_);
}

void GpuEnvironmentService::appendLog(const QString &message)
{
    if (!message.trimmed().isEmpty()) emit logAppended(message.left(16000).trimmed());
}

void GpuEnvironmentService::publishState()
{
    const bool busy = isBusy();
    emit stateChanged();
    if (busy != lastBusy_) {
        lastBusy_ = busy;
        emit busyChanged(busy);
    }
}

bool GpuEnvironmentService::checkEnvironment()
{
    if (shuttingDown_ || isBusy()) return false;
    const QString python = launcherPython(), script = scriptPath(false);
    if (python.isEmpty() || !QFileInfo(script).isFile()) {
        probeKnown_ = true;
        environment_ = {{"ok", false}, {"reason", "未找到 GPU 检查工具或 Python；CPU 仍可使用。"}};
        setStatus(environment_.value("reason").toString());
        emit environmentChanged(environment_);
        publishState();
        return false;
    }
    auto *process = createProcess("gpuProbeProcess");
    probeProcess_ = process;
    probeOutput_.clear(); probeErrors_.clear();
    setStatus("正在检查显卡与本地 GPU 推理环境…");
    publishState();
    if (shuttingDown_) {
        probeProcess_ = nullptr;
        process->deleteLater();
        publishState();
        return false;
    }
    connect(process, &QProcess::readyReadStandardOutput, this, [this, process] {
        if (probeProcess_ != process) return;
        probeOutput_ += process->readAllStandardOutput();
        if (probeOutput_.size() > 512 * 1024) process->kill();
    });
    connect(process, &QProcess::readyReadStandardError, this, [this, process] {
        if (probeProcess_ == process)
            probeErrors_ = (probeErrors_ + process->readAllStandardError()).right(16000);
    });
    connect(process, qOverload<int, QProcess::ExitStatus>(&QProcess::finished), this,
            [this, process](int code, QProcess::ExitStatus status) {
                finishProbe(process, code == 0 && status == QProcess::NormalExit);
            });
    connect(process, &QProcess::errorOccurred, this, [this, process](QProcess::ProcessError error) {
        if (error == QProcess::FailedToStart) finishProbe(process, false);
    });
    auto environment = QProcessEnvironment::systemEnvironment();
    environment.insert("PYTHONDONTWRITEBYTECODE", "1");
    process->setProcessEnvironment(environment);
    process->start(python, {script, "--runtime-dir", runtimeDirectory()});
    const QPointer<QProcess> target(process);
    QTimer::singleShot(options_.probeTimeoutMs, this, [this, target] {
        if (target && probeProcess_ == target && target->state() != QProcess::NotRunning) {
            probeErrors_ = "检查超时，请重试；CPU 仍可使用。";
            target->kill();
        }
    });
    return true;
}

void GpuEnvironmentService::finishProbe(QProcess *process, bool success)
{
    if (probeProcess_ != process) return;
    probeOutput_ += process->readAllStandardOutput();
    const auto response = QJsonDocument::fromJson(probeOutput_).object();
    probeProcess_ = nullptr;
    probeKnown_ = true;
    environment_ = success && response.contains("cuda_available")
        ? response : QJsonObject{{"ok", false}, {"reason", QString("GPU 环境检查未完成：%1")
            .arg(probeErrors_.isEmpty() ? process->errorString() : QString::fromUtf8(probeErrors_).left(500))}};
    const bool ready = environment_.value("ok").toBool() && environment_.value("prepared").toBool() &&
                       environment_.value("cuda_available").toBool();
    QStringList hardware;
    for (const auto &value : environment_.value("devices").toArray()) {
        const auto device = value.toObject();
        hardware << QString("GPU %1 · %2 · %3 GB").arg(device.value("index").toInt())
            .arg(device.value("name").toString().left(120))
            .arg(device.value("total_memory_mb").toDouble() / 1024, 0, 'f', 1);
    }
    const QString reason = environment_.value("reason").toString();
    setStatus((ready ? QString("GPU 推理环境已就绪") : QString("GPU 推理环境尚未就绪")) +
              (hardware.isEmpty() ? QString("\n未检测到 NVIDIA 显卡") : "\n" + hardware.join("\n")) +
              (reason.isEmpty() ? QString() : "\n" + reason));
    emit environmentChanged(environment_);
    process->deleteLater();
    publishState();
}

bool GpuEnvironmentService::prepareEnvironment()
{
    if (shuttingDown_ || isBusy()) return false;
    const QString python = launcherPython(), script = scriptPath(true);
    if (python.isEmpty() || !QFileInfo(script).isFile()) {
        setStatus("未找到 GPU 准备工具或 Python。CPU 仍可使用。");
        return false;
    }
    auto *process = createProcess("gpuSetupProcess");
    setupProcess_ = process; setupCancelling_ = false; setupOutput_.clear();
    emit logReset();
    appendLog("GPU 依赖将安装到：" + runtimeDirectory());
    setStatus("正在准备 GPU 支持。首次下载约 4 GB，耗时取决于网络；CPU 环境和显卡驱动保持原样。");
    emit progressChanged(0, 0, 0, true);
    publishState();
    if (shuttingDown_ || setupCancelling_) {
        finishSetup(process, false);
        return false;
    }
    connect(process, &QProcess::readyReadStandardOutput, this, &GpuEnvironmentService::readSetupOutput);
    connect(process, &QProcess::readyReadStandardError, this, [this, process] {
        if (setupProcess_ == process) appendLog(QString::fromUtf8(process->readAllStandardError()));
    });
    connect(process, qOverload<int, QProcess::ExitStatus>(&QProcess::finished), this,
            [this, process](int code, QProcess::ExitStatus status) {
                finishSetup(process, code == 0 && status == QProcess::NormalExit);
            });
    connect(process, &QProcess::errorOccurred, this, [this, process](QProcess::ProcessError error) {
        if (error == QProcess::FailedToStart) finishSetup(process, false);
    });
    auto environment = QProcessEnvironment::systemEnvironment();
    environment.insert("PYTHONUNBUFFERED", "1");
    environment.insert("PYTHONDONTWRITEBYTECODE", "1");
    process->setProcessEnvironment(environment);
    process->start(python, {"-u", script, "--runtime-dir", runtimeDirectory()});
    return true;
}

void GpuEnvironmentService::readSetupOutput()
{
    if (!setupProcess_) return;
    setupOutput_ += setupProcess_->readAllStandardOutput();
    while (setupOutput_.contains('\n')) {
        const int newline = setupOutput_.indexOf('\n');
        const auto line = setupOutput_.left(newline).trimmed();
        setupOutput_.remove(0, newline + 1);
        const auto event = QJsonDocument::fromJson(line).object();
        if (event.isEmpty()) appendLog(QString::fromUtf8(line));
        else {
            const QString message = event.value("message").toString();
            appendLog(message.isEmpty() ? event.value("event").toString() : message);
            if (!message.isEmpty()) setStatus(message.left(600));
            const auto percent = event.value("percent").isDouble() ? event.value("percent") : event.value("progress");
            if (percent.isDouble()) emit progressChanged(0, 100, qBound(0, percent.toInt(), 100), true);
        }
    }
    if (setupOutput_.size() > 256 * 1024) {
        appendLog(QString::fromUtf8(setupOutput_.left(16000)));
        setupOutput_.clear();
    }
}

void GpuEnvironmentService::finishSetup(QProcess *process, bool success)
{
    if (setupProcess_ != process) return;
    readSetupOutput();
    if (!setupOutput_.trimmed().isEmpty()) appendLog(QString::fromUtf8(setupOutput_));
    const bool cancelled = setupCancelling_;
    setupProcess_ = nullptr; setupCancelling_ = false;
    emit progressChanged(0, 100, 0, false);
    const QString message = cancelled ? QStringLiteral("GPU 准备已取消，CPU 仍可使用。") : success
        ? QStringLiteral("GPU 准备完成，正在重新检查环境…")
        : QStringLiteral("GPU 准备未完成。请查看日志后重试；CPU 仍可使用。");
    setStatus(message); appendLog(message);
    process->deleteLater();
    publishState();
    emit preparationFinished(success && !cancelled, cancelled);
    if (success && !cancelled && !shuttingDown_)
        QTimer::singleShot(0, this, [this] { checkEnvironment(); });
}

void GpuEnvironmentService::cancelPreparation()
{
    if (!setupProcess_ || setupCancelling_) return;
    setupCancelling_ = true;
    setStatus("正在取消本窗口启动的 GPU 准备任务…");
    appendLog("正在取消 GPU 准备任务。");
    terminateAsync(setupProcess_, options_.preparationKillDelayMs);
    publishState();
}

void GpuEnvironmentService::requestShutdown()
{
    if (shuttingDown_) return;
    shuttingDown_ = true;
    if (setupProcess_) cancelPreparation();
    if (probeProcess_) terminateAsync(probeProcess_, options_.shutdownKillDelayMs);
    publishState();
    // Also await deleteLater of already finished helpers. A queued completion
    // keeps an idle shutdown request from reentering its owner's close handler.
    QTimer::singleShot(0, this, &GpuEnvironmentService::maybeFinishShutdown);
}

void GpuEnvironmentService::maybeFinishShutdown()
{
    if (isShutdownReady() && !shutdownSignalled_) {
        shutdownSignalled_ = true;
        emit shutdownReady();
    }
}
} // namespace vision::app
