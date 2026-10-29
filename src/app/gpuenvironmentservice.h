// SPDX-License-Identifier: AGPL-3.0-only
#pragma once

#include <QJsonObject>
#include <QObject>
#include <QPointer>
#include <QSet>
#include <QString>

class QProcess;

namespace vision::app
{
// Owns only the GPU environment helper processes. Inference device selection
// remains in VisionEngine; a successful probe is not proof of CUDA inference.
// Call on the owning event-loop thread and keep alive until shutdownReady.
class GpuEnvironmentService final : public QObject
{
    Q_OBJECT
  public:
    struct Options
    {
        QString projectRoot;
        // Optional launch seams for hermetic helper-process tests. Empty values
        // retain the application's existing runtime/script discovery rules.
        QString launcherPython;
        QString probeScript;
        QString setupScript;
        QString runtimeDirectory;
        int probeTimeoutMs = 45000;
        int preparationKillDelayMs = 5000;
        int shutdownKillDelayMs = 5000;
    };

    explicit GpuEnvironmentService(const QString &projectRoot, QObject *parent = nullptr);
    explicit GpuEnvironmentService(Options options, QObject *parent = nullptr);
    ~GpuEnvironmentService() override;

    QJsonObject environment() const { return environment_; }
    bool probeKnown() const { return probeKnown_; }
    QString status() const { return status_; }
    QString runtimeDirectory() const;
    bool isChecking() const;
    bool isPreparing() const;
    bool isCancelling() const { return setupCancelling_; }
    bool isBusy() const;
    bool shutdownRequested() const { return shuttingDown_; }
    bool isShutdownReady() const { return shuttingDown_ && processes_.isEmpty(); }

    // The caller handles task permissions and any CPU-runtime download UI.
    // False means no helper was started (busy, shutting down, or unavailable).
    bool checkEnvironment();
    bool prepareEnvironment();
    void cancelPreparation();
    void requestShutdown();

  signals:
    void environmentChanged(const QJsonObject &environment);
    void statusChanged(const QString &status);
    void logReset();
    void logAppended(const QString &message);
    void progressChanged(int minimum, int maximum, int value, bool visible);
    void stateChanged();
    void busyChanged(bool busy);
    void preparationFinished(bool success, bool cancelled);
    void shutdownReady();

  private:
    QString launcherPython() const;
    QString scriptPath(bool setup) const;
    QProcess *createProcess(const QString &name);
    void setStatus(const QString &status);
    void appendLog(const QString &message);
    void publishState();
    void finishProbe(QProcess *process, bool success);
    void readSetupOutput();
    void finishSetup(QProcess *process, bool success);
    void maybeFinishShutdown();

    Options options_;
    QPointer<QProcess> probeProcess_, setupProcess_;
    QSet<QProcess *> processes_;
    QJsonObject environment_;
    QByteArray probeOutput_, probeErrors_, setupOutput_;
    QString status_;
    bool probeKnown_ = false, setupCancelling_ = false;
    bool shuttingDown_ = false, shutdownSignalled_ = false, lastBusy_ = false;
};
} // namespace vision::app
