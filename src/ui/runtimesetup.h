// SPDX-License-Identifier: AGPL-3.0-only
#pragma once
#include <QObject>
#include <functional>
class QWidget;
namespace ui {
class RuntimeResources;

// Application shutdown must keep the event loop running until shutdownReady().
// This gate also covers preparations whose dialog has already been destroyed.
class RuntimeSetup final : public QObject
{
    Q_OBJECT
public:
    ~RuntimeSetup() override;
    bool isShutdownReady() const;

public slots:
    void cancelPendingPreparations();
    void requestShutdown();

signals:
    void shutdownReady();

private:
    explicit RuntimeSetup(QObject *parent);
    void trackPreparation(QObject *preparation);
    void maybeEmitShutdownReady();
    bool shuttingDown_ = false;
    bool shutdownEmitted_ = false;
    int preparations_ = 0;
    friend class RuntimeResources;
    friend RuntimeSetup *runtimeSetup();
    friend bool ensureCpuRuntime(QWidget *, std::function<void()>);
};

RuntimeSetup *runtimeSetup();

// Returns true when the caller can continue now. Otherwise queues a retry that
// runs only after the shared, explicitly started preparation succeeds.
bool ensureCpuRuntime(QWidget *parent, std::function<void()> retry);
}
