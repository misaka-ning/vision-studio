// SPDX-License-Identifier: AGPL-3.0-only
#pragma once

#include <QCoreApplication>
#include <QHash>
#include <QObject>
#include <QPointer>
#include <QThread>
#include <QThreadPool>
#include <QTimer>
#include <atomic>
#include <functional>
#include <memory>
#include <thread>
#include <system_error>
#include <utility>

namespace vision::app
{
// QThread::finished/isFinished include Qt's finishing phase. Deferred deletes
// and thread-local destructors can still run then. A watched thread is owned by
// a self-retaining ticket until a background native wait has really completed.
// This is header-only so standalone QtCore runtime setup needs no app/UI link.
class AsyncThreadRetirement final
{
    struct State { std::atomic_bool joined{false}; };
    static QHash<QThread *, QPointer<QObject>> &registry()
    {
        static QHash<QThread *, QPointer<QObject>> watched;
        return watched;
    }
    class Ticket final : public QObject
    {
        QThread *thread_;
        QPointer<QObject> owner_;
        std::function<void()> callback_;
        std::shared_ptr<State> state_ = std::make_shared<State>();
        QTimer timer_;
        void check()
        {
            if (!state_->joined.load(std::memory_order_acquire)) return;
            timer_.stop();
            auto *thread = std::exchange(thread_, nullptr);
            // First release the owner's maps/pointers: QObject::destroyed may
            // synchronously reenter that owner during QThread deletion. Native
            // cleanup is already complete, but the application-level ticket
            // remains active throughout the callback and object destruction.
            const auto callback = std::move(callback_);
            if (owner_ && callback) callback();
            delete thread;
            registry().remove(thread);
            deleteLater();
        }
      public:
        Ticket(QThread *thread, QObject *owner, std::function<void()> callback)
            : thread_(thread), owner_(owner), callback_(std::move(callback))
        {
            setObjectName("asyncThreadRetirementTicket");
            thread_->setParent(nullptr);
            registry().insert(thread_, owner);
            timer_.setInterval(2);
            connect(&timer_, &QTimer::timeout, this, [this] { check(); });
            connect(thread_, &QThread::finished, thread_, [thread, state = state_] {
                const auto reap = [thread, state] {
                    thread->wait(); // Only this detached background reaper waits.
                    state->joined.store(true, std::memory_order_release);
                };
                try {
                    std::thread(reap).detach();
                } catch (const std::system_error &) {
                    // No exception may cross Qt's finished notification. The
                    // existing pool is a resource-pressure scheduling fallback.
                    QThreadPool::globalInstance()->start(reap);
                }
            }, Qt::DirectConnection);
            connect(thread_, &QThread::finished, this, [this] {
                timer_.start();
                check();
            }, Qt::QueuedConnection);
        }
    };
  public:
    // Register on the owner's event loop before the first start(). Ownership
    // includes safe reaping if owner is destroyed; only its callback is skipped.
    // Callback runs after native join but before safe QThread deletion, so owner
    // clears its pointers/maps before any reentrant QObject::destroyed observer.
    // Callers must remove any finished->thread.deleteLater/delete connection.
    static void watch(QThread *thread, QObject *owner, std::function<void()> afterFullyJoined)
    {
        Q_ASSERT(thread && owner && QCoreApplication::instance());
        Q_ASSERT(QThread::currentThread() == owner->thread() && thread->thread() == owner->thread());
        Q_ASSERT(!thread->isRunning() && !thread->isFinished() && !registry().contains(thread));
        new Ticket(thread, owner, std::move(afterFullyJoined));
    }
    // Read-only observations. Like registration, query on the application loop.
    static QList<QThread *> watchedThreads(const QObject *owner = nullptr)
    {
        QList<QThread *> result;
        for (auto it = registry().cbegin(); it != registry().cend(); ++it)
            if (!owner || it.value() == owner) result.append(it.key());
        return result;
    }
    static int activeCount() { return registry().size(); }
};
}
