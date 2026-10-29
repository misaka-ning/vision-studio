// SPDX-License-Identifier: AGPL-3.0-only
#pragma once
#include <QObject>
#include <QSemaphore>
#include <atomic>
#include <memory>

namespace vision::test
{
struct ThreadTailState
{
    std::atomic_bool entered{false}, completed{false}, released{false};
    QSemaphore release;
    void resume() { if (!released.exchange(true)) release.release(); }
    void finish() { entered.store(true); release.acquire(); completed.store(true); }
};
// Also releases on a failed test assertion, so a diagnostic never strands a
// native worker or its background retirement task.
struct ThreadTailRelease
{
    std::shared_ptr<ThreadTailState> state;
    ~ThreadTailRelease() { state->resume(); }
};
struct ThreadLocalTail
{
    std::shared_ptr<ThreadTailState> state;
    ~ThreadLocalTail() { state->finish(); }
};
inline void establishThreadLocalTail(const std::shared_ptr<ThreadTailState> &state)
{
    thread_local ThreadLocalTail tail{state};
    (void)tail.state;
}
class DeferredDeleteTail final : public QObject
{
    std::shared_ptr<ThreadTailState> state_;
  public:
    explicit DeferredDeleteTail(std::shared_ptr<ThreadTailState> state) : state_(std::move(state)) {}
    ~DeferredDeleteTail() override { state_->finish(); }
};
}
