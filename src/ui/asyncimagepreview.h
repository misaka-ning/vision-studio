// SPDX-License-Identifier: AGPL-3.0-only
#pragma once

#include "core/visiontypes.h"
#include <QObject>
#include <functional>
#include <memory>

class QThreadPool;
class QTimer;

// Application-owned gate also accounts for reads whose preview widget has
// already been destroyed. It never queues work or waits on the GUI thread.
class AsyncImagePreviewPool final : public QObject
{
    Q_OBJECT
  public:
    void requestShutdown();
    bool isShutdownReady() const;
    bool shutdownRequested() const { return shuttingDown_; }
    int activeRequestCount() const;

  signals:
    void shutdownStarted();
    void shutdownReady();

  private:
    friend class AsyncImagePreview;
    explicit AsyncImagePreviewPool(QObject *parent);
    bool tryStart(std::function<void()> task);
    void checkShutdown();
    QThreadPool *pool_ = nullptr;
    QTimer *poll_ = nullptr;
    bool shuttingDown_ = false, shutdownSignalled_ = false;
};

// Full-resolution selected-image decoding. All methods are called on the GUI
// thread; workers own only immutable requests and a shared result mailbox.
class AsyncImagePreview final : public QObject
{
    Q_OBJECT
  public:
    explicit AsyncImagePreview(QObject *parent = nullptr);
    ~AsyncImagePreview() override;

    void request(quint64 generation, const QString &primaryPath, const QString &fallbackPath,
                 vision::InputColorMode colorMode);
    void cancel();
    void requestShutdown();
    bool isShutdownReady() const;
    static AsyncImagePreviewPool *applicationPool();

    int activeRequestCount() const;
    bool hasPendingRequest() const;

  signals:
    // path is the file actually decoded (or the final attempted path on error).
    // primaryUsed distinguishes a valid snapshot from the original fallback.
    // Only the latest non-cancelled request can emit ready.
    void ready(quint64 generation, QString path, QImage original, QImage preview,
               bool primaryUsed, QString error);
    void shutdownReady();

  private:
    struct Private;
    std::unique_ptr<Private> d_;

    void startLatestRequest();
    bool deliverCompletedRequest();
    void checkShutdown();
};
