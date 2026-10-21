// SPDX-License-Identifier: AGPL-3.0-only
#pragma once

#include "core/visiontypes.h"
#include <QObject>
#include <memory>

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

    int activeRequestCount() const;
    bool hasPendingRequest() const;

  signals:
    // path is the file actually decoded (or the final attempted path on error).
    // primaryUsed distinguishes a valid snapshot from the original fallback.
    // Only the latest non-cancelled request can emit ready.
    void ready(quint64 generation, QString path, QImage original, QImage preview,
               bool primaryUsed, QString error);

  private:
    struct Private;
    std::unique_ptr<Private> d_;

    void startLatestRequest();
    bool deliverCompletedRequest();
};
