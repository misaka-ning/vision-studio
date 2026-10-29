// SPDX-License-Identifier: AGPL-3.0-only
#include "asyncimagepreview.h"

#include <QCoreApplication>
#include <QImageReader>
#include <QPointer>
#include <QThreadPool>
#include <QTimer>
#include <atomic>
#include <exception>
#include <mutex>
#include <optional>

namespace
{
constexpr int MaximumPreviewWorkers = 2;

struct PreviewCompletion
{
    quint64 sequence = 0;
    quint64 generation = 0;
    QString path;
    QImage original;
    QImage preview;
    bool primaryUsed = false;
    QString error;
};

struct PreviewMailbox
{
    std::atomic<bool> alive{true};
    std::atomic<quint64> latest{0};
    std::atomic<int> active{0};
    std::mutex mutex;
    std::optional<PreviewCompletion> completed;
};

QImage readPreviewSource(const QString &path, QString &error)
{
    if (path.isEmpty())
        return {};
    QImageReader reader(path);
    reader.setAutoTransform(true);
    QImage image = reader.read();
    if (image.isNull())
        error = reader.errorString();
    return image;
}
} // namespace

AsyncImagePreviewPool::AsyncImagePreviewPool(QObject *parent) : QObject(parent)
{
    pool_ = new QThreadPool(this);
    pool_->setObjectName("visionImagePreviewPool");
    pool_->setMaxThreadCount(MaximumPreviewWorkers);
    pool_->setExpiryTimeout(15000);
    poll_ = new QTimer(this);
    poll_->setInterval(16);
    connect(poll_, &QTimer::timeout, this, &AsyncImagePreviewPool::checkShutdown);
}

AsyncImagePreviewPool *AsyncImagePreview::applicationPool()
{
    static auto *pool = new AsyncImagePreviewPool(QCoreApplication::instance());
    return pool;
}

bool AsyncImagePreviewPool::tryStart(std::function<void()> task)
{
    return !shuttingDown_ && pool_->tryStart(std::move(task));
}

int AsyncImagePreviewPool::activeRequestCount() const { return pool_->activeThreadCount(); }
bool AsyncImagePreviewPool::isShutdownReady() const { return shuttingDown_ && activeRequestCount() == 0; }

void AsyncImagePreviewPool::requestShutdown()
{
    if (shuttingDown_) return;
    shuttingDown_ = true;
    emit shutdownStarted();
    poll_->start();
    QTimer::singleShot(0, this, &AsyncImagePreviewPool::checkShutdown);
}

void AsyncImagePreviewPool::checkShutdown()
{
    if (isShutdownReady() && !shutdownSignalled_) {
        shutdownSignalled_ = true;
        poll_->stop();
        emit shutdownReady();
    }
}

struct AsyncImagePreview::Private
{
    struct Request
    {
        quint64 sequence = 0;
        quint64 generation = 0;
        QString primary;
        QString fallback;
        vision::InputColorMode mode = vision::InputColorMode::Color;
    };
    std::shared_ptr<PreviewMailbox> mailbox = std::make_shared<PreviewMailbox>();
    QTimer *dispatch = nullptr;
    QTimer *poll = nullptr;
    quint64 sequence = 0;
    std::optional<Request> pending;
    bool shuttingDown = false, shutdownSignalled = false;
};

AsyncImagePreview::AsyncImagePreview(QObject *parent) : QObject(parent), d_(std::make_unique<Private>())
{
    d_->dispatch = new QTimer(this);
    d_->dispatch->setSingleShot(true);
    d_->poll = new QTimer(this);
    d_->poll->setInterval(16);
    connect(d_->dispatch, &QTimer::timeout, this, &AsyncImagePreview::startLatestRequest);
    connect(d_->poll, &QTimer::timeout, this, [this]
    {
        if (d_->shuttingDown) { checkShutdown(); return; }
        if (deliverCompletedRequest())
            startLatestRequest();
    });
    connect(applicationPool(), &AsyncImagePreviewPool::shutdownStarted, this, &AsyncImagePreview::requestShutdown);
    if (applicationPool()->shutdownRequested()) requestShutdown();
}

AsyncImagePreview::~AsyncImagePreview()
{
    d_->dispatch->stop();
    d_->poll->stop();
    d_->mailbox->alive.store(false, std::memory_order_release);
    d_->mailbox->latest.fetch_add(1, std::memory_order_acq_rel);
    // No join: workers retain the non-QObject mailbox, never this object.
}

void AsyncImagePreview::request(quint64 generation, const QString &primaryPath, const QString &fallbackPath,
                                vision::InputColorMode colorMode)
{
    if (d_->shuttingDown || applicationPool()->shutdownRequested()) return;
    const quint64 sequence = ++d_->sequence;
    d_->mailbox->latest.store(sequence, std::memory_order_release);
    {
        const std::lock_guard<std::mutex> lock(d_->mailbox->mutex);
        d_->mailbox->completed.reset();
    }
    // Repeated clicks replace this single pending record, never append jobs.
    d_->pending = Private::Request{sequence, generation, primaryPath, fallbackPath, colorMode};
    if (!d_->dispatch->isActive())
        d_->dispatch->start(0);
}

void AsyncImagePreview::cancel()
{
    d_->mailbox->latest.store(++d_->sequence, std::memory_order_release);
    d_->pending.reset();
    d_->dispatch->stop();
    d_->poll->stop();
    const std::lock_guard<std::mutex> lock(d_->mailbox->mutex);
    d_->mailbox->completed.reset();
    if (d_->shuttingDown) d_->poll->start();
}

void AsyncImagePreview::requestShutdown()
{
    if (d_->shuttingDown) return;
    d_->shuttingDown = true;
    cancel();
    QTimer::singleShot(0, this, &AsyncImagePreview::checkShutdown);
}

bool AsyncImagePreview::isShutdownReady() const
{
    return d_->shuttingDown && !d_->pending && activeRequestCount() == 0;
}

void AsyncImagePreview::checkShutdown()
{
    if (isShutdownReady() && !d_->shutdownSignalled) {
        d_->shutdownSignalled = true;
        d_->poll->stop();
        emit shutdownReady();
    }
}

int AsyncImagePreview::activeRequestCount() const
{
    return d_->mailbox->active.load(std::memory_order_acquire);
}

bool AsyncImagePreview::hasPendingRequest() const
{
    return d_->pending.has_value();
}

void AsyncImagePreview::startLatestRequest()
{
    if (d_->shuttingDown || applicationPool()->shutdownRequested()) { requestShutdown(); return; }
    const auto mailbox = d_->mailbox;
    bool started = false;
    if (d_->pending && mailbox->active.load(std::memory_order_acquire) < MaximumPreviewWorkers)
    {
        const auto request = *d_->pending;
        mailbox->active.fetch_add(1, std::memory_order_acq_rel);
        started = applicationPool()->tryStart([mailbox, request]
        {
            struct ActiveGuard
            {
                std::shared_ptr<PreviewMailbox> mailbox;
                ~ActiveGuard() { mailbox->active.fetch_sub(1, std::memory_order_acq_rel); }
            } activeGuard{mailbox};
            const auto current = [&]
            {
                return mailbox->alive.load(std::memory_order_acquire) &&
                       mailbox->latest.load(std::memory_order_acquire) == request.sequence;
            };
            if (!current())
                return;
            PreviewCompletion result;
            result.sequence = request.sequence;
            result.generation = request.generation;
            QString primaryError;
            try
            {
                result.path = request.primary;
                result.original = readPreviewSource(request.primary, primaryError);
                if (!current())
                    return;
                result.primaryUsed = !result.original.isNull();
                if (result.original.isNull() && !request.fallback.isEmpty() && request.fallback != request.primary)
                {
                    result.path = request.fallback;
                    result.original = readPreviewSource(request.fallback, result.error);
                    if (!current())
                        return;
                }
                if (result.original.isNull())
                    result.error = QStringLiteral("无法读取预览图片：") +
                                   (result.error.isEmpty() ? primaryError : result.error);
                else
                {
                    result.preview = vision::inputPreviewImage(result.original, request.mode);
                    result.error.clear();
                }
            }
            catch (const std::exception &error)
            {
                result.original = {};
                result.preview = {};
                result.error = QStringLiteral("无法生成图片预览：") + QString::fromUtf8(error.what());
            }
            catch (...)
            {
                result.original = {};
                result.preview = {};
                result.error = QStringLiteral("无法生成图片预览。");
            }
            if (!current())
                return;
            const std::lock_guard<std::mutex> lock(mailbox->mutex);
            if (current())
                mailbox->completed = std::move(result);
        });
        if (started)
            d_->pending.reset();
        else
            mailbox->active.fetch_sub(1, std::memory_order_acq_rel);
    }
    // started also covers a tiny image completing before tryStart returns.
    if (started || d_->pending || mailbox->active.load(std::memory_order_acquire) > 0)
    {
        if (!d_->poll->isActive())
            d_->poll->start();
    }
    else
        d_->poll->stop();
}

bool AsyncImagePreview::deliverCompletedRequest()
{
    const auto mailbox = d_->mailbox;
    std::optional<PreviewCompletion> result;
    {
        const std::lock_guard<std::mutex> lock(mailbox->mutex);
        result.swap(mailbox->completed);
    }
    if (!result || result->sequence != mailbox->latest.load(std::memory_order_acquire))
        return true;
    QPointer<AsyncImagePreview> guard(this);
    emit ready(result->generation, result->path, result->original, result->preview,
               result->primaryUsed, result->error);
    return !guard.isNull();
}
