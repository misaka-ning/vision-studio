// SPDX-License-Identifier: AGPL-3.0-only
#include "lazyimagequeue.h"
#include "icons.h"

#include <QCoreApplication>
#include <QHash>
#include <QImageReader>
#include <QPersistentModelIndex>
#include <QPointer>
#include <QResizeEvent>
#include <QScrollBar>
#include <QSet>
#include <QShowEvent>
#include <QThreadPool>
#include <QTimer>
#include <algorithm>
#include <atomic>
#include <deque>
#include <list>
#include <mutex>

namespace
{
constexpr int MaximumCachedThumbnails = 128;
constexpr int MaximumVisibleRequests = 96;
constexpr int NeighbourRows = 2;
constexpr int MaximumActiveJobs = 2;

// This dedicated pool belongs to the application, not the widget. Destroying a
// queue never waits for a disk read. tryStart() below deliberately never queues
// QRunnables: at most two reads can be running and all pending requests remain
// cheap, replaceable GUI-side path/index records.
QThreadPool *thumbnailPool()
{
    static auto *pool = []
    {
        auto *result = new QThreadPool(QCoreApplication::instance());
        result->setObjectName("visionThumbnailPool");
        result->setMaxThreadCount(MaximumActiveJobs);
        result->setExpiryTimeout(15000);
        return result;
    }();
    return pool;
}

struct ThumbnailCompletion
{
    quint64 generation = 0;
    quint64 request = 0;
    QString path;
    QImage image;
};

struct ThumbnailMailbox
{
    std::atomic<bool> cancelled{false};
    std::mutex mutex;
    std::deque<ThumbnailCompletion> completed;
};

QImage readThumbnail(const QString &path, const QSize &target)
{
    QImageReader reader(path);
    reader.setAutoTransform(true);
    const QSize original = reader.size();
    if (original.isValid() && (original.width() > target.width() || original.height() > target.height()))
        reader.setScaledSize(original.scaled(target, Qt::KeepAspectRatio));
    QImage image = reader.read();
    // Some plugins do not support reduced decoding. Keep only the small result
    // after the background read, including after EXIF orientation is applied.
    if (!image.isNull() && (image.width() > target.width() || image.height() > target.height()))
        image = image.scaled(target, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    return image;
}
} // namespace

struct LazyImageQueue::Private
{
    struct Request
    {
        QPersistentModelIndex index;
        QString path;
    };
    struct Cached
    {
        QImage image;
        QList<QPersistentModelIndex> attached;
        std::list<QString>::iterator recency;
    };

    vision::InputColorMode colorMode = vision::InputColorMode::Color;
    QIcon placeholder = ui::icon("image");
    QTimer *dispatch = nullptr;
    QTimer *poll = nullptr;
    quint64 generation = 1;
    quint64 nextRequest = 1;
    std::shared_ptr<ThumbnailMailbox> mailbox = std::make_shared<ThumbnailMailbox>();
    QList<Request> pending;
    QHash<quint64, Request> active;
    QSet<QString> activeSources;
    QHash<QString, Cached> cache;
    std::list<QString> recency;
};

LazyImageQueue::LazyImageQueue(QWidget *parent) : QListWidget(parent), d_(std::make_unique<Private>())
{
    setUniformItemSizes(true);
    setLayoutMode(QListView::Batched);
    setBatchSize(128);
    d_->dispatch = new QTimer(this);
    d_->dispatch->setSingleShot(true);
    d_->poll = new QTimer(this);
    d_->poll->setInterval(16);
    connect(d_->dispatch, &QTimer::timeout, this, &LazyImageQueue::refreshVisibleThumbnails);
    connect(d_->poll, &QTimer::timeout, this, [this]
    {
        if (collectCompletedThumbnails())
            startThumbnailJobs();
    });
    connect(horizontalScrollBar(), &QScrollBar::valueChanged, this, &LazyImageQueue::scheduleVisibleThumbnails);
    connect(verticalScrollBar(), &QScrollBar::valueChanged, this, &LazyImageQueue::scheduleVisibleThumbnails);
    connect(model(), &QAbstractItemModel::rowsInserted, this, &LazyImageQueue::scheduleVisibleThumbnails);
    connect(model(), &QAbstractItemModel::rowsRemoved, this, &LazyImageQueue::scheduleVisibleThumbnails);
    connect(model(), &QAbstractItemModel::modelAboutToBeReset, this, &LazyImageQueue::resetThumbnails);
    connect(model(), &QAbstractItemModel::layoutChanged, this, &LazyImageQueue::scheduleVisibleThumbnails);
    connect(model(), &QAbstractItemModel::dataChanged, this,
            [this](const QModelIndex &, const QModelIndex &, const QList<int> &roles)
    {
        if (roles.isEmpty() || roles.contains(Qt::UserRole))
            scheduleVisibleThumbnails();
    });
    connect(this, &QAbstractItemView::iconSizeChanged, this, &LazyImageQueue::resetThumbnails);
}

LazyImageQueue::~LazyImageQueue()
{
    d_->dispatch->stop();
    d_->poll->stop();
    d_->mailbox->cancelled.store(true, std::memory_order_release);
    // QListWidget can reset its model while its base destructor runs, after
    // d_ has already been destroyed. Remove those incoming connections first.
    disconnect(model(), nullptr, this, nullptr);
    disconnect(horizontalScrollBar(), nullptr, this, nullptr);
    disconnect(verticalScrollBar(), nullptr, this, nullptr);
    disconnect(this, nullptr, this, nullptr);
    // Background reads own only the mailbox and immutable values, no QWidget,
    // QModelIndex or raw this pointer. Their late results can be discarded.
}

void LazyImageQueue::setColorMode(vision::InputColorMode mode)
{
    if (d_->colorMode == mode)
        return;
    d_->colorMode = mode;
    // At most 128 tiny cached images are transformed. Never read source files
    // or walk the full dataset when toggling colour/grayscale.
    const auto paths = d_->cache.keys();
    for (const QString &path : paths)
    {
        const auto attached = d_->cache.value(path).attached;
        for (const auto &index : attached)
            displayCachedThumbnail(path, index);
    }
    scheduleVisibleThumbnails();
}

void LazyImageQueue::scheduleVisibleThumbnails()
{
    if (!d_->dispatch->isActive())
        d_->dispatch->start(0);
}

void LazyImageQueue::cancelPendingThumbnails()
{
    d_->dispatch->stop();
    d_->poll->stop();
    d_->mailbox->cancelled.store(true, std::memory_order_release);
    d_->mailbox = std::make_shared<ThumbnailMailbox>();
    ++d_->generation;
    d_->pending.clear();
    d_->active.clear();
    d_->activeSources.clear();
}

void LazyImageQueue::discardThumbnailCache()
{
    // Only entries that actually received a cached thumbnail need clearing.
    // Persistent indices cannot refer to a deleted row or another model.
    for (auto it = d_->cache.cbegin(); it != d_->cache.cend(); ++it)
        for (const auto &index : it->attached)
            if (index.isValid() && index.model() == model() && index.data(Qt::UserRole).toString() == it.key())
                if (auto *entry = item(index.row()))
                {
                    entry->setData(Qt::UserRole + 1, QVariant());
                    entry->setIcon(d_->placeholder);
                }
    d_->cache.clear();
    d_->recency.clear();
}

void LazyImageQueue::resetThumbnails()
{
    cancelPendingThumbnails();
    discardThumbnailCache();
    scheduleVisibleThumbnails();
}

void LazyImageQueue::clear()
{
    cancelPendingThumbnails();
    discardThumbnailCache();
    QListWidget::clear();
}

int LazyImageQueue::cachedThumbnailCount() const
{
    return d_->cache.size();
}

int LazyImageQueue::activeThumbnailJobs() const
{
    return d_->active.size();
}

void LazyImageQueue::displayCachedThumbnail(const QString &path, const QPersistentModelIndex &index)
{
    if (!index.isValid() || index.model() != model() || index.data(Qt::UserRole).toString() != path)
        return;
    auto found = d_->cache.find(path);
    if (found == d_->cache.end())
        return;
    auto *entry = item(index.row());
    if (!entry)
        return;
    d_->recency.splice(d_->recency.begin(), d_->recency, found->recency);
    if (!found->attached.contains(index))
        found->attached.append(index);
    if (found->image.isNull())
    {
        entry->setData(Qt::UserRole + 1, QVariant());
        entry->setIcon(d_->placeholder);
    }
    else
    {
        entry->setData(Qt::UserRole + 1, found->image);
        entry->setIcon(QPixmap::fromImage(vision::inputPreviewImage(found->image, d_->colorMode)));
    }
}

void LazyImageQueue::refreshVisibleThumbnails()
{
    if (!collectCompletedThumbnails())
        return;
    d_->pending.clear();
    const QRect visible = viewport()->rect();
    if (!isVisible() || !visible.isValid() || count() == 0)
    {
        startThumbnailJobs();
        return;
    }

    // Probe viewport cells, not all rows. The half-cell steps also catch
    // partially visible edge tiles. A hard bound handles pathological grids.
    const QSize grid = gridSize().isValid() ? gridSize() : QSize(100, 64);
    const int stepX = std::max(8, grid.width() / 2);
    const int stepY = std::max(8, grid.height() / 2);
    QSet<int> visibleRows;
    int probes = 0;
    for (int y = 0; y <= visible.bottom() && probes < 1024; y += stepY)
        for (int x = 0; x <= visible.right() && probes < 1024; x += stepX, ++probes)
        {
            const auto index = indexAt(QPoint(x, y));
            if (index.isValid())
                visibleRows.insert(index.row());
        }
    const auto edge = indexAt(visible.bottomRight());
    if (edge.isValid())
        visibleRows.insert(edge.row());
    auto orderedRows = visibleRows.values();
    std::sort(orderedRows.begin(), orderedRows.end());
    QList<int> wantedRows = orderedRows;
    QSet<int> wanted = visibleRows;
    for (int row : orderedRows)
        for (int distance = 1; distance <= NeighbourRows; ++distance)
            for (int adjacent : {row - distance, row + distance})
                if (adjacent >= 0 && adjacent < count() && !wanted.contains(adjacent))
                {
                    wanted.insert(adjacent);
                    wantedRows.append(adjacent);
                }
    QSet<QString> queuedSources;
    int requested = 0;
    for (int row : wantedRows)
    {
        if (requested >= MaximumVisibleRequests)
            break;
        auto *entry = item(row);
        if (!entry || entry->isHidden())
            continue;
        const QString path = entry->data(Qt::UserRole).toString();
        if (path.isEmpty())
            continue;
        ++requested;
        const QPersistentModelIndex index = indexFromItem(entry);
        if (d_->cache.contains(path))
            displayCachedThumbnail(path, index);
        else if (!d_->activeSources.contains(path) && !queuedSources.contains(path))
        {
            d_->pending.append({index, path});
            queuedSources.insert(path);
        }
    }
    startThumbnailJobs();
}

bool LazyImageQueue::collectCompletedThumbnails()
{
    std::deque<ThumbnailCompletion> completed;
    {
        const std::lock_guard<std::mutex> lock(d_->mailbox->mutex);
        completed.swap(d_->mailbox->completed);
    }
    QStringList loaded;
    for (auto &result : completed)
    {
        if (result.generation != d_->generation)
            continue;
        const auto found = d_->active.find(result.request);
        if (found == d_->active.end())
            continue;
        const auto request = found.value();
        d_->active.erase(found);
        d_->activeSources.remove(request.path);
        if (result.path != request.path || !request.index.isValid() || request.index.model() != model() ||
            request.index.data(Qt::UserRole).toString() != request.path)
            continue;
        if (!d_->cache.contains(result.path))
        {
            while (d_->cache.size() >= MaximumCachedThumbnails && !d_->recency.empty())
            {
                const QString oldest = d_->recency.back();
                auto evicted = d_->cache.find(oldest);
                if (evicted != d_->cache.end())
                {
                    for (const auto &index : evicted->attached)
                        if (index.isValid() && index.model() == model() && index.data(Qt::UserRole).toString() == oldest)
                            if (auto *entry = item(index.row()))
                            {
                                entry->setData(Qt::UserRole + 1, QVariant());
                                entry->setIcon(d_->placeholder);
                            }
                    d_->cache.erase(evicted);
                }
                d_->recency.pop_back();
            }
            d_->recency.push_front(result.path);
            d_->cache.insert(result.path, {result.image, {}, d_->recency.begin()});
        }
        displayCachedThumbnail(result.path, request.index);
        if (!result.image.isNull())
            loaded.append(result.path);
    }
    if (!loaded.isEmpty())
        scheduleVisibleThumbnails();
    QPointer<LazyImageQueue> guard(this);
    for (const auto &path : loaded)
    {
        emit thumbnailLoaded(path);
        if (!guard)
            return false;
    }
    return true;
}

void LazyImageQueue::startThumbnailJobs()
{
    const qreal ratio = devicePixelRatioF();
    const QSize target(qBound(64, qRound(iconSize().width() * ratio), 512),
                       qBound(46, qRound(iconSize().height() * ratio), 512));
    while (!d_->pending.isEmpty() && d_->active.size() < MaximumActiveJobs)
    {
        const auto request = d_->pending.front();
        if (!request.index.isValid() || request.index.model() != model() ||
            request.index.data(Qt::UserRole).toString() != request.path || d_->cache.contains(request.path))
        {
            d_->pending.removeFirst();
            continue;
        }
        const auto mailbox = d_->mailbox;
        const quint64 generation = d_->generation;
        const quint64 requestId = d_->nextRequest++;
        const QString path = request.path;
        // Capture only immutable values and a shared non-QObject mailbox.
        if (!thumbnailPool()->tryStart([mailbox, generation, requestId, path, target]
        {
            if (mailbox->cancelled.load(std::memory_order_acquire))
                return;
            QImage image = readThumbnail(path, target);
            if (mailbox->cancelled.load(std::memory_order_acquire))
                return;
            const std::lock_guard<std::mutex> lock(mailbox->mutex);
            if (!mailbox->cancelled.load(std::memory_order_acquire))
                mailbox->completed.push_back({generation, requestId, path, std::move(image)});
        }))
            break;
        d_->active.insert(requestId, request);
        d_->activeSources.insert(path);
        d_->pending.removeFirst();
    }
    if (d_->pending.isEmpty() && d_->active.isEmpty())
        d_->poll->stop();
    else if (!d_->poll->isActive())
        d_->poll->start();
}

void LazyImageQueue::resizeEvent(QResizeEvent *event)
{
    QListWidget::resizeEvent(event);
    scheduleVisibleThumbnails();
}

void LazyImageQueue::showEvent(QShowEvent *event)
{
    QListWidget::showEvent(event);
    scheduleVisibleThumbnails();
}

void LazyImageQueue::scrollContentsBy(int dx, int dy)
{
    QListWidget::scrollContentsBy(dx, dy);
    scheduleVisibleThumbnails();
}
