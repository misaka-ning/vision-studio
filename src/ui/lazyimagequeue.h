// SPDX-License-Identifier: AGPL-3.0-only
#pragma once

#include "core/visiontypes.h"
#include <QListWidget>
#include <QPersistentModelIndex>
#include <memory>

// Queue entries contain source paths, never eagerly decoded full-size images.
// Only the viewport and a small neighbouring strip request thumbnails.
class LazyImageQueue final : public QListWidget
{
    Q_OBJECT
  public:
    explicit LazyImageQueue(QWidget *parent = nullptr);
    ~LazyImageQueue() override;

    void setColorMode(vision::InputColorMode mode);
    void scheduleVisibleThumbnails();
    void cancelPendingThumbnails();
    void resetThumbnails();
    void clear();

    int cachedThumbnailCount() const;
    int activeThumbnailJobs() const;

  signals:
    // Emitted after a successful real background decode is installed on the GUI thread.
    void thumbnailLoaded(const QString &path);

  protected:
    void resizeEvent(QResizeEvent *event) override;
    void showEvent(QShowEvent *event) override;
    void scrollContentsBy(int dx, int dy) override;

  private:
    struct Private;
    std::unique_ptr<Private> d_;

    void refreshVisibleThumbnails();
    bool collectCompletedThumbnails();
    void startThumbnailJobs();
    void displayCachedThumbnail(const QString &path, const QPersistentModelIndex &index);
    void discardThumbnailCache();
};
