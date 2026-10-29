// SPDX-License-Identifier: AGPL-3.0-only
#pragma once
#include <QWidget>
class QLabel;
class ModelViewer;
class ModelDisplayPage final : public QWidget
{
    Q_OBJECT
  public:
    struct Controls { QLabel *modelName = nullptr, *metadata = nullptr, *hint = nullptr, *status = nullptr; ModelViewer *viewer = nullptr; };
    explicit ModelDisplayPage(QWidget *parent = nullptr);
    ~ModelDisplayPage() override;
    const Controls &controls() const { return controls_; }
    void setSelectedModelHeader(const QString &path, const QString &displayName, bool needsRuntime = false);
  signals:
    void modelLoaded(const QString &path);
    void loadFailed(const QString &error);
  private:
    Controls controls_;
};
