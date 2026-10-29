// SPDX-License-Identifier: AGPL-3.0-only
#pragma once
#include "modelsessioncontroller.h"

namespace vision::app
{
struct WorkbenchDraft
{
    // Editors and evaluation requests use doubles. ModelConfig's float fields
    // are derived only by compose, retaining the existing backend contract.
    double confidence = .25, iou = .45;
    ModelConfig parameters;
    QString classFilterText, labelsPath;
    int cameraIndex = 0;
    StereoView stereoView = StereoView::Full;
    bool automaticExport = false;
};
class WorkbenchConfiguration final : public QObject
{
    Q_OBJECT
  public:
    explicit WorkbenchConfiguration(QObject *parent = nullptr) : QObject(parent) {}
    WorkbenchDraft snapshot() const { return draft_; }
    void replaceDraft(const WorkbenchDraft &draft) { draft_ = draft; emit changed(); }
    void setLabels(const QStringList &labels, const QString &path)
    { draft_.parameters.labels = labels; draft_.labelsPath = path; emit changed(); }
    ModelConfig compose(const ModelSessionSnapshot &model, bool convertedProfileAvailable) const;
  signals:
    void changed();
  private:
    WorkbenchDraft draft_;
};
}
